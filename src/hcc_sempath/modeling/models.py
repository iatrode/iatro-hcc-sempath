from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file as load_safetensors_file
from timm.models.vision_transformer import checkpoint_filter_fn

from hcc_sempath.spatial_schema import (
    CELL_INSTANCE_DENSITY,
    DEFAULT_SPATIAL_COMPONENTS,
    spatial_component_specs,
)


# Soft ceiling applied to prototype/attribute logits before any softmax/sigmoid.
# A finite tanh bound keeps probabilities away from the saturated 0.9-0.99 band
# at training and deployment time. Shared by the loss modules so student and
# teacher responses use the same response scale. The default limit of 4.0 caps
# attribute sigmoids at ~0.982 and 4-class L1 softmax at ~0.999.
LOGIT_LIMIT = 4.0
PROB_EPS = 1e-4
STUDENT_BACKBONE_NAME = "vit_small_patch14_dinov2"
STUDENT_IMAGE_SIZE = 224
STUDENT_PATCH_SIZE = 14
SPATIAL_OUTPUT_STRIDE = 7
SPATIAL_PATCH_PADDING = 4
STUDENT_PRETRAINED_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "pretrained"
STUDENT_PRETRAINED_ARTIFACTS = (
    (
        STUDENT_PRETRAINED_DIR / "dinov2_vits14_pretrain.safetensors",
        "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081",
    ),
    (
        STUDENT_PRETRAINED_DIR / "dinov2_vits14_pretrain.pth",
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    ),
)


def _resolve_fixed_student_pretraining(
    artifacts: Sequence[tuple[Path, str]] = STUDENT_PRETRAINED_ARTIFACTS,
) -> tuple[Path, str]:
    for path, digest in artifacts:
        if path.is_file():
            return path, digest
    return artifacts[0]


STUDENT_PRETRAINED_PATH, STUDENT_PRETRAINED_SHA256 = _resolve_fixed_student_pretraining()


def _read_fixed_student_pretraining(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_safetensors_file(str(path), device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("model", state)


def _load_fixed_student_pretraining(backbone: nn.Module) -> None:
    if not STUDENT_PRETRAINED_PATH.is_file():
        expected = ", ".join(str(path) for path, _ in STUDENT_PRETRAINED_ARTIFACTS)
        raise FileNotFoundError(
            f"fixed DINOv2-S/14 pretrained weight is missing; expected one of: {expected}"
        )
    with STUDENT_PRETRAINED_PATH.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != STUDENT_PRETRAINED_SHA256:
        raise ValueError(
            "fixed DINOv2-S/14 pretrained weight checksum mismatch: "
            f"expected={STUDENT_PRETRAINED_SHA256} got={digest} path={STUDENT_PRETRAINED_PATH}"
        )
    state = _read_fixed_student_pretraining(STUDENT_PRETRAINED_PATH)
    state = checkpoint_filter_fn(state, backbone)
    backbone.load_state_dict(state, strict=True)


class StudentEncoder(nn.Module):
    """Lightweight tile encoder that produces the reusable HCC embedding."""

    def __init__(
        self,
        backbone_name: str = STUDENT_BACKBONE_NAME,
        embedding_dim: int = 256,
        pretrained: bool = True,
        projector_type: str = "linear",
        projector_hidden_dim: int = 2048,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        backbone_kwargs = {
            "pretrained": False,
            "num_classes": 0,
            "global_pool": "token" if backbone_name == STUDENT_BACKBONE_NAME else "avg",
        }
        if backbone_name == STUDENT_BACKBONE_NAME:
            backbone_kwargs["img_size"] = STUDENT_IMAGE_SIZE
        self.backbone = timm.create_model(backbone_name, **backbone_kwargs)
        if pretrained:
            if backbone_name != STUDENT_BACKBONE_NAME:
                raise ValueError(f"pretraining is fixed to {STUDENT_BACKBONE_NAME}, got {backbone_name}")
            _load_fixed_student_pretraining(self.backbone)
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        student_dim = int(self.backbone.num_features)
        if projector_type == "linear":
            self.projector = nn.Sequential(
                nn.LayerNorm(student_dim),
                nn.Linear(student_dim, embedding_dim),
            )
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.LayerNorm(student_dim),
                nn.Linear(student_dim, int(projector_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(projector_hidden_dim), embedding_dim),
            )
        else:
            raise ValueError(f"unsupported projector_type: {projector_type}")
        self.embedding_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projector(features)

    def forward_with_patch_tokens(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Return the unchanged global embedding plus spatial ViT patch tokens."""
        tokens = self.backbone.forward_features(images)
        if tokens.ndim != 3:
            raise ValueError(f"ROI-guided L2 requires token-shaped backbone features, got {tuple(tokens.shape)}")
        global_features = self.backbone.forward_head(tokens, pre_logits=True)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, prefix_tokens:]
        grid = tuple(int(value) for value in self.backbone.patch_embed.grid_size)
        if patch_tokens.shape[1] != grid[0] * grid[1]:
            raise ValueError(
                f"patch-token/grid mismatch: tokens={patch_tokens.shape[1]} grid={grid[0]}x{grid[1]}"
            )
        return self.projector(global_features), patch_tokens, grid


class TeacherProjectionHead(nn.Module):
    """Teacher-specific alignment head used during distillation."""

    def __init__(
        self,
        embedding_dim: int,
        teacher_dim: int,
        head_type: str = "linear",
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if head_type == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, teacher_dim),
            )
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), teacher_dim),
            )
        else:
            raise ValueError(f"unsupported teacher_head_type: {head_type}")

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


def _pointwise_conv_as_linear(
    convolution: nn.Conv2d,
    features: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a 1x1 convolution as one GEMM without changing its state."""

    if convolution.kernel_size != (1, 1) or convolution.groups != 1:
        raise ValueError("pointwise linear route requires an ungrouped 1x1 convolution")
    channels_last = features.permute(0, 2, 3, 1)
    projected = F.linear(
        channels_last,
        convolution.weight[:, :, 0, 0],
        convolution.bias,
    )
    return projected.permute(0, 3, 1, 2)


def _depthwise_conv_as_shift_sum(
    convolution: nn.Conv2d,
    features: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the fixed 3x3 depthwise kernel without fragmented cuDNN calls."""

    if (
        convolution.kernel_size != (3, 3)
        or convolution.groups != features.shape[1]
        or convolution.in_channels != convolution.out_channels
    ):
        raise ValueError("shift-sum route requires a channel-preserving 3x3 depthwise convolution")
    dilation_h, dilation_w = convolution.dilation
    padding_h, padding_w = convolution.padding
    if (padding_h, padding_w) != (dilation_h, dilation_w):
        raise ValueError("shift-sum route expects same-size dilated padding")
    padded = F.pad(
        features,
        (padding_w, padding_w, padding_h, padding_h),
    )
    height, width = features.shape[-2:]
    weight = convolution.weight[:, 0]
    result = None
    for kernel_row in range(3):
        row_start = kernel_row * dilation_h
        for kernel_col in range(3):
            col_start = kernel_col * dilation_w
            shifted = padded[
                :,
                :,
                row_start : row_start + height,
                col_start : col_start + width,
            ]
            term = shifted * weight[
                :,
                kernel_row,
                kernel_col,
            ].view(1, -1, 1, 1)
            result = term if result is None else result + term
    assert result is not None
    if convolution.bias is not None:
        result = result + convolution.bias.view(1, -1, 1, 1)
    return result


class SpatialContextBlock(nn.Module):
    """Residual multi-cell context at the dense L2 output grid."""

    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        groups = min(32, dim)
        while dim % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, dim)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=dim,
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.norm(features)
        features = F.gelu(features)
        features = _depthwise_conv_as_shift_sum(
            self.depthwise,
            features,
        )
        features = _pointwise_conv_as_linear(
            self.pointwise,
            features,
        )
        return residual + features


class SpatialMorphometryHead(nn.Module):
    """Nine-component class-routed spatial morphometry.

    The local branch reuses the pretrained 14x14 DINO patch projection with a
    seven-pixel stride. This preserves the intended cell-scale observation
    window while doubling coordinate sampling density. Final Transformer tokens
    are fused back in so vessels, ducts, vacuoles, and other structures may be
    recognized through context spanning multiple local cells.

    The instance head is decoded only for countable components. Point/circle
    marks provide one centre; connected brush support provides one centre for
    a large discrete structure even when it was completed with several
    overlapping pen strokes, but never for continuous areas or pigment. The
    second head represents density for cell components and occupied-area
    evidence for the remaining area-capable components.
    """

    def __init__(
        self,
        student_dim: int,
        component_count: int,
        spatial_dim: int,
        *,
        output_stride: int = SPATIAL_OUTPUT_STRIDE,
        patch_padding: int = SPATIAL_PATCH_PADDING,
    ) -> None:
        super().__init__()
        if component_count <= 0:
            raise ValueError(f"component_count must be positive, got {component_count}")
        if output_stride <= 0:
            raise ValueError(f"output_stride must be positive, got {output_stride}")
        self.component_count = int(component_count)
        self.output_stride = int(output_stride)
        self.patch_padding = int(patch_padding)
        if self.component_count == len(DEFAULT_SPATIAL_COMPONENTS):
            specs = spatial_component_specs(DEFAULT_SPATIAL_COMPONENTS)
            instance_valid = torch.tensor(
                [
                    spec.supports_instance_count
                    for spec in specs
                ],
                dtype=torch.bool,
            )
            density_valid = torch.tensor(
                [spec.supports_density for spec in specs],
                dtype=torch.bool,
            )
        else:
            instance_valid = torch.ones(
                self.component_count,
                dtype=torch.bool,
            )
            density_valid = torch.ones(
                self.component_count,
                dtype=torch.bool,
            )
        self.register_buffer(
            "instance_valid",
            instance_valid,
            persistent=True,
        )
        self.register_buffer(
            "density_valid",
            density_valid,
            persistent=False,
        )
        groups = min(32, spatial_dim)
        while spatial_dim % groups:
            groups -= 1
        self.local_projection = nn.Conv2d(student_dim, spatial_dim, kernel_size=1)
        self.semantic_projection = nn.Conv2d(student_dim, spatial_dim, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(spatial_dim * 2, spatial_dim, kernel_size=1),
            nn.GroupNorm(groups, spatial_dim),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            SpatialContextBlock(spatial_dim, dilation=1),
            SpatialContextBlock(spatial_dim, dilation=2),
            SpatialContextBlock(spatial_dim, dilation=4),
        )
        self.register_buffer(
            "instance_prototypes",
            torch.zeros(self.component_count, spatial_dim),
            persistent=True,
        )
        self.register_buffer(
            "instance_negative_prototypes",
            torch.zeros(self.component_count, spatial_dim),
            persistent=True,
        )
        self.register_buffer(
            "instance_implicit_negative_prototypes",
            torch.zeros(self.component_count, spatial_dim),
            persistent=True,
        )
        self.register_buffer(
            "measurement_prototypes",
            torch.zeros(self.component_count, spatial_dim),
            persistent=True,
        )
        self.register_buffer(
            "measurement_negative_prototypes",
            torch.zeros(self.component_count, spatial_dim),
            persistent=True,
        )
        self.register_buffer(
            "measurement_implicit_negative_prototypes",
            torch.zeros(self.component_count, spatial_dim),
            persistent=True,
        )
        for name in (
            "instance_prototype_counts",
            "instance_negative_prototype_counts",
            "instance_implicit_negative_prototype_counts",
            "measurement_prototype_counts",
            "measurement_negative_prototype_counts",
            "measurement_implicit_negative_prototype_counts",
        ):
            self.register_buffer(
                name,
                torch.zeros(self.component_count),
                persistent=True,
            )
        self.instance_log_temperature = nn.Parameter(
            torch.full((self.component_count,), math.log(0.1))
        )
        self.measurement_log_temperature = nn.Parameter(
            torch.full((self.component_count,), math.log(0.1))
        )
        self.instance_bias = nn.Parameter(
            torch.full((self.component_count,), -2.19)
        )
        self.measurement_bias = nn.Parameter(
            torch.full((self.component_count,), -2.19)
        )

    @staticmethod
    def _masked_pair_centroid_sums(
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return additive sufficient statistics for exact bank centroids.

        Every supervised tile/component pair contributes one local observation
        irrespective of marked area. Keeping the unnormalised sum makes the
        result invariant to how the fixed expert bank is split into compute
        batches.
        """

        normalized = F.normalize(features.detach().float(), dim=1)
        selected = mask.to(device=features.device, dtype=torch.bool)
        selected_float = selected.to(dtype=normalized.dtype)
        cell_counts = selected_float.flatten(2).sum(dim=2)
        pair_valid = cell_counts > 0
        pair_sums = torch.einsum(
            "bdhw,bkhw->bkd",
            normalized,
            selected_float,
        )
        pair_centroids = pair_sums / cell_counts.clamp_min(1).unsqueeze(-1)
        sums = (
            pair_centroids
            * pair_valid.to(dtype=pair_centroids.dtype).unsqueeze(-1)
        ).sum(dim=0)
        counts = pair_valid.sum(dim=0).to(dtype=torch.float32)
        return sums, counts

    def prototype_observation_sums(
        self,
        features: torch.Tensor,
        *,
        point_centers: torch.Tensor,
        brush_bag_ids: torch.Tensor,
        area_positive: torch.Tensor,
        explicit_negative: torch.Tensor,
        implicit_negative: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Collect exact local-anchor sufficient statistics for one chunk."""

        countable = self.instance_valid.view(1, -1, 1, 1)
        density = self.density_valid.view(1, -1, 1, 1)
        instance_positive = (point_centers > 0) & countable
        measurement_positive = (
            ((point_centers > 0) & density)
            | (brush_bag_ids > 0)
            | area_positive.to(dtype=torch.bool)
        )
        explicit = explicit_negative.to(dtype=torch.bool)
        implicit = implicit_negative.to(dtype=torch.bool)
        positive_support = (
            (point_centers > 0)
            | (brush_bag_ids > 0)
            | area_positive.to(dtype=torch.bool)
        )
        full_implicit = implicit.flatten(2).all(dim=2)
        instance_pair_valid = (
            positive_support.flatten(2).any(dim=2)
            | full_implicit
        ).view(*full_implicit.shape, 1, 1)
        measurement_pair_valid = (
            measurement_positive.flatten(2).any(dim=2)
            | full_implicit
        ).view(*full_implicit.shape, 1, 1)
        return {
            "instance": self._masked_pair_centroid_sums(
                features,
                instance_positive,
            ),
            "measurement": self._masked_pair_centroid_sums(
                features,
                measurement_positive,
            ),
            "instance_negative": self._masked_pair_centroid_sums(
                features,
                explicit & countable,
            ),
            "measurement_negative": self._masked_pair_centroid_sums(
                features,
                explicit,
            ),
            "instance_implicit_negative": (
                self._masked_pair_centroid_sums(
                    features,
                    implicit & countable & instance_pair_valid,
                )
            ),
            "measurement_implicit_negative": (
                self._masked_pair_centroid_sums(
                    features,
                    implicit & measurement_pair_valid,
                )
            ),
        }

    @torch.no_grad()
    def replace_prototypes(
        self,
        observations: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Replace local anchors with exact current-bank centroids."""

        expected = {
            "instance",
            "measurement",
            "instance_negative",
            "measurement_negative",
            "instance_implicit_negative",
            "measurement_implicit_negative",
        }
        if set(observations) != expected:
            raise ValueError(
                "spatial prototype observations differ from the fixed "
                f"contract: got={sorted(observations)}"
            )
        for name, (sums, counts) in observations.items():
            destination = getattr(self, f"{name}_prototypes")
            destination_counts = getattr(self, f"{name}_prototype_counts")
            sums = sums.to(device=destination.device, dtype=torch.float32)
            counts = counts.to(
                device=destination_counts.device,
                dtype=torch.float32,
            )
            if sums.shape != destination.shape:
                raise ValueError(
                    f"{name} prototype shape mismatch: "
                    f"got={tuple(sums.shape)} expected={tuple(destination.shape)}"
                )
            if counts.shape != destination_counts.shape:
                raise ValueError(
                    f"{name} prototype-count shape mismatch: "
                    f"got={tuple(counts.shape)} "
                    f"expected={tuple(destination_counts.shape)}"
                )
            normalized = F.normalize(
                sums / counts.clamp_min(1).unsqueeze(-1),
                dim=1,
            )
            destination.copy_(
                torch.where(
                    (counts > 0).unsqueeze(1),
                    normalized,
                    torch.zeros_like(normalized),
                )
            )
            destination_counts.copy_(counts)

    @staticmethod
    def _prototype_response(
        features: torch.Tensor,
        positive: torch.Tensor,
        positive_counts: torch.Tensor,
        negative: torch.Tensor,
        negative_counts: torch.Tensor,
        implicit_negative: torch.Tensor,
        implicit_negative_counts: torch.Tensor,
        log_temperature: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        normalized_features = F.normalize(features.float(), dim=1)
        positive_similarity = torch.einsum(
            "bdhw,kd->bkhw",
            normalized_features,
            F.normalize(positive.float(), dim=1),
        )
        negative_similarity = torch.einsum(
            "bdhw,kd->bkhw",
            normalized_features,
            F.normalize(negative.float(), dim=1),
        )
        implicit_negative_similarity = torch.einsum(
            "bdhw,kd->bkhw",
            normalized_features,
            F.normalize(implicit_negative.float(), dim=1),
        )
        positive_ready = (positive_counts > 0).view(1, -1, 1, 1)
        negative_ready = (negative_counts > 0).view(1, -1, 1, 1)
        implicit_negative_ready = (
            implicit_negative_counts > 0
        ).view(1, -1, 1, 1)
        # Explicit expert negatives define the decision boundary whenever they
        # exist. Weak implicit background is a fallback, not a centroid pool
        # that may dilute the explicit evidence. "Weak" refers to its direct
        # per-cell loss weight. This full-bank pair-averaged anchor is a contrastive
        # coordinate, not another set of strong negative labels, so its
        # similarity must not be scaled by the direct-loss coefficient.
        negative_response = torch.where(
            negative_ready,
            negative_similarity,
            torch.where(
                implicit_negative_ready,
                implicit_negative_similarity,
                torch.zeros_like(implicit_negative_similarity),
            ),
        )
        response = torch.where(
            positive_ready,
            positive_similarity,
            torch.zeros_like(positive_similarity),
        ) - negative_response
        temperature = log_temperature.exp().clamp(0.03, 1.0).view(
            1, -1, 1, 1
        )
        return bounded_logits(
            response / temperature + bias.view(1, -1, 1, 1)
        )

    def prototype_logits(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instance_logits = self._prototype_response(
            features,
            self.instance_prototypes,
            self.instance_prototype_counts,
            self.instance_negative_prototypes,
            self.instance_negative_prototype_counts,
            self.instance_implicit_negative_prototypes,
            self.instance_implicit_negative_prototype_counts,
            self.instance_log_temperature,
            self.instance_bias,
        ).masked_fill(
            ~self.instance_valid.view(1, -1, 1, 1),
            -20.0,
        )
        measurement_logits = self._prototype_response(
            features,
            self.measurement_prototypes,
            self.measurement_prototype_counts,
            self.measurement_negative_prototypes,
            self.measurement_negative_prototype_counts,
            self.measurement_implicit_negative_prototypes,
            self.measurement_implicit_negative_prototype_counts,
            self.measurement_log_temperature,
            self.measurement_bias,
        )
        return instance_logits, measurement_logits

    def forward(
        self,
        images: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int],
        patch_projection: nn.Conv2d,
        *,
        detach_backbone: bool,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        weight = patch_projection.weight.detach() if detach_backbone else patch_projection.weight
        bias = patch_projection.bias
        if bias is not None and detach_backbone:
            bias = bias.detach()
        local_tokens = F.conv2d(
            images,
            weight,
            bias,
            stride=self.output_stride,
            padding=self.patch_padding,
        )
        batch, token_count, dim = patch_tokens.shape
        if token_count != patch_grid[0] * patch_grid[1]:
            raise ValueError(
                f"patch-token/grid mismatch: tokens={token_count} grid={patch_grid[0]}x{patch_grid[1]}"
            )
        semantic_tokens = patch_tokens.detach() if detach_backbone else patch_tokens
        semantic_map = semantic_tokens.transpose(1, 2).reshape(
            batch,
            dim,
            patch_grid[0],
            patch_grid[1],
        )
        semantic_map = F.interpolate(
            semantic_map,
            size=local_tokens.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat(
            [
                _pointwise_conv_as_linear(
                    self.local_projection,
                    local_tokens,
                ),
                _pointwise_conv_as_linear(
                    self.semantic_projection,
                    semantic_map,
                ),
            ],
            dim=1,
        )
        features = self.context(
            F.gelu(
                self.fusion[1](
                    _pointwise_conv_as_linear(
                        self.fusion[0],
                        features,
                    )
                )
            )
        )
        instance_logits, abundance_logits = self.prototype_logits(features)
        outputs = {
            "l2_instance_logits": instance_logits,
            "l2_instance_probabilities": torch.sigmoid(instance_logits),
            "l2_instance_valid": self.instance_valid,
            "l2_abundance_logits": abundance_logits,
            "l2_abundance_probabilities": torch.sigmoid(abundance_logits),
        }
        if return_features:
            outputs["l2_spatial_features"] = features
        return outputs


class TeacherL2PrototypeState(nn.Module):
    """Frozen-teacher global component centroids used only for adjudication."""

    def __init__(self, component_count: int, teacher_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "prototypes",
            torch.zeros(component_count, teacher_dim),
            persistent=True,
        )
        self.register_buffer(
            "counts",
            torch.zeros(component_count),
            persistent=True,
        )

    @torch.no_grad()
    def replace(
        self,
        sums: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Install exact frozen-teacher centroids from the complete L2 bank."""

        sums = sums.to(device=self.prototypes.device, dtype=torch.float32)
        counts = counts.to(device=self.counts.device, dtype=torch.float32)
        if sums.shape != self.prototypes.shape:
            raise ValueError(
                "teacher L2 prototype shape mismatch: "
                f"got={tuple(sums.shape)} "
                f"expected={tuple(self.prototypes.shape)}"
            )
        if counts.shape != self.counts.shape:
            raise ValueError(
                "teacher L2 prototype-count shape mismatch: "
                f"got={tuple(counts.shape)} expected={tuple(self.counts.shape)}"
            )
        normalized = F.normalize(
            sums / counts.clamp_min(1).unsqueeze(-1),
            dim=1,
        )
        self.prototypes.copy_(
            torch.where(
                (counts > 0).unsqueeze(1),
                normalized,
                torch.zeros_like(normalized),
            )
        )
        self.counts.copy_(counts)


class HCCSemPathModel(nn.Module):
    """Teacher-distilled HCC encoder with L1 and spatial L2 output heads."""

    def __init__(
        self,
        backbone_name: str = STUDENT_BACKBONE_NAME,
        embedding_dim: int = 256,
        teacher_dims: dict[str, int] | None = None,
        pretrained: bool = True,
        projector_type: str = "linear",
        projector_hidden_dim: int = 2048,
        teacher_head_type: str = "linear",
        grad_checkpointing: bool = False,
        l1_num_classes: int = 0,
        spatial_num_components: int = 0,
        spatial_dim: int = 256,
        spatial_output_stride: int = SPATIAL_OUTPUT_STRIDE,
    ) -> None:
        super().__init__()
        if teacher_dims is None:
            teacher_dims = {"teacher": embedding_dim}
        self.encoder = StudentEncoder(
            backbone_name=backbone_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            projector_type=projector_type,
            projector_hidden_dim=projector_hidden_dim,
            grad_checkpointing=grad_checkpointing,
        )
        self.teacher_heads = nn.ModuleDict(
            {
                name: TeacherProjectionHead(
                    embedding_dim,
                    int(dim),
                    head_type=teacher_head_type,
                    hidden_dim=projector_hidden_dim,
                )
                for name, dim in teacher_dims.items()
            }
        )
        self.l1_num_classes = int(l1_num_classes)
        self.register_buffer(
            "l1_prototypes",
            (
                torch.zeros(self.l1_num_classes, embedding_dim)
                if self.l1_num_classes > 0
                else None
            ),
            persistent=True,
        )
        self.register_buffer(
            "l1_prototype_counts",
            (
                torch.zeros(self.l1_num_classes)
                if self.l1_num_classes > 0
                else None
            ),
            persistent=True,
        )
        self.l1_log_temperature = (
            nn.Parameter(torch.tensor(math.log(0.1)))
            if self.l1_num_classes > 0
            else None
        )
        self.spatial_num_components = int(spatial_num_components)
        self.spatial_head = (
            SpatialMorphometryHead(
                student_dim=int(self.encoder.backbone.num_features),
                component_count=self.spatial_num_components,
                spatial_dim=int(spatial_dim),
                output_stride=int(spatial_output_stride),
            )
            if self.spatial_num_components > 0
            else None
        )
        self.register_buffer(
            "global_l2_prototypes",
            (
                torch.zeros(self.spatial_num_components, embedding_dim)
                if self.spatial_num_components > 0
                else None
            ),
            persistent=True,
        )
        self.register_buffer(
            "global_l2_prototype_counts",
            (
                torch.zeros(self.spatial_num_components)
                if self.spatial_num_components > 0
                else None
            ),
            persistent=True,
        )
        self.teacher_l2_prototypes = nn.ModuleDict(
            {
                name: TeacherL2PrototypeState(
                    self.spatial_num_components,
                    int(dim),
                )
                for name, dim in teacher_dims.items()
            }
            if self.spatial_num_components > 0
            else {}
        )

    @property
    def teacher_names(self) -> list[str]:
        return list(self.teacher_heads.keys())

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def project_teachers(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(embedding) for name, head in self.teacher_heads.items()}

    @torch.no_grad()
    def replace_l1_prototypes(
        self,
        sums: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Install exact current-student L1 centroids from the full bank."""

        if self.l1_prototypes is None or self.l1_prototype_counts is None:
            return
        sums = sums.to(device=self.l1_prototypes.device, dtype=torch.float32)
        counts = counts.to(
            device=self.l1_prototype_counts.device,
            dtype=torch.float32,
        )
        if sums.shape != self.l1_prototypes.shape:
            raise ValueError(
                f"L1 prototype shape mismatch: got={tuple(sums.shape)} "
                f"expected={tuple(self.l1_prototypes.shape)}"
            )
        if counts.shape != self.l1_prototype_counts.shape:
            raise ValueError(
                "L1 prototype-count shape mismatch: "
                f"got={tuple(counts.shape)} "
                f"expected={tuple(self.l1_prototype_counts.shape)}"
            )
        if bool((counts <= 0).any()):
            raise ValueError(
                "the complete expert bank must contain every L1 class"
            )
        self.l1_prototypes.copy_(
            F.normalize(
                sums / counts.unsqueeze(-1),
                dim=1,
            )
        )
        self.l1_prototype_counts.copy_(counts)

    def l1_prototype_logits(
        self,
        embedding_norm: torch.Tensor,
    ) -> torch.Tensor:
        if self.l1_prototypes is None or self.l1_log_temperature is None:
            raise RuntimeError("L1 prototype readout is not configured")
        similarity = self.l1_prototype_similarity(embedding_norm)
        temperature = self.l1_log_temperature.exp().clamp(0.03, 1.0)
        return bounded_logits(similarity / temperature)

    def l1_prototype_similarity(
        self,
        embedding_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw cosine coordinate shared by L1 and PAMT-D.

        The deployable L1 classifier applies its learned temperature in
        :meth:`l1_prototype_logits`. PAMT-D instead applies its fixed research
        temperature exactly once to this common cosine coordinate.
        """

        if self.l1_prototypes is None:
            raise RuntimeError("L1 prototype readout is not configured")
        similarity = normalized_prototype_logits(
            embedding_norm,
            self.l1_prototypes,
        )
        ready = (self.l1_prototype_counts > 0).view(1, -1)
        similarity = torch.where(
            ready,
            similarity,
            torch.zeros_like(similarity),
        )
        return similarity

    @torch.no_grad()
    def replace_global_l2_prototypes(
        self,
        student_sums: torch.Tensor,
        counts: torch.Tensor,
        teacher_sums: dict[str, torch.Tensor],
    ) -> None:
        """Install exact global L2 reliability anchors from the full bank."""

        if (
            self.global_l2_prototypes is None
            or self.global_l2_prototype_counts is None
        ):
            return
        if set(teacher_sums) != set(self.teacher_l2_prototypes):
            raise ValueError(
                "teacher L2 anchor names differ from model teachers: "
                f"got={sorted(teacher_sums)} "
                f"expected={sorted(self.teacher_l2_prototypes)}"
            )
        student_sums = student_sums.to(
            device=self.global_l2_prototypes.device,
            dtype=torch.float32,
        )
        counts = counts.to(
            device=self.global_l2_prototype_counts.device,
            dtype=torch.float32,
        )
        if student_sums.shape != self.global_l2_prototypes.shape:
            raise ValueError(
                "global L2 prototype shape mismatch: "
                f"got={tuple(student_sums.shape)} "
                f"expected={tuple(self.global_l2_prototypes.shape)}"
            )
        if counts.shape != self.global_l2_prototype_counts.shape:
            raise ValueError(
                "global L2 prototype-count shape mismatch: "
                f"got={tuple(counts.shape)} "
                f"expected={tuple(self.global_l2_prototype_counts.shape)}"
            )
        if bool((counts <= 0).any()):
            raise ValueError(
                "the complete spatial expert bank must contain every L2 "
                "component"
            )
        self.global_l2_prototypes.copy_(
            F.normalize(
                student_sums / counts.unsqueeze(-1),
                dim=1,
            )
        )
        self.global_l2_prototype_counts.copy_(counts)
        for name, state in self.teacher_l2_prototypes.items():
            state.replace(teacher_sums[name], counts)

    def global_l2_response(
        self,
        embedding_norm: torch.Tensor,
        *,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        if (
            self.global_l2_prototypes is None
            or self.global_l2_prototype_counts is None
        ):
            return embedding_norm.new_zeros((embedding_norm.shape[0], 0))
        logits = bounded_logits(
            normalized_prototype_logits(
                embedding_norm,
                self.global_l2_prototypes,
            )
            / float(temperature)
        )
        ready = (self.global_l2_prototype_counts > 0).view(1, -1)
        return torch.where(
            ready,
            clamp_probability(torch.sigmoid(logits)),
            torch.full_like(logits, 0.5),
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        spatial_detach_backbone: bool = False,
        return_spatial_features: bool = False,
        run_spatial: bool = True,
        spatial_sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.spatial_head is None or not run_spatial:
            embedding = self.encode(images)
            patch_tokens = None
            patch_grid = None
        else:
            embedding, patch_tokens, patch_grid = self.encoder.forward_with_patch_tokens(images)
        embedding_norm = F.normalize(embedding, dim=-1)
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "embedding": embedding,
            "embedding_norm": embedding_norm,
            "teacher_outputs": self.project_teachers(embedding),
        }
        if patch_tokens is not None and patch_grid is not None and self.spatial_head is not None:
            spatial_images = images
            spatial_patch_tokens = patch_tokens
            if spatial_sample_mask is not None:
                spatial_sample_mask = spatial_sample_mask.to(
                    device=images.device,
                    dtype=torch.bool,
                )
                if spatial_sample_mask.shape != (images.shape[0],):
                    raise ValueError(
                        "spatial_sample_mask must have shape "
                        f"({images.shape[0]},), got "
                        f"{tuple(spatial_sample_mask.shape)}"
                    )
                if not torch.compiler.is_compiling():
                    has_spatial_sample = spatial_sample_mask.any()
                    if spatial_sample_mask.device.type == "cuda":
                        torch._assert_async(
                            has_spatial_sample,
                            "spatial_sample_mask must select at least one sample",
                        )
                    elif not bool(has_spatial_sample):
                        raise ValueError(
                            "spatial_sample_mask must select at least one sample"
                        )
                spatial_images = images[spatial_sample_mask]
                spatial_patch_tokens = patch_tokens[spatial_sample_mask]
            outputs.update(
                self.spatial_head(
                    spatial_images,
                    spatial_patch_tokens,
                    patch_grid,
                    self.encoder.backbone.patch_embed.proj,
                    detach_backbone=spatial_detach_backbone,
                    return_features=return_spatial_features,
                )
            )
        if self.l1_prototypes is not None:
            l1_similarity = self.l1_prototype_similarity(embedding_norm)
            assert self.l1_log_temperature is not None
            l1_temperature = self.l1_log_temperature.exp().clamp(0.03, 1.0)
            l1_logits = bounded_logits(l1_similarity / l1_temperature)
            outputs["l1_similarity"] = l1_similarity
            outputs["l1_logits"] = l1_logits
            outputs["l1_probabilities"] = F.softmax(l1_logits, dim=-1)
        return outputs


class ToyTeacherEncoder(nn.Module):
    """Small frozen teacher used only for local smoke tests."""

    def __init__(self, teacher_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, teacher_dim)
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images).flatten(1)
        return self.head(x)


def normalized_prototype_logits(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    return features @ prototypes.transpose(0, 1)


def bounded_logits(value: torch.Tensor, limit: float = LOGIT_LIMIT) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0:
        raise ValueError(f"logit limit must be positive, got {limit}")
    value = value.float()
    return limit * torch.tanh(value / limit)


def clamp_probability(value: torch.Tensor, *, normalize: bool = False, eps: float = PROB_EPS) -> torch.Tensor:
    """Pull a probability/target tensor away from the degenerate 0/1 boundary.

    Upcasts to float32. Categorical targets are projected to the probability
    simplex and mixed with an ``eps`` uniform floor, so they remain normalized
    as well as finite. Independent probabilities are clamped directly.
    """
    value = value.float()
    if normalize:
        if value.shape[-1] <= 0 or eps * value.shape[-1] >= 1.0:
            raise ValueError(
                "categorical probability floor is incompatible with class count"
            )
        value = value.clamp_min(0.0)
        denominator = value.sum(dim=-1, keepdim=True)
        invalid = (
            (denominator <= 0)
            | ~torch.isfinite(denominator)
        ).any()
        assert_async = getattr(torch, "_assert_async", None)
        if invalid.device.type == "cuda" and assert_async is not None:
            assert_async(
                ~invalid,
                "categorical probabilities must have positive finite mass",
            )
        elif bool(invalid):
            raise ValueError(
                "categorical probabilities must have positive finite mass"
            )
        value = value / denominator
        return value * (1.0 - eps * value.shape[-1]) + eps
    return value.clamp(eps, 1.0 - eps)


def canonical_payload_sha256(payload: object) -> str:
    """Hash a JSON-compatible scientific contract deterministically."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash normalized state-dict names, metadata, and exact tensor bytes."""

    normalized = {
        str(key).removeprefix("_orig_mod."): value
        for key, value in state.items()
    }
    digest = hashlib.sha256()
    for key in sorted(normalized):
        tensor = normalized[key].detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(
                order="C"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def validate_spatial_decoder_calibration(
    payload: dict,
    component_names: Sequence[str],
    *,
    expected_output_stride: int | None = None,
    expected_model_state_sha256: str | None = None,
    expected_research_contract_sha256: str | None = None,
    expected_optimizer_visible_contract_sha256: str | None = None,
    expected_supervision_assets_sha256: str | None = None,
) -> dict:
    """Validate and normalize the frozen spatial-analysis decoder contract."""

    names = [str(name) for name in component_names]
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != 1:
        raise ValueError("spatial decoder calibration must have version=1")
    calibrated_names = [
        str(name)
        for name in payload.get("spatial_component_names", [])
    ]
    if calibrated_names != names:
        raise ValueError(
            "spatial decoder calibration component order mismatch: "
            f"expected={names} got={calibrated_names}"
        )

    def thresholds(key: str) -> list[float]:
        values = [float(value) for value in payload.get(key, [])]
        if len(values) != len(names) or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                f"spatial decoder calibration {key} requires one finite "
                "value in [0, 1] per component"
            )
        return values

    kernels = [int(value) for value in payload.get("nms_kernel", [])]
    if len(kernels) != len(names) or any(
        value <= 0 or value % 2 == 0 for value in kernels
    ):
        raise ValueError(
            "spatial decoder calibration nms_kernel requires one positive "
            "odd value per component"
        )
    minimum_focus_cells = int(payload.get("minimum_focus_cells", 0))
    if minimum_focus_cells <= 0:
        raise ValueError(
            "spatial decoder calibration minimum_focus_cells must be positive"
        )
    output_stride = int(payload.get("spatial_output_stride", 0))
    if output_stride <= 0:
        raise ValueError(
            "spatial decoder calibration spatial_output_stride must be positive"
        )
    if (
        expected_output_stride is not None
        and output_stride != int(expected_output_stride)
    ):
        raise ValueError(
            "spatial decoder calibration output stride mismatch: "
            f"expected={expected_output_stride} got={output_stride}"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "spatial decoder calibration requires checkpoint provenance"
        )
    digest_fields = (
        "checkpoint_model_sha256",
        "research_contract_sha256",
        "validation_annotation_sha256",
        "validation_protocol_sha256",
        "validation_cohort_sha256",
        "optimizer_visible_contract_sha256",
        "supervision_assets_sha256",
    )
    normalized_provenance: dict[str, object] = {}
    for key in digest_fields:
        value = str(provenance.get(key, ""))
        if (
            len(value) != 64
            or value.lower() != value
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(
                "spatial decoder calibration provenance "
                f"{key} must be a lowercase SHA-256 digest"
            )
        normalized_provenance[key] = value
    for key in ("terminal_epoch", "expected_epochs"):
        value = int(provenance.get(key, -1))
        if value <= 0:
            raise ValueError(
                "spatial decoder calibration provenance "
                f"{key} must be positive"
            )
        normalized_provenance[key] = value
    if (
        normalized_provenance["terminal_epoch"]
        != normalized_provenance["expected_epochs"]
    ):
        raise ValueError(
            "spatial decoder calibration provenance is not terminal"
        )
    if (
        expected_model_state_sha256 is not None
        and normalized_provenance["checkpoint_model_sha256"]
        != expected_model_state_sha256
    ):
        raise ValueError(
            "spatial decoder calibration belongs to a different checkpoint"
        )
    if (
        expected_research_contract_sha256 is not None
        and normalized_provenance["research_contract_sha256"]
        != expected_research_contract_sha256
    ):
        raise ValueError(
            "spatial decoder calibration research contract mismatch"
        )
    if (
        expected_optimizer_visible_contract_sha256 is not None
        and normalized_provenance[
            "optimizer_visible_contract_sha256"
        ]
        != expected_optimizer_visible_contract_sha256
    ):
        raise ValueError(
            "spatial decoder calibration optimizer-visible contract mismatch"
        )
    if (
        expected_supervision_assets_sha256 is not None
        and normalized_provenance["supervision_assets_sha256"]
        != expected_supervision_assets_sha256
    ):
        raise ValueError(
            "spatial decoder calibration supervision-asset mismatch"
        )
    return {
        "version": 1,
        "spatial_component_names": names,
        "instance_threshold": thresholds("instance_threshold"),
        "abundance_threshold": thresholds("abundance_threshold"),
        "nms_kernel": kernels,
        "minimum_focus_cells": minimum_focus_cells,
        "spatial_output_stride": output_stride,
        "provenance": normalized_provenance,
    }


def _collapse_peak_plateaus(
    candidates: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Keep one deterministic representative per 8-connected max plateau."""

    candidate_cpu = candidates.detach().to(device="cpu", dtype=torch.bool)
    probability_cpu = probabilities.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    selected = torch.zeros_like(candidate_cpu)
    for batch_idx in range(candidate_cpu.shape[0]):
        for component_idx in range(candidate_cpu.shape[1]):
            mask = candidate_cpu[batch_idx, component_idx]
            visited = torch.zeros_like(mask)
            height, width = mask.shape
            for start_row, start_col in mask.nonzero().tolist():
                if bool(visited[start_row, start_col]):
                    continue
                stack = [(int(start_row), int(start_col))]
                visited[start_row, start_col] = True
                plateau: list[tuple[int, int]] = []
                while stack:
                    row, col = stack.pop()
                    plateau.append((row, col))
                    for delta_row in (-1, 0, 1):
                        for delta_col in (-1, 0, 1):
                            if delta_row == 0 and delta_col == 0:
                                continue
                            next_row = row + delta_row
                            next_col = col + delta_col
                            if not (
                                0 <= next_row < height
                                and 0 <= next_col < width
                            ):
                                continue
                            if (
                                bool(mask[next_row, next_col])
                                and not bool(visited[next_row, next_col])
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                best_row, best_col = max(
                    plateau,
                    key=lambda point: (
                        float(
                            probability_cpu[
                                batch_idx,
                                component_idx,
                                point[0],
                                point[1],
                            ]
                        ),
                        -point[0],
                        -point[1],
                    ),
                )
                selected[
                    batch_idx,
                    component_idx,
                    best_row,
                    best_col,
                ] = True
    return selected.to(device=candidates.device)


@torch.no_grad()
def decode_spatial_morphometry(
    outputs: dict[str, torch.Tensor],
    *,
    component_names: list[str] | tuple[str, ...] | None = None,
    calibration: dict | None = None,
    instance_threshold: float | Sequence[float] | torch.Tensor | None = None,
    abundance_threshold: float | Sequence[float] | torch.Tensor | None = None,
    output_stride: int | None = None,
    nms_kernel: int | Sequence[int] | None = None,
    minimum_focus_cells: int | None = None,
) -> dict[str, torch.Tensor | list[list[list[tuple[float, float, float]]]]]:
    """Decode L2 maps according to each component's measurement contract.

    Area-only components receive no fabricated instance count. Bile-pigment
    focus density is a thresholded connected-component descriptor, not a
    biological instance count or a directly supervised target.
    """

    instance_probability = outputs["l2_instance_probabilities"].float()
    abundance_probability = outputs["l2_abundance_probabilities"].float()
    if instance_probability.shape != abundance_probability.shape:
        raise ValueError(
            "instance/abundance map shape mismatch: "
            f"instance={tuple(instance_probability.shape)} "
            f"abundance={tuple(abundance_probability.shape)}"
        )
    component_count = int(instance_probability.shape[1])
    if component_names is None:
        if component_count == len(DEFAULT_SPATIAL_COMPONENTS):
            resolved_names = DEFAULT_SPATIAL_COMPONENTS
            specs = spatial_component_specs(resolved_names)
        else:
            resolved_names = tuple(
                f"synthetic-{index}" for index in range(component_count)
            )
            specs = spatial_component_specs(
                resolved_names,
                unknown_mode=CELL_INSTANCE_DENSITY,
            )
    else:
        resolved_names = tuple(str(name) for name in component_names)
        if len(resolved_names) != component_count:
            raise ValueError(
                "component_names/probability channel mismatch: "
                f"names={len(resolved_names)} channels={component_count}"
            )
        specs = spatial_component_specs(
            resolved_names,
            unknown_mode=CELL_INSTANCE_DENSITY,
        )
    explicit_decoder_values = any(
        value is not None
        for value in (
            instance_threshold,
            abundance_threshold,
            output_stride,
            nms_kernel,
            minimum_focus_cells,
        )
    )
    if calibration is not None:
        if explicit_decoder_values:
            raise ValueError(
                "pass either calibration or explicit decoder values, not both"
            )
        frozen = validate_spatial_decoder_calibration(
            calibration,
            resolved_names,
        )
        instance_threshold = frozen["instance_threshold"]
        abundance_threshold = frozen["abundance_threshold"]
        output_stride = int(frozen["spatial_output_stride"])
        nms_kernel = frozen["nms_kernel"]
        minimum_focus_cells = int(frozen["minimum_focus_cells"])
    elif not all(
        value is not None
        for value in (
            instance_threshold,
            abundance_threshold,
            output_stride,
            nms_kernel,
            minimum_focus_cells,
        )
    ):
        raise ValueError(
            "spatial analysis requires a frozen decoder calibration or all "
            "explicit decoder values"
        )
    assert instance_threshold is not None
    assert abundance_threshold is not None
    assert output_stride is not None
    assert nms_kernel is not None
    assert minimum_focus_cells is not None
    if minimum_focus_cells <= 0:
        raise ValueError(
            "minimum_focus_cells must be positive, "
            f"got {minimum_focus_cells}"
        )

    def component_threshold(
        value: float | Sequence[float] | torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        threshold = torch.as_tensor(
            value,
            device=instance_probability.device,
            dtype=torch.float32,
        ).view(-1)
        if threshold.numel() == 1:
            threshold = threshold.expand(component_count)
        if threshold.shape != (component_count,):
            raise ValueError(
                f"{label} must be scalar or have one value per component"
            )
        if not bool(torch.isfinite(threshold).all()) or bool(
            ((threshold < 0) | (threshold > 1)).any()
        ):
            raise ValueError(f"{label} values must be finite and in [0, 1]")
        return threshold.view(1, -1, 1, 1)

    if isinstance(nms_kernel, int):
        nms_kernels = [int(nms_kernel)] * component_count
    else:
        nms_kernels = [int(value) for value in nms_kernel]
    if len(nms_kernels) != component_count or any(
        value <= 0 or value % 2 == 0 for value in nms_kernels
    ):
        raise ValueError(
            "nms_kernel must be one positive odd value per component"
        )
    count_valid = torch.tensor(
        [spec.supports_instance_count for spec in specs],
        dtype=torch.bool,
        device=instance_probability.device,
    )
    density_valid = torch.tensor(
        [spec.supports_density for spec in specs],
        dtype=torch.bool,
        device=instance_probability.device,
    )
    area_valid = torch.tensor(
        [spec.supports_area for spec in specs],
        dtype=torch.bool,
        device=instance_probability.device,
    )
    focus_valid = torch.tensor(
        [spec.supports_focus_density for spec in specs],
        dtype=torch.bool,
        device=instance_probability.device,
    )
    pooled = torch.cat(
        [
            F.max_pool2d(
                instance_probability[:, index : index + 1],
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
            for index, kernel in enumerate(nms_kernels)
        ],
        dim=1,
    )
    instance_candidates = (
        instance_probability
        >= component_threshold(
            instance_threshold,
            label="instance_threshold",
        )
    ) & (
        instance_probability >= pooled
    ) & count_valid.view(1, -1, 1, 1)
    instance_mask = _collapse_peak_plateaus(
        instance_candidates,
        instance_probability,
    )
    raw_instance_counts = instance_mask.flatten(2).sum(dim=2)
    instance_counts = raw_instance_counts.float().masked_fill(
        ~count_valid.view(1, -1),
        float("nan"),
    )
    abundance_mass = abundance_probability.flatten(2).sum(dim=2)
    mean_abundance = abundance_probability.flatten(2).mean(dim=2)
    high_abundance_mask = (
        abundance_probability
        >= component_threshold(
            abundance_threshold,
            label="abundance_threshold",
        )
    )
    high_abundance_fraction = high_abundance_mask.flatten(2).float().mean(dim=2)
    density_mass = abundance_mass.masked_fill(
        ~density_valid.view(1, -1),
        float("nan"),
    )
    mean_density = mean_abundance.masked_fill(
        ~density_valid.view(1, -1),
        float("nan"),
    )
    area_fraction = high_abundance_fraction.masked_fill(
        ~area_valid.view(1, -1),
        float("nan"),
    )
    area_pixels = (
        high_abundance_mask.flatten(2).sum(dim=2).float()
        * float(output_stride) ** 2
    ).masked_fill(
        ~area_valid.view(1, -1),
        float("nan"),
    )

    focus_counts = torch.full(
        (instance_probability.shape[0], component_count),
        float("nan"),
        dtype=torch.float32,
        device=instance_probability.device,
    )
    for batch_idx in range(instance_probability.shape[0]):
        for component_idx in focus_valid.nonzero(as_tuple=False).flatten().tolist():
            mask = high_abundance_mask[batch_idx, component_idx].detach().cpu()
            visited = torch.zeros_like(mask, dtype=torch.bool)
            component_total = 0
            height, width = mask.shape
            for row in range(height):
                for col in range(width):
                    if not bool(mask[row, col]) or bool(visited[row, col]):
                        continue
                    stack = [(row, col)]
                    visited[row, col] = True
                    cell_count = 0
                    while stack:
                        current_row, current_col = stack.pop()
                        cell_count += 1
                        for delta_row in (-1, 0, 1):
                            for delta_col in (-1, 0, 1):
                                if delta_row == 0 and delta_col == 0:
                                    continue
                                next_row = current_row + delta_row
                                next_col = current_col + delta_col
                                if not (
                                    0 <= next_row < height
                                    and 0 <= next_col < width
                                ):
                                    continue
                                if bool(visited[next_row, next_col]) or not bool(
                                    mask[next_row, next_col]
                                ):
                                    continue
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                    if cell_count >= int(minimum_focus_cells):
                        component_total += 1
            focus_counts[batch_idx, component_idx] = float(component_total)
    decoded_pixel_area = (
        float(instance_probability.shape[-2])
        * float(instance_probability.shape[-1])
        * float(output_stride) ** 2
    )
    focus_density_per_megapixel = (
        focus_counts * (1_000_000.0 / decoded_pixel_area)
    )
    coordinates: list[list[list[tuple[float, float, float]]]] = []
    for batch_idx in range(instance_probability.shape[0]):
        per_component: list[list[tuple[float, float, float]]] = []
        for component_idx in range(instance_probability.shape[1]):
            rows, cols = instance_mask[batch_idx, component_idx].nonzero(
                as_tuple=True
            )
            component_points = [
                (
                    (float(col) + 0.5) * float(output_stride),
                    (float(row) + 0.5) * float(output_stride),
                    float(
                        instance_probability[
                            batch_idx,
                            component_idx,
                            row,
                            col,
                        ]
                    ),
                )
                for row, col in zip(
                    rows.tolist(),
                    cols.tolist(),
                    strict=True,
                )
            ]
            per_component.append(component_points)
        coordinates.append(per_component)
    return {
        "instance_mask": instance_mask,
        "instance_counts": instance_counts,
        "instance_count_valid": count_valid,
        "instance_coordinates": coordinates,
        "abundance_map": abundance_probability,
        "abundance_mass": abundance_mass,
        "mean_abundance": mean_abundance,
        "high_abundance_fraction": high_abundance_fraction,
        "density_mass": density_mass,
        "mean_density": mean_density,
        "density_valid": density_valid,
        "area_fraction": area_fraction,
        "area_pixels": area_pixels,
        "area_valid": area_valid,
        "focus_counts": focus_counts,
        "focus_density_per_megapixel": focus_density_per_megapixel,
        "focus_density_valid": focus_valid,
        "output_stride": torch.tensor(
            int(output_stride),
            device=instance_probability.device,
        ),
    }


def load_hcc_sempath_release(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[HCCSemPathModel, dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (
        config.get("format") != "hcc-sempath-l1-spatial-state-dict"
        or int(config.get("version", -1)) != 3
    ):
        raise ValueError(
            "unsupported HCC-SemPath release; expected "
            "hcc-sempath-l1-spatial-state-dict version 3"
        )
    model_config = config["model"]
    config["spatial_decoder_calibration"] = (
        validate_spatial_decoder_calibration(
            config.get("spatial_decoder_calibration"),
            config.get("spatial_component_names", []),
            expected_output_stride=int(
                model_config.get(
                    "spatial_output_stride",
                    SPATIAL_OUTPUT_STRIDE,
                )
            ),
        )
    )
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    provenance = config.get("training_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("release has no training provenance")
    expected_release_digest = str(
        provenance.get("release_model_sha256", "")
    )
    if (
        len(expected_release_digest) != 64
        or model_state_sha256(state) != expected_release_digest
    ):
        raise ValueError("release model-state digest mismatch")
    model = HCCSemPathModel(
        backbone_name=model_config["backbone_name"],
        embedding_dim=int(model_config["embedding_dim"]),
        teacher_dims={},
        pretrained=False,
        projector_type=model_config.get("projector_type", "linear"),
        projector_hidden_dim=int(model_config.get("projector_hidden_dim", 2048)),
        l1_num_classes=int(model_config["l1_num_classes"]),
        spatial_num_components=int(model_config["spatial_num_components"]),
        spatial_dim=int(model_config.get("spatial_dim", 256)),
        spatial_output_stride=int(
            model_config.get("spatial_output_stride", SPATIAL_OUTPUT_STRIDE)
        ),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config
