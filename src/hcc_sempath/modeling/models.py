from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import checkpoint_filter_fn


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
STUDENT_PRETRAINED_PATH = (
    Path(__file__).resolve().parents[3] / "artifacts" / "pretrained" / "dinov2_vits14_pretrain.pth"
)
STUDENT_PRETRAINED_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def _load_fixed_student_pretraining(backbone: nn.Module) -> None:
    if not STUDENT_PRETRAINED_PATH.is_file():
        raise FileNotFoundError(
            f"fixed DINOv2-S/14 pretrained weight is missing: {STUDENT_PRETRAINED_PATH}"
        )
    with STUDENT_PRETRAINED_PATH.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != STUDENT_PRETRAINED_SHA256:
        raise ValueError(
            "fixed DINOv2-S/14 pretrained weight checksum mismatch: "
            f"expected={STUDENT_PRETRAINED_SHA256} got={digest} path={STUDENT_PRETRAINED_PATH}"
        )
    state = torch.load(STUDENT_PRETRAINED_PATH, map_location="cpu", weights_only=True)
    state = state.get("model", state)
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
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="token",
            img_size=STUDENT_IMAGE_SIZE,
        )
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


class HCCSemPathModel(nn.Module):
    """Shared HCC encoder with optional training heads and deployable prototype readouts."""

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
        l2_num_attributes: int = 0,
        roi_l2_num_attributes: int = 0,
        roi_patch_dim: int | None = None,
        roi_top_q: float = 0.1,
        roi_patch_temperature: float = 0.1,
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
        self.register_buffer(
            "l1_prototypes",
            torch.zeros(l1_num_classes, embedding_dim) if l1_num_classes > 0 else None,
        )
        self.register_buffer(
            "l2_prototypes",
            torch.zeros(l2_num_attributes, embedding_dim) if l2_num_attributes > 0 else None,
        )
        self.register_buffer(
            "l2_biases",
            torch.zeros(l2_num_attributes) if l2_num_attributes > 0 else None,
        )
        self.register_buffer("l1_temperature", torch.tensor(0.1) if l1_num_classes > 0 else None)
        self.register_buffer(
            "l2_temperature",
            torch.full((l2_num_attributes,), 0.1) if l2_num_attributes > 0 else None,
        )
        self.roi_l2_num_attributes = int(roi_l2_num_attributes)
        self.roi_top_q = float(roi_top_q)
        self.roi_patch_temperature = float(roi_patch_temperature)
        if self.roi_l2_num_attributes > 0:
            if not 0 < self.roi_top_q <= 1:
                raise ValueError(f"roi_top_q must be in (0, 1], got {roi_top_q}")
            if self.roi_patch_temperature <= 0:
                raise ValueError(f"roi_patch_temperature must be positive, got {roi_patch_temperature}")
            patch_dim = int(roi_patch_dim or embedding_dim)
            student_dim = int(self.encoder.backbone.num_features)
            self.roi_patch_projector = nn.Sequential(nn.LayerNorm(student_dim), nn.Linear(student_dim, patch_dim))
            # These queries are never trained from global pseudo-labels. ROI token loss is
            # their semantic anchor, so local maps cannot learn to echo global context alone.
            self.roi_attribute_queries = nn.Parameter(torch.empty(self.roi_l2_num_attributes, patch_dim))
            nn.init.trunc_normal_(self.roi_attribute_queries, std=0.02)
        else:
            self.roi_patch_projector = None
            self.register_parameter("roi_attribute_queries", None)

    @property
    def teacher_names(self) -> list[str]:
        return list(self.teacher_heads.keys())

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def project_teachers(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(embedding) for name, head in self.teacher_heads.items()}

    def forward(
        self,
        images: torch.Tensor,
        *,
        roi_detach_backbone: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.roi_patch_projector is None:
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
        if patch_tokens is not None and patch_grid is not None and self.roi_attribute_queries is not None:
            if roi_detach_backbone:
                patch_tokens = patch_tokens.detach()
            patch_embedding = F.normalize(self.roi_patch_projector(patch_tokens), dim=-1)
            queries = F.normalize(self.roi_attribute_queries, dim=-1)
            patch_logits = (patch_embedding @ queries.T) / self.roi_patch_temperature
            patch_logits = bounded_logits(patch_logits)
            top_count = max(1, int(round(patch_logits.shape[1] * self.roi_top_q)))
            local_logits = patch_logits.topk(top_count, dim=1).values.mean(dim=1)
            outputs["roi_patch_logits"] = patch_logits.transpose(1, 2).reshape(
                patch_logits.shape[0], self.roi_l2_num_attributes, patch_grid[0], patch_grid[1]
            )
            outputs["roi_local_logits"] = local_logits
        if self.l1_prototypes is not None and self.l1_temperature is not None:
            l1_logits = (embedding_norm @ self.l1_prototypes.T) / self.l1_temperature
            l1_logits = bounded_logits(l1_logits)
            outputs["l1_probabilities"] = F.softmax(l1_logits, dim=-1)
        if (
            self.l2_prototypes is not None
            and self.l2_biases is not None
            and self.l2_temperature is not None
        ):
            l2_cosine_scores = embedding_norm @ self.l2_prototypes.T
            outputs["l2_cosine_scores"] = l2_cosine_scores
            l2_centered_scores = centered_attribute_scores(
                l2_cosine_scores,
                self.l2_biases,
                self.l2_temperature,
            )
            outputs["l2_centered_scores"] = l2_centered_scores
            outputs["l2_scores"] = l2_centered_scores
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


def calibrated_attribute_scores(
    cosine_scores: torch.Tensor,
    biases: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Map per-attribute cosine scores to centered relative retrieval scores."""
    temperatures = temperatures.to(device=cosine_scores.device, dtype=cosine_scores.dtype).clamp_min(1e-6)
    biases = biases.to(device=cosine_scores.device, dtype=cosine_scores.dtype)
    return torch.sigmoid((cosine_scores - biases) / temperatures)


def bounded_logits(value: torch.Tensor, limit: float = LOGIT_LIMIT) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0:
        raise ValueError(f"logit limit must be positive, got {limit}")
    value = value.float()
    return limit * torch.tanh(value / limit)


def clamp_probability(value: torch.Tensor, *, normalize: bool = False, eps: float = PROB_EPS) -> torch.Tensor:
    """Pull a probability/target tensor away from the degenerate 0/1 boundary.

    Upcasts to float32, optionally renormalizes across the last dim, then clamps
    into ``[eps, 1 - eps]`` so downstream KL/BCE terms stay finite under fp16.
    """
    value = value.float()
    if normalize:
        value = value / value.sum(dim=-1, keepdim=True).clamp_min(eps)
    return value.clamp(eps, 1.0 - eps)


def centered_attribute_scores(
    cosine_scores: torch.Tensor,
    biases: torch.Tensor,
    temperatures: torch.Tensor,
    limit: float = LOGIT_LIMIT,
) -> torch.Tensor:
    """Map calibrated cosine scores to bounded non-probabilistic L2 evidence.

    The output is a relative morphology score, not a calibrated probability.
    This keeps deployment aligned with training-time v2 semantics: pull the
    teacher/prototype response scale back before interpretation instead of
    hiding saturated logits behind a final sigmoid.
    """
    cosine_scores = cosine_scores.float()
    temperatures = temperatures.to(device=cosine_scores.device, dtype=cosine_scores.dtype).clamp_min(1e-6)
    biases = biases.to(device=cosine_scores.device, dtype=cosine_scores.dtype)
    scaled = (cosine_scores - biases) / temperatures
    return bounded_logits(scaled, limit=limit)


def load_hcc_sempath_release(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[HCCSemPathModel, dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_config = config["model"]
    model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=int(model_config["embedding_dim"]),
        teacher_dims={},
        pretrained=False,
        projector_type=model_config.get("projector_type", "linear"),
        projector_hidden_dim=int(model_config.get("projector_hidden_dim", 2048)),
        l1_num_classes=int(model_config["l1_num_classes"]),
        l2_num_attributes=int(model_config["l2_num_attributes"]),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config
