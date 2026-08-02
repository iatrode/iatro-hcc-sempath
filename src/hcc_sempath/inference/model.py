from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from hcc_sempath.modeling.models import HCCSemPathModel, model_state_sha256


RELEASE_CONFIG_NAME = "config.json"
RELEASE_WEIGHTS_NAME = "hcc_sempath_release.pt"
RELEASE_FORMAT = "hcc-sempath-classification-spatial-state-dict"
RELEASE_VERSION = 3


@dataclass(frozen=True)
class ReleaseModel:
    model: HCCSemPathModel
    config: dict
    model_dir: Path
    weights_path: Path
    model_digest: str
    classification_names: tuple[str, ...]
    spatial_component_names: tuple[str, ...]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"release preprocessing.{name} must contain three values")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def load_release_model(
    model_dir: str | Path,
    *,
    device: torch.device,
) -> ReleaseModel:
    model_dir = Path(model_dir)
    config_path = model_dir / RELEASE_CONFIG_NAME
    weights_path = model_dir / RELEASE_WEIGHTS_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != RELEASE_FORMAT or int(config.get("version", -1)) != RELEASE_VERSION:
        raise ValueError(
            f"unsupported SemPath release contract: {config.get('format')} "
            f"version={config.get('version')}"
        )
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("release config has no model contract")
    classification_names = tuple(str(name) for name in config.get("classification_names", []))
    spatial_names = tuple(str(name) for name in config.get("spatial_component_names", []))
    if len(classification_names) != int(model_config.get("classification_num_classes", -1)):
        raise ValueError("release classification-name count does not match model contract")
    if len(spatial_names) != int(model_config.get("spatial_num_components", -1)):
        raise ValueError("release spatial-name count does not match model contract")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("release weights must contain a non-empty state dict")
    model_digest = model_state_sha256(state)
    expected_digest = str(
        config.get("training_provenance", {}).get("release_model_sha256", "")
    )
    if expected_digest and model_digest != expected_digest:
        raise ValueError(
            "release model digest mismatch: "
            f"expected={expected_digest} actual={model_digest}"
        )
    model = HCCSemPathModel(
        backbone_name=str(model_config["backbone_name"]),
        embedding_dim=int(model_config["embedding_dim"]),
        teacher_dims={},
        pretrained=False,
        projector_type=str(model_config.get("projector_type", "linear")),
        projector_hidden_dim=int(model_config.get("projector_hidden_dim", 2048)),
        classification_num_classes=len(classification_names),
        spatial_num_components=len(spatial_names),
        spatial_dim=int(model_config.get("spatial_dim", 256)),
        spatial_output_stride=int(model_config.get("spatial_output_stride", 7)),
    )
    model.load_state_dict(state, strict=True)
    if model.spatial_head is not None:
        model.spatial_head.use_local_branch = bool(
            model_config.get("spatial_use_local_branch", True)
        )
        model.spatial_head.use_semantic_branch = bool(
            model_config.get("spatial_use_semantic_branch", True)
        )
        model.spatial_head.use_context = bool(
            model_config.get("spatial_use_context", True)
        )
    model.to(device).eval()
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("release config has no preprocessing contract")
    return ReleaseModel(
        model=model,
        config=config,
        model_dir=model_dir,
        weights_path=weights_path,
        model_digest=model_digest,
        classification_names=classification_names,
        spatial_component_names=spatial_names,
        mean=_triplet(preprocessing.get("mean"), "mean"),
        std=_triplet(preprocessing.get("std"), "std"),
    )


def prepare_images(
    images: torch.Tensor,
    release: ReleaseModel,
    device: torch.device,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected uint8 NHWC images, got shape={tuple(images.shape)}")
    images = images.to(device, non_blocking=device.type == "cuda")
    images = images.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean = torch.tensor(release.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(release.std, device=device).view(1, 3, 1, 1)
    return images.sub_(mean).div_(std)
