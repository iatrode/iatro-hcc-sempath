from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors_file

from hcc_sempath.modeling.models import (
    HCCSemPathModel,
    SPATIAL_OUTPUT_STRIDE,
    STUDENT_BACKBONE_NAME,
    model_state_sha256,
    validate_spatial_decoder_calibration,
)
from hcc_sempath.spatial_schema import DEFAULT_SPATIAL_COMPONENTS, spatial_component_metadata
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.prototype_labels import DEFAULT_CLASSIFICATION_CLASSES


def _checkpoint_config(payload: dict) -> dict:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no resolved training config")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint has no model state dict")
    return config


def _release_model(
    payload: dict,
) -> tuple[HCCSemPathModel, dict, list[str], list[str]]:
    config = _checkpoint_config(payload)
    teachers = teacher_names(config)
    classification_names = [
        str(name)
        for name in config["model"].get(
            "classification_class_names",
            DEFAULT_CLASSIFICATION_CLASSES,
        )
    ]
    spatial_names = [
        str(name)
        for name in config["data"].get(
            "spatial_component_names",
            DEFAULT_SPATIAL_COMPONENTS,
        )
    ]
    spatial_dim = int(config["model"].get("spatial_dim", 256))
    spatial_stride = int(
        config["model"].get("spatial_output_stride", SPATIAL_OUTPUT_STRIDE)
    )
    common = {
        "backbone_name": STUDENT_BACKBONE_NAME,
        "embedding_dim": embedding_dim(config),
        "pretrained": False,
        "projector_type": config["model"].get("projector_type", "linear"),
        "projector_hidden_dim": int(
            config["model"].get("projector_hidden_dim", 2048)
        ),
        "classification_num_classes": len(classification_names),
        "spatial_num_components": len(spatial_names),
        "spatial_dim": spatial_dim,
        "spatial_output_stride": spatial_stride,
    }
    training_model = HCCSemPathModel(
        **common,
        teacher_dims=teacher_dims(config, teachers),
        teacher_head_type=config["model"].get("teacher_head_type", "linear"),
    )
    training_model.load_state_dict(
        {
            key.removeprefix("_orig_mod."): value
            for key, value in payload["model"].items()
        },
        strict=True,
    )
    release_model = HCCSemPathModel(**common, teacher_dims={})
    release_model.encoder.load_state_dict(training_model.encoder.state_dict())
    assert release_model.classification_prototypes is not None
    assert training_model.classification_prototypes is not None
    assert release_model.classification_prototype_counts is not None
    assert training_model.classification_prototype_counts is not None
    assert release_model.classification_log_temperature is not None
    assert training_model.classification_log_temperature is not None
    assert release_model.spatial_head is not None
    assert training_model.spatial_head is not None
    with torch.no_grad():
        release_model.classification_prototypes.copy_(
            training_model.classification_prototypes
        )
        release_model.classification_prototype_counts.copy_(
            training_model.classification_prototype_counts
        )
        release_model.classification_log_temperature.copy_(
            training_model.classification_log_temperature
        )
    release_model.spatial_head.load_state_dict(
        training_model.spatial_head.state_dict()
    )
    release_model.eval()
    return release_model, config, classification_names, spatial_names


def export_release(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    spatial_calibration: str | Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    release_model, config, classification_names, spatial_names = _release_model(payload)
    release_state = release_model.state_dict()
    release_digest = model_state_sha256(release_state)
    training_digest = model_state_sha256(payload["model"])
    spatial_stride = int(
        config["model"].get("spatial_output_stride", SPATIAL_OUTPUT_STRIDE)
    )
    calibration = None
    if spatial_calibration is not None:
        calibration = validate_spatial_decoder_calibration(
            json.loads(Path(spatial_calibration).read_text(encoding="utf-8")),
            spatial_names,
            expected_output_stride=spatial_stride,
            expected_model_state_sha256=training_digest,
        )
    release_config = {
        "format": "hcc-sempath-classification-spatial-state-dict",
        "version": 4,
        "model": {
            "backbone_name": STUDENT_BACKBONE_NAME,
            "embedding_dim": embedding_dim(config),
            "projector_type": config["model"].get("projector_type", "linear"),
            "projector_hidden_dim": int(
                config["model"].get("projector_hidden_dim", 2048)
            ),
            "classification_num_classes": len(classification_names),
            "spatial_num_components": len(spatial_names),
            "spatial_dim": int(config["model"].get("spatial_dim", 256)),
            "spatial_output_stride": spatial_stride,
            "spatial_use_local_branch": bool(
                config["model"].get("spatial_use_local_branch", True)
            ),
            "spatial_use_semantic_branch": bool(
                config["model"].get("spatial_use_semantic_branch", True)
            ),
            "spatial_use_context": bool(
                config["model"].get("spatial_use_context", True)
            ),
        },
        "preprocessing": {
            "mean": config["data"].get("mean", [0.485, 0.456, 0.406]),
            "std": config["data"].get("std", [0.229, 0.224, 0.225]),
        },
        "classification_names": classification_names,
        "spatial_component_names": spatial_names,
        "spatial_component_contracts": spatial_component_metadata(spatial_names),
        "spatial_decoder_calibration": calibration,
        "training_provenance": {
            "source_checkpoint": checkpoint.name,
            "selected_epoch": int(payload.get("epoch", -1)),
            "selection_finalized": bool(payload.get("selection_finalized", False)),
            "training_model_sha256": training_digest,
            "release_model_sha256": release_digest,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    config_path = output_dir / "config.json"
    temporary_weights = weights_path.with_suffix(".safetensors.tmp")
    temporary_config = config_path.with_suffix(".json.tmp")
    serializable_state = {
        key: value.detach().cpu().contiguous()
        for key, value in release_state.items()
    }
    save_safetensors_file(
        serializable_state,
        str(temporary_weights),
        metadata={
            "format": "pt",
            "hcc_sempath_release_version": "4",
            "release_model_sha256": release_digest,
        },
    )
    temporary_config.write_text(
        json.dumps(release_config, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_weights.replace(weights_path)
    temporary_config.replace(config_path)
    return release_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a training checkpoint as a standalone SemPath model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--spatial-calibration",
        help="Optional independently produced spatial decoder calibration JSON.",
    )
    args = parser.parse_args()
    config = export_release(
        args.checkpoint,
        args.output,
        spatial_calibration=args.spatial_calibration,
    )
    print(
        "export_ok "
        f"output={Path(args.output)} "
        f"release_model_sha256={config['training_provenance']['release_model_sha256']}"
    )


if __name__ == "__main__":
    main()
