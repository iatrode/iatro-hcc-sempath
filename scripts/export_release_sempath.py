from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.modeling.models import (  # noqa: E402
    HCCSemPathModel,
    SPATIAL_OUTPUT_STRIDE,
    STUDENT_BACKBONE_NAME,
    canonical_payload_sha256,
    model_state_sha256,
    validate_spatial_decoder_calibration,
)
from hcc_sempath.spatial_schema import spatial_component_metadata  # noqa: E402
from hcc_sempath.training.config import (  # noqa: E402
    embedding_dim,
    load_config,
    teacher_dims,
    teacher_names,
)
from hcc_sempath.training.prototype_labels import DEFAULT_CLASSIFICATION_CLASSES  # noqa: E402
from hcc_sempath.training.roi import (  # noqa: E402
    DEFAULT_SPATIAL_COMPONENTS,
    spatial_component_names,
)

def _resolved_spatial_names(cfg: dict) -> list[str]:
    frozen = cfg.get("data", {}).get("spatial_component_names")
    if frozen:
        return [str(name) for name in frozen]
    path = cfg.get("data", {}).get("spatial_manifest_path")
    return (
        spatial_component_names(path)
        if path
        else list(DEFAULT_SPATIAL_COMPONENTS)
    )


def _release_contract(cfg: dict) -> dict:
    names = teacher_names(cfg)
    return {
        "model": {
            "embedding_dim": embedding_dim(cfg),
            "projector_type": cfg["model"].get("projector_type", "linear"),
            "projector_hidden_dim": int(
                cfg["model"].get("projector_hidden_dim", 2048)
            ),
            "teacher_head_type": cfg["model"].get(
                "teacher_head_type",
                "linear",
            ),
            "teacher_dims": teacher_dims(cfg, names),
            "classification_class_names": [
                str(name)
                for name in cfg["model"].get(
                    "classification_class_names",
                    DEFAULT_CLASSIFICATION_CLASSES,
                )
            ],
            "spatial_dim": int(cfg["model"].get("spatial_dim", 256)),
            "spatial_output_stride": int(
                cfg["model"].get(
                    "spatial_output_stride",
                    SPATIAL_OUTPUT_STRIDE,
                )
            ),
        },
        "data": {
            "teachers": names,
            "mean": cfg["data"].get(
                "mean",
                [0.485, 0.456, 0.406],
            ),
            "std": cfg["data"].get(
                "std",
                [0.229, 0.224, 0.225],
            ),
            "spatial_component_names": _resolved_spatial_names(cfg),
        },
    }


def _require_finalized_checkpoint(payload: dict) -> dict:
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint has no resolved training config")
    expected = int(cfg.get("train", {}).get("epochs", -1))
    checkpoint_epoch = int(payload.get("epoch", -1))
    expected_matches = (
        int(payload.get("expected_epochs", -1)) == expected
    )
    legacy_terminal = bool(
        payload.get("training_complete", False)
        and checkpoint_epoch == expected
        and expected_matches
    )
    finalized_selection = bool(
        payload.get("run_complete", False)
        and payload.get("selection_finalized", False)
        and checkpoint_epoch
        == int(payload.get("best_selection_epoch", -1))
        and expected_matches
    )
    joint_selection_run = bool(
        cfg.get("train", {}).get("selection_early_stop", False)
        or cfg.get("data", {}).get(
            "require_complete_expert_validation",
            False,
        )
    )
    accepted = (
        finalized_selection
        if joint_selection_run
        else (legacy_terminal or finalized_selection)
    )
    if not accepted:
        raise ValueError(
            "spatial release requires a finalized joint-selection "
            "checkpoint from a completed run"
        )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the classification + spatial HCC-SemPath release."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument("--config")
    parser.add_argument("--spatial-calibration", required=True)
    parser.add_argument("--output-dir", default="artifacts/release")
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    cfg = _require_finalized_checkpoint(payload)
    expected_epochs = int(cfg.get("train", {}).get("epochs", -1))
    if args.config:
        requested_cfg = load_config(Path(args.config))
        if _release_contract(requested_cfg) != _release_contract(cfg):
            raise ValueError(
                "export config differs from the checkpoint release contract"
            )
    names = teacher_names(cfg)
    dims = teacher_dims(cfg, names)
    classification_names = [
        str(name)
        for name in cfg["model"].get("classification_class_names", DEFAULT_CLASSIFICATION_CLASSES)
    ]
    spatial_names = _resolved_spatial_names(cfg)
    spatial_dim = int(cfg["model"].get("spatial_dim", 256))
    spatial_stride = int(
        cfg["model"].get("spatial_output_stride", SPATIAL_OUTPUT_STRIDE)
    )
    research_contract = cfg.get("research_contract")
    if not isinstance(research_contract, dict):
        raise ValueError("checkpoint has no frozen research contract")
    with Path(args.spatial_calibration).open(
        "r",
        encoding="utf-8",
    ) as handle:
        calibration = validate_spatial_decoder_calibration(
            json.load(handle),
            spatial_names,
            expected_output_stride=spatial_stride,
            expected_model_state_sha256=model_state_sha256(
                payload["model"]
            ),
            expected_research_contract_sha256=(
                canonical_payload_sha256(research_contract)
            ),
            expected_optimizer_visible_contract_sha256=str(
                canonical_payload_sha256(
                    {
                        "packages": cfg["data"].get(
                            "optimizer_visible_tile_packages",
                            [],
                        ),
                        "sizes": cfg["data"].get(
                            "optimizer_visible_tile_package_sizes",
                            [],
                        ),
                        "sha256": cfg["data"].get(
                            "optimizer_visible_tile_package_sha256",
                            [],
                        ),
                        "expert_split_exclusion_sha256": cfg["data"].get(
                            "expert_split_exclusion_sha256"
                        ),
                    }
                )
            ),
            expected_supervision_assets_sha256=(
                canonical_payload_sha256(
                    cfg["data"].get(
                        "supervision_asset_sha256",
                        {},
                    )
                )
            ),
            expected_formal_asset_contract_sha256=str(
                cfg["data"].get(
                    "formal_asset_contract_sha256",
                    "",
                )
            ),
            expected_source_tree_sha256=str(
                cfg["data"].get("formal_source", {}).get(
                    "source_tree_sha256",
                    "",
                )
            ),
            expected_study_contract_sha256=str(
                cfg["data"].get(
                    "formal_study_contract_sha256",
                    "",
                )
            ),
        )
    calibration_provenance = calibration["provenance"]
    expected_epoch_provenance = {
        "selected_epoch": int(payload["epoch"]),
        "terminal_epoch": int(
            payload.get("run_terminal_epoch", payload["epoch"])
        ),
        "expected_epochs": int(payload["expected_epochs"]),
    }
    for key, expected_value in expected_epoch_provenance.items():
        if int(calibration_provenance.get(key, -1)) != expected_value:
            raise ValueError(
                "spatial calibration checkpoint epoch provenance mismatch: "
                f"{key} expected={expected_value} "
                f"got={calibration_provenance.get(key)}"
            )
    if not bool(
        calibration_provenance.get("selection_finalized", False)
    ):
        raise ValueError(
            "spatial calibration was not produced from a finalized "
            "selection checkpoint"
        )

    training_model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        classification_num_classes=len(classification_names),
        spatial_num_components=len(spatial_names),
        spatial_dim=spatial_dim,
        spatial_output_stride=spatial_stride,
    )
    contract = research_contract
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no frozen research contract")
    if contract.get("student_backbone") != STUDENT_BACKBONE_NAME:
        raise ValueError(
            "checkpoint backbone contract does not match the spatial release"
        )
    model_state = {
        key.removeprefix("_orig_mod."): value
        for key, value in payload["model"].items()
    }
    training_model.load_state_dict(model_state, strict=True)

    release_model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims={},
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        classification_num_classes=len(classification_names),
        spatial_num_components=len(spatial_names),
        spatial_dim=spatial_dim,
        spatial_output_stride=spatial_stride,
    )
    release_model.encoder.load_state_dict(training_model.encoder.state_dict())
    assert (
        release_model.classification_prototypes is not None
        and training_model.classification_prototypes is not None
        and release_model.classification_prototype_counts is not None
        and training_model.classification_prototype_counts is not None
        and release_model.classification_log_temperature is not None
        and training_model.classification_log_temperature is not None
    )
    assert release_model.spatial_head is not None and training_model.spatial_head is not None
    with torch.no_grad():
        release_model.classification_prototypes.copy_(training_model.classification_prototypes)
        release_model.classification_prototype_counts.copy_(
            training_model.classification_prototype_counts
        )
        release_model.classification_log_temperature.copy_(
            training_model.classification_log_temperature
        )
    release_model.spatial_head.load_state_dict(training_model.spatial_head.state_dict())
    release_model.eval()
    release_state = release_model.state_dict()
    release_model_digest = model_state_sha256(release_state)

    release_config = {
        "format": "hcc-sempath-classification-spatial-state-dict",
        "version": 3,
        "model": {
            "backbone_name": STUDENT_BACKBONE_NAME,
            "embedding_dim": embedding_dim(cfg),
            "projector_type": cfg["model"].get("projector_type", "linear"),
            "projector_hidden_dim": int(
                cfg["model"].get("projector_hidden_dim", 2048)
            ),
            "classification_num_classes": len(classification_names),
            "spatial_num_components": len(spatial_names),
            "spatial_dim": spatial_dim,
            "spatial_output_stride": spatial_stride,
        },
        "preprocessing": {
            "mean": cfg["data"].get("mean", [0.485, 0.456, 0.406]),
            "std": cfg["data"].get("std", [0.229, 0.224, 0.225]),
        },
        "classification_names": classification_names,
        "spatial_component_names": spatial_names,
        "spatial_component_contracts": spatial_component_metadata(spatial_names),
        "spatial_decoder_calibration": calibration,
        "training_provenance": {
            "selected_epoch": int(payload["epoch"]),
            "terminal_epoch": int(
                payload.get("run_terminal_epoch", payload["epoch"])
            ),
            "expected_epochs": expected_epochs,
            "selection_finalized": bool(
                payload.get(
                    "selection_finalized",
                    int(payload["epoch"]) == int(expected_epochs),
                )
            ),
            "formal_asset_contract_sha256": calibration_provenance[
                "formal_asset_contract_sha256"
            ],
            "source_tree_sha256": calibration_provenance[
                "source_tree_sha256"
            ],
            "study_contract_sha256": calibration_provenance[
                "study_contract_sha256"
            ],
            "training_model_sha256": calibration["provenance"][
                "checkpoint_model_sha256"
            ],
            "release_model_sha256": release_model_digest,
        },
        "spatial_measurements": {
            "readout": (
                "dynamic expert-updated positive/negative prototype responses; "
                "training-only global PAMT-D component centroids are excluded"
            ),
            "instance_head": (
                "decoded only for cell-instance and discrete-structure components"
            ),
            "measurement_head": (
                "density for cell components; occupied-area probability for "
                "continuous, pigment, and discrete-structure components"
            ),
            "brush_circle_supervision": (
                "class-specific: cell brush=density bag; circle=large instance; "
                "structure brush=one instance with extent; continuous brush=area"
            ),
            "bile_pigment": (
                "area/burden primary; thresholded focus density is derived and "
                "is not an instance count"
            ),
            "bile_focus_connectivity": 8,
            "bile_focus_density_unit": "connected foci per 1e6 input pixels",
            "bile_focus_minimum_cells": (
                "decoder parameter; freeze on independent spatial validation"
            ),
            "physical_area": (
                "pixel area is emitted for area-capable components; conversion "
                "to physical area requires verified MPP and frozen calibration"
            ),
            "output_stride_pixels": spatial_stride,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(release_state, output_dir / "hcc_sempath_release.pt")
    (output_dir / "config.json").write_text(
        json.dumps(release_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"release_ok output={output_dir}")


if __name__ == "__main__":
    main()
