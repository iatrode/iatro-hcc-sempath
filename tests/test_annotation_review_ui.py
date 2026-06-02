from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "annotation_review_ui.py"
    spec = importlib.util.spec_from_file_location("annotation_review_ui", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_l1_review_candidates_prioritize_unstable_without_label_guessing() -> None:
    module = _load_module()
    payload = {
        "annotations": {
            "stable": {"tile_id": "a", "l1": "HCC-tumor", "l2": []},
            "indeterminate": {
                "tile_id": "b",
                "l1": "Indeterminate-region",
                "l2": ["fibrous-stroma-present"],
            },
            "fibrous": {"tile_id": "c", "l1": "Fibrous-stromal", "l2": ["fibrous-stroma-present"]},
        }
    }

    candidates = module.build_candidates(payload)

    assert candidates[0].key == "indeterminate"
    assert candidates[0].suggested_l1 == "Indeterminate-region"
    assert candidates[0].uncertainty > candidates[1].uncertainty
    fibrous = next(candidate for candidate in candidates if candidate.key == "fibrous")
    assert fibrous.suggested_l1 == "Fibrous-stromal"


def test_review_state_accept_and_adjust_write_review_fields(tmp_path: Path) -> None:
    module = _load_module()
    annotation_json = tmp_path / "annotations.json"
    annotation_json.write_text(
        json.dumps(
            {
                "annotations": {
                    "tumor": {
                        "tile_id": "b",
                        "row": 0,
                        "iac_path": "missing.iac",
                        "l1": "HCC-tumor",
                        "l2": ["hepatocellular-parenchyma-present"],
                    },
                    "artifact": {
                        "tile_id": "a",
                        "row": 1,
                        "iac_path": "missing.iac",
                        "l1": "Artifact-non-tissue",
                        "l2": [],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "reviewed.json"
    state = module.ReviewState(annotation_json, output_json, mode="l1", binary_a="Artifact-non-tissue", binary_b="Degenerative-material")

    accepted = state.review("tumor", "accept")
    adjusted = state.review("artifact", "adjust", "Degenerative-material")

    assert accepted["reviewed"] is True
    assert accepted["l1"] == "HCC-tumor"
    assert accepted["review_previous_l1"] == "HCC-tumor"
    assert adjusted["reviewed"] is True
    assert adjusted["l1"] == "Degenerative-material"
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["annotations"]["tumor"]["review_decision"] == "accept"
    assert (tmp_path / "reviewed.review.csv").exists()


def test_binary_mode_filters_to_requested_classes() -> None:
    module = _load_module()
    payload = {
        "annotations": {
            "artifact": {"tile_id": "a", "l1": "Artifact-non-tissue", "l2": []},
            "fibrous": {"tile_id": "f", "l1": "Fibrous-stromal", "l2": []},
            "tumor": {"tile_id": "t", "l1": "HCC-tumor", "l2": []},
        }
    }

    candidates = module.build_candidates(payload, mode="binary", binary_a="HCC-tumor", binary_b="Fibrous-stromal")

    assert {candidate.key for candidate in candidates} == {"fibrous", "tumor"}
