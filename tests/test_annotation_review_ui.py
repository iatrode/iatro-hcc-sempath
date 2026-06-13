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


def test_l1_review_candidates_prioritize_degenerative_without_label_guessing() -> None:
    module = _load_module()
    payload = {
        "annotations": {
            "stable": {"tile_id": "a", "l1": "HCC-tumor", "l2": []},
            "degenerative": {
                "tile_id": "b",
                "l1": "Degenerative-material",
                "l2": ["necrosis-present"],
            },
            "stromal": {"tile_id": "c", "l1": "Inflammatory-stromal", "l2": ["fibrous-stroma-present"]},
        }
    }

    candidates = module.build_candidates(payload)

    assert candidates[0].key == "degenerative"
    assert candidates[0].suggested_l1 == "Degenerative-material"
    assert candidates[0].uncertainty > candidates[1].uncertainty
    stromal = next(candidate for candidate in candidates if candidate.key == "stromal")
    assert stromal.suggested_l1 == "Inflammatory-stromal"


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
                    "stromal": {
                        "tile_id": "a",
                        "row": 1,
                        "iac_path": "missing.iac",
                        "l1": "Inflammatory-stromal",
                        "l2": ["fibrous-stroma-present"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "reviewed.json"
    state = module.ReviewState(annotation_json, output_json, mode="l1", binary_a="HCC-tumor", binary_b="Degenerative-material")

    accepted = state.review("tumor", "accept")
    adjusted = state.review("stromal", "adjust", "Degenerative-material")

    assert accepted["reviewed"] is True
    assert accepted["l1"] == "HCC-tumor"
    assert accepted["review_previous_l1"] == "HCC-tumor"
    assert adjusted["reviewed"] is True
    assert adjusted["l1"] == "Degenerative-material"
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["annotations"]["tumor"]["review_decision"] == "accept"
    assert (tmp_path / "reviewed.review.csv").exists()


def test_review_state_resumes_existing_output_and_updates_candidate_cache(tmp_path: Path) -> None:
    module = _load_module()
    annotation_json = tmp_path / "annotations.json"
    output_json = tmp_path / "reviewed.json"
    source_payload = {
        "annotations": {
            "done": {"tile_id": "done", "l1": "HCC-tumor"},
            "todo": {"tile_id": "todo", "l1": "Inflammatory-stromal"},
        }
    }
    resumed_payload = {
        "annotations": {
            "done": {"tile_id": "done", "l1": "HCC-tumor", "reviewed": True},
            "todo": {"tile_id": "todo", "l1": "Inflammatory-stromal"},
        }
    }
    annotation_json.write_text(json.dumps(source_payload), encoding="utf-8")
    output_json.write_text(json.dumps(resumed_payload), encoding="utf-8")
    state = module.ReviewState(annotation_json, output_json, mode="l1", binary_a="HCC-tumor", binary_b="Inflammatory-stromal")

    assert [candidate.key for candidate in state.candidates()] == ["todo"]

    state.review("todo", "reject")
    assert state.candidates() == []

    repeated = state.review("todo", "reject")
    assert repeated["review_decision"] == "reject"


def test_binary_mode_filters_to_requested_classes() -> None:
    module = _load_module()
    payload = {
        "annotations": {
            "background": {"tile_id": "a", "l1": "Background-liver", "l2": []},
            "stromal": {"tile_id": "f", "l1": "Inflammatory-stromal", "l2": []},
            "tumor": {"tile_id": "t", "l1": "HCC-tumor", "l2": []},
        }
    }

    candidates = module.build_candidates(payload, mode="binary", binary_a="HCC-tumor", binary_b="Inflammatory-stromal")

    assert {candidate.key for candidate in candidates} == {"stromal", "tumor"}


def test_disagreement_csvs_build_blinded_resumable_queue(tmp_path: Path) -> None:
    module = _load_module()
    fields = [
        "rank",
        "split",
        "tile_id",
        "package_path",
        "row_idx",
        "disagreement_score",
        "gigapath_l1",
    ]
    val_csv = tmp_path / "val.csv"
    exval_csv = tmp_path / "exval.csv"
    for path, split, tile_id in ((val_csv, "val", "v1"), (exval_csv, "exval", "e1")):
        path.write_text(
            ",".join(fields)
            + "\n"
            + f"1,{split},{tile_id},/tmp/{tile_id}.iac,3,1.2,HCC-tumor\n",
            encoding="utf-8",
        )

    output_json = tmp_path / "review.json"
    state = module.ReviewState(
        None,
        output_json,
        mode="disagreement",
        binary_a="HCC-tumor",
        binary_b="Inflammatory-stromal",
        disagreement_csvs=[val_csv, exval_csv],
        blind_seed=7,
    )

    candidates = state.candidates()
    assert {candidate.tile_id for candidate in candidates} == {"TD-0001", "TD-0002"}
    public = state.public_item(candidates[0].key)
    assert set(public) == {"review_id", "reviewed", "l1", "source_group"}
    assert "gigapath_l1" not in public

    reviewed = state.review(candidates[0].key, "adjust", "Background-liver")
    assert reviewed["adjudication_status"] == "adjudicated"
    assert reviewed["l1"] == "Background-liver"
    assert output_json.exists()
    review_csv = tmp_path / "review.review.csv"
    assert review_csv.exists()
    assert "gigapath_l1" in review_csv.read_text(encoding="utf-8")

    resumed = module.ReviewState(
        None,
        output_json,
        mode="disagreement",
        binary_a="HCC-tumor",
        binary_b="Inflammatory-stromal",
        disagreement_csvs=[val_csv, exval_csv],
    )
    assert len(resumed.candidates()) == 1


def test_disagreement_review_can_mark_tile_uncertain(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source.csv"
    source.write_text(
        "rank,split,tile_id,package_path,row_idx\n"
        "1,val,tile-1,/tmp/tile-1.iac,0\n",
        encoding="utf-8",
    )
    state = module.ReviewState(
        None,
        tmp_path / "review.json",
        mode="disagreement",
        binary_a="HCC-tumor",
        binary_b="Inflammatory-stromal",
        disagreement_csvs=[source],
    )

    item = state.review("TD-0001", "uncertain")

    assert item["reviewed"] is True
    assert item["adjudication_status"] == "uncertain"
    assert item["l1"] == ""
