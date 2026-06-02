from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "anchor_information_curve.py"
    spec = importlib.util.spec_from_file_location("anchor_information_curve", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    anchors = root / "anchors.csv"
    locked = root / "locked_val.csv"
    contract = root / "prototype_contract.json"
    anchors.write_text(
        "tile_id,slide_id,patient_id,level1_label,level2_labels,source_split\n"
        "a1,s1,p1,tumor,necrosis,train\n"
        "a2,s1,p1,tumor,necrosis,train\n"
        "a3,s2,p2,stroma,fibrosis,train\n"
        "a4,s2,p2,stroma,fibrosis,train\n"
        "a5,s3,p3,tumor,necrosis;fibrosis,train\n"
        "a6,s3,p3,stroma,,train\n"
        "v_skip,s4,p4,tumor,necrosis,val\n",
        encoding="utf-8",
    )
    locked.write_text(
        "tile_id,slide_id,patient_id,level1_label,level2_labels,source_split\n"
        "v1,s4,p4,tumor,necrosis,val\n"
        "v2,s5,p5,stroma,fibrosis,val\n",
        encoding="utf-8",
    )
    contract.write_text(
        json.dumps({"l1_prototypes": ["tumor", "stroma"], "l2_prototypes": ["necrosis", "fibrosis"]}),
        encoding="utf-8",
    )
    return anchors, locked, contract


def test_anchor_information_curve_outputs_nested_feature_metrics(tmp_path: Path) -> None:
    module = _load_module()
    anchors, locked, contract = _write_inputs(tmp_path)
    output_root = tmp_path / "curve"
    vectors = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a2": np.array([0.9, 0.1], dtype=np.float32),
        "a3": np.array([0.0, 1.0], dtype=np.float32),
        "a4": np.array([0.1, 0.9], dtype=np.float32),
        "a5": np.array([0.8, 0.2], dtype=np.float32),
        "a6": np.array([0.2, 0.8], dtype=np.float32),
        "v1": np.array([1.0, 0.05], dtype=np.float32),
        "v2": np.array([0.05, 1.0], dtype=np.float32),
    }

    class FakeFeatureStore:
        def __init__(self, teacher_paths):
            self.teacher_paths = teacher_paths

        def read(self, teacher: str, tile_id: str):
            scale = 1.0 if teacher == "t1" else 1.1
            return vectors[tile_id] * scale

        def close(self) -> None:
            return None

    module.FeatureStore = FakeFeatureStore
    result = module.run(
        argparse.Namespace(
            teacher_feature_root="",
            teacher_feature_packages="t1=/tmp/t1.features.iac,t2=/tmp/t2.features.iac",
            annotation_json="",
            anchor_manifest=str(anchors),
            locked_val_manifest=str(locked),
            candidate_manifest="",
            prototype_contract=str(contract),
            anchor_counts="2,4,6,8",
            teachers="t1,t2",
            output_root=str(output_root),
            seed=13,
            split_seed=13,
            locked_val_fraction=0.2,
            locked_val_count=0,
            anchor_group_key="tile_id",
            bootstrap_iterations=20,
            plateau_delta_epsilon=0.001,
            plateau_drift_threshold=0.05,
            plateau_redundancy_threshold=0.9,
            plot_formats="png",
            no_plots=True,
        )
    )

    plan = json.loads((output_root / "anchor_information_plan.json").read_text(encoding="utf-8"))
    assert plan["sweep_type"] == "anchor_information_curve"
    assert plan["anchor_counts_available"] == [2, 4, 6]
    assert plan["does_not_train"] is True
    assert plan["nested_subsets"] is True
    assert plan["locked_validation_reused_for_all_counts"] is True
    assert plan["split_seed"] == 13

    subsets = _rows(output_root / "nested_subsets.csv")
    assert [row["anchor_count"] for row in subsets] == ["2", "4", "6"]
    n2 = {row["tile_id"] for row in _rows(output_root / "N2" / "anchors.csv")}
    n4 = {row["tile_id"] for row in _rows(output_root / "N4" / "anchors.csv")}
    n6 = {row["tile_id"] for row in _rows(output_root / "N6" / "anchors.csv")}
    assert n2 < n4 < n6
    assert "v_skip" not in n6

    teacher_rows = _rows(output_root / "anchor_information_by_teacher.csv")
    prototype_rows = _rows(output_root / "anchor_information_by_prototype.csv")
    summary_rows = _rows(output_root / "anchor_information_summary.csv")
    assert len(teacher_rows) == 6
    assert len(prototype_rows) == 24
    assert len(summary_rows) == 3
    assert teacher_rows[0]["teacher"] in {"t1", "t2"}
    assert "coverage" in teacher_rows[0]
    assert "val_agreement_l1" in teacher_rows[0]
    assert "prototype_drift" in teacher_rows[0]
    assert "redundancy" in teacher_rows[0]
    assert {"level", "prototype", "train_anchor_count", "center_available", "val_agreement"}.issubset(prototype_rows[0])
    assert summary_rows[-1]["plateau_consensus"] in {"true", "false"}
    assert result["recommendation"]["recommended_anchor_count"] in {2, 4, 6}


def test_missing_centers_are_reported_without_global_mean_fill(tmp_path: Path) -> None:
    module = _load_module()
    anchors, locked, contract = _write_inputs(tmp_path)
    output_root = tmp_path / "curve"
    vectors = {
        tile_id: np.array([idx + 1.0, 1.0], dtype=np.float32)
        for idx, tile_id in enumerate(["a1", "a2", "a3", "a4", "a5", "a6", "v1", "v2"])
    }

    class FakeFeatureStore:
        def __init__(self, teacher_paths):
            pass

        def read(self, teacher: str, tile_id: str):
            return vectors[tile_id]

        def close(self) -> None:
            return None

    module.FeatureStore = FakeFeatureStore
    module.run(
        argparse.Namespace(
            teacher_feature_root="",
            teacher_feature_packages="t=/tmp/t.features.iac",
            annotation_json="",
            anchor_manifest=str(anchors),
            locked_val_manifest=str(locked),
            candidate_manifest="",
            prototype_contract=str(contract),
            anchor_counts="1",
            teachers="t",
            output_root=str(output_root),
            seed=13,
            split_seed=13,
            locked_val_fraction=0.2,
            locked_val_count=0,
            anchor_group_key="tile_id",
            bootstrap_iterations=0,
            plateau_delta_epsilon=0.001,
            plateau_drift_threshold=0.05,
            plateau_redundancy_threshold=0.9,
            plot_formats="png",
            no_plots=True,
        )
    )

    row = _rows(output_root / "anchor_information_by_teacher.csv")[0]
    assert int(row["available_l1_centers"]) == 1
    assert int(row["available_l2_centers"]) <= 1
    assert row["missing_l1_centers"] or row["missing_l2_centers"]


def test_default_anchor_counts_stop_at_three_thousand() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(["--teachers", "t"])
    assert args.anchor_counts == "100,200,400,800,1200,1600,2000,3000"
