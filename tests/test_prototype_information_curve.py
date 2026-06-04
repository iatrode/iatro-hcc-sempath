from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prototype_information_curve.py"
    spec = importlib.util.spec_from_file_location("prototype_information_curve", script_path)
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
    prototype_samples = root / "prototype_samples.csv"
    locked = root / "locked_val.csv"
    contract = root / "prototype_contract.json"
    prototype_samples.write_text(
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
    return prototype_samples, locked, contract


def test_prototype_information_curve_outputs_nested_feature_metrics(tmp_path: Path) -> None:
    module = _load_module()
    prototype_samples, locked, contract = _write_inputs(tmp_path)
    output_root = tmp_path / "curve"
    vectors = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a2": np.array([0.9, 0.1], dtype=np.float32),
        "a3": np.array([0.0, 1.0], dtype=np.float32),
        "a4": np.array([0.1, 0.9], dtype=np.float32),
        "a5": np.array([0.8, 0.2], dtype=np.float32),
        "a6": np.array([0.2, 0.8], dtype=np.float32),
        "v_skip": np.array([0.7, 0.3], dtype=np.float32),
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
            prototype_sample_manifest=str(prototype_samples),
            locked_val_manifest=str(locked),
            candidate_manifest="",
            prototype_contract=str(contract),
            prototype_sample_counts="2,4,6,8",
            teachers="t1,t2",
            output_root=str(output_root),
            seed=13,
            split_seed=13,
            locked_val_fraction=0.2,
            locked_val_count=0,
            prototype_sample_group_key="tile_id",
            bootstrap_iterations=20,
            plateau_delta_epsilon=0.001,
            plateau_drift_threshold=0.05,
            plateau_redundancy_threshold=0.9,
            plot_formats="png",
            no_plots=True,
        )
    )

    report = json.loads((output_root / "infospace_information_report.json").read_text(encoding="utf-8"))
    assert report["sweep_type"] == "infospace_information_curve"
    assert report["prototype_sample_counts_available"] == [2, 4, 6]
    assert report["does_not_train"] is True
    assert report["nested_subsets"] is True
    assert report["seed"] == 13
    assert "recommendation" in report

    subsets = _rows(output_root / "nested_subsets.csv")
    assert [row["prototype_sample_count"] for row in subsets] == ["2", "4", "6"]
    n2 = {row["tile_id"] for row in _rows(output_root / "N2" / "prototype_samples.csv")}
    n4 = {row["tile_id"] for row in _rows(output_root / "N4" / "prototype_samples.csv")}
    n6 = {row["tile_id"] for row in _rows(output_root / "N6" / "prototype_samples.csv")}
    assert n2 < n4 < n6

    teacher_rows = _rows(output_root / "infospace_information_by_teacher.csv")
    prototype_rows = _rows(output_root / "infospace_information_by_prototype.csv")
    summary_rows = _rows(output_root / "infospace_information_summary.csv")
    assert len(teacher_rows) == 6
    assert len(prototype_rows) == 24
    assert len(summary_rows) == 3
    assert teacher_rows[0]["teacher"] in {"t1", "t2"}
    assert "infospace_novelty" in teacher_rows[0]
    assert "prototype_drift" in teacher_rows[0]
    assert "infospace_redundancy" in teacher_rows[0]
    assert {"level", "prototype", "prototype_sample_count", "prototype_tile_count", "center_available"}.issubset(
        prototype_rows[0]
    )
    assert summary_rows[-1]["plateau_consensus"] in {"true", "false"}
    assert result["recommendation"]["recommended_prototype_sample_count"] in {2, 4, 6}


def test_missing_centers_are_reported_without_global_mean_fill(tmp_path: Path) -> None:
    module = _load_module()
    prototype_samples, locked, contract = _write_inputs(tmp_path)
    output_root = tmp_path / "curve"
    vectors = {
        tile_id: np.array([idx + 1.0, 1.0], dtype=np.float32)
        for idx, tile_id in enumerate(["a1", "a2", "a3", "a4", "a5", "a6", "v_skip", "v1", "v2"])
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
            prototype_sample_manifest=str(prototype_samples),
            locked_val_manifest=str(locked),
            candidate_manifest="",
            prototype_contract=str(contract),
            prototype_sample_counts="1",
            teachers="t",
            output_root=str(output_root),
            seed=13,
            split_seed=13,
            locked_val_fraction=0.2,
            locked_val_count=0,
            prototype_sample_group_key="tile_id",
            bootstrap_iterations=0,
            plateau_delta_epsilon=0.001,
            plateau_drift_threshold=0.05,
            plateau_redundancy_threshold=0.9,
            plot_formats="png",
            no_plots=True,
        )
    )

    row = _rows(output_root / "infospace_information_by_teacher.csv")[0]
    assert int(row["available_l1_centers"]) == 1
    assert int(row["available_l2_centers"]) <= 1
    assert row["missing_l1_centers"] or row["missing_l2_centers"]


def test_default_prototype_sample_counts_stop_at_three_thousand() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.seed == 13


def test_teacher_aliases_resolve_local_feature_directories(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "features"
    h1_dir = root / "h1"
    uni2_dir = root / "uni2"
    h1_dir.mkdir(parents=True)
    uni2_dir.mkdir(parents=True)
    h1_package = h1_dir / "slide_a.h_optimus_1.features.iac"
    uni2_package = uni2_dir / "slide_a.uni2_h.features.iac"
    h1_package.write_text("", encoding="utf-8")
    uni2_package.write_text("", encoding="utf-8")

    paths = module._resolve_teacher_paths(
        argparse.Namespace(
            teacher_feature_root=str(root),
            teacher_feature_packages="",
            teachers="h1,uni2",
        )
    )

    assert sorted(paths) == ["h_optimus_1", "uni2_h"]
    assert paths["h_optimus_1"] == [h1_package]
    assert paths["uni2_h"] == [uni2_package]


def test_explicit_teacher_package_aliases_are_canonicalized() -> None:
    module = _load_module()
    paths = module._teacher_paths_from_arg("h1=/tmp/h1.features.iac,uni2=/tmp/uni2.features.iac")
    assert sorted(paths) == ["h_optimus_1", "uni2_h"]
