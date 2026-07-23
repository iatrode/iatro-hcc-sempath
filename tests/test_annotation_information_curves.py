from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_annotation_information_curves.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_annotation_information_curves",
        script,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _l1_fixture(*, growing_class: str | None = None):
    classes = ["a", "b"]
    checkpoints = [100, 200, 300, 400]
    fixed = {
        "status": "provisionally_stable",
        "enough_now": True,
        "global": {
            "status": "provisionally_stable",
            "enough_now": True,
            "reason": "stable",
        },
        "classes": {
            name: {
                "status": (
                    "still_growing"
                    if name == growing_class
                    else "provisionally_stable"
                ),
                "enough_now": name != growing_class,
                "reason": (
                    "growing" if name == growing_class else "stable"
                ),
                "positive_tile_count": 40,
            }
            for name in classes
        },
    }
    result = {
        "teacher": [
            {
                "prototype_sample_count": checkpoints[-1],
                "teacher": teacher,
                "missing_l1_centers": "",
            }
            for teacher in ["t1", "t2"]
        ],
        "fixed_probe_information": fixed,
    }
    report = {
        "prototype_sample_counts_available": checkpoints,
        "prototype_sample_pool_count": 420,
        "fixed_probe_information": fixed,
    }
    return result, report


def test_l1_stop_requires_global_and_every_class_curve() -> None:
    module = _load_module()
    result, report = _l1_fixture()
    decision = module.decide_l1(result, report)

    assert decision["decision"] == "stop"
    assert set(decision["classes"]) == {"a", "b"}
    assert all(
        value["decision"] == "stop"
        for value in decision["classes"].values()
    )

    result, report = _l1_fixture(growing_class="b")
    decision = module.decide_l1(result, report)

    assert decision["decision"] == "continue"
    assert decision["classes"]["b"]["decision"] == "continue"
    assert "unconfirmed L1 class curves: b" in decision["blockers"]


def test_l2_stop_requires_every_component() -> None:
    module = _load_module()
    stable = {
        "status": "provisionally_stable",
        "enough_now": True,
        "reason": "stable",
        "coverage": {
            "positive_tile_count": 20,
            "positive_slide_count": 8,
        },
        "recommended_reference_tile_count_by_ratio": {"0.35": 20},
    }
    decision = module.decide_l2(
        {"attributes": {"a": stable, "b": stable}}
    )
    assert decision["decision"] == "stop"
    assert decision["components"]["a"][
        "candidate_reference_tile_count"
    ] == 20

    growing = {
        **stable,
        "status": "still_growing",
        "enough_now": False,
    }
    decision = module.decide_l2(
        {"attributes": {"a": stable, "b": growing}}
    )
    assert decision["decision"] == "continue"
    assert decision["blockers"] == ["b"]
    assert decision["components"]["b"][
        "candidate_reference_tile_count"
    ] is None


def test_manifest_feature_roots_resolve_packages(tmp_path: Path) -> None:
    module = _load_module()
    roots = {}
    for teacher in ["t1", "t2"]:
        root = tmp_path / teacher
        root.mkdir()
        (root / f"slide.{teacher}.features.iac").write_bytes(b"")
        roots[teacher] = str(root)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"feature_roots": roots}),
        encoding="utf-8",
    )

    packages = module._feature_packages_from_manifest(manifest)

    assert set(packages) == {"t1", "t2"}
    assert all(len(paths) == 1 for paths in packages.values())


def test_unified_entry_has_no_task_switches() -> None:
    module = _load_module()
    assert not hasattr(module, "build_parser")
