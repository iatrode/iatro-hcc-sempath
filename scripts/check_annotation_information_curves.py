#!/usr/bin/env python3
"""Compute L1/L2 annotation information curves and issue one stopping decision."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_L1_ANNOTATION = (
    REPO_ROOT
    / "annotations"
    / "hcc_prototype_review.final_l1.json"
)
DEFAULT_L2_ANNOTATION = REPO_ROOT / "annotations" / "hcc_l2_roi_v2.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "local" / "mac" / "manifest.yaml"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "diagnostics"
    / "annotation_information_curve_current"
)


def _load_script_module(stem: str) -> Any:
    path = REPO_ROOT / "scripts" / f"{stem}.py"
    module_name = f"hcc_sempath_script_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _feature_packages_from_manifest(path: Path) -> dict[str, list[Path]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    roots = payload.get("feature_roots")
    if not isinstance(roots, dict) or not roots:
        raise ValueError(f"manifest has no feature_roots mapping: {path}")
    packages: dict[str, list[Path]] = {}
    for teacher, raw_root in roots.items():
        root = Path(str(raw_root)).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(
                f"teacher feature root is missing: teacher={teacher} root={root}"
            )
        matches = sorted(root.rglob("*.features.iac"))
        if not matches:
            matches = sorted(root.rglob("*features*.iac"))
        if not matches:
            raise FileNotFoundError(
                f"no teacher feature IAC packages: teacher={teacher} root={root}"
            )
        packages[str(teacher)] = matches
    return packages


def _encode_feature_packages(packages: dict[str, list[Path]]) -> str:
    return ",".join(
        f"{teacher}={'|'.join(str(path) for path in paths)}"
        for teacher, paths in packages.items()
    )


def decide_l1(
    result: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    teacher_rows = list(result.get("teacher", []))
    available_counts = [
        int(value)
        for value in report.get("prototype_sample_counts_available", [])
    ]
    evaluated_count = max(available_counts, default=0)
    latest_teacher_rows = [
        row
        for row in teacher_rows
        if int(row.get("prototype_sample_count", 0)) == evaluated_count
    ]
    missing_centres = sorted(
        {
            name
            for row in latest_teacher_rows
            for name in str(row.get("missing_l1_centers", "")).split(";")
            if name
        }
    )
    fixed = dict(
        report.get("fixed_probe_information")
        or result.get("fixed_probe_information")
        or {}
    )
    global_curve = dict(fixed.get("global", {}))
    classes = {
        name: {
            **value,
            "decision": (
                "stop" if value.get("enough_now", False) else "continue"
            ),
            "current_class_tile_count": int(
                value.get("positive_tile_count", 0)
            ),
        }
        for name, value in sorted(
            dict(fixed.get("classes", {})).items()
        )
    }
    class_blockers = [
        name
        for name, decision in classes.items()
        if decision["decision"] != "stop"
    ]
    global_ready = bool(global_curve.get("enough_now", False))
    ready = (
        global_ready
        and not missing_centres
        and bool(classes)
        and not class_blockers
    )
    blockers: list[str] = []
    if not global_ready:
        blockers.append("global L1 information curve is not confirmed")
    if missing_centres:
        blockers.append(
            "missing teacher-space centres: " + ", ".join(missing_centres)
        )
    if class_blockers:
        blockers.append(
            "unconfirmed L1 class curves: " + ", ".join(class_blockers)
        )
    return {
        "decision": "stop" if ready else "continue",
        "current_annotation_tile_count": int(
            report.get("prototype_sample_pool_count", 0)
        ),
        "evaluated_tile_count": evaluated_count,
        "global_curve": {
            **global_curve,
            "decision": "stop" if global_ready else "continue",
        },
        "classes": classes,
        "blockers": blockers,
    }


def decide_l2(report: dict[str, Any]) -> dict[str, Any]:
    attributes = dict(report.get("attributes", {}))
    components: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for name, value in sorted(attributes.items()):
        status = str(value.get("status", "not_assessable"))
        ready = status == "provisionally_stable" and bool(
            value.get("enough_now", False)
        )
        coverage = value.get("coverage", {})
        components[name] = {
            "decision": "stop" if ready else "continue",
            "status": status,
            "reason": str(value.get("reason", "")),
            "positive_tile_count": int(
                coverage.get("positive_tile_count", 0)
            ),
            "positive_slide_count": int(
                coverage.get("positive_slide_count", 0)
            ),
            "candidate_reference_tile_count": (
                value.get(
                    "recommended_reference_tile_count_by_ratio",
                    {},
                ).get("0.35")
                if ready
                else None
            ),
            "teacher_low_gain_support": value.get(
                "teacher_low_gain_support_by_teacher_ratio", {}
            ).get("0.35", {}),
            "candidate_reference_tile_count_by_teacher": value.get(
                "recommended_reference_tile_count_by_teacher_ratio", {}
            ).get("0.35", {}),
        }
        if not ready:
            blockers.append(name)
    ready = bool(components) and not blockers
    return {
        "decision": "stop" if ready else "continue",
        "components": components,
        "blockers": blockers,
    }


def _run_l1(output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _load_script_module("prototype_information_curve")
    command = [
        "--annotation-json",
        str(DEFAULT_L1_ANNOTATION),
        "--prototype-contract",
        str(DEFAULT_L1_ANNOTATION),
        "--prototype-levels",
        "l1",
        "--output-root",
        str(output_root),
        "--prototype-sample-counts",
        "100,200,400,800,1200,1600,2000,3000",
        "--seed",
        "13",
        "--workers",
        "0",
        "--bootstrap-iterations",
        "500",
        "--fixed-probe-resamples",
        "16",
        "--no-pca",
    ]
    packages = _feature_packages_from_manifest(DEFAULT_MANIFEST)
    command.extend(
        ["--teacher-feature-packages", _encode_feature_packages(packages)]
    )
    command.extend(["--teachers", ",".join(packages)])
    l1_args = module.build_parser().parse_args(command)
    result = module.run(l1_args)
    report_path = output_root / "infospace_information_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return result, report


def _run_l2(output_root: Path) -> dict[str, Any]:
    module = _load_script_module("roi_information_curve")
    packages = _feature_packages_from_manifest(DEFAULT_MANIFEST)
    command = [
        "--annotation-json",
        str(DEFAULT_L2_ANNOTATION),
        "--output-root",
        str(output_root),
        "--teacher-feature-packages",
        _encode_feature_packages(packages),
        "--sample-counts",
        (
            "5,10,15,20,30,40,50,60,70,80,90,100,"
            "120,140,160,180,200,225,250,275,300,350,400,450,500"
        ),
        "--seed",
        "13",
        "--resamples",
        "32",
        "--elbow-ratio",
        "0.35",
        "--elbow-support",
        "0.80",
        "--confirmation-increments",
        "3",
    ]
    l2_args = module.build_parser().parse_args(command)
    return module.run(l2_args)


def run() -> dict[str, Any]:
    output_root = DEFAULT_OUTPUT_ROOT.resolve()
    l1_output = output_root / "l1"
    l2_output = output_root / "l2"
    l1_result, l1_report = _run_l1(l1_output)
    l2_report = _run_l2(l2_output)

    l1_decision = decide_l1(l1_result, l1_report)
    l2_decision = decide_l2(l2_report)
    overall_ready = (
        l1_decision["decision"] == "stop"
        and l2_decision["decision"] == "stop"
    )

    l1_annotation = DEFAULT_L1_ANNOTATION.resolve()
    l2_annotation = DEFAULT_L2_ANNOTATION.resolve()
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "stop_annotation"
            if overall_ready
            else "continue_annotation"
        ),
        "decision_rule": (
            "stop only when one fixed slide-separated probe shows confirmed "
            "low marginal gain under nested reference growth for the global "
            "L1 curve and every L1 class; every L2 component must pass "
            "separately in all four frozen teacher spaces, and L2 coverage QC "
            "must also pass"
        ),
        "l1": l1_decision,
        "l2": l2_decision,
        "inputs": {
            "l1_annotation_sha256": _sha256(l1_annotation),
            "l2_annotation_sha256": _sha256(l2_annotation),
        },
        "curve_reports": {
            "l1": str(
                l1_output / "infospace_information_report.json"
            ),
            "l2": str(l2_output / "roi_information_report.json"),
        },
    }
    _write_json(output_root / "annotation_stop_report.json", report)

    summary_rows: list[dict[str, Any]] = [
        {
            "level": "L1",
            "unit": "all",
            "decision": l1_decision["decision"],
            "status": l1_decision["global_curve"]["decision"],
            "positive_tile_count": l1_decision[
                "current_annotation_tile_count"
            ],
            "reason": "; ".join(l1_decision["blockers"])
            or "all L1 information gates passed",
        }
    ]
    summary_rows.extend(
        {
            "level": "L1",
            "unit": name,
            "decision": value["decision"],
            "status": (
                "confirmed"
                if value["decision"] == "stop"
                else "unconfirmed"
            ),
            "positive_tile_count": value["current_class_tile_count"],
            "reason": value["reason"],
        }
        for name, value in l1_decision["classes"].items()
    )
    summary_rows.append(
        {
            "level": "L2",
            "unit": "all",
            "decision": l2_decision["decision"],
            "status": l2_decision["decision"],
            "positive_tile_count": "",
            "reason": (
                "unconfirmed components: "
                + ", ".join(l2_decision["blockers"])
                if l2_decision["blockers"]
                else "all L2 information gates passed"
            ),
        }
    )
    summary_rows.extend(
        {
            "level": "L2",
            "unit": name,
            "decision": value["decision"],
            "status": value["status"],
            "positive_tile_count": value["positive_tile_count"],
            "reason": value["reason"],
        }
        for name, value in l2_decision["components"].items()
    )
    _write_csv(output_root / "annotation_stop_summary.csv", summary_rows)

    print(
        "[annotation-curves] "
        f"L1={l1_decision['decision'].upper()} "
        f"L2={l2_decision['decision'].upper()} "
        f"OVERALL={report['decision'].upper()}",
        flush=True,
    )
    if l1_decision["blockers"]:
        print(
            "[annotation-curves] L1 blockers: "
            + "; ".join(l1_decision["blockers"]),
            flush=True,
        )
    if l2_decision["blockers"]:
        print(
            "[annotation-curves] L2 blockers: "
            + ", ".join(l2_decision["blockers"]),
            flush=True,
        )
    print(
        f"[annotation-curves] report={output_root / 'annotation_stop_report.json'}",
        flush=True,
    )
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
