#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from iatro.iac.adapters.features import FeatureCacheReader


TILE_SUFFIX = ".tiles.iac"
TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")


def _strip_suffix(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise ValueError(f"name does not end with {suffix}: {name}")
    return name[: -len(suffix)]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(denom, 1e-12, None)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    denom = float(np.linalg.norm(vector))
    if denom <= 1e-12:
        raise ValueError("prototype vector has near-zero norm")
    return vector / denom


def _mean_unit(features: list[np.ndarray], label: str) -> np.ndarray:
    if not features:
        raise ValueError(f"no features available for prototype label={label}")
    matrix = _normalize_rows(np.stack(features, axis=0))
    return _normalize_vector(matrix.mean(axis=0))


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"unsupported training manifest: {path}")
    return payload


def _load_annotations(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"annotation JSON missing annotations object: {path}")
    rows = [item for item in annotations.values() if item.get("tile_id") and item.get("l1")]
    if not rows:
        raise ValueError(f"annotation JSON has no usable annotations: {path}")
    return payload, rows


def _feature_path(manifest: dict[str, Any], item: dict[str, Any], teacher: str) -> Path:
    feature_roots = manifest.get("feature_roots")
    if not isinstance(feature_roots, dict) or teacher not in feature_roots:
        raise ValueError(f"manifest feature_roots missing teacher={teacher}")
    dataset = str(item.get("dataset") or "").strip()
    if not dataset:
        iac = str(item.get("iac") or "")
        dataset = Path(iac).parent.name
    if not dataset:
        raise ValueError(f"annotation row missing dataset/iac: tile_id={item.get('tile_id')}")
    iac_name = Path(str(item.get("iac") or "")).name
    if not iac_name:
        raise ValueError(f"annotation row missing iac name: tile_id={item.get('tile_id')}")
    stem = _strip_suffix(iac_name, str(manifest.get("tile_suffix", TILE_SUFFIX)))
    feature_dir = Path(feature_roots[teacher]) / dataset
    matches = sorted(feature_dir.glob(f"{stem}.*.features.iac"))
    if not matches:
        raise FileNotFoundError(
            f"missing feature package teacher={teacher} tile_id={item.get('tile_id')} expected={feature_dir}/{stem}.*.features.iac"
        )
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous feature packages teacher={teacher} tile_id={item.get('tile_id')}: {matches}")
    return matches[0]


def _read_feature(path: Path, row: int) -> np.ndarray:
    reader = FeatureCacheReader(path)
    try:
        return reader.read_feature_at(int(row)).astype(np.float32, copy=False).reshape(-1)
    finally:
        reader.close()


def _write_registry(
    path: Path,
    *,
    prototypes: list[np.ndarray],
    names: list[str],
    levels: list[int],
    exclusive: list[bool],
    counts: list[int],
    source: dict[str, Any],
) -> None:
    groups = ["primary_state" if level == 1 else "attribute_presence" for level in levels]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "prototypes": torch.from_numpy(np.stack(prototypes, axis=0).astype(np.float32)),
            "names": names,
            "groups": groups,
            "levels": levels,
            "exclusive": exclusive,
            "counts": counts,
            "source": source,
        },
        path,
    )


def _write_supervision_csv(path: Path, rows: list[dict[str, Any]], source_split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "slide_id",
                "patient_id",
                "level1_label",
                "source_split",
                "adjudicated",
                "dataset",
                "iac",
                "row",
            ],
        )
        writer.writeheader()
        for item in rows:
            tile_id = str(item["tile_id"]).strip()
            if tile_id in seen:
                raise ValueError(f"duplicate annotated tile_id: {tile_id}")
            seen.add(tile_id)
            slide_id = str(item.get("slide") or item.get("slide_id") or tile_id).strip()
            writer.writerow(
                {
                    "tile_id": tile_id,
                    "slide_id": slide_id,
                    "patient_id": str(item.get("patient_id") or slide_id).strip(),
                    "level1_label": str(item["l1"]).strip(),
                    "source_split": source_split,
                    "adjudicated": "true",
                    "dataset": str(item.get("dataset") or Path(str(item.get("iac") or "")).parent.name).strip(),
                    "iac": str(item.get("iac") or "").strip(),
                    "row": str(item.get("row")),
                }
            )


def _collect_teacher_features(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    teacher: str,
) -> tuple[list[np.ndarray], int]:
    features: list[np.ndarray] = []
    dim = 0
    for item in rows:
        feature = _read_feature(_feature_path(manifest, item, teacher), int(item["row"]))
        if dim == 0:
            dim = int(feature.shape[0])
        elif int(feature.shape[0]) != dim:
            raise ValueError(f"feature dim mismatch teacher={teacher}: got={feature.shape[0]} expected={dim}")
        features.append(feature)
    return features, dim


def _build_label_prototypes(
    rows: list[dict[str, Any]],
    features: list[np.ndarray],
    l1_names: list[str],
    l2_names: list[str],
) -> tuple[list[np.ndarray], list[int]]:
    prototypes: list[np.ndarray] = []
    counts: list[int] = []
    for label in l1_names:
        selected = [feature for item, feature in zip(rows, features) if str(item.get("l1")) == label]
        prototypes.append(_mean_unit(selected, label))
        counts.append(len(selected))
    for label in l2_names:
        selected = [
            feature
            for item, feature in zip(rows, features)
            if label in {str(value) for value in (item.get("l2") or item.get("level2_labels") or [])}
        ]
        prototypes.append(_mean_unit(selected, label))
        counts.append(len(selected))
    return prototypes, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training prototype assets from final annotation JSON.")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--source-split", default="train")
    args = parser.parse_args()

    annotation_path = Path(args.annotation_json)
    manifest_path = Path(args.training_manifest)
    output_dir = Path(args.output_dir)
    manifest = _load_manifest(manifest_path)
    payload, rows = _load_annotations(annotation_path)
    l1_names = [str(name) for name in payload.get("l1_prototypes", [])]
    if not l1_names:
        raise ValueError("annotation JSON missing l1_prototypes")
    # V2 teacher prototypes preserve only the four-way L1 semantic axis.
    # Spatial L2 is supervised exclusively from the geometry manifest.
    l2_names: list[str] = []
    names = list(l1_names)
    levels = [1] * len(l1_names)
    exclusive = [True] * len(l1_names)

    teacher_dims: dict[str, int] = {}
    for teacher in TEACHERS:
        features, dim = _collect_teacher_features(manifest, rows, teacher)
        teacher_dims[teacher] = dim
        prototypes, counts = _build_label_prototypes(rows, features, l1_names, l2_names)
        _write_registry(
            output_dir / f"{teacher}_hcc_semantic_prototypes.pt",
            prototypes=prototypes,
            names=names,
            levels=levels,
            exclusive=exclusive,
            counts=counts,
            source={
                "annotation_json": str(annotation_path),
                "training_manifest": str(manifest_path),
                "teacher": teacher,
                "builder": "build_prototype_assets_from_annotations.py",
            },
        )

    supervision_path = output_dir / "hcc_prototype_supervision_manifest.csv"
    _write_supervision_csv(supervision_path, rows, source_split=str(args.source_split))
    summary = {
        "annotation_json": str(annotation_path),
        "training_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "annotations": len(rows),
        "l1_prototypes": l1_names,
        "teacher_dims": teacher_dims,
        "supervision_manifest": str(supervision_path),
    }
    (output_dir / "prototype_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "prototype_assets_ok "
        f"annotations={len(rows)} output_dir={output_dir} "
        f"l1_classes={len(l1_names)}"
    )


if __name__ == "__main__":
    main()
