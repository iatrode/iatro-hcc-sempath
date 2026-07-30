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

from hcc_sempath.training.feature_pack_merge import (
    MergedTeacherFeatureCacheReader,
    is_merged_teacher_feature_package,
)


TILE_SUFFIX = ".tiles.iac"
TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")
TARGET_PER_CLASS = 400
SEPARATION_WEIGHT = 32.0


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
    rows = sorted(
        (
            item
            for item in annotations.values()
            if item.get("tile_id") and item.get("classification")
        ),
        key=lambda item: str(item["tile_id"]),
    )
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


def _read_features(
    path: Path,
    rows: list[int],
    teacher: str,
) -> np.ndarray:
    if is_merged_teacher_feature_package(path):
        merged_reader = MergedTeacherFeatureCacheReader(path)
        try:
            return merged_reader.read_features_many_at(rows, [teacher])[teacher]
        finally:
            merged_reader.close()
    reader = FeatureCacheReader(path)
    try:
        return reader.read_features_at(
            [int(row) for row in rows]
        ).astype(np.float32, copy=False)
    finally:
        reader.close()


def _write_registry(
    path: Path,
    *,
    prototypes: list[np.ndarray],
    names: list[str],
    counts: list[int],
    source: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "prototypes": torch.from_numpy(np.stack(prototypes, axis=0).astype(np.float32)),
            "names": names,
            "counts": counts,
            "source": source,
        },
        path,
    )


def _write_supervision_csv(
    path: Path,
    split_rows: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "slide_id",
                "patient_id",
                "classification_label",
                "source_split",
                "adjudicated",
                "dataset",
                "iac",
                "row",
            ],
        )
        writer.writeheader()
        for source_split, rows in split_rows:
            for item in rows:
                tile_id = str(item["tile_id"]).strip()
                if tile_id in seen:
                    raise ValueError(
                        "train/validation annotation overlap or duplicate: "
                        f"{tile_id}"
                    )
                seen.add(tile_id)
                slide_id = str(
                    item.get("slide")
                    or item.get("slide_id")
                    or tile_id
                ).strip()
                writer.writerow(
                    {
                        "tile_id": tile_id,
                        "slide_id": slide_id,
                        "patient_id": str(
                            item.get("patient_id") or slide_id
                        ).strip(),
                        "classification_label": str(
                            item["classification"]
                        ).strip(),
                        "source_split": source_split,
                        "adjudicated": "true",
                        "dataset": str(
                            item.get("dataset")
                            or Path(
                                str(item.get("iac") or "")
                            ).parent.name
                        ).strip(),
                        "iac": str(item.get("iac") or "").strip(),
                        "row": str(item.get("row")),
                    }
                )


def _collect_teacher_features(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    teacher: str,
) -> tuple[list[np.ndarray], int]:
    grouped: dict[Path, list[tuple[int, int]]] = {}
    for index, item in enumerate(rows):
        path = _feature_path(manifest, item, teacher)
        grouped.setdefault(path, []).append((index, int(item["row"])))

    collected: list[np.ndarray | None] = [None] * len(rows)
    dim = 0
    for path, indexed_rows in grouped.items():
        matrix = _read_features(
            path,
            [row for _, row in indexed_rows],
            teacher,
        )
        for (index, _), feature in zip(indexed_rows, matrix, strict=True):
            if dim == 0:
                dim = int(feature.shape[0])
            elif int(feature.shape[0]) != dim:
                raise ValueError(
                    f"feature dim mismatch teacher={teacher}: "
                    f"got={feature.shape[0]} expected={dim}"
                )
            collected[index] = feature
    if any(feature is None for feature in collected):
        raise RuntimeError(f"incomplete feature collection for teacher={teacher}")
    return [feature for feature in collected if feature is not None], dim


def _build_label_prototypes(
    rows: list[dict[str, Any]],
    features: list[np.ndarray],
    classification_names: list[str],
    spatial_names: list[str],
) -> tuple[list[np.ndarray], list[int]]:
    prototypes: list[np.ndarray] = []
    counts: list[int] = []
    for label in classification_names:
        selected = [feature for item, feature in zip(rows, features) if str(item.get("classification")) == label]
        prototypes.append(_mean_unit(selected, label))
        counts.append(len(selected))
    for label in spatial_names:
        selected = [
            feature
            for item, feature in zip(rows, features)
            if label in {str(value) for value in (item.get("spatial") or item.get("spatial_labels") or [])}
        ]
        prototypes.append(_mean_unit(selected, label))
        counts.append(len(selected))
    return prototypes, counts


def _facility_order(
    similarity: np.ndarray,
    count: int,
    *,
    margin_rank: np.ndarray,
) -> list[int]:
    sample_count = int(similarity.shape[0])
    covered = np.zeros(sample_count, dtype=np.float32)
    selected = np.zeros(sample_count, dtype=bool)
    order: list[int] = []
    for _ in range(min(count, sample_count)):
        gain = np.maximum(
            similarity - covered[:, None],
            0.0,
        ).sum(axis=0, dtype=np.float64)
        score = gain * (
            1.0 + SEPARATION_WEIGHT * margin_rank
        )
        score[selected] = -np.inf
        chosen = int(np.argmax(score))
        order.append(chosen)
        selected[chosen] = True
        covered = np.maximum(covered, similarity[:, chosen])
    return order


def _global_class_margins(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    labels: list[str],
) -> np.ndarray:
    row_labels = np.asarray(
        [str(row["classification"]) for row in rows]
    )
    margin = np.zeros(len(rows), dtype=np.float32)
    for teacher in TEACHERS:
        centroids: list[np.ndarray] = []
        for label in labels:
            centroid = features[teacher][row_labels == label].mean(
                axis=0
            )
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids.append(centroid)
        scores = features[teacher] @ np.stack(centroids).T
        for label_index, label in enumerate(labels):
            mask = row_labels == label
            competitor = np.max(
                np.delete(scores[mask], label_index, axis=1),
                axis=1,
            )
            margin[mask] += (
                scores[mask, label_index] - competitor
            ) / len(TEACHERS)
    return margin


def _rank_margin(
    indices: list[int],
    margin: np.ndarray,
) -> np.ndarray:
    values = margin[np.asarray(indices)]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(indices), dtype=np.float32)
    ranks[order] = np.linspace(
        0.0,
        1.0,
        len(indices),
        dtype=np.float32,
    )
    return ranks


def _select_fixed_training_bank(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    classification_names: list[str],
) -> list[int]:
    normalized_features = {
        teacher: _normalize_rows(features[teacher])
        for teacher in TEACHERS
    }
    global_margin = _global_class_margins(
        rows,
        normalized_features,
        classification_names,
    )
    selected_indices: list[int] = []
    for label in classification_names:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row["classification"]) == label
        ]
        if len(indices) < TARGET_PER_CLASS:
            raise ValueError(
                f"{label} has {len(indices)} accepted tiles; "
                f"{TARGET_PER_CLASS} are required"
            )
        combined = np.zeros(
            (len(indices), len(indices)),
            dtype=np.float32,
        )
        for teacher in TEACHERS:
            matrix = normalized_features[teacher][indices]
            combined += np.clip(
                (matrix @ matrix.T + 1.0) * 0.5,
                0.0,
                1.0,
            ) / len(TEACHERS)
        local_order = _facility_order(
            combined,
            TARGET_PER_CLASS,
            margin_rank=_rank_margin(indices, global_margin),
        )
        selected_indices.extend(
            indices[index]
            for index in local_order
        )
    return sorted(selected_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training prototype assets from final annotation JSON.")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-annotation-json", default="")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--source-split", default="train")
    args = parser.parse_args()

    annotation_path = Path(args.annotation_json)
    manifest_path = Path(args.training_manifest)
    output_dir = Path(args.output_dir)
    manifest = _load_manifest(manifest_path)
    payload, rows = _load_annotations(annotation_path)
    classification_names = [str(name) for name in payload.get("classification_prototypes", [])]
    if not classification_names:
        raise ValueError("annotation JSON missing classification_prototypes")
    # Teacher prototypes define the frozen exclusive classification semantic axis; the
    # geometry manifest supplies the independent spatial supervision.
    spatial_names: list[str] = []
    names = list(classification_names)

    teacher_dims: dict[str, int] = {}
    teacher_features: dict[str, np.ndarray] = {}
    for teacher in TEACHERS:
        features, dim = _collect_teacher_features(manifest, rows, teacher)
        teacher_dims[teacher] = dim
        teacher_features[teacher] = np.stack(
            features,
            axis=0,
        ).astype(np.float32)

    selected_indices = _select_fixed_training_bank(
        rows,
        teacher_features,
        classification_names,
    )
    selected_rows = [rows[index] for index in selected_indices]
    for teacher in TEACHERS:
        selected_features = [
            teacher_features[teacher][index]
            for index in selected_indices
        ]
        prototypes, counts = _build_label_prototypes(
            selected_rows,
            selected_features,
            classification_names,
            spatial_names,
        )
        _write_registry(
            output_dir / f"{teacher}_hcc_semantic_prototypes.pt",
            prototypes=prototypes,
            names=names,
            counts=counts,
            source={
                "annotation_json": str(annotation_path),
                "training_manifest": str(manifest_path),
                "teacher": teacher,
                "builder": "build_prototype_assets_from_annotations.py",
                "selection": (
                    "four-teacher fixed greedy facility coverage"
                ),
                "target_per_class": TARGET_PER_CLASS,
            },
        )

    validation_rows: list[dict[str, Any]] = []
    validation_path: Path | None = None
    if str(args.validation_annotation_json).strip():
        validation_path = Path(args.validation_annotation_json)
        validation_payload, validation_rows = _load_annotations(
            validation_path
        )
        validation_names = [
            str(name)
            for name in validation_payload.get(
                "classification_prototypes",
                [],
            )
        ]
        if validation_names != classification_names:
            raise ValueError(
                "training/validation classification schemas differ"
            )
    supervision_path = output_dir / "hcc_prototype_supervision_manifest.csv"
    _write_supervision_csv(
        supervision_path,
        [
            (str(args.source_split), selected_rows),
            ("val", validation_rows),
        ],
    )
    summary = {
        "annotation_json": str(annotation_path),
        "validation_annotation_json": (
            None
            if validation_path is None
            else str(validation_path)
        ),
        "training_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "accepted_training_annotations": len(rows),
        "selected_training_annotations": len(selected_rows),
        "validation_annotations": len(validation_rows),
        "target_per_class": TARGET_PER_CLASS,
        "classification_prototypes": classification_names,
        "teacher_dims": teacher_dims,
        "supervision_manifest": str(supervision_path),
    }
    (output_dir / "prototype_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "prototype_assets_ok "
        f"train_selected={len(selected_rows)} "
        f"validation={len(validation_rows)} output_dir={output_dir} "
        f"classification_classes={len(classification_names)}"
    )


if __name__ == "__main__":
    main()
