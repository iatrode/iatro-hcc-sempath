from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.prototypes import load_prototype_registry
from hcc_sempath.training.config import load_config, manifest_data_paths, teacher_names
from hcc_sempath.training.datasets import _open_feature_source
from hcc_sempath.training.manifest import load_training_manifest


STRATA = [
    "high_confidence_teacher_consensus",
    "high_teacher_disagreement",
    "low_margin",
    "intermediate",
]
TEACHERS = ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]


def _label_short(label: str) -> str:
    return (
        label.replace("HCC-tumor", "HCC")
        .replace("Background-liver", "BG")
        .replace("Inflammatory-stromal", "Infl/Strom")
        .replace("Degenerative-material", "Deg")
        .replace("-present", "")
        .replace("hepatocellular-parenchyma", "hep-par")
        .replace("inflammatory-cell", "inflam")
        .replace("fibrous-stroma", "fib-strom")
        .replace("steatosis-vacuolation", "steat")
        .replace("vascular-structure", "vascular")
        .replace("ductular-portal", "duct/portal")
    )


def _short_tile(tile_id: str) -> str:
    if "_" in tile_id:
        prefix, suffix = tile_id.rsplit("_", 1)
        slide = prefix.split(".")[0]
        return f"{slide}\n{suffix}"
    return tile_id[:36]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tile_image(meta: dict, readers: dict[str, TilePackageReader]):
    path = meta["tile_package_path"]
    reader = readers.get(path)
    if reader is None:
        reader = TilePackageReader(path)
        readers[path] = reader
    return reader.read_image(meta["tile_id"]).convert("RGB")


def _tile_to_teacher_paths(cfg_path: Path, split: str) -> dict[str, dict[str, str]]:
    cfg = load_config(cfg_path)
    cfg["data"]["exval_tile_fraction"] = 1.0
    cfg["data"]["val_tile_fraction"] = 1.0
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, split)
    names = teacher_names(cfg)
    return {
        str(tile_path): {name: teacher_packages[name][idx] for name in names}
        for idx, tile_path in enumerate(tile_packages)
    }


def _read_feature(source, row: int, teacher: str) -> np.ndarray:
    if source.__class__.__name__ == "MergedTeacherFeatureCacheReader":
        return source.read_feature_at(row, teacher)
    return source.read_feature_at(row)


def _prototype_semantics(
    *,
    tile_ids: set[str],
    metadata: dict[str, dict],
    cfg_path: Path,
    prototype_dir: Path,
    split: str,
) -> dict[str, dict]:
    tile_to_teacher_paths = _tile_to_teacher_paths(cfg_path, split)
    result = {tile_id: {} for tile_id in tile_ids}
    for teacher in TEACHERS:
        registry = load_prototype_registry(prototype_dir / f"{teacher}_hcc_semantic_prototypes.pt")
        prototypes = F.normalize(registry.prototypes.float(), dim=1)
        primary_idx = torch.tensor(registry.primary_indices, dtype=torch.long)
        attr_idx = torch.tensor(registry.attribute_indices, dtype=torch.long)
        grouped: dict[str, list[tuple[str, int]]] = {}
        for tile_id in tile_ids:
            meta = metadata[tile_id]
            path = tile_to_teacher_paths[meta["tile_package_path"]][teacher]
            grouped.setdefault(path, []).append((tile_id, int(meta["source_row"])))
        for path, items in grouped.items():
            source = _open_feature_source(Path(path))
            try:
                for tile_id, source_row in items:
                    feature = torch.from_numpy(_read_feature(source, source_row, teacher).astype("float32")).view(1, -1)
                    feature = F.normalize(feature, dim=1)
                    logits = (feature @ prototypes.T)[0]
                    p_logits = logits.index_select(0, primary_idx)
                    p_local = int(torch.argmax(p_logits).item())
                    p_idx = int(primary_idx[p_local].item())
                    a_logits = logits.index_select(0, attr_idx)
                    a_local = int(torch.argmax(a_logits).item())
                    a_idx = int(attr_idx[a_local].item())
                    result[tile_id][f"{teacher}_l1"] = registry.names[p_idx]
                    result[tile_id][f"{teacher}_l1_score"] = float(p_logits[p_local].item())
                    result[tile_id][f"{teacher}_l2"] = registry.names[a_idx]
                    result[tile_id][f"{teacher}_l2_score"] = float(a_logits[a_local].item())
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
    for tile_id, values in result.items():
        l1_values = [values.get(f"{teacher}_l1", "") for teacher in TEACHERS]
        l2_values = [values.get(f"{teacher}_l2", "") for teacher in TEACHERS]
        l1_counts = sorted(
            ((label, l1_values.count(label)) for label in set(l1_values) if label),
            key=lambda item: (-item[1], item[0]),
        )
        l2_counts = sorted(
            ((label, l2_values.count(label)) for label in set(l2_values) if label),
            key=lambda item: (-item[1], item[0]),
        )
        values["l1_summary"] = ", ".join(f"{_label_short(label)}x{count}" for label, count in l1_counts[:2])
        values["l2_summary"] = ", ".join(f"{_label_short(label)}x{count}" for label, count in l2_counts[:2])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="experiments/07_full_exval_cache/results/manifests/exval_sampled_z_hcc_metadata.csv")
    parser.add_argument("--strata", default="experiments/09_representation_audit/results/query_failure_strata.csv")
    parser.add_argument("--retrieval", default="experiments/08_pre_review_gate/results/retrieval_z_hcc_exval.csv")
    parser.add_argument("--config", default="experiments/shared/configs/local_sampled_eval.yaml")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--split", default="exval")
    parser.add_argument("--output-dir", default="experiments/reports/case_qc")
    parser.add_argument("--queries-per-stratum", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=3)
    args = parser.parse_args()

    metadata = {row["tile_id"]: row for row in _read_csv(Path(args.metadata))}
    strata_rows = _read_csv(Path(args.strata))
    retrieval_rows = _read_csv(Path(args.retrieval))
    retrieval_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval_rows:
        retrieval_by_query[row["query_id"]].append(row)
    for rows in retrieval_by_query.values():
        rows.sort(key=lambda row: int(row["rank"]))

    selected = []
    selected_by_stratum: dict[str, list[dict]] = {}
    for stratum in STRATA:
        candidates = [row for row in strata_rows if row["stratum"] == stratum]
        candidates.sort(
            key=lambda row: (
                -float(row["z_hcc_margin"]) if stratum != "low_margin" else float(row["z_hcc_margin"]),
                row["query_id"],
            )
        )
        selected_by_stratum[stratum] = candidates[: args.queries_per_stratum]
        selected.extend(selected_by_stratum[stratum])

    semantic_tile_ids = {row["query_tile_id"] for row in selected}
    for row in selected:
        for neighbor in retrieval_by_query[row["query_id"]][: args.neighbors]:
            semantic_tile_ids.add(neighbor["neighbor_tile_id"])
    semantics = _prototype_semantics(
        tile_ids=semantic_tile_ids,
        metadata=metadata,
        cfg_path=Path(args.config),
        prototype_dir=Path(args.prototype_dir),
        split=args.split,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = []
    readers: dict[str, TilePackageReader] = {}
    try:
        n_cols = 1 + int(args.neighbors)
        sheet_paths = []
        for stratum, queries in selected_by_stratum.items():
            if not queries:
                continue
            fig, axes = plt.subplots(
                len(queries),
                n_cols,
                figsize=(3.05 * n_cols, 3.15 * len(queries)),
                squeeze=False,
            )
            for row_idx, query in enumerate(queries):
                query_id = query["query_id"]
                q_tile = query["query_tile_id"]
                q_meta = metadata[q_tile]
                tiles = [("query", q_tile, "", q_meta)]
                for neighbor in retrieval_by_query[query_id][: args.neighbors]:
                    n_tile = neighbor["neighbor_tile_id"]
                    tiles.append((f"rank {neighbor['rank']}", n_tile, neighbor["cosine"], metadata[n_tile]))
                for col_idx, (label, tile_id, cosine, meta) in enumerate(tiles):
                    ax = axes[row_idx, col_idx]
                    ax.imshow(_tile_image(meta, readers))
                    if col_idx == 0:
                        title = (
                            f"QUERY q={query_id}\n"
                            f"margin={float(query['z_hcc_margin']):.3f} "
                            f"teacher={float(query['mean_teacher_pair_cosine']):.3f} "
                            f"dis={float(query['mean_teacher_disagreement']):.3f}\n"
                            f"{_short_tile(tile_id)}"
                        )
                    else:
                        title = f"{label} cos={float(cosine):.3f}\n{_short_tile(tile_id)}"
                    ax.set_title(title, fontsize=8)
                    footer = (
                        f"L1={semantics[tile_id].get('l1_summary', '')}\n"
                        f"L2={semantics[tile_id].get('l2_summary', '')}\n"
                        f"patient={meta['patient_id']}\n"
                        f"x={meta['x']} y={meta['y']}\n"
                        f"row={meta['source_row']}"
                    )
                    ax.text(
                        0.01,
                        0.01,
                        footer,
                        transform=ax.transAxes,
                        fontsize=6.5,
                        va="bottom",
                        ha="left",
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.62, "pad": 2, "edgecolor": "none"},
                    )
                    ax.axis("off")
                    case_rows.append({
                        "query_id": query_id,
                        "stratum": query["stratum"],
                        "role": label,
                        "tile_id": tile_id,
                        "slide_id": meta["slide_id"],
                        "source_row": meta["source_row"],
                        "x": meta["x"],
                        "y": meta["y"],
                        "z_hcc_margin": query["z_hcc_margin"],
                        "mean_teacher_pair_cosine": query["mean_teacher_pair_cosine"],
                        "mean_teacher_disagreement": query["mean_teacher_disagreement"],
                        "retrieval_cosine": cosine,
                        **semantics[tile_id],
                    })
            fig.suptitle(stratum, y=0.995, fontsize=11)
            fig.tight_layout()
            sheet_path = out_dir / f"case_qc_{stratum}.png"
            fig.savefig(sheet_path, dpi=180)
            plt.close(fig)
            sheet_paths.append(sheet_path.name)
    finally:
        for reader in readers.values():
            reader.close()

    _write_csv(out_dir / "case_qc_items.csv", case_rows, list(case_rows[0]))
    report = out_dir / "case_qc_summary.md"
    summary_text = (
        "# Case QC Summary\n\n"
        f"Selected query strata: {', '.join(STRATA)}.\n\n"
        f"Queries per stratum: {args.queries_per_stratum}.\n\n"
        f"Neighbors shown per query: {args.neighbors}.\n\n"
        "Outputs:\n\n"
        + "".join(f"- `{name}`\n" for name in sheet_paths)
        + "- `case_qc_items.csv`\n"
    )
    report.write_text(
        summary_text,
        encoding="utf-8",
    )
    print(f"case_qc_ok queries={len(selected)} output={out_dir}")


if __name__ == "__main__":
    main()
