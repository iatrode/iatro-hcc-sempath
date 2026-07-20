from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from iatro.iac.adapters.features import build_teacher_feature_package
from iatro.iac.adapters.tiles import TilePackageReader, read_package_manifest, read_package_metadata
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.config import embedding_dim, load_config, manifest_data_paths, teacher_dims, teacher_names
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.manifest import load_training_manifest, package_stem


def _load_model(cfg: dict, checkpoint: Path, device: torch.device) -> HCCSemPathModel:
    names = teacher_names(cfg)
    dims = teacher_dims(cfg, names)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=False,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _localize_cfg(cfg: dict, manifest: str, prototype_dir: str, device: str, batch_size: int) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("runtime", {})["device"] = device
    cfg.setdefault("data", {})["train_manifest_path"] = manifest
    cfg["data"]["num_workers"] = 0
    cfg["data"]["val_tile_fraction"] = 1.0
    cfg["data"]["exval_tile_fraction"] = 1.0
    cfg["data"]["prototype_paths"] = {
        "gigapath": f"{prototype_dir}/gigapath_hcc_semantic_prototypes.pt",
        "h_optimus_1": f"{prototype_dir}/h_optimus_1_hcc_semantic_prototypes.pt",
        "uni2_h": f"{prototype_dir}/uni2_h_hcc_semantic_prototypes.pt",
        "virchow2": f"{prototype_dir}/virchow2_hcc_semantic_prototypes.pt",
    }
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["amp"] = False
    cfg["model"]["grad_checkpointing"] = False
    return cfg


def _sample_plan(counts: list[int], target: int, per_package_min: int, per_package_cap: int) -> list[int]:
    if target <= 0 or target >= sum(counts):
        return counts[:]
    mins = [min(count, per_package_min) for count in counts]
    if sum(mins) > target:
        # Fall back to at least one row per package, then distribute by size.
        mins = [1 if count > 0 else 0 for count in counts]
    caps = [min(count, per_package_cap) for count in counts]
    takes = mins[:]
    remaining = target - sum(takes)
    while remaining > 0:
        capacity = np.asarray([cap - take for cap, take in zip(caps, takes)], dtype=np.int64)
        candidates = np.flatnonzero(capacity > 0)
        if len(candidates) == 0:
            break
        weights = np.asarray([counts[i] for i in candidates], dtype=np.float64)
        raw = weights / weights.sum() * remaining
        add = np.floor(raw).astype(np.int64)
        add = np.minimum(add, capacity[candidates])
        if int(add.sum()) == 0:
            order = candidates[np.argsort(-(raw - np.floor(raw)))]
            for idx in order[:remaining]:
                if capacity[idx] > 0:
                    takes[int(idx)] += 1
                    remaining -= 1
                    if remaining <= 0:
                        break
            continue
        for idx, value in zip(candidates, add):
            takes[int(idx)] += int(value)
        remaining -= int(add.sum())
    return takes


def _even_rows(count: int, take: int) -> np.ndarray:
    if take >= count:
        return np.arange(count, dtype=np.int64)
    if take <= 0:
        return np.empty((0,), dtype=np.int64)
    rows = np.floor(np.linspace(0, count - 1, num=take, dtype=np.float64)).astype(np.int64)
    rows = np.unique(rows)
    while len(rows) < take:
        missing = take - len(rows)
        extra = np.linspace(0, count - 1, num=take + missing, dtype=np.float64).astype(np.int64)
        rows = np.unique(np.concatenate([rows, extra]))
    return rows[:take]


def _batch_tensor(arrays: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()


def _write_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "tile_id",
        "patient_id",
        "slide_id",
        "dataset",
        "split",
        "tile_package_path",
        "student_feature_package_path",
        "source_row",
        "sample_row",
        "x",
        "y",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache sampled exval z_hcc embeddings as IAC feature packages.")
    parser.add_argument("--config", default="artifacts/models/hcc-sempath-full/resolved_config.json")
    parser.add_argument("--checkpoint", default="artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt")
    parser.add_argument("--manifest", default="configs/local/mac/manifest.yaml")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--split", default="exval")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-tiles", type=int, default=200000)
    parser.add_argument("--per-package-min", type=int, default=256)
    parser.add_argument("--per-package-cap", type=int, default=2048)
    parser.add_argument("--output-dir", default="experiments/07_full_exval_cache/results")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    cfg = _localize_cfg(load_config(args.config), args.manifest, args.prototype_dir, args.device, args.batch_size)
    device = torch.device(cfg["runtime"]["device"])
    model = _load_model(cfg, Path(args.checkpoint), device)
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, _ = manifest_data_paths(cfg, manifest, args.split)
    counts = [len(read_package_manifest(path)) for path in tile_packages]
    takes = _sample_plan(counts, int(args.target_tiles), int(args.per_package_min), int(args.per_package_cap))

    out_dir = Path(args.output_dir)
    feature_root = out_dir / "student_z_hcc"
    progress_log = out_dir / "logs" / f"{args.split}_sampled_z_hcc_progress.log"
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text(
        "event=cache_start "
        f"split={args.split} target_tiles={args.target_tiles} batch_size={args.batch_size} device={device}\n",
        encoding="utf-8",
    )
    metadata_rows: list[dict] = []
    package_rows: list[dict] = []
    total_tiles = 0
    total_read_seconds = 0.0
    total_infer_seconds = 0.0
    emb_dim = embedding_dim(cfg)

    for package_idx, (tile_path_raw, count, take) in enumerate(zip(tile_packages, counts, takes, strict=True)):
        if take <= 0:
            continue
        tile_path = Path(tile_path_raw)
        dataset = tile_path.parent.name
        stem = package_stem(tile_path, str(manifest.get("tile_suffix", ".tiles.iac")))
        output_path = feature_root / dataset / f"{stem}.hcc_sempath_z_hcc.features.iac"
        if output_path.exists() and not args.overwrite:
            package_rows.append({
                "package_idx": package_idx,
                "tile_package_path": str(tile_path),
                "student_feature_package_path": str(output_path),
                "available_tiles": count,
                "sampled_tiles": take,
                "status": "exists_skipped",
            })
            continue

        source_records = read_package_manifest(tile_path)
        rows = _even_rows(count, take)
        sampled_records = [source_records[int(row)] for row in rows]
        tile_header = read_package_metadata(tile_path)
        reader = TilePackageReader(tile_path)

        def features():
            nonlocal total_read_seconds, total_infer_seconds, total_tiles
            arrays: list[np.ndarray] = []
            for source_row in rows:
                t0 = time.perf_counter()
                arrays.append(reader.read_array_at(int(source_row)))
                total_read_seconds += time.perf_counter() - t0
                if len(arrays) >= int(args.batch_size):
                    batch = {"images": _batch_tensor(arrays), "images_uint8": True}
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        values = model(_prepare_images(batch, cfg, device))["embedding_norm"].detach().cpu().numpy()
                    total_infer_seconds += time.perf_counter() - t1
                    total_tiles += len(arrays)
                    for value in values:
                        yield value.astype("float32", copy=False)
                    arrays.clear()
            if arrays:
                batch = {"images": _batch_tensor(arrays), "images_uint8": True}
                t1 = time.perf_counter()
                with torch.no_grad():
                    values = model(_prepare_images(batch, cfg, device))["embedding_norm"].detach().cpu().numpy()
                total_infer_seconds += time.perf_counter() - t1
                total_tiles += len(arrays)
                for value in values:
                    yield value.astype("float32", copy=False)

        try:
            build_teacher_feature_package(
                sampled_records,
                features(),
                output_path,
                teacher_name="hcc_sempath_z_hcc",
                dtype="float32",
                feature_dim=emb_dim,
                tile_width=int(tile_header["tile_width"]),
                tile_height=int(tile_header["tile_height"]),
                stride_x=int(tile_header["stride_x"]),
                stride_y=int(tile_header["stride_y"]),
                overwrite=bool(args.overwrite),
            )
        finally:
            reader.close()

        for sample_row, source_row in enumerate(rows):
            record = source_records[int(source_row)]
            metadata_rows.append({
                "tile_id": record.tile_id,
                "patient_id": record.patient_id,
                "slide_id": record.slide_id,
                "dataset": dataset,
                "split": args.split,
                "tile_package_path": str(tile_path),
                "student_feature_package_path": str(output_path),
                "source_row": int(source_row),
                "sample_row": int(sample_row),
                "x": record.x,
                "y": record.y,
            })
        package_rows.append({
            "package_idx": package_idx,
            "tile_package_path": str(tile_path),
            "student_feature_package_path": str(output_path),
            "available_tiles": count,
            "sampled_tiles": take,
            "status": "written",
        })
        elapsed = time.perf_counter() - started
        message = (
            f"package_done idx={package_idx + 1}/{len(tile_packages)} "
            f"dataset={dataset} sampled={take} total={total_tiles} "
            f"elapsed_sec={elapsed:.1f}"
        )
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

    metadata_path = out_dir / "manifests" / f"{args.split}_sampled_z_hcc_metadata.csv"
    _write_metadata(metadata_path, metadata_rows)
    package_manifest_path = out_dir / "manifests" / f"{args.split}_sampled_z_hcc_packages.csv"
    with package_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "package_idx",
                "tile_package_path",
                "student_feature_package_path",
                "available_tiles",
                "sampled_tiles",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(package_rows)

    elapsed = time.perf_counter() - started
    manifest_payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "target_tiles": int(args.target_tiles),
        "available_tiles": int(sum(counts)),
        "sampled_tiles_planned": int(sum(takes)),
        "sampled_tiles_written": int(total_tiles),
        "package_count": len(tile_packages),
        "read_decode_seconds": total_read_seconds,
        "inference_seconds": total_infer_seconds,
        "elapsed_seconds": elapsed,
        "tiles_per_sec_end_to_end": total_tiles / elapsed if elapsed > 0 else 0.0,
        "tiles_per_sec_inference": total_tiles / total_infer_seconds if total_infer_seconds > 0 else 0.0,
        "feature_root": str(feature_root),
        "metadata_path": str(metadata_path),
        "package_manifest_path": str(package_manifest_path),
        "progress_log": str(progress_log),
    }
    manifest_path = out_dir / "manifests" / f"{args.split}_sampled_z_hcc_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    report_path = Path("experiments/07_full_exval_cache/reports/sampled_exval_z_hcc_cache_qc.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Sampled Exval z_hcc IAC Cache QC\n\n"
        f"Sampled tiles written: {total_tiles}\n\n"
        f"Available exval tiles: {sum(counts)}\n\n"
        f"Package count: {len(tile_packages)}\n\n"
        f"End-to-end tiles/s: {manifest_payload['tiles_per_sec_end_to_end']:.2f}\n\n"
        f"Inference tiles/s: {manifest_payload['tiles_per_sec_inference']:.2f}\n\n"
        f"Metadata: `{metadata_path}`\n\n"
        f"Package manifest: `{package_manifest_path}`\n",
        encoding="utf-8",
    )
    message = (
        "sampled_exval_z_hcc_cache_ok "
        f"tiles={total_tiles} packages={len(tile_packages)} "
        f"end_to_end_tiles_per_sec={manifest_payload['tiles_per_sec_end_to_end']:.2f} "
        f"manifest={manifest_path}"
    )
    with progress_log.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


if __name__ == "__main__":
    main()
