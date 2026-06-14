from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import load_hcc_sempath_release
from hcc_sempath.training.config import load_config, manifest_data_paths
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.manifest import load_training_manifest
from hcc_sempath.training.prototype_images import load_prototype_image_bank


def _batch_tensor(arrays: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _write_thresholds(
    output: Path,
    scores: np.memmap,
    valid: np.memmap,
    prototype_path: Path,
    l2_names: list[str],
    *,
    phase: str,
    total_tiles: int,
) -> None:
    selected = np.flatnonzero(np.asarray(valid, dtype=bool))
    if selected.size == 0:
        raise ValueError("no cached scores available for threshold calculation")
    bank = load_prototype_image_bank(prototype_path)
    priors = bank.level2.float().mean(dim=0).cpu().numpy()
    thresholds = [
        float(np.quantile(np.asarray(scores[selected, idx]), 1.0 - float(prior)))
        for idx, prior in enumerate(priors)
    ]
    payload = {
        "version": 1,
        "method": "prototype-prior-matched training-score quantile",
        "status": "temporary_1_of_10" if phase == "tenth" else "full_training",
        "phase": phase,
        "cached_tiles": int(selected.size),
        "total_training_tiles": int(total_tiles),
        "l2_names": l2_names,
        "target_prevalence": priors.tolist(),
        "thresholds": thresholds,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache train L2 cosine scores and derive prior-matched thresholds.")
    parser.add_argument("--config", default="artifacts/models/hcc-sempath-full/resolved_config.json")
    parser.add_argument("--release-config", default="artifacts/release/config.json")
    parser.add_argument("--release-checkpoint", default="artifacts/release/hcc_sempath_release.pt")
    parser.add_argument("--manifest", default="configs/local/mac/manifest.yaml")
    parser.add_argument("--prototype-images", default="artifacts/prototypes/zhcc_hcc_prototype_images.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--phase", choices=("tenth", "remainder"), required=True)
    parser.add_argument("--output-dir", default="artifacts/caches/local_cache/train_l2_thresholds")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "train_l2_cosine_scores.float32.mmap"
    valid_path = output_dir / "train_l2_cosine_scores.valid.uint8.mmap"
    state_path = output_dir / "progress.json"
    threshold_path = output_dir / "thresholds.json"

    cfg = load_config(args.config)
    cfg = json.loads(json.dumps(cfg))
    cfg["runtime"]["device"] = args.device
    cfg["data"]["train_manifest_path"] = args.manifest
    cfg["train"]["amp"] = False
    device = torch.device(args.device)
    model, release_config = load_hcc_sempath_release(
        args.release_config,
        args.release_checkpoint,
        device,
    )
    l2_names = list(release_config["l2_names"])

    manifest = load_training_manifest(args.manifest)
    packages, _ = manifest_data_paths(cfg, manifest, "train")
    counts: list[int] = []
    for path in packages:
        reader = TilePackageReader(path)
        try:
            counts.append(reader.record_count)
        finally:
            reader.close()
    offsets = np.cumsum([0, *counts[:-1]], dtype=np.int64)
    total_tiles = int(sum(counts))

    scores = np.memmap(scores_path, mode="r+" if scores_path.exists() else "w+", dtype=np.float32, shape=(total_tiles, len(l2_names)))
    valid = np.memmap(valid_path, mode="r+" if valid_path.exists() else "w+", dtype=np.uint8, shape=(total_tiles,))
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"version": 1, "total_tiles": total_tiles, "completed": {"tenth": [], "remainder": []}}
    completed = set(int(value) for value in state["completed"][args.phase])
    started = time.perf_counter()
    processed = 0

    for package_idx, (path, count, offset) in enumerate(zip(packages, counts, offsets, strict=True)):
        if package_idx in completed:
            continue
        if args.phase == "tenth":
            rows = np.arange(0, count, 10, dtype=np.int64)
        else:
            rows = np.flatnonzero(np.arange(count, dtype=np.int64) % 10 != 0)
        reader = TilePackageReader(path)
        arrays: list[np.ndarray] = []
        batch_rows: list[int] = []

        def flush() -> None:
            nonlocal processed
            if not arrays:
                return
            batch = {"images": _batch_tensor(arrays), "images_uint8": True}
            with torch.inference_mode():
                values = model(_prepare_images(batch, cfg, device))["l2_cosine_scores"].cpu().numpy()
            indices = offset + np.asarray(batch_rows, dtype=np.int64)
            scores[indices] = values.astype(np.float32, copy=False)
            valid[indices] = 1
            processed += len(batch_rows)
            arrays.clear()
            batch_rows.clear()

        try:
            for row in rows:
                arrays.append(reader.read_array_at(int(row)))
                batch_rows.append(int(row))
                if len(arrays) >= int(args.batch_size):
                    flush()
            flush()
        finally:
            reader.close()
        scores.flush()
        valid.flush()
        completed.add(package_idx)
        state["completed"][args.phase] = sorted(completed)
        state["cached_tiles"] = int(valid.sum())
        state["updated_phase"] = args.phase
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        elapsed = time.perf_counter() - started
        print(
            f"package_done phase={args.phase} package={package_idx + 1}/{len(packages)} "
            f"processed={processed} cached={state['cached_tiles']} elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    _write_thresholds(
        threshold_path,
        scores,
        valid,
        Path(args.prototype_images),
        l2_names,
        phase=args.phase,
        total_tiles=total_tiles,
    )
    print(f"thresholds_ok phase={args.phase} output={threshold_path}", flush=True)


if __name__ == "__main__":
    main()
