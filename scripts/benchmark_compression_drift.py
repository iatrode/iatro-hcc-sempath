from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config

from hcc_sempath.teachers import TimmTeacherEncoder
from hcc_sempath.tile_package import TilePackageReader, read_package_manifest, read_package_metadata


def _psnr(rmse: float) -> float:
    return 99.0 if rmse == 0 else 20 * math.log10(255.0 / rmse)


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


@torch.no_grad()
def _features(model, transform, images, device: str, batch_size: int) -> np.ndarray:
    feats = []
    model = model.to(device).eval()
    for start in range(0, len(images), batch_size):
        batch = torch.stack([transform(image.convert("RGB")) for image in images[start : start + batch_size]]).to(device)
        feats.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def compare_package(
    reference_package: Path,
    candidate_package: Path,
    tile_ids: list[str],
    model,
    transform,
    device: str,
    batch_size: int,
) -> dict:
    ref_reader = TilePackageReader(reference_package)
    cand_reader = TilePackageReader(candidate_package)
    try:
        ref_images = [ref_reader.read_image(tile_id) for tile_id in tile_ids]
        cand_images = [cand_reader.read_image(tile_id) for tile_id in tile_ids]
    finally:
        ref_reader.close()
        cand_reader.close()

    maes = []
    rmses = []
    psnrs = []
    max_abs = []
    for ref_image, cand_image in zip(ref_images, cand_images):
        ref = np.asarray(ref_image.convert("RGB"), dtype=np.float32)
        cand = np.asarray(cand_image.convert("RGB"), dtype=np.float32)
        diff = cand - ref
        mae = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff * diff).mean()))
        maes.append(mae)
        rmses.append(rmse)
        psnrs.append(_psnr(rmse))
        max_abs.append(float(np.abs(diff).max()))

    ref_features = _features(model, transform, ref_images, device, batch_size)
    cand_features = _features(model, transform, cand_images, device, batch_size)
    ref_norm = ref_features / np.maximum(np.linalg.norm(ref_features, axis=1, keepdims=True), 1e-12)
    cand_norm = cand_features / np.maximum(np.linalg.norm(cand_features, axis=1, keepdims=True), 1e-12)
    cosine = (ref_norm * cand_norm).sum(axis=1)
    l2 = np.linalg.norm(cand_features - ref_features, axis=1)

    return {
        "package": str(candidate_package),
        "package_bytes": candidate_package.stat().st_size,
        "header": read_package_metadata(candidate_package),
        "image_mae": _summarize(maes),
        "image_rmse": _summarize(rmses),
        "image_psnr": _summarize(psnrs),
        "image_max_abs": _summarize(max_abs),
        "feature_cosine": _summarize(cosine.tolist()),
        "feature_l2": _summarize(l2.tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image and teacher-feature drift across compressed IatroCache packages.")
    parser.add_argument("--reference-package", required=True)
    parser.add_argument("--candidate-package", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tiles", type=int, default=256)
    args = parser.parse_args()

    reference_package = Path(args.reference_package)
    records = read_package_manifest(reference_package)
    if args.max_tiles > 0 and len(records) > args.max_tiles:
        indices = np.linspace(0, len(records) - 1, args.max_tiles, dtype=int).tolist()
        records = [records[i] for i in indices]
    tile_ids = [record.tile_id for record in records]

    model = TimmTeacherEncoder(args.model_name, pretrained=args.pretrained)
    data_config = resolve_model_data_config(model.model)
    data_config["input_size"] = (3, args.image_size, args.image_size)
    transform = create_transform(**data_config, is_training=False)

    result = {
        "reference_package": str(reference_package),
        "reference_bytes": reference_package.stat().st_size,
        "reference_header": read_package_metadata(reference_package),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "image_size": args.image_size,
        "device": args.device,
        "num_tiles": len(tile_ids),
        "comparisons": [
            compare_package(
                reference_package,
                Path(candidate),
                tile_ids,
                model,
                transform,
                args.device,
                args.batch_size,
            )
            for candidate in args.candidate_package
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"compression_drift_ok output={output}")


if __name__ == "__main__":
    main()
