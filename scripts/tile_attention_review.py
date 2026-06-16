from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hcc_sempath_mpl"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel, normalized_prototype_logits
from hcc_sempath.modeling.prototypes import PrototypeRegistry, load_prototype_registry
from hcc_sempath.training.config import embedding_dim, manifest_data_paths, teacher_dims, teacher_names
from hcc_sempath.training.datasets import _open_feature_source, _read_teacher_features_at
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.manifest import load_training_manifest
from hcc_sempath.training.prototype_images import (
    PrototypeImageBank,
    build_student_prototype_registry,
    load_prototype_image_bank,
)


@dataclass(frozen=True)
class TileSample:
    package_idx: int
    row: int
    tile_id: str
    image: Image.Image
    teacher_features: dict[str, np.ndarray]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        return yaml.safe_load(handle) or {}


def _localize_config(cfg: dict[str, Any], manifest_path: Path, prototype_dir: Path) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("runtime", {})["device"] = "auto"
    cfg.setdefault("data", {})["train_manifest_path"] = str(manifest_path)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["prototype_paths"] = {
        "gigapath": str(prototype_dir / "gigapath_hcc_semantic_prototypes.pt"),
        "h_optimus_1": str(prototype_dir / "h_optimus_1_hcc_semantic_prototypes.pt"),
        "uni2_h": str(prototype_dir / "uni2_h_hcc_semantic_prototypes.pt"),
        "virchow2": str(prototype_dir / "virchow2_hcc_semantic_prototypes.pt"),
    }
    image_bank_path = prototype_dir / "zhcc_hcc_prototype_images.pt"
    if image_bank_path.exists():
        cfg["data"]["zhcc_prototype_image_path"] = str(image_bank_path)
    supervision_path = prototype_dir / "hcc_prototype_supervision_manifest.csv"
    if supervision_path.exists():
        cfg["data"]["prototype_supervision_manifest_path"] = str(supervision_path)
    cfg.setdefault("train", {})["batch_size"] = min(int(cfg.get("train", {}).get("batch_size", 128)), 128)
    return cfg


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> HCCSemPathModel:
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
    _disable_fused_attention(model)
    model.eval()
    return model


def _disable_fused_attention(model: HCCSemPathModel) -> None:
    backbone = model.encoder.backbone
    for block in getattr(backbone, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "fused_attn"):
            attn.fused_attn = False


def _load_teacher_prototypes(cfg: dict[str, Any], device: torch.device) -> dict[str, PrototypeRegistry]:
    dims = teacher_dims(cfg, teacher_names(cfg))
    paths = cfg["data"]["prototype_paths"]
    return {
        name: load_prototype_registry(paths[name], expected_dim=dims[name]).to(device)
        for name in teacher_names(cfg)
    }


def _subset_image_bank(bank: PrototypeImageBank, max_images: int, seed: int) -> PrototypeImageBank:
    if max_images <= 0 or max_images >= bank.count:
        return bank
    rng = random.Random(seed)
    selected: set[int] = set()
    level1_count = bank.primary_count
    level2_count = bank.attribute_count
    for idx in range(level1_count):
        rows = torch.nonzero(bank.level1 == idx, as_tuple=False).flatten().tolist()
        if rows:
            selected.add(rng.choice(rows))
    for idx in range(level2_count):
        rows = torch.nonzero(bank.level2[:, idx] > 0.5, as_tuple=False).flatten().tolist()
        if rows:
            selected.add(rng.choice(rows))
    remaining = [idx for idx in range(bank.count) if idx not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, int(max_images) - len(selected))])
    indices = torch.tensor(sorted(selected), dtype=torch.long)
    return PrototypeImageBank(
        images=bank.images.index_select(0, indices).contiguous(),
        tile_ids=[bank.tile_ids[int(i)] for i in indices.tolist()],
        level1=bank.level1.index_select(0, indices).contiguous(),
        level2=bank.level2.index_select(0, indices).contiguous(),
        names=list(bank.names),
        groups=list(bank.groups),
        levels=list(bank.levels),
        exclusive=list(bank.exclusive),
        source={**dict(bank.source or {}), "subset_max_images": int(max_images)},
    )


def _maybe_build_zhcc_registry(
    model: HCCSemPathModel,
    cfg: dict[str, Any],
    device: torch.device,
    max_images: int,
    batch_size: int,
    seed: int,
) -> PrototypeRegistry | None:
    image_path = cfg["data"].get("zhcc_prototype_image_path")
    if not image_path or max_images <= 0:
        return None
    bank = _subset_image_bank(load_prototype_image_bank(image_path), max_images=max_images, seed=seed)
    return build_student_prototype_registry(
        model=model,
        image_bank=bank,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
    ).to(device)


def _open_sources(paths: dict[str, list[str]], package_idx: int) -> tuple[dict[Path, object], dict[str, Path]]:
    teacher_paths = {name: Path(paths[name][package_idx]) for name in paths}
    sources = {path: _open_feature_source(path) for path in sorted(set(teacher_paths.values()))}
    return sources, teacher_paths


def _read_sample(
    tile_readers: dict[int, TilePackageReader],
    tile_packages: list[str],
    teacher_packages: dict[str, list[str]],
    package_idx: int,
    row: int,
) -> TileSample:
    reader = tile_readers.get(package_idx)
    if reader is None:
        reader = TilePackageReader(tile_packages[package_idx])
        tile_readers[package_idx] = reader
    image = reader.read_image_at(row).convert("RGB")
    tile_id = reader.tile_id_at(row)
    sources, teacher_paths = _open_sources(teacher_packages, package_idx)
    try:
        teacher_features = _read_teacher_features_at(sources, teacher_paths, row)
    finally:
        for source in sources.values():
            close = getattr(source, "close", None)
            if close is not None:
                close()
    return TileSample(
        package_idx=package_idx,
        row=row,
        tile_id=tile_id,
        image=image,
        teacher_features={name: np.asarray(feature, dtype=np.float32) for name, feature in teacher_features.items()},
    )


def _sample_package_rows(tile_packages: list[str], count: int, seed: int) -> list[tuple[int, int]]:
    readers = [TilePackageReader(path) for path in tile_packages]
    try:
        counts = [reader.record_count for reader in readers]
        total = sum(counts)
        cumulative = np.cumsum(counts)
        rng = np.random.default_rng(seed)
        rows = []
        seen: set[tuple[int, int]] = set()
        while len(rows) < count:
            global_row = int(rng.integers(0, total))
            package_idx = int(np.searchsorted(cumulative, global_row, side="right"))
            package_start = 0 if package_idx == 0 else int(cumulative[package_idx - 1])
            row = global_row - package_start
            key = (package_idx, row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
        return rows
    finally:
        for reader in readers:
            reader.close()


def _read_selection(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"package_idx", "row_idx"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"selection CSV must contain {sorted(required)}: {path}")
    return rows


def _primary_top(registry: PrototypeRegistry, logits: torch.Tensor) -> tuple[str, float, float]:
    idx = torch.tensor(registry.primary_indices, device=logits.device)
    values = logits.index_select(0, idx)
    order = values.argsort(descending=True)
    best_local = int(order[0].item())
    best_idx = int(idx[best_local].item())
    best = float(values[best_local].detach().cpu())
    second = float(values[int(order[1].item())].detach().cpu()) if values.numel() > 1 else best
    return registry.names[best_idx], best, best - second


def _attribute_top(registry: PrototypeRegistry, logits: torch.Tensor) -> tuple[str, float]:
    if not registry.attribute_indices:
        return "", 0.0
    idx = torch.tensor(registry.attribute_indices, device=logits.device)
    values = logits.index_select(0, idx)
    best_local = int(values.argmax().item())
    best_idx = int(idx[best_local].item())
    return registry.names[best_idx], float(values[best_local].detach().cpu())


def _image_tensor(sample: TileSample, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    arr = torch.from_numpy(np.array(sample.image, dtype=np.uint8, copy=True)).permute(2, 0, 1).unsqueeze(0).contiguous()
    batch = {"images": arr, "images_uint8": True}
    return _prepare_images(batch, cfg, device)


def _score_sample(
    model: HCCSemPathModel,
    sample: TileSample,
    cfg: dict[str, Any],
    teacher_prototypes: dict[str, PrototypeRegistry],
    zhcc_registry: PrototypeRegistry | None,
    device: torch.device,
) -> dict[str, Any]:
    names = teacher_names(cfg)
    with torch.no_grad():
        images = _image_tensor(sample, cfg, device)
        outputs = model(images)
        embedding = outputs["embedding_norm"][0]
        result: dict[str, Any] = {
            "tile_id": sample.tile_id,
            "package_idx": sample.package_idx,
            "row": sample.row,
        }
        cosines = []
        student_l1 = []
        teacher_l1 = []
        margins = []
        for name in names:
            registry = teacher_prototypes[name]
            teacher_feature = torch.from_numpy(sample.teacher_features[name]).to(device=device, dtype=torch.float32)
            student_feature = outputs["teacher_outputs"][name][0]
            cosine = float(F.cosine_similarity(student_feature, teacher_feature, dim=0).detach().cpu())
            cosines.append(cosine)
            student_logits = normalized_prototype_logits(student_feature.unsqueeze(0), registry.prototypes)[0]
            teacher_logits = normalized_prototype_logits(teacher_feature.unsqueeze(0), registry.prototypes)[0]
            s_l1, s_score, s_margin = _primary_top(registry, student_logits)
            t_l1, t_score, _ = _primary_top(registry, teacher_logits)
            s_attr, s_attr_score = _attribute_top(registry, student_logits)
            t_attr, t_attr_score = _attribute_top(registry, teacher_logits)
            student_l1.append(s_l1)
            teacher_l1.append(t_l1)
            margins.append(s_margin)
            result[f"{name}_cosine"] = cosine
            result[f"{name}_student_l1"] = s_l1
            result[f"{name}_student_l1_score"] = s_score
            result[f"{name}_student_l1_margin"] = s_margin
            result[f"{name}_student_attr"] = s_attr
            result[f"{name}_student_attr_score"] = s_attr_score
            result[f"{name}_teacher_l1"] = t_l1
            result[f"{name}_teacher_l1_score"] = t_score
            result[f"{name}_teacher_attr"] = t_attr
            result[f"{name}_teacher_attr_score"] = t_attr_score
        result["mean_teacher_head_cosine"] = float(np.mean(cosines))
        result["teacher_l1_unique"] = len(set(teacher_l1))
        result["student_l1_unique"] = len(set(student_l1))
        result["student_l1_margin_mean"] = float(np.mean(margins))
        if zhcc_registry is not None:
            z_logits = normalized_prototype_logits(embedding.unsqueeze(0), zhcc_registry.prototypes)[0]
            z_l1, z_score, z_margin = _primary_top(zhcc_registry, z_logits)
            z_attr, z_attr_score = _attribute_top(zhcc_registry, z_logits)
            result["zhcc_student_l1"] = z_l1
            result["zhcc_student_l1_score"] = z_score
            result["zhcc_student_l1_margin"] = z_margin
            result["zhcc_student_attr"] = z_attr
            result["zhcc_student_attr_score"] = z_attr_score
        else:
            result["zhcc_student_l1"] = ""
            result["zhcc_student_l1_score"] = ""
            result["zhcc_student_l1_margin"] = ""
            result["zhcc_student_attr"] = ""
            result["zhcc_student_attr_score"] = ""
        result["selection_score"] = (
            result["mean_teacher_head_cosine"]
            + 0.02 * result["teacher_l1_unique"]
            + 0.10 * result["student_l1_margin_mean"]
        )
        return result


def _occlusion_sensitivity(
    model: HCCSemPathModel,
    sample: TileSample,
    cfg: dict[str, Any],
    zhcc_registry: PrototypeRegistry,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    images = _image_tensor(sample, cfg, device)
    primary_idx = torch.tensor(zhcc_registry.primary_indices, device=device)
    primary_prototypes = zhcc_registry.prototypes.index_select(0, primary_idx)
    with torch.no_grad():
        baseline_embedding = model(images)["embedding_norm"]
        baseline_logits = normalized_prototype_logits(baseline_embedding, primary_prototypes)[0]
    top_order = baseline_logits.argsort(descending=True)
    target_idx = int(top_order[0].item())
    competitor_indices = [idx for idx in range(len(primary_idx)) if idx != target_idx]
    baseline_margin = float(
        baseline_logits[target_idx] - baseline_logits[competitor_indices].max()
    )

    patch_size = int(getattr(model.encoder.backbone.patch_embed, "patch_size", (16, 16))[0])
    height, width = images.shape[-2:]
    grid_h = height // patch_size
    grid_w = width // patch_size
    variants = []
    for row in range(grid_h):
        for col in range(grid_w):
            variant = images[0].clone()
            y0 = row * patch_size
            x0 = col * patch_size
            variant[:, y0 : y0 + patch_size, x0 : x0 + patch_size] = 0.0
            variants.append(variant)

    importance = []
    with torch.no_grad():
        for start in range(0, len(variants), batch_size):
            batch = torch.stack(variants[start : start + batch_size]).to(device)
            embeddings = model(batch)["embedding_norm"]
            logits = normalized_prototype_logits(embeddings, primary_prototypes)
            margins = logits[:, target_idx] - logits[:, competitor_indices].max(dim=1).values
            importance.extend((baseline_margin - margins).detach().cpu().tolist())
    heatmap = torch.tensor(importance, dtype=torch.float32).reshape(1, 1, grid_h, grid_w)
    heatmap = F.interpolate(heatmap, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    heatmap = heatmap.clamp_min(0)
    heatmap = heatmap / heatmap.max().clamp_min(1e-6)
    return heatmap.numpy()


def _overlay(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    base = np.asarray(image.resize((heatmap.shape[1], heatmap.shape[0])), dtype=np.float32) / 255.0
    color = plt.get_cmap("magma")(heatmap)[..., :3]
    return np.clip(0.58 * base + 0.42 * color, 0.0, 1.0)


def _write_outputs(
    samples: list[TileSample],
    rows: list[dict[str, Any]],
    occlusion_maps: list[np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "tile_attention_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, max(3.0, 2.7 * len(samples))), squeeze=False)
    for idx, (sample, row, occlusion_map) in enumerate(zip(samples, rows, occlusion_maps, strict=True)):
        axes[idx, 0].imshow(sample.image)
        axes[idx, 0].set_title("tile")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(_overlay(sample.image, occlusion_map))
        axes[idx, 1].set_title("decision-margin occlusion")
        axes[idx, 1].axis("off")
        text_lines = [
            f"{sample.tile_id.encode('ascii', 'replace').decode('ascii')}",
            f"pkg={sample.package_idx} row={sample.row}",
            f"mean head cos={float(row['mean_teacher_head_cosine']):.3f}",
            f"teacher L1 unique={row['teacher_l1_unique']}",
            f"student L1 unique={row['student_l1_unique']}",
        ]
        if row.get("selection_stratum"):
            text_lines.insert(1, f"stratum={row['selection_stratum']}")
        if row.get("selection_expert_l1"):
            text_lines.append(f"expert={row['selection_expert_l1']}")
        if row.get("selection_teacher_plurality"):
            text_lines.append(f"plurality={row['selection_teacher_plurality']}")
        if row.get("selection_full_prediction"):
            text_lines.append(f"full={row['selection_full_prediction']}")
        if row.get("zhcc_student_l1"):
            text_lines.append(
                f"z_hcc={row['zhcc_student_l1']} "
                f"margin={float(row['zhcc_student_l1_margin']):.3f}"
            )
            text_lines.append(f"attr={row['zhcc_student_attr']}")
        axes[idx, 2].text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=9)
        axes[idx, 2].axis("off")
    fig.suptitle("Final-decision occlusion sensitivity", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "tile_attention_sheet.png", dpi=180)
    plt.close(fig)
    for sample, occlusion_map in zip(samples, occlusion_maps, strict=True):
        Image.fromarray((np.asarray(_overlay(sample.image, occlusion_map)) * 255).astype(np.uint8)).save(
            out_dir / f"{sample.tile_id}.occlusion.png"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small tile-level saliency/prototype review for a trained trial.")
    parser.add_argument("--config", default="outputs/trail/trial_0002/resolved_config.json")
    parser.add_argument("--checkpoint", default="outputs/trail/trial_0002/checkpoints/last.pt")
    parser.add_argument("--manifest", default="configs/local/mac/manifest.yaml")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs/trail/local_tile_attention/trial_0002_last")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--selection-csv",
        default="",
        help="Optional fixed package_idx,row_idx case list; bypasses random candidate selection.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--zhcc-bank-max", type=int, default=512)
    parser.add_argument("--zhcc-bank-batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = _localize_config(_load_config(Path(args.config)), Path(args.manifest), Path(args.prototype_dir))
    device = _resolve_device(args.device)
    cfg["runtime"]["device"] = str(device)
    model = _load_model(cfg, Path(args.checkpoint), device)
    teacher_prototypes = _load_teacher_prototypes(cfg, device)
    zhcc_registry = _maybe_build_zhcc_registry(
        model,
        cfg,
        device,
        max_images=args.zhcc_bank_max,
        batch_size=args.zhcc_bank_batch_size,
        seed=args.seed,
    )

    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, args.split)
    selection_rows = _read_selection(Path(args.selection_csv)) if args.selection_csv else []
    package_rows = (
        [(int(row["package_idx"]), int(row["row_idx"])) for row in selection_rows]
        if selection_rows
        else _sample_package_rows(tile_packages, count=max(args.candidates, args.samples), seed=args.seed)
    )
    tile_readers: dict[int, TilePackageReader] = {}
    try:
        candidates = [
            _read_sample(tile_readers, tile_packages, teacher_packages, package_idx, row)
            for package_idx, row in package_rows
        ]
    finally:
        for reader in tile_readers.values():
            reader.close()

    scored = [
        _score_sample(model, sample, cfg, teacher_prototypes, zhcc_registry, device)
        for sample in candidates
    ]
    if selection_rows:
        for result, selection in zip(scored, selection_rows, strict=True):
            for key, value in selection.items():
                result[f"selection_{key}"] = value
        take = list(range(len(scored)))
    else:
        order = sorted(range(len(scored)), key=lambda idx: float(scored[idx]["selection_score"]), reverse=True)
        take = order[: max(1, args.samples)]
    selected_samples = [candidates[idx] for idx in take]
    selected_rows = [scored[idx] for idx in take]
    if zhcc_registry is None:
        raise RuntimeError("decision-margin occlusion requires the z_hcc prototype registry")
    occlusion_maps = [
        _occlusion_sensitivity(model, sample, cfg, zhcc_registry, device)
        for sample in selected_samples
    ]
    out_dir = Path(args.output_dir)
    _write_outputs(selected_samples, selected_rows, occlusion_maps, out_dir)
    print(
        f"tile_attention_ok device={device} candidates={len(candidates)} samples={len(selected_samples)} "
        f"zhcc_registry={'yes' if zhcc_registry is not None else 'no'} output={out_dir}"
    )


if __name__ == "__main__":
    main()
