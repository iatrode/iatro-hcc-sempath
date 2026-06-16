"""Symmetric representation-quality evaluation for HCC-SemPath.

Builds frozen-feature tables for the student (z_HCC) and the four teachers on the
1000-tile expert-adjudicated eval asset, then evaluates all five models under the
SAME readout protocols (linear probe, kNN, neighborhood purity) so the comparison
is method-symmetric. Teacher features are read on-demand from IatroCache by
(package_path, row_idx) recorded in the expert review CSV; student features come
from a forward pass of the frozen checkpoint on the same tiles.

This file only assembles features + labels and runs a wiring sanity check. Probe
logic lives in run_symmetric_probe.py.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
REVIEW_CSV = REPO / "annotations/reviews/teacher_disagreement/exval_1000/review.csv"
EXPORT_CFG = REPO / "experiments/02_embedding_export/configs/local_sampled_export.yaml"
PROTO_DIR = REPO / "artifacts/prototypes"

TEACHERS = ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]
L1_CLASSES = ["HCC-tumor", "Background-liver", "Inflammatory-stromal", "Degenerative-material"]
L2_ATTRS = [
    "bile-pigment-present", "ductular-portal-present", "fibrous-stroma-present",
    "hemorrhage-present", "hepatocellular-parenchyma-present", "hyaline-change-present",
    "inflammatory-cell-present", "necrosis-present", "steatosis-vacuolation-present",
    "vascular-structure-present",
]


def load_review_rows() -> list[dict]:
    with REVIEW_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> int:
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "t"} else 0


def build_labels(rows: list[dict]) -> dict:
    """Return tile_id-ordered label arrays + queue masks."""
    tile_ids = [r["tile_id"] for r in rows]
    l1 = np.array([L1_CLASSES.index(r["l1"]) for r in rows], dtype=np.int64)
    l2 = np.array(
        [[_parse_bool(r[f"l2_{a}"]) for a in L2_ATTRS] for r in rows], dtype=np.int64
    )
    group = np.array([r["source_group"] for r in rows])
    return {
        "tile_id": tile_ids,
        "l1": l1,
        "l2": l2,
        "group": group,
        "random500": group == "random500",
        "top500": group == "top500",
    }


def load_teacher_features(rows: list[dict], teacher: str) -> np.ndarray:
    """Read teacher features by (package_path, row_idx), mapping Tiles->Features path."""
    from hcc_sempath.training.config import load_config, manifest_data_paths, teacher_names
    from hcc_sempath.training.datasets import _open_feature_source
    from hcc_sempath.training.manifest import load_training_manifest

    cfg = load_config(EXPORT_CFG)
    cfg["data"]["exval_tile_fraction"] = 1.0
    cfg["data"]["val_tile_fraction"] = 1.0
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, "exval")
    names = teacher_names(cfg)
    tile_to_teacher = {
        str(tp): {n: teacher_packages[n][i] for n in names}
        for i, tp in enumerate(tile_packages)
    }

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for idx, r in enumerate(rows):
        src = tile_to_teacher[r["package_path"]][teacher]
        grouped[src].append((idx, int(r["row_idx"])))

    feats: dict[int, np.ndarray] = {}
    for src_path, items in grouped.items():
        source = _open_feature_source(Path(src_path))
        try:
            is_merged = source.__class__.__name__ == "MergedTeacherFeatureCacheReader"
            for idx, row in items:
                vec = source.read_feature_at(row, teacher) if is_merged else source.read_feature_at(row)
                feats[idx] = np.asarray(vec, dtype=np.float32)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
    return np.stack([feats[i] for i in range(len(rows))])


CHECKPOINT = REPO / "artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"


def load_student_features(rows: list[dict], batch_size: int = 64) -> np.ndarray:
    """Forward the frozen student on the same 1000 tiles, return embedding_norm.

    Reads tile images directly from each tile's IatroCache package via the same
    PackReader the training dataset uses, so preprocessing matches training."""
    import torch
    from hcc_sempath.io.tile_package import TilePackageReader, read_package_metadata
    from hcc_sempath.training.config import embedding_dim, load_config, teacher_dims, teacher_names
    from hcc_sempath.training.datasets import _build_image_transform
    from hcc_sempath.modeling.models import HCCSemPathModel

    cfg = load_config(EXPORT_CFG)
    device = torch.device(cfg["runtime"]["device"])
    meta0 = read_package_metadata(rows[0]["package_path"])
    image_size = (int(meta0["tile_height"]), int(meta0["tile_width"]))
    transform = _build_image_transform(image_size, cfg["data"].get("mean"), cfg["data"].get("std"))

    names = teacher_names(cfg)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=teacher_dims(cfg, names),
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=False,
    ).to(device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    readers: dict[str, "TilePackageReader"] = {}
    out = np.zeros((len(rows), embedding_dim(cfg)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            imgs = []
            for r in chunk:
                pkg = r["package_path"]
                reader = readers.get(pkg) or readers.setdefault(pkg, TilePackageReader(pkg))
                image = reader.read_image(r["tile_id"]).convert("RGB")
                imgs.append(transform(image))
            batch = torch.stack(imgs).to(device)
            emb = model(batch)["embedding_norm"].detach().cpu().numpy().astype("float32")
            out[start : start + len(chunk)] = emb
    for reader in readers.values():
        close = getattr(reader, "close", None)
        if close is not None:
            close()
    return out


def load_or_cache_features(rows: list[dict], cache_dir: Path) -> dict[str, np.ndarray]:
    """Build (or load cached) feature matrices for all 5 models."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats: dict[str, np.ndarray] = {}
    for model_name, loader in [
        ("z_hcc", lambda: load_student_features(rows)),
        *[(t, (lambda t=t: load_teacher_features(rows, t))) for t in TEACHERS],
    ]:
        path = cache_dir / f"feat_{model_name}.npy"
        if path.exists():
            feats[model_name] = np.load(path)
        else:
            arr = loader()
            np.save(path, arr)
            feats[model_name] = arr
    return feats


def sanity_check_teacher_wiring(rows: list[dict], teacher: str, feats: np.ndarray) -> dict:
    """Cross-check: nearest-prototype argmax of read features should match the
    teacher's recorded L1 in review.csv. High agreement => row indexing is correct."""
    import torch
    import torch.nn.functional as F
    from hcc_sempath.modeling.prototypes import load_prototype_registry

    reg = load_prototype_registry(PROTO_DIR / f"{teacher}_hcc_semantic_prototypes.pt")
    protos = F.normalize(reg.prototypes.float(), dim=1)
    primary = torch.tensor(reg.primary_indices, dtype=torch.long)
    primary_protos = protos.index_select(0, primary)
    names = [reg.names[i] for i in reg.primary_indices]

    arr = F.normalize(torch.from_numpy(feats.astype("float32")), dim=1)
    pred_idx = (arr @ primary_protos.T).argmax(dim=1).tolist()
    pred_names = [names[i] for i in pred_idx]

    col = f"{teacher}_l1"
    recorded = [r.get(col, "") for r in rows]
    have = [(p, q) for p, q in zip(pred_names, recorded) if q]
    if not have:
        return {"teacher": teacher, "checked": 0, "agreement": float("nan")}
    agree = sum(1 for p, q in have if p == q) / len(have)
    return {"teacher": teacher, "checked": len(have), "agreement": round(agree, 4)}


if __name__ == "__main__":
    rows = load_review_rows()
    labels = build_labels(rows)
    print(f"loaded {len(rows)} tiles | random500={labels['random500'].sum()} top500={labels['top500'].sum()}")
    print(f"L1 dist: {dict(zip(*np.unique(labels['l1'], return_counts=True)))}")
    print("=== teacher wiring sanity check (nearest-prototype vs recorded *_l1) ===")
    for t in TEACHERS:
        feats = load_teacher_features(rows, t)
        chk = sanity_check_teacher_wiring(rows, t, feats)
        print(f"  {t:14} dim={feats.shape[1]:5} {chk}")
