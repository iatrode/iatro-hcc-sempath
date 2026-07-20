from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iatro.iac.adapters.tiles import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel, normalized_prototype_logits
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.prototype_images import (
    build_student_prototype_registry,
    load_prototype_image_bank,
)


MODELS = {
    "a2": "artifacts/experiments/ablation/a2_no_adjudication",
    "a6": "artifacts/experiments/ablation/a6_full_filter",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_model(cfg: dict, checkpoint: Path, device: torch.device) -> HCCSemPathModel:
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
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _load_images(rows: list[dict[str, str]]) -> torch.Tensor:
    grouped: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[Path(row["package_path"])].append((idx, int(row["row_idx"])))
    loaded: list[tuple[int, torch.Tensor]] = []
    for package_path, items in grouped.items():
        reader = TilePackageReader(package_path)
        for idx, row_idx in items:
            image = np.array(reader.read_image_at(row_idx).convert("RGB"), dtype=np.uint8, copy=True)
            loaded.append((idx, torch.from_numpy(image).permute(2, 0, 1)))
        reader.close()
    loaded.sort(key=lambda item: item[0])
    if len(loaded) != len(rows):
        raise RuntimeError(f"loaded {len(loaded)} of {len(rows)} reviewed tiles")
    return torch.stack([image for _, image in loaded])


def _infer_scores(
    rows: list[dict[str, str]],
    images: torch.Tensor,
    model_root: Path,
    image_bank: dict,
    image_bank_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, list[str], float]:
    with (model_root / "resolved_config.json").open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["data"]["zhcc_prototype_image_path"] = str(image_bank_path)
    model = _load_model(
        cfg,
        model_root / "checkpoints" / "best_scientific_score.pt",
        device,
    )
    registry = build_student_prototype_registry(
        model=model,
        image_bank=image_bank,
        cfg=cfg,
        device=device,
        batch_size=256,
    ).to(device)
    names = [registry.names[idx] for idx in registry.primary_indices]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = images[start : start + 64].to(device)
            prepared = [
                _prepare_images(
                    {"images": image.unsqueeze(0), "images_uint8": True},
                    cfg,
                    device,
                )
                for image in batch
            ]
            embedding = model(torch.cat(prepared))["embedding_norm"]
            chunks.append(
                normalized_prototype_logits(embedding, registry.primary_prototypes)
                .cpu()
                .numpy()
            )
    temperature = float(cfg["loss"].get("zhcc_primary_temperature", 0.1))
    return np.concatenate(chunks), names, temperature


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 13, rounds: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(rounds, dtype=np.float64)
    for idx in range(rounds):
        sample = rng.integers(0, len(values), size=len(values))
        means[idx] = values[sample].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _metrics(scores: np.ndarray, truth: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    true_score = scores[np.arange(len(scores)), truth]
    masked = scores.copy()
    masked[np.arange(len(scores)), truth] = -np.inf
    margin = true_score - masked.max(axis=1)
    scaled = scores / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    nll = -np.log(np.clip(probabilities[np.arange(len(scores)), truth], 1e-12, 1.0))
    one_hot = np.eye(scores.shape[1], dtype=np.float64)[truth]
    brier = ((probabilities - one_hot) ** 2).sum(axis=1)
    return {"margin": margin, "nll": nll, "brier": brier}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        default="annotations/reviews/teacher_disagreement/exval_1000/review.csv",
    )
    parser.add_argument(
        "--prototype-bank",
        default="artifacts/prototypes/zhcc_hcc_prototype_images.pt",
    )
    parser.add_argument(
        "--output",
        default="experiments/10_teacher_disagreement_review/tables/a6_vs_a2_l1_strength.csv",
    )
    args = parser.parse_args()

    rows = _read_csv(REPO_ROOT / args.review)
    images = _load_images(rows)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    image_bank_path = REPO_ROOT / args.prototype_bank
    image_bank = load_prototype_image_bank(image_bank_path)

    scores = {}
    names = None
    temperature = None
    for key, relative_root in MODELS.items():
        model_scores, model_names, model_temperature = _infer_scores(
            rows,
            images,
            REPO_ROOT / relative_root,
            image_bank,
            image_bank_path,
            device,
        )
        if names is not None and model_names != names:
            raise RuntimeError("A2 and A6 prototype class order differs")
        names = model_names
        temperature = model_temperature
        scores[key] = model_scores

    name_to_idx = {name.lower(): idx for idx, name in enumerate(names)}
    truth = np.asarray([name_to_idx[row["l1"].lower()] for row in rows], dtype=np.int64)
    model_metrics = {key: _metrics(value, truth, temperature) for key, value in scores.items()}

    output = []
    groups = ("random500", "top500", "all")
    for group in groups:
        idx = np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if group == "all" or row["source_group"] == group
            ],
            dtype=np.int64,
        )
        for metric in ("margin", "nll", "brier"):
            a2 = model_metrics["a2"][metric][idx]
            a6 = model_metrics["a6"][metric][idx]
            delta = a6 - a2
            lo, hi = _bootstrap_mean_ci(delta)
            output.append(
                {
                    "source_group": group,
                    "metric": metric,
                    "tiles": len(idx),
                    "a2_mean": float(a2.mean()),
                    "a6_mean": float(a6.mean()),
                    "a6_minus_a2_mean": float(delta.mean()),
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                    "fraction_a6_improved": float(
                        (delta > 0).mean() if metric == "margin" else (delta < 0).mean()
                    ),
                }
            )

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"a6_vs_a2_l1_strength_ok rows={len(rows)} device={device} output={output_path}")


if __name__ == "__main__":
    main()
