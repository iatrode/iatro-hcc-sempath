from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter

from .models import load_hcc_sempath_release


POINT_ATTRIBUTES = {
    "hepatocellular-parenchyma-present": 16,
    "inflammatory-cell-present": 10,
}


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.quantile(values, [0.05, 0.95])
    if high <= low + 1e-8:
        low, high = float(values.min()), float(values.max())
    if high <= low + 1e-8:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def detect_hematoxylin_centers(image: Image.Image, *, max_candidates: int = 240) -> list[dict]:
    """Return stain-derived nucleus centers without assigning a semantic class."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    optical_density = -np.log((rgb + 1.0) / 256.0)
    hematoxylin = (
        0.650 * optical_density[..., 0]
        + 0.704 * optical_density[..., 1]
        + 0.286 * optical_density[..., 2]
    )
    tissue = rgb.mean(axis=-1) < 245.0
    if not tissue.any():
        return []
    threshold = float(np.quantile(hematoxylin[tissue], 0.96))
    peaks = tissue & (hematoxylin >= threshold) & (hematoxylin == maximum_filter(hematoxylin, size=5))
    ys, xs = np.nonzero(peaks)
    if not len(xs):
        return []
    strengths = _normalize_map(hematoxylin)[ys, xs]
    order = np.argsort(strengths)[::-1]
    selected: list[dict] = []
    min_distance_sq = 5.0**2
    for index in order:
        x, y = int(xs[index]), int(ys[index])
        if any((x - item["pixel_x"]) ** 2 + (y - item["pixel_y"]) ** 2 < min_distance_sq for item in selected):
            continue
        selected.append(
            {
                "x": (x + 0.5) / rgb.shape[1],
                "y": (y + 0.5) / rgb.shape[0],
                "pixel_x": x,
                "pixel_y": y,
                "stain": float(strengths[index]),
            }
        )
        if len(selected) >= max_candidates:
            break
    return selected


def _rank_nucleus_points(
    candidates: list[dict],
    saliency: np.ndarray,
    *,
    limit: int,
    stain_weight: float,
) -> list[dict]:
    if not candidates:
        return []
    height, width = saliency.shape
    ranked = []
    for item in candidates:
        x = min(width - 1, max(0, int(item["x"] * width)))
        y = min(height - 1, max(0, int(item["y"] * height)))
        score = stain_weight * item["stain"] + (1.0 - stain_weight) * float(saliency[y, x])
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    chosen: list[dict] = []
    min_distance_sq = 0.025**2
    for score, item in ranked:
        if any((item["x"] - old["x"]) ** 2 + (item["y"] - old["y"]) ** 2 < min_distance_sq for old in chosen):
            continue
        chosen.append({"x": item["x"], "y": item["y"], "confidence": float(np.clip(score, 0.0, 1.0))})
        if len(chosen) >= limit:
            break
    return chosen


def _saliency_regions(saliency: np.ndarray, *, limit: int = 3) -> list[dict]:
    peaks = saliency == maximum_filter(saliency, size=3)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(saliency[ys, xs])[::-1]
    height, width = saliency.shape
    chosen: list[dict] = []
    for index in order:
        confidence = float(saliency[ys[index], xs[index]])
        if confidence < 0.35:
            break
        x, y = (xs[index] + 0.5) / width, (ys[index] + 0.5) / height
        if any((x - old["x"]) ** 2 + (y - old["y"]) ** 2 < 0.14**2 for old in chosen):
            continue
        chosen.append({"x": x, "y": y, "confidence": confidence})
        if len(chosen) >= limit:
            break
    return chosen


class RoiPlanGenerator:
    """Lazy, thread-safe ROI proposal generator for the annotation UI.

    The release classifier supplies class-specific feature-gradient maps. H&E
    stain peaks supply candidate cell centers. The result remains a preview;
    the UI decides whether it becomes annotation state.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = _auto_device() if device == "auto" else device
        self._model = None
        self._config: dict | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            self._model, self._config = load_hcc_sempath_release(
                self.config_path, self.checkpoint_path, self.device
            )
        return self._model, self._config

    def _feature_gradient_maps(self, image: Image.Image) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        model, config = self._load()
        names = list(config["l2_names"])
        preprocessing = config["preprocessing"]
        rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor(preprocessing["mean"], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(preprocessing["std"], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.enable_grad():
            features = model.encoder.backbone.forward_features(tensor).detach().requires_grad_(True)
            pooled = model.encoder.backbone.forward_head(features, pre_logits=True)
            embedding = model.encoder.projector(pooled)
            scores = F.normalize(embedding, dim=-1) @ model.l2_prototypes.T
            prefix = int(getattr(model.encoder.backbone, "num_prefix_tokens", 1))
            patch_tokens = features[:, prefix:]
            grid = tuple(int(value) for value in model.encoder.backbone.patch_embed.grid_size)
            maps: dict[str, np.ndarray] = {}
            for index, name in enumerate(names):
                gradient = torch.autograd.grad(
                    scores[0, index], features, retain_graph=index < len(names) - 1
                )[0][:, prefix:]
                contribution = (gradient * patch_tokens).sum(dim=-1).reshape(1, 1, *grid)
                contribution = F.relu(contribution)
                if float(contribution.detach().max()) <= 1e-8:
                    contribution = (gradient * patch_tokens).abs().sum(dim=-1).reshape(1, 1, *grid)
                upsampled = F.interpolate(
                    contribution, size=image.size[::-1], mode="bilinear", align_corners=False
                )[0, 0]
                maps[name] = _normalize_map(upsampled.detach().float().cpu().numpy())
        score_values = scores.detach().float().cpu().numpy()[0]
        return dict(zip(names, (float(value) for value in score_values))), maps

    def generate(self, image: Image.Image | bytes) -> dict:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        with self._lock:
            scores, maps = self._feature_gradient_maps(image)
            assert self._config is not None
            thresholds = dict(zip(self._config["l2_names"], self._config["l2_decision_thresholds"]))
            nuclei = detect_hematoxylin_centers(image)

            suggestions: list[dict] = []
            class_scores: dict[str, dict] = {}
            for attribute, score in scores.items():
                threshold = float(thresholds[attribute])
                predicted_positive = score >= threshold
                class_scores[attribute] = {
                    "score": score,
                    "threshold": threshold,
                    "predicted_positive": predicted_positive,
                }
            planned_attributes = {
                attribute
                for _margin, attribute in sorted(
                    (
                        (item["score"] - item["threshold"], attribute)
                        for attribute, item in class_scores.items()
                        if item["predicted_positive"] and attribute != "hyaline-change-present"
                    ),
                    reverse=True,
                )[:4]
            }
            for attribute, score in scores.items():
                if attribute not in planned_attributes:
                    continue
                if attribute in POINT_ATTRIBUTES:
                    stain_weight = 0.65 if attribute == "inflammatory-cell-present" else 0.35
                    points = _rank_nucleus_points(
                        nuclei,
                        maps[attribute],
                        limit=POINT_ATTRIBUTES[attribute],
                        stain_weight=stain_weight,
                    )
                    for point in points:
                        suggestions.append(
                            {
                                "attribute": attribute,
                                "state": "positive",
                                "review_complete": False,
                                "confidence": point["confidence"],
                                "source": "feature-gradient+hematoxylin",
                                "geometry": {
                                    "type": "point",
                                    "coordinate_space": "normalized",
                                    "point": [point["x"], point["y"]],
                                    "radius": 0.018,
                                },
                            }
                        )
                else:
                    for region in _saliency_regions(maps[attribute]):
                        suggestions.append(
                            {
                                "attribute": attribute,
                                "state": "positive",
                                "review_complete": False,
                                "confidence": region["confidence"],
                                "source": "feature-gradient",
                                "geometry": {
                                    "type": "circle",
                                    "coordinate_space": "normalized",
                                    "center": [region["x"], region["y"]],
                                    "radius": 0.07,
                                },
                            }
                        )

        counts: dict[str, int] = {}
        for item in suggestions:
            counts[item["attribute"]] = counts.get(item["attribute"], 0) + 1
        return {
            "version": 1,
            "method": "release-feature-gradient+he-nuclei-v1",
            "device": self.device,
            "suggestions": suggestions,
            "class_scores": class_scores,
            "summary": {
                "suggestion_count": len(suggestions),
                "counts": counts,
                "predicted_positive": [
                    name for name, item in class_scores.items() if item["predicted_positive"]
                ],
                "planned_positive": sorted(planned_attributes),
            },
        }
