from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label, maximum_filter

from .models import load_hcc_sempath_release


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


def _hematoxylin_map(rgb: np.ndarray) -> np.ndarray:
    optical_density = -np.log((rgb + 1.0) / 256.0)
    return (
        0.650 * optical_density[..., 0]
        + 0.704 * optical_density[..., 1]
        + 0.286 * optical_density[..., 2]
    )


def detect_hematoxylin_centers(image: Image.Image, *, max_candidates: int = 240) -> list[dict]:
    """Return stain-derived nucleus centers without assigning a semantic class."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    hematoxylin = _hematoxylin_map(rgb)
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


def _estimate_seed_spacing(
    image: Image.Image,
    seeds: list[tuple[float, float]],
    *,
    fallback_pixels: float = 7.0,
) -> tuple[float, float]:
    """Estimate one-center-per-target spacing from this class's seed morphology."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    hematoxylin = _hematoxylin_map(rgb)
    diameters: list[float] = []
    window_radius = max(8, min(18, round(min(width, height) * 0.065)))

    for x, y in seeds:
        pixel_x = min(width - 1, max(0, int(x * width)))
        pixel_y = min(height - 1, max(0, int(y * height)))
        left, right = max(0, pixel_x - window_radius), min(width, pixel_x + window_radius + 1)
        top, bottom = max(0, pixel_y - window_radius), min(height, pixel_y + window_radius + 1)
        patch = hematoxylin[top:bottom, left:right]
        local_x, local_y = pixel_x - left, pixel_y - top
        near = patch[
            max(0, local_y - 2) : min(patch.shape[0], local_y + 3),
            max(0, local_x - 2) : min(patch.shape[1], local_x + 3),
        ]
        peak = float(near.max())
        background = float(np.quantile(patch, 0.50))
        high = float(np.quantile(patch, 0.95))
        contrast = max(peak, high) - background
        if contrast <= 1e-4:
            continue
        mask = patch >= background + 0.45 * contrast
        components, _count = label(mask)
        near_components = components[
            max(0, local_y - 2) : min(components.shape[0], local_y + 3),
            max(0, local_x - 2) : min(components.shape[1], local_x + 3),
        ]
        component_ids = near_components[near_components > 0]
        if not component_ids.size:
            continue
        values, counts = np.unique(component_ids, return_counts=True)
        component_id = int(values[np.argmax(counts)])
        area = int(np.count_nonzero(components == component_id))
        if area:
            diameters.append(2.0 * float(np.sqrt(area / np.pi)))

    spacing_pixels = float(np.median(diameters)) if diameters else float(fallback_pixels)
    spacing_pixels = float(np.clip(spacing_pixels, 5.0, 18.0))
    return spacing_pixels / min(width, height), spacing_pixels


def _local_color_descriptor(rgb: np.ndarray, x: float, y: float, *, radius: int = 8) -> np.ndarray:
    height, width = rgb.shape[:2]
    pixel_x = min(width - 1, max(0, int(x * width)))
    pixel_y = min(height - 1, max(0, int(y * height)))
    crop = rgb[
        max(0, pixel_y - radius) : min(height, pixel_y + radius + 1),
        max(0, pixel_x - radius) : min(width, pixel_x + radius + 1),
    ]
    scaled = crop / 255.0
    optical_density = -np.log((crop + 1.0) / 256.0)
    hematoxylin = (
        0.650 * optical_density[..., 0]
        + 0.704 * optical_density[..., 1]
        + 0.286 * optical_density[..., 2]
    )
    return np.concatenate(
        (
            scaled.mean(axis=(0, 1)),
            scaled.std(axis=(0, 1)),
            np.quantile(scaled, 0.25, axis=(0, 1)),
            np.quantile(scaled, 0.75, axis=(0, 1)),
            np.asarray([hematoxylin.mean(), hematoxylin.std()]),
        )
    ).astype(np.float32)


def _relative_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median))) * 1.4826
    if scale <= 1e-6:
        scale = float(values.std())
    return (values - median) / max(scale, 1e-6)


def _rank_similar_centers(
    centers: list[tuple[float, float]],
    scores: np.ndarray,
    seeds: list[tuple[float, float]],
    *,
    limit: int = 120,
    min_distance: float = 7 / 224,
) -> list[tuple[float, float, float]]:
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float]] = []
    minimum_sq = min_distance**2
    for index in order:
        x, y = centers[int(index)]
        if any((x - sx) ** 2 + (y - sy) ** 2 < minimum_sq for sx, sy in seeds):
            continue
        if any((x - old_x) ** 2 + (y - old_y) ** 2 < minimum_sq for old_x, old_y in selected):
            continue
        selected.append((x, y))
        if len(selected) >= limit:
            break
    denominator = max(1, len(selected) - 1)
    return [
        (x, y, 1.0 - rank / denominator)
        for rank, (x, y) in enumerate(selected)
    ]


class RoiPlanGenerator:
    """Find same-tile image centers similar to user-confirmed seed marks."""

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

    def _patch_tokens(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        model, config = self._load()
        preprocessing = config["preprocessing"]
        rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor(preprocessing["mean"], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(preprocessing["std"], device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            features = model.encoder.backbone.forward_features((tensor - mean) / std)
        prefix = int(getattr(model.encoder.backbone, "num_prefix_tokens", 1))
        tokens = F.normalize(features[0, prefix:].float(), dim=-1).cpu().numpy()
        grid = tuple(int(value) for value in model.encoder.backbone.patch_embed.grid_size)
        return tokens, grid

    @staticmethod
    def _token_at(tokens: np.ndarray, grid: tuple[int, int], x: float, y: float) -> np.ndarray:
        grid_y = min(grid[0] - 1, max(0, int(y * grid[0])))
        grid_x = min(grid[1] - 1, max(0, int(x * grid[1])))
        return tokens[grid_y * grid[1] + grid_x]

    @staticmethod
    def _candidate_centers(image: Image.Image) -> list[tuple[float, float]]:
        width, height = image.size
        centers = [(item["x"], item["y"]) for item in detect_hematoxylin_centers(image)]
        # A dense fallback retains pigment, hemorrhage, and vacuolation patterns
        # that are not reliably represented by hematoxylin maxima.
        centers.extend(
            ((x + 0.5) / width, (y + 0.5) / height)
            for y in range(4, height, 7)
            for x in range(4, width, 7)
        )
        return centers

    def generate_similar(
        self,
        image: Image.Image | bytes,
        *,
        attribute: str,
        seeds: list[list[float] | tuple[float, float]],
        occupied: list[list[float] | tuple[float, float]] | None = None,
    ) -> dict:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        normalized_seeds = [(float(seed[0]), float(seed[1])) for seed in seeds]
        if not normalized_seeds or len(normalized_seeds) > 20:
            raise ValueError("similarity propagation requires 1-20 seed centers")
        if any(not 0 <= value <= 1 for seed in normalized_seeds for value in seed):
            raise ValueError("seed centers must use normalized coordinates")
        normalized_occupied = [
            (float(point[0]), float(point[1])) for point in (occupied or [])
        ]
        if any(not 0 <= value <= 1 for point in normalized_occupied for value in point):
            raise ValueError("occupied centers must use normalized coordinates")

        with self._lock:
            tokens, grid = self._patch_tokens(image)
            rgb = np.asarray(image, dtype=np.float32)
            centers = self._candidate_centers(image)
            seed_tokens = np.stack(
                [self._token_at(tokens, grid, x, y) for x, y in normalized_seeds]
            )
            token_centroid = seed_tokens.mean(axis=0)
            token_centroid /= max(float(np.linalg.norm(token_centroid)), 1e-8)
            seed_colors = np.stack(
                [_local_color_descriptor(rgb, x, y) for x, y in normalized_seeds]
            )
            color_centroid = seed_colors.mean(axis=0)

            candidate_tokens = np.stack(
                [self._token_at(tokens, grid, x, y) for x, y in centers]
            )
            candidate_colors = np.stack(
                [_local_color_descriptor(rgb, x, y) for x, y in centers]
            )
            color_scale = candidate_colors.std(axis=0) + 1e-4
            token_similarity = candidate_tokens @ token_centroid
            color_similarity = -np.mean(
                ((candidate_colors - color_centroid) / color_scale) ** 2,
                axis=1,
            )
            combined = _relative_scores(token_similarity) + _relative_scores(color_similarity)
            minimum_spacing, minimum_spacing_pixels = _estimate_seed_spacing(
                image, normalized_seeds
            )
            ranked = _rank_similar_centers(
                centers,
                combined,
                [*normalized_seeds, *normalized_occupied],
                min_distance=minimum_spacing,
            )

        suggestions = [
            {
                "attribute": attribute,
                "state": "positive",
                "review_complete": False,
                "confidence": confidence,
                "source": "same-tile-seed-similarity",
                "geometry": {
                    "type": "point",
                    "coordinate_space": "normalized",
                    "point": [x, y],
                    "radius": 0.018,
                },
            }
            for x, y, confidence in ranked
        ]
        return {
            "version": 1,
            "method": "same-tile-local-seed-centroid-v1",
            "device": self.device,
            "suggestions": suggestions,
            "summary": {
                "attribute": attribute,
                "seed_count": len(normalized_seeds),
                "occupied_count": len(normalized_occupied),
                "candidate_count": len(centers),
                "suggestion_count": len(suggestions),
                "minimum_spacing": minimum_spacing,
                "minimum_spacing_pixels": minimum_spacing_pixels,
            },
        }
