from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image
from scipy.ndimage import correlate, gaussian_filter, label, maximum_filter, uniform_filter


_RGB_FROM_HED = np.asarray(
    (
        (0.650, 0.700, 0.290),
        (0.070, 0.990, 0.110),
        (0.270, 0.570, 0.780),
    ),
    dtype=np.float32,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)
_NUCLEUS_MATCH_SIGMA = {
    "hepatocellular-parenchyma": 2.0,
    "inflammatory-cell": 0.0,
}


def _stain_channels(rgb: np.ndarray) -> np.ndarray:
    """Separate H&E-like optical density into HED stain channels."""
    scaled = np.clip((rgb.astype(np.float32) + 1.0) / 256.0, 1e-6, 1.0)
    optical_density = -np.log(scaled)
    return (optical_density @ _HED_FROM_RGB).astype(np.float32)


def _hematoxylin_map(rgb: np.ndarray) -> np.ndarray:
    return _stain_channels(rgb)[..., 0]


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


def _estimate_seed_exclusion_distance(
    image: Image.Image,
    seeds: list[tuple[float, float]],
    *,
    fallback_pixels: float,
) -> tuple[float, float]:
    """Estimate a safe default from user-confirmed center-to-center distances."""
    width, height = image.size
    if len(seeds) < 2:
        exclusion_pixels = float(fallback_pixels)
    else:
        points = np.asarray(
            [(x * width, y * height) for x, y in seeds], dtype=np.float32
        )
        differences = points[:, None, :] - points[None, :, :]
        distances = np.sqrt(np.sum(differences**2, axis=-1))
        np.fill_diagonal(distances, np.inf)
        nearest_distances = distances.min(axis=1)
        # Half the typical nearest-neighbor distance stays below the spacing
        # that the user has already judged to represent two separate cells.
        exclusion_pixels = min(
            0.5 * float(np.median(nearest_distances)),
            float(fallback_pixels) * 1.25,
        )
    exclusion_pixels = float(np.clip(exclusion_pixels, 3.0, 40.0))
    return exclusion_pixels / min(width, height), exclusion_pixels


def _image_matching_channels(rgb: np.ndarray) -> np.ndarray:
    """Build stain-aware channels directly from pixels, without learned features."""
    return _stain_channels(rgb)


def _extract_centered_patch(
    channels: np.ndarray,
    x: float,
    y: float,
    *,
    radius: int,
) -> np.ndarray:
    height, width = channels.shape[:2]
    pixel_x = min(width - 1, max(0, int(round(x * (width - 1)))))
    pixel_y = min(height - 1, max(0, int(round(y * (height - 1)))))
    padded = np.pad(channels, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    return padded[
        pixel_y : pixel_y + 2 * radius + 1,
        pixel_x : pixel_x + 2 * radius + 1,
    ]


def _normalized_template_response(channels: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return channel-wise zero-mean normalized cross-correlation."""
    template = np.asarray(template, dtype=np.float32)
    height, width, channel_count = template.shape
    sample_count = float(height * width)
    response_sum = np.zeros(channels.shape[:2], dtype=np.float32)
    response_count = np.zeros_like(response_sum)
    for channel in range(channel_count):
        values = channels[..., channel]
        centered_template = template[..., channel] - float(template[..., channel].mean())
        template_norm = float(np.sqrt(np.sum(centered_template**2)))
        if template_norm <= 1e-8:
            continue
        numerator = correlate(
            values,
            centered_template,
            mode="reflect",
        )
        local_sum = uniform_filter(values, size=(height, width), mode="reflect") * sample_count
        local_sum_sq = uniform_filter(values**2, size=(height, width), mode="reflect") * sample_count
        local_variance_sum = np.maximum(local_sum_sq - local_sum**2 / sample_count, 0.0)
        denominator = template_norm * np.sqrt(local_variance_sum)
        valid = denominator > 1e-8
        channel_response = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=valid,
        ).clip(-1.0, 1.0)
        response_sum += channel_response
        response_count += valid
    return np.divide(
        response_sum,
        response_count,
        out=np.full_like(response_sum, -1.0),
        where=response_count > 0,
    ).clip(-1.0, 1.0)


def _seed_template_response(
    channels: np.ndarray,
    seed: tuple[float, float],
    *,
    radius: int,
) -> np.ndarray:
    """Match one seed independently, allowing the local pattern to rotate."""
    template = _extract_centered_patch(channels, *seed, radius=radius)
    responses = [
        _normalized_template_response(channels, np.rot90(template, turns, axes=(0, 1)))
        for turns in range(4)
    ]
    return np.maximum.reduce(responses)


def _detect_nucleus_centers(
    hematoxylin: np.ndarray,
    *,
    nucleus_spacing_pixels: float,
    border: int,
    limit: int = 360,
) -> list[tuple[int, int]]:
    """Return one stain-derived center for each nucleus-sized response peak."""
    height, width = hematoxylin.shape
    smoothed = gaussian_filter(hematoxylin, sigma=1.0)
    window = max(3, int(round(nucleus_spacing_pixels * 0.75)))
    if window % 2 == 0:
        window += 1
    threshold = float(np.quantile(smoothed, 0.55))
    local_maxima = (
        (smoothed >= threshold)
        & (smoothed >= maximum_filter(smoothed, size=window, mode="nearest") - 1e-7)
    )
    components, component_count = label(local_maxima)
    candidates: list[tuple[int, int, float]] = []
    for component_id in range(1, component_count + 1):
        ys, xs = np.nonzero(components == component_id)
        if not len(xs):
            continue
        values = smoothed[ys, xs]
        best = int(np.argmax(values))
        pixel_x, pixel_y = int(xs[best]), int(ys[best])
        if not border <= pixel_x < width - border or not border <= pixel_y < height - border:
            continue
        candidates.append((pixel_x, pixel_y, float(values[best])))
    candidates.sort(key=lambda item: item[2], reverse=True)

    minimum_distance_sq = max(3.0, nucleus_spacing_pixels * 0.65) ** 2
    selected: list[tuple[int, int]] = []
    for pixel_x, pixel_y, _strength in candidates:
        if any(
            (pixel_x - old_x) ** 2 + (pixel_y - old_y) ** 2 < minimum_distance_sq
            for old_x, old_y in selected
        ):
            continue
        selected.append((pixel_x, pixel_y))
        if len(selected) >= limit:
            break
    return selected


def _rank_nucleus_matches(
    centers: list[tuple[int, int]],
    response: np.ndarray,
    *,
    occupied: list[tuple[float, float]],
    occupied_distance_pixels: float,
    limit: int = 240,
) -> list[tuple[float, float, float]]:
    height, width = response.shape
    occupied_pixels = [(x * width, y * height) for x, y in occupied]
    occupied_distance_sq = max(2.0, occupied_distance_pixels) ** 2
    ranked = sorted(
        (
            (pixel_x, pixel_y, float(response[pixel_y, pixel_x]))
            for pixel_x, pixel_y in centers
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    selected: list[tuple[float, float, float]] = []
    for pixel_x, pixel_y, score in ranked:
        if any(
            (pixel_x - known_x) ** 2 + (pixel_y - known_y) ** 2 < occupied_distance_sq
            for known_x, known_y in occupied_pixels
        ):
            continue
        selected.append(
            (
                (pixel_x + 0.5) / width,
                (pixel_y + 0.5) / height,
                float(np.clip(score, 0.0, 1.0)),
            )
        )
        if len(selected) >= limit:
            break
    return selected


class RoiPlanGenerator:
    """Find same-tile centers using only local image-template similarity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

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
        if attribute not in _NUCLEUS_MATCH_SIGMA:
            raise ValueError(
                "image-only nucleus matching currently supports only "
                "hepatocellular-parenchyma and inflammatory-cell"
            )

        with self._lock:
            rgb = np.asarray(image, dtype=np.float32)
            channels = _image_matching_channels(rgb)
            _morphology_spacing, morphology_spacing_pixels = _estimate_seed_spacing(
                image, normalized_seeds
            )
            minimum_spacing, minimum_spacing_pixels = _estimate_seed_exclusion_distance(
                image,
                normalized_seeds,
                fallback_pixels=morphology_spacing_pixels,
            )
            template_radius_pixels = int(
                np.clip(round(morphology_spacing_pixels * 0.75), 4, 10)
            )
            match_sigma = _NUCLEUS_MATCH_SIGMA[attribute]
            matching_map = channels[..., 0]
            if match_sigma > 0:
                matching_map = gaussian_filter(matching_map, sigma=match_sigma)
            seed_responses = [
                maximum_filter(
                    _seed_template_response(
                        matching_map[..., None],
                        seed,
                        radius=template_radius_pixels,
                    ),
                    size=5,
                    mode="nearest",
                )
                for seed in normalized_seeds
            ]
            # Each seed remains an independent example. Requiring agreement from
            # up to three seeds suppresses accidental one-template background hits
            # without averaging the seed images into a synthetic morphology.
            consensus_seed_count = min(3, len(seed_responses))
            response_stack = np.stack(seed_responses, axis=0)
            response = np.sort(response_stack, axis=0)[-consensus_seed_count:].mean(axis=0)
            # Nucleus centers may be a couple of pixels away from the user's
            # exact click. Keep the similarity local while tolerating that offset.
            nucleus_spacing_pixels = max(5.0, morphology_spacing_pixels)
            centers = _detect_nucleus_centers(
                channels[..., 0],
                nucleus_spacing_pixels=nucleus_spacing_pixels,
                border=template_radius_pixels,
            )
            ranked = _rank_nucleus_matches(
                centers,
                response,
                occupied=[*normalized_seeds, *normalized_occupied],
                occupied_distance_pixels=max(2.0, nucleus_spacing_pixels * 0.5),
                limit=240,
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
        ranked_scores = [confidence for _x, _y, confidence in ranked]
        if ranked_scores:
            preview_target = max(1, min(80, 4 * len(normalized_seeds)))
            recommended_index = min(preview_target - 1, len(ranked_scores) - 1)
            recommended_similarity = float(
                np.clip(ranked_scores[recommended_index], 0.20, 0.85)
            )
            maximum_similarity = float(ranked_scores[0])
        else:
            recommended_similarity = 0.20
            maximum_similarity = 0.0
        return {
            "version": 1,
            "method": "same-tile-nucleus-image-match-v4",
            "suggestions": suggestions,
            "summary": {
                "attribute": attribute,
                "seed_count": len(normalized_seeds),
                "occupied_count": len(normalized_occupied),
                "candidate_count": len(centers),
                "suggestion_count": len(suggestions),
                "maximum_similarity": maximum_similarity,
                "recommended_similarity": recommended_similarity,
                "preview_target": max(1, min(80, 4 * len(normalized_seeds))),
                "minimum_spacing": minimum_spacing,
                "minimum_spacing_pixels": minimum_spacing_pixels,
                "morphology_spacing_pixels": morphology_spacing_pixels,
                "template_radius_pixels": template_radius_pixels,
                "peak_spacing_pixels": nucleus_spacing_pixels,
                "consensus_seed_count": consensus_seed_count,
                "match_sigma": match_sigma,
            },
        }
