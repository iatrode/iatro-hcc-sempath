from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import logging
import math
import os
import random
import re
import secrets
import socket
import tempfile
import threading
import webbrowser
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from hcc_sempath.cli._annotation_tiles import AnnotationTilePackageReader, IacRecord
from hcc_sempath.training.prototype_labels import (
    DEFAULT_CLASSIFICATION_CLASSES,
)
from iatro.iac import read_header
from iatro.iac.adapters.tiles import decode_jxl


LOG = logging.getLogger("hcc_sempath.annotate_prototypes")
OVERVIEW_CELL_PX = 4
OVERVIEW_WORKERS = 8
CONTEXT_VISIBLE_RADIUS = 2
CONTEXT_PREFETCH_RADIUS = 5
CONTEXT_MAX_RADIUS = 8
CONTEXT_WORKERS = 8
MAX_OPEN_IAC_VIEWERS = 8

CLASSIFICATION_PROTOTYPES = list(DEFAULT_CLASSIFICATION_CLASSES)

SPATIAL_PROTOTYPES = [
    "hepatocellular-parenchyma",
    "necrosis",
    "hemorrhage",
    "bile-pigment",
    "inflammatory-cell",
    "fibroblast",
    "fibrous-stroma",
    "steatosis-vacuolation",
    "small-vessel",
    "large-vessel",
    "ductular-portal",
]

# Spatial ROI taxonomy.
ROI_SPATIAL_PROTOTYPES = list(SPATIAL_PROTOTYPES)
ROI_INFORMATION_SUPPORT_THRESHOLD = 0.80


def _validate_brush_geometry(geometry: dict) -> None:
    points = geometry.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("brush geometry requires at least one point")
    try:
        width = float(geometry["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("brush geometry requires a finite positive width") from exc
    if not math.isfinite(width) or width <= 0:
        raise ValueError("brush geometry requires a finite positive width")

    parsed_points: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        try:
            if isinstance(point, dict):
                x, y = float(point["x"]), float(point["y"])
            else:
                x, y = float(point[0]), float(point[1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"brush geometry point {index} requires finite x/y coordinates"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"brush geometry point {index} requires finite x/y coordinates"
            )
        parsed_points.append((x, y))

    if str(geometry.get("coordinate_space", "pixel")).lower() == "normalized":
        radius = width / 2.0
        xs = [point[0] for point in parsed_points]
        ys = [point[1] for point in parsed_points]
        if (
            max(xs) + radius < 0.0
            or min(xs) - radius > 1.0
            or max(ys) + radius < 0.0
            or min(ys) - radius > 1.0
        ):
            raise ValueError(
                "normalized brush geometry does not intersect the tile canvas"
            )


class RoiCandidateQueue:
    def __init__(
        self,
        path: str | Path,
        *,
        information_report_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("ROI candidate pool requires a non-empty candidates list")
        self.attributes = list(payload.get("spatial_prototypes") or ROI_SPATIAL_PROTOTYPES)
        if self.attributes != ROI_SPATIAL_PROTOTYPES:
            raise ValueError(f"ROI candidate pool taxonomy must be {ROI_SPATIAL_PROTOTYPES}")
        self.candidates: list[dict] = []
        self.by_tile_id: dict[str, dict] = {}
        for rank, item in enumerate(candidates):
            tile_id = str(item.get("tile_id") or "").strip()
            if not tile_id or tile_id in self.by_tile_id:
                raise ValueError(f"invalid or duplicate ROI candidate tile_id: {tile_id!r}")
            record = dict(item)
            record["tile_id"] = tile_id
            record["rank"] = int(item.get("rank", rank))
            record["source_spatial"] = [
                name for name in item.get("source_spatial", []) if name in self.attributes
            ]
            record["priority_attributes"] = [
                name for name in item.get("priority_attributes", record["source_spatial"])
                if name in self.attributes
            ]
            if not record["priority_attributes"]:
                record["priority_attributes"] = list(record["source_spatial"])
            self.candidates.append(record)
            self.by_tile_id[tile_id] = record
        self.information_report_path = (
            Path(information_report_path)
            if information_report_path
            else None
        )
        self._information_report_mtime_ns: int | None = None
        self._information_policy: dict[str, dict] = {}
        self._last_sampling_policy: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _reload_information_policy(self) -> dict[str, dict]:
        path = self.information_report_path
        if path is None or not path.is_file():
            return {
                name: {
                    "status": "unmeasured",
                    "urgency": 1.0,
                    "information_support": 0.0,
                }
                for name in self.attributes
            }
        mtime_ns = path.stat().st_mtime_ns
        with self._lock:
            if (
                self._information_policy
                and self._information_report_mtime_ns == mtime_ns
            ):
                return self._information_policy
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                LOG.warning(
                    "roi_information_policy_read_failed path=%s",
                    path,
                    exc_info=True,
                )
                return self._information_policy or {
                    name: {
                        "status": "unmeasured",
                        "urgency": 1.0,
                        "information_support": 0.0,
                    }
                    for name in self.attributes
                }
            raw_attributes = payload.get("attributes", {})
            policy: dict[str, dict] = {}
            for name in self.attributes:
                value = raw_attributes.get(name, {})
                status = str(value.get("status", "not_assessable"))
                pooled_support = value.get(
                    "tail_plateau_support_by_ratio",
                    {},
                ).get("0.35")
                if isinstance(pooled_support, (int, float)):
                    information_support = float(pooled_support)
                else:
                    information_support = 0.0
                if status == "provisionally_stable":
                    urgency = 0.0
                elif status == "coverage_limited":
                    urgency = 0.75
                elif status == "still_growing":
                    support_gap = max(
                        0.0,
                        ROI_INFORMATION_SUPPORT_THRESHOLD
                        - information_support,
                    ) / ROI_INFORMATION_SUPPORT_THRESHOLD
                    urgency = 1.0 + support_gap
                else:
                    urgency = 2.0
                policy[name] = {
                    "status": status,
                    "urgency": urgency,
                    "information_support": information_support,
                }
            self._information_report_mtime_ns = mtime_ns
            self._information_policy = policy
            LOG.info(
                "roi_information_policy_reload path=%s priorities=%s",
                path,
                ",".join(
                    name
                    for name in self.attributes
                    if policy[name]["urgency"] >= 0.75
                ),
            )
            return policy

    def _source_target_probabilities(
        self,
        roi_positive_by_tile: dict[str, set[str]],
    ) -> dict[str, dict[str, float]]:
        reviewed = Counter()
        positive = {
            source: Counter()
            for source in self.attributes
        }
        for tile_id, targets in roi_positive_by_tile.items():
            candidate = self.by_tile_id.get(tile_id)
            if candidate is None:
                continue
            for source in candidate["source_spatial"]:
                reviewed[source] += 1
                positive[source].update(
                    target for target in targets if target in self.attributes
                )
        result: dict[str, dict[str, float]] = {}
        for source in self.attributes:
            result[source] = {}
            for target in self.attributes:
                if source == target:
                    prior_mean, prior_strength = 0.65, 4.0
                else:
                    prior_mean, prior_strength = 0.10, 6.0
                result[source][target] = (
                    positive[source][target]
                    + prior_mean * prior_strength
                ) / (reviewed[source] + prior_strength)
        return result

    def sampling_policy(self) -> dict[str, dict]:
        return {
            name: dict(value)
            for name, value in self._last_sampling_policy.items()
        }

    def information_policy(self) -> dict[str, dict]:
        return {
            name: dict(value)
            for name, value in self._reload_information_policy().items()
        }

    def direct_hint_status(
        self,
        processed_tile_ids: set[str],
        *,
        allowed_tile_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int | bool]]:
        relevant = [
            item
            for item in self.candidates
            if allowed_tile_ids is None
            or item["tile_id"] in allowed_tile_ids
        ]
        result = {}
        for name in self.attributes:
            direct = [
                item
                for item in relevant
                if name in item["source_spatial"]
            ]
            reviewed = sum(
                item["tile_id"] in processed_tile_ids
                for item in direct
            )
            total = len(direct)
            result[name] = {
                "total": total,
                "reviewed": reviewed,
                "remaining": total - reviewed,
                "exhausted": total > 0 and reviewed == total,
            }
        return result

    def contains(self, tile_id: str) -> bool:
        return tile_id in self.by_tile_id

    def candidate(self, tile_id: str) -> dict | None:
        return self.by_tile_id.get(tile_id)

    def weighted_remaining(
        self,
        *,
        processed_tile_ids: set[str],
        roi_positive_counts: dict[str, int],
        roi_positive_by_tile: dict[str, set[str]] | None = None,
        priority_attributes: list[str] | tuple[str, ...] | None = None,
        allowed_tile_ids: set[str] | None = None,
        exclude_tile_id: str | None = None,
    ) -> list[dict]:
        """Return a dynamically weighted order of component-presence hints.

        Component-presence labels choose which image is shown next.
        Current information-curve deficits set the target urgency. The
        observed mapping from presence hints to spatial positives
        is updated after every review, allowing related labels to
        recover a target whose direct hint inventory has been exhausted.
        """
        explicit_priority = bool(priority_attributes)
        active = [
            name for name in (priority_attributes or self.attributes)
            if name in self.attributes
        ]
        active_set = set(active)
        rank_weight = {
            name: len(active) - index if explicit_priority else 1.0
            for index, name in enumerate(active)
        }
        information_policy = self._reload_information_policy()
        deficient = {
            name
            for name in active
            if float(information_policy[name]["urgency"]) > 0.0
        }
        candidates = [
            item for item in self.candidates
            if item["tile_id"] not in processed_tile_ids
            and item["tile_id"] != exclude_tile_id
            and (allowed_tile_ids is None or item["tile_id"] in allowed_tile_ids)
            and item["source_spatial"]
        ]
        probabilities = self._source_target_probabilities(
            roi_positive_by_tile or {},
        )
        candidate_target_probabilities: dict[str, dict[str, float]] = {}
        predicted_inventory = Counter()
        for item in candidates:
            target_probabilities = {
                target: max(
                    (
                        probabilities[source][target]
                        for source in item["source_spatial"]
                        if source in probabilities
                    ),
                    default=0.0,
                )
                for target in active
            }
            candidate_target_probabilities[item["tile_id"]] = (
                target_probabilities
            )
            predicted_inventory.update(target_probabilities)
        weighted: list[tuple[bool, float, dict]] = []
        for item in candidates:
            target_probabilities = candidate_target_probabilities[
                item["tile_id"]
            ]
            contributions = {
                target: (
                    float(information_policy[target]["urgency"])
                    * rank_weight[target]
                    * target_probabilities[target]
                    / (1.0 + int(roi_positive_counts.get(target, 0)))
                    ** 0.25
                    / max(1.0, predicted_inventory[target]) ** 0.5
                )
                for target in active
                if (
                    target in active_set
                    and float(information_policy[target]["urgency"]) > 0.0
                )
            }
            score = sum(contributions.values())
            if score <= 0:
                continue
            # Weighted sampling without replacement. The random key keeps
            # "next" varied while retaining a strong priority/rarity bias.
            key = random.random() ** (1.0 / score)
            ranked_targets = sorted(
                contributions,
                key=contributions.get,
                reverse=True,
            )
            direct_targets = [
                target
                for target in ranked_targets
                if (
                    target in deficient
                    and target in item["source_spatial"]
                )
            ]
            weighted.append(
                (
                    bool(direct_targets),
                    key,
                    {
                        **item,
                        "direct_priority_attributes": direct_targets,
                        "dynamic_priority_attributes": ranked_targets,
                        "dynamic_score": score,
                    },
                )
            )
        weighted.sort(
            key=lambda pair: (pair[0], pair[1]),
            reverse=True,
        )
        remaining_direct = {
            name: sum(
                name in item["source_spatial"]
                for item in candidates
            )
            for name in active
        }
        with self._lock:
            self._last_sampling_policy = {
                name: {
                    **information_policy[name],
                    "positive_tile_count": int(
                        roi_positive_counts.get(name, 0)
                    ),
                    "remaining_direct_hint_count": remaining_direct[name],
                    "predicted_remaining_yield": float(
                        predicted_inventory[name]
                    ),
                }
                for name in active
            }
        return [item for _direct, _key, item in weighted]


class SharedPriorityQueue:
    """A mutable tile boundary shared by every annotation mode and version."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("shared priority manifest requires a non-empty candidates list")
        self._lock = threading.RLock()
        self.candidates: list[dict] = []
        self.by_tile_id: dict[str, dict] = {}
        for rank, item in enumerate(candidates):
            self._append_normalized(item, rank=rank)

    def _append_normalized(self, item: dict, *, rank: int) -> dict:
        tile_id = str(item.get("tile_id") or "").strip()
        iac = str(item.get("iac") or item.get("iac_path") or "").strip()
        row = int(item.get("row", -1))
        if not tile_id or not iac or row < 0:
            raise ValueError(f"invalid shared priority tile: {item!r}")
        existing = self.by_tile_id.get(tile_id)
        if existing is not None:
            if Path(existing["iac"]).as_posix() != Path(iac).as_posix() or existing["row"] != row:
                raise ValueError(f"conflicting shared priority tile_id: {tile_id}")
            return existing
        record = {
            "tile_id": tile_id,
            "iac": iac,
            "row": row,
            "slide": str(item.get("slide") or item.get("slide_id") or ""),
            "rank": rank,
        }
        self.candidates.append(record)
        self.by_tile_id[tile_id] = record
        return record

    def contains(self, tile_id: str) -> bool:
        with self._lock:
            return tile_id in self.by_tile_id

    def add(self, package: AnnotationPackage, record: IacRecord) -> bool:
        item = {
            "tile_id": record.tile_id,
            "iac": package.rel_path,
            "row": record.row,
            "slide": record.slide_label,
        }
        with self._lock:
            if record.tile_id in self.by_tile_id:
                return False
            self._append_normalized(item, rank=len(self.candidates))
            self._flush_locked()
            LOG.info(
                "shared_priority_add path=%s iac=%s row=%d tile_id=%s total=%d",
                self.path,
                package.rel_path,
                record.row,
                record.tile_id,
                len(self.candidates),
            )
            return True

    def add_annotations(self, annotations: list[dict]) -> int:
        added = 0
        with self._lock:
            for item in annotations:
                if not item.get("tile_id") or item["tile_id"] in self.by_tile_id:
                    continue
                try:
                    self._append_normalized(item, rank=len(self.candidates))
                except (TypeError, ValueError):
                    LOG.warning("shared_priority_existing_annotation_skip item=%r", item)
                    continue
                added += 1
            if added:
                self._flush_locked()
                LOG.info("shared_priority_extend path=%s added=%d total=%d", self.path, added, len(self.candidates))
        return added

    def progress(self, state: AnnotationState) -> dict[str, int]:
        with self._lock:
            priority_ids = set(self.by_tile_id)
        reviewed_ids = {
            str(item.get("tile_id"))
            for item in state.annotations.values()
            if item.get("tile_id") and _is_counted_annotation(item)
        }
        skipped_ids = {
            parts[1]
            for key in state.skipped
            if len(parts := key.split("::")) >= 2
        }
        reviewed = len(priority_ids & reviewed_ids)
        skipped = len((priority_ids & skipped_ids) - reviewed_ids)
        total = len(priority_ids)
        return {
            "reviewed": reviewed,
            "skipped": skipped,
            "total": total,
            "remaining": max(0, total - reviewed - skipped),
        }

    def _flush_locked(self) -> None:
        payload = {"version": 1, "candidate_count": len(self.candidates), "candidates": self.candidates}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.path.parent) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            tmp_path = Path(handle.name)
        tmp_path.replace(self.path)


class StrictReviewQueue:
    """A read-only, ordered tile boundary for one explicit review pass."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("strict review manifest requires a non-empty candidates list")
        self.review_id = str(payload.get("review_id") or self.path.stem).strip()
        if not self.review_id:
            raise ValueError("strict review manifest requires a non-empty review_id")
        self.candidates: list[dict] = []
        self.by_tile_id: dict[str, dict] = {}
        for rank, item in enumerate(candidates):
            tile_id = str(item.get("tile_id") or "").strip()
            iac = str(item.get("iac") or item.get("iac_path") or "").strip()
            row = int(item.get("row", -1))
            if not tile_id or not iac or row < 0:
                raise ValueError(f"invalid strict review tile: {item!r}")
            if tile_id in self.by_tile_id:
                raise ValueError(f"duplicate strict review tile_id: {tile_id}")
            record = {
                "tile_id": tile_id,
                "iac": iac,
                "row": row,
                "slide": str(item.get("slide") or item.get("slide_id") or ""),
                "rank": rank,
            }
            self.candidates.append(record)
            self.by_tile_id[tile_id] = record

    def progress(self, state: AnnotationState) -> dict[str, int | str]:
        reviewed, skipped = state.review_completion(self.review_id)
        candidate_ids = set(self.by_tile_id)
        reviewed_count = len(candidate_ids & reviewed)
        skipped_count = len((candidate_ids & skipped) - reviewed)
        total = len(candidate_ids)
        return {
            "review_id": self.review_id,
            "reviewed": reviewed_count,
            "skipped": skipped_count,
            "total": total,
            "remaining": max(0, total - reviewed_count - skipped_count),
        }


@dataclass(frozen=True)
class AnnotationPackage:
    path: Path
    rel_path: str
    dataset: str
    total: int


def _find_free_port(host: str, preferred: int) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _auth_ok(provided: str, expected: str) -> bool:
    return not expected or (
        bool(provided) and hmac.compare_digest(provided, expected)
    )


def _request_auth_token(query: dict[str, list[str]]) -> str:
    for key in ("token", "auth_token", "access_token"):
        value = query.get(key, [""])[0]
        if value:
            return value
    return ""


def _auth_token_path(state_path: str | Path) -> Path:
    return Path(state_path).with_suffix(".auth-token")


def _load_or_create_auth_token(state_path: str | Path, provided: str = "") -> str:
    if provided:
        return provided
    token_path = _auth_token_path(state_path)
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(f"{token}\n", encoding="utf-8")
    token_path.chmod(0o600)
    return token


def _candidate_iac_paths(
    root: Path,
    include_packages: tuple[str, ...] = (),
) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".iac" else []

    if include_packages:
        paths = []
        seen: set[Path] = set()
        for relative in include_packages:
            matches = sorted(root.glob(relative))
            if not matches:
                raise FileNotFoundError(
                    f"included image-tile IAC package is absent: {root / relative}"
                )
            for candidate in matches:
                if (
                    candidate.suffix != ".iac"
                    or not candidate.is_file()
                    or candidate in seen
                ):
                    continue
                paths.append(candidate)
                seen.add(candidate)
        if not paths:
            raise FileNotFoundError(
                f"no included image-tile IAC packages found under: {root}"
            )
        return paths

    paths = []
    visited_dirs: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real_dir = os.path.realpath(dirpath)
        if real_dir in visited_dirs:
            dirnames[:] = []
            continue
        visited_dirs.add(real_dir)

        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if os.path.realpath(Path(dirpath) / dirname) not in visited_dirs
        )
        for filename in sorted(filenames):
            if filename.endswith(".iac"):
                paths.append(Path(dirpath) / filename)
    return paths


def _package_from_path(root: Path, path: Path) -> AnnotationPackage | None:
    header = read_header(path)
    if header.get("payload_type") != "image_tiles":
        if root.is_file():
            raise ValueError(f"annotation requires image-tile IAC package: {path}")
        LOG.info("discover_iac_skip path=%s payload_type=%s", path, header.get("payload_type"))
        return None
    total = int(header.get("num_records", 0))
    rel = path.name if root.is_file() else str(path.relative_to(root))
    dataset = ""
    if not root.is_file():
        parent = path.parent.relative_to(root)
        dataset = "" if str(parent) == "." else parent.parts[0]
    return AnnotationPackage(path=path, rel_path=rel, dataset=dataset, total=total)


def discover_iac_packages(
    input_path: str | Path,
    include_packages: tuple[str, ...] = (),
) -> list[AnnotationPackage]:
    root = Path(input_path).resolve()
    LOG.info("discover_iac_start input=%s", root)
    packages = []
    for path in _candidate_iac_paths(root, include_packages):
        package = _package_from_path(root, path)
        if package is None:
            continue
        packages.append(package)
        LOG.info("discover_iac_add rel_path=%s dataset=%s records=%d", package.rel_path, package.dataset or "-", package.total)
    if not packages:
        raise FileNotFoundError(f"no image-tile .iac packages found under: {root}")
    LOG.info("discover_iac_done input=%s packages=%d", root, len(packages))
    return packages


def _annotation_key(package: AnnotationPackage, record: IacRecord) -> str:
    return f"{package.rel_path}::{record.tile_id}::{record.display_x},{record.display_y}"


def _is_counted_annotation(item: dict) -> bool:
    if bool(item.get("skipped")) or bool(item.get("skip")):
        return False
    skip_values = {"skip", "skipped"}
    decision = str(item.get("decision") or "").strip().lower()
    review_decision = str(item.get("review_decision") or "").strip().lower()
    if decision in skip_values or review_decision in skip_values:
        return False
    return bool(str(item.get("classification") or item.get("classification_label") or "").strip())


def _ordered_unique(values) -> list[str]:
    result = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def _state_prototypes(payload: dict, key: str, fallback: list[str], annotation_field: str) -> list[str]:
    values = list(payload.get(key) or fallback)
    for item in payload.get("annotations", {}).values():
        observed = item.get(annotation_field)
        if isinstance(observed, list):
            values.extend(observed)
        else:
            values.append(observed)
    return _ordered_unique(values)


def _default_label_definitions(task: str, values: list[str]) -> list[dict]:
    return [
        {"id": value, "name": value, "task": task, "active": True, "order": index}
        for index, value in enumerate(values)
    ]


def _normalized_label_definitions(task: str, values: list[str], raw: object) -> list[dict]:
    definitions: list[dict] = []
    seen: set[str] = set()
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            label_id = str(item.get("id") or "").strip()
            if not label_id or label_id in seen:
                continue
            definitions.append({
                "id": label_id,
                "name": str(item.get("name") or label_id).strip() or label_id,
                "task": task,
                "active": bool(item.get("active", True)),
                "order": int(item.get("order", index)),
            })
            seen.add(label_id)
    for value in values:
        if value not in seen:
            definitions.append({
                "id": value, "name": value, "task": task,
                "active": True, "order": len(definitions),
            })
            seen.add(value)
    return sorted(definitions, key=lambda item: (item["order"], item["id"]))


class AnnotationState:
    def __init__(
        self,
        state_path: str | Path,
        input_path: str | Path,
        *,
        spatial_prototypes: list[str] | None = None,
        require_complete_roi: bool = False,
    ) -> None:
        self.state_path = Path(state_path)
        self.input_path = str(Path(input_path).resolve())
        self.annotations: dict[str, dict] = {}
        self.skipped: set[str] = set()
        self.extra_payload: dict = {}
        self.last_iac = ""
        self.last_row: int | None = None
        self.classification_prototypes = list(CLASSIFICATION_PROTOTYPES)
        self.spatial_prototypes = list(spatial_prototypes or SPATIAL_PROTOTYPES)
        self.require_complete_roi = bool(require_complete_roi)
        self.label_definitions = {
            "classification": _default_label_definitions("classification", self.classification_prototypes),
            "spatial": _default_label_definitions("spatial", self.spatial_prototypes),
        }
        self.revision = 0
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.input_path = str(Path(payload.get("input_path", self.input_path)).resolve())
            self.annotations = dict(payload.get("annotations", {}))
            self.skipped = set(str(item) for item in payload.get("skipped", []))
            self.last_iac = str(payload.get("last_iac") or "")
            raw_last_row = payload.get("last_row")
            self.last_row = int(raw_last_row) if raw_last_row is not None else None
            self.classification_prototypes = _state_prototypes(payload, "classification_prototypes", CLASSIFICATION_PROTOTYPES, "classification")
            if spatial_prototypes is None:
                self.spatial_prototypes = _state_prototypes(payload, "spatial_prototypes", SPATIAL_PROTOTYPES, "spatial")
            else:
                self.spatial_prototypes = list(spatial_prototypes)
            raw_definitions = payload.get("label_definitions") or {}
            self.label_definitions = {
                "classification": _normalized_label_definitions("classification", self.classification_prototypes, raw_definitions.get("classification")),
                "spatial": _normalized_label_definitions("spatial", self.spatial_prototypes, raw_definitions.get("spatial")),
            }
            self._sync_active_prototypes()
            known_keys = {
                "version", "input_path", "classification_prototypes", "spatial_prototypes",
                "annotations", "skipped", "last_iac", "last_row", "roi_mode", "label_definitions",
            }
            self.extra_payload = {key: value for key, value in payload.items() if key not in known_keys}
            LOG.info(
                "state_load path=%s annotations=%d skipped=%d classification=%d spatial=%d input=%s",
                self.state_path,
                len(self.annotations),
                len(self.skipped),
                len(self.classification_prototypes),
                len(self.spatial_prototypes),
                self.input_path,
            )
        else:
            LOG.info("state_new path=%s input=%s", self.state_path, self.input_path)

    def _sync_active_prototypes(self) -> None:
        self.classification_prototypes = [item["id"] for item in self.label_definitions["classification"] if item["active"]]
        self.spatial_prototypes = [item["id"] for item in self.label_definitions["spatial"] if item["active"]]

    def labels_json(self) -> dict:
        return {"revision": self.revision, "label_sets": self.label_definitions}

    def review_completion(self, review_id: str) -> tuple[set[str], set[str]]:
        passes = self.extra_payload.get("review_passes")
        if not isinstance(passes, dict):
            return set(), set()
        value = passes.get(review_id)
        if not isinstance(value, dict):
            return set(), set()
        return (
            {str(item) for item in value.get("reviewed", [])},
            {str(item) for item in value.get("skipped", [])},
        )

    def _mark_review_completion(
        self,
        review_id: str | None,
        tile_id: str,
        *,
        skipped: bool,
    ) -> None:
        if not review_id:
            return
        reviewed, skipped_ids = self.review_completion(review_id)
        if skipped:
            reviewed.discard(tile_id)
            skipped_ids.add(tile_id)
        else:
            skipped_ids.discard(tile_id)
            reviewed.add(tile_id)
        passes = self.extra_payload.setdefault("review_passes", {})
        passes[review_id] = {
            "reviewed": sorted(reviewed),
            "skipped": sorted(skipped_ids),
        }

    def _label_references(self, task: str, label_id: str) -> int:
        if task == "classification":
            return sum(1 for item in self.annotations.values() if item.get("classification") == label_id)
        return sum(1 for item in self.annotations.values() if label_id in item.get("spatial", []))

    def change_label(self, task: str, operation: str, *, label_id: str = "", name: str = "") -> dict:
        task = task.lower()
        if task not in {"classification", "spatial"}:
            raise ValueError("label task must be classification or spatial")
        definitions = self.label_definitions[task]
        current = next((item for item in definitions if item["id"] == label_id), None)
        clean_name = name.strip()
        if operation == "add":
            if not clean_name:
                raise ValueError("label name is required")
            if any(item["name"].casefold() == clean_name.casefold() for item in definitions):
                raise ValueError(f"duplicate {task.upper()} label name: {clean_name}")
            slug = re.sub(r"[^a-z0-9]+", "-", clean_name.lower()).strip("-") or "label"
            base = f"{task}_{slug}"
            label_id = base
            suffix = 2
            known = {item["id"] for item in definitions}
            while label_id in known:
                label_id = f"{base}-{suffix}"
                suffix += 1
            definitions.append({"id": label_id, "name": clean_name, "task": task, "active": True, "order": len(definitions)})
        elif current is None:
            raise ValueError(f"unknown {task.upper()} label: {label_id}")
        elif operation == "rename":
            if not clean_name:
                raise ValueError("label name is required")
            if any(item["id"] != label_id and item["name"].casefold() == clean_name.casefold() for item in definitions):
                raise ValueError(f"duplicate {task.upper()} label name: {clean_name}")
            current["name"] = clean_name
        elif operation == "archive":
            if task == "classification" and sum(bool(item["active"]) for item in definitions) <= 1 and current["active"]:
                raise ValueError("classification requires at least one active label")
            current["active"] = False
        elif operation == "restore":
            current["active"] = True
        elif operation == "delete":
            references = self._label_references(task, label_id)
            if references:
                raise ValueError(f"label has {references} annotation reference(s); archive it instead")
            if task == "classification" and current["active"] and sum(bool(item["active"]) for item in definitions) <= 1:
                raise ValueError("classification requires at least one active label")
            definitions.remove(current)
        else:
            raise ValueError(f"unknown label operation: {operation}")
        self._sync_active_prototypes()
        self.revision += 1
        self.flush()
        return self.labels_json()

    @property
    def csv_path(self) -> Path:
        return self.state_path.with_suffix(".csv")

    def is_annotated(self, package: AnnotationPackage, record: IacRecord) -> bool:
        key = _annotation_key(package, record)
        return key in self.annotations or key in self.skipped

    def annotation_for(self, package: AnnotationPackage, record: IacRecord) -> dict | None:
        value = self.annotations.get(_annotation_key(package, record))
        return dict(value) if value is not None else None

    def roi_positive_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.spatial_prototypes}
        expected = set(self.spatial_prototypes)
        for item in self.annotations.values():
            roi = list(item.get("roi") or [])
            positive = {
                str(entry.get("attribute"))
                for entry in roi
                if entry.get("geometry") is not None and entry.get("state", "positive") == "positive"
            }
            for name in positive & expected:
                counts[name] += 1
        return counts

    def counts_for_package(self, package: AnnotationPackage, records: list[IacRecord]) -> dict:
        annotated = sum(1 for record in records if self.is_counted_record(package, record))
        skipped = sum(1 for record in records if _annotation_key(package, record) in self.skipped)
        return {"annotated": annotated, "total": len(records), "remaining": max(0, len(records) - annotated - skipped), "skipped": skipped}

    def annotations_for_package(self, package: AnnotationPackage) -> list[dict]:
        prefix = f"{package.rel_path}::"
        return [value for key, value in self.annotations.items() if key.startswith(prefix)]

    def counted_annotations_for_package(self, package: AnnotationPackage) -> list[dict]:
        prefix = f"{package.rel_path}::"
        return [item for key, item in self.annotations.items() if key.startswith(prefix) and key not in self.skipped and _is_counted_annotation(item)]

    def skipped_for_package(self, package: AnnotationPackage) -> set[str]:
        prefix = f"{package.rel_path}::"
        return {key for key in self.skipped if key.startswith(prefix)}

    def is_counted_record(self, package: AnnotationPackage, record: IacRecord) -> bool:
        key = _annotation_key(package, record)
        return key not in self.skipped and _is_counted_annotation(self.annotations.get(key, {}))

    def lightweight_counts_for_package(self, package: AnnotationPackage) -> dict:
        annotated = len(self.counted_annotations_for_package(package))
        skipped = len(self.skipped_for_package(package))
        return {"annotated": annotated, "total": package.total, "remaining": max(0, package.total - annotated - skipped), "skipped": skipped}

    def save_annotation(
        self,
        package: AnnotationPackage,
        record: IacRecord,
        classification: str,
        spatial: list[str],
        roi: list[dict] | None = None,
        *,
        review_id: str | None = None,
    ) -> None:
        known_classification = {item["id"] for item in self.label_definitions["classification"]}
        known_spatial = {item["id"] for item in self.label_definitions["spatial"]}
        if classification not in known_classification:
            raise ValueError(f"unknown classification prototype: {classification}")
        unknown_spatial = sorted(set(spatial) - known_spatial)
        if unknown_spatial:
            raise ValueError(f"unknown spatial prototype(s): {unknown_spatial}")
        roi = list(roi or [])
        for item in roi:
            if item.get("attribute") not in known_spatial:
                raise ValueError(f"unknown spatial attribute: {item.get('attribute')}")
            if item.get("state", "positive") not in {"positive", "negative"}:
                raise ValueError(f"unknown ROI state: {item.get('state')}")
            geometry = item.get("geometry")
            if geometry is None and not bool(item.get("review_complete", False)):
                raise ValueError("ROI item requires geometry or review_complete=true")
            if geometry is not None and geometry.get("type") not in {"point", "brush", "circle", "polygon"}:
                raise ValueError(f"unsupported ROI geometry: {geometry.get('type')}")
            if geometry is not None and geometry.get("type") == "brush":
                _validate_brush_geometry(geometry)
        if self.require_complete_roi:
            positive_roi = {
                str(item.get("attribute"))
                for item in roi
                if item.get("geometry") is not None and item.get("state", "positive") == "positive"
            }
            negative_reviewed = {
                str(item.get("attribute"))
                for item in roi
                if item.get("geometry") is None
                and item.get("state") == "negative"
                and item.get("review_complete")
            }
            conflict = sorted(positive_roi & negative_reviewed)
            if conflict:
                raise ValueError(f"ROI class cannot be both positive and negative: {conflict}")
            if set(spatial) != positive_roi:
                raise ValueError(
                    f"spatial selections must equal positive ROI attributes: "
                    f"selected={sorted(spatial)} roi={sorted(positive_roi)}"
                )
        payload = {
            "dataset": package.dataset,
            "iac": package.rel_path,
            "iac_path": str(package.path),
            "tile_id": record.tile_id,
            "row": record.row,
            "slide": record.slide_label,
            "split": record.split,
            "x": record.display_x,
            "y": record.display_y,
            "classification": classification,
            "spatial": list(spatial),
            "roi": roi,
            "roi_complete_all": False,
            "roi_reviewed": self.require_complete_roi,
        }
        key = _annotation_key(package, record)
        self.annotations[key] = payload
        self.skipped.discard(key)
        self._mark_review_completion(
            review_id,
            record.tile_id,
            skipped=False,
        )
        self.last_iac = package.rel_path
        self.last_row = record.row
        self.revision += 1
        self.flush()
        LOG.info(
            "annotation_save iac=%s row=%d tile_id=%s x=%d y=%d classification=%s spatial=%s total_annotations=%d",
            package.rel_path,
            record.row,
            record.tile_id,
            record.display_x,
            record.display_y,
            classification,
            ",".join(spatial) if spatial else "-",
            len(self.annotations),
        )

    def save_skip(
        self,
        package: AnnotationPackage,
        record: IacRecord,
        *,
        review_id: str | None = None,
    ) -> None:
        key = _annotation_key(package, record)
        if review_id:
            self._mark_review_completion(
                review_id,
                record.tile_id,
                skipped=True,
            )
            self.last_iac = package.rel_path
            self.last_row = record.row
            self.revision += 1
            self.flush()
            LOG.info(
                "review_skip review_id=%s iac=%s row=%d tile_id=%s",
                review_id,
                package.rel_path,
                record.row,
                record.tile_id,
            )
            return
        if key in self.annotations:
            return
        self.skipped.add(key)
        self.last_iac = package.rel_path
        self.last_row = record.row
        self.revision += 1
        self.flush()
        LOG.info(
            "annotation_skip iac=%s row=%d tile_id=%s x=%d y=%d total_skipped=%d",
            package.rel_path,
            record.row,
            record.tile_id,
            record.display_x,
            record.display_y,
            len(self.skipped),
        )

    def flush(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.extra_payload)
        payload.update({
            "version": 2,
            "input_path": self.input_path,
            "classification_prototypes": self.classification_prototypes,
            "spatial_prototypes": self.spatial_prototypes,
            "last_iac": self.last_iac,
            "last_row": self.last_row,
            "annotations": self.annotations,
            "skipped": sorted(self.skipped),
            "roi_mode": self.require_complete_roi,
            "label_definitions": self.label_definitions,
        })
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.state_path.parent) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.state_path)
        self._write_csv()
        LOG.info("state_flush json=%s csv=%s annotations=%d", self.state_path, self.csv_path, len(self.annotations))

    def _write_csv(self) -> None:
        base_fields = ["dataset", "iac", "iac_path", "tile_id", "row", "slide", "split", "x", "y", "classification", "classification_name", "spatial", "spatial_names"]
        extra_fields = sorted(
            {
                key
                for item in self.annotations.values()
                for key in item.keys()
                if key not in set(base_fields)
            }
        )
        fieldnames = base_fields + extra_fields
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in sorted(self.annotations.values(), key=lambda value: (value["iac"], value["row"])):
                row = dict(item)
                classification_names = {entry["id"]: entry["name"] for entry in self.label_definitions["classification"]}
                spatial_names = {entry["id"]: entry["name"] for entry in self.label_definitions["spatial"]}
                row["classification_name"] = classification_names.get(row.get("classification", ""), row.get("classification", ""))
                row["spatial_names"] = ";".join(spatial_names.get(value, value) for value in row.get("spatial", []))
                row["spatial"] = ";".join(row.get("spatial", []))
                writer.writerow(row)


class AnnotationData:
    def __init__(
        self,
        input_path: str | Path,
        state_path: str | Path,
        *,
        async_scan: bool = False,
        roi_candidate_manifest: str | Path | None = None,
        roi_information_report: str | Path | None = None,
        roi_priority_attributes: list[str] | tuple[str, ...] | None = None,
        roi_mode: bool = False,
        priority_queue: SharedPriorityQueue | None = None,
        review_queue: StrictReviewQueue | None = None,
        include_packages: list[str] | tuple[str, ...] | None = None,
        min_tissue_fraction: float = 0.30,
    ) -> None:
        self.input_root = Path(input_path).resolve()
        self.cache_root = (self.input_root if self.input_root.is_dir() else self.input_root.parent) / ".hcc_sempath_annotation_cache"
        self.packages: list[AnnotationPackage] = []
        self.roi_queue = (
            RoiCandidateQueue(
                roi_candidate_manifest,
                information_report_path=roi_information_report,
            )
            if roi_candidate_manifest
            else None
        )
        self.roi_information_report = (
            str(roi_information_report)
            if roi_information_report
            else None
        )
        self.roi_mode = bool(roi_mode or self.roi_queue is not None)
        requested_priority = tuple(dict.fromkeys(
            str(name).strip() for name in (roi_priority_attributes or ()) if str(name).strip()
        ))
        unknown_priority = set(requested_priority) - set(ROI_SPATIAL_PROTOTYPES)
        if unknown_priority:
            raise ValueError(f"unknown ROI navigation priority attributes: {sorted(unknown_priority)}")
        self.roi_priority_attributes = requested_priority
        self.priority_queue = priority_queue
        self.review_queue = review_queue
        self.include_packages = tuple(dict.fromkeys(include_packages or ()))
        if not 0.0 <= min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be between 0 and 1")
        self.min_tissue_fraction = float(min_tissue_fraction)
        self._tissue_fraction_cache: dict[tuple[str, int], float] = {}
        self._auto_filtered: set[tuple[str, int]] = set()
        self.state = AnnotationState(
            state_path,
            input_path,
            spatial_prototypes=ROI_SPATIAL_PROTOTYPES if self.roi_mode else None,
            require_complete_roi=self.roi_mode,
        )
        if self.priority_queue is not None:
            self.priority_queue.add_annotations(list(self.state.annotations.values()))
        self._viewers: OrderedDict[int, AnnotationTilePackageReader] = OrderedDict()
        self._lock = threading.RLock()
        self._scan_done = False
        self._scan_error = ""
        self._scan_thread: threading.Thread | None = None
        self._thumbnail_token = ""
        self._review_cursor_by_package: dict[int, int] = {}
        if async_scan:
            self._scan_thread = threading.Thread(target=self._scan_packages, name="iac-scan", daemon=True)
            self._scan_thread.start()
        else:
            self.packages = discover_iac_packages(
                self.input_root,
                self.include_packages,
            )
            self._scan_done = True

    def _scan_packages(self) -> None:
        LOG.info("discover_iac_background_start input=%s", self.input_root)
        try:
            count = 0
            for path in _candidate_iac_paths(
                self.input_root,
                self.include_packages,
            ):
                package = _package_from_path(self.input_root, path)
                if package is None:
                    continue
                with self._lock:
                    self.packages.append(package)
                    count = len(self.packages)
                LOG.info("discover_iac_add rel_path=%s dataset=%s records=%d count=%d", package.rel_path, package.dataset or "-", package.total, count)
            if count == 0:
                self._scan_error = f"no image-tile .iac packages found under: {self.input_root}"
                LOG.error("discover_iac_background_empty input=%s", self.input_root)
        except Exception as exc:
            self._scan_error = str(exc)
            LOG.exception("discover_iac_background_failed input=%s", self.input_root)
        finally:
            self._scan_done = True
            LOG.info("discover_iac_background_done input=%s packages=%d error=%s", self.input_root, len(self.packages), self._scan_error or "-")

    def close(self) -> None:
        for viewer in self._viewers.values():
            viewer.close()
        self._viewers.clear()

    def scan_status(self) -> dict:
        with self._lock:
            return {"done": self._scan_done, "error": self._scan_error, "packages": len(self.packages), "last_iac": self.state.last_iac, "last_row": self.state.last_row}

    def package(self, index: int) -> AnnotationPackage:
        with self._lock:
            return self.packages[index]

    def activate_thumbnail_token(self, token: str) -> None:
        with self._lock:
            if token and token != self._thumbnail_token:
                LOG.info("thumbnail_token_activate")
                self._thumbnail_token = token

    def thumbnail_token_active(self, token: str | None) -> bool:
        if not token:
            return True
        with self._lock:
            return token == self._thumbnail_token

    def overview_cache_path(self, package: AnnotationPackage) -> Path:
        digest = hashlib.sha1(package.rel_path.encode("utf-8")).hexdigest()[:16]
        name = f"{Path(package.rel_path).stem}.{digest}.overview.jpg"
        return self.cache_root / name

    def viewer(self, index: int) -> AnnotationTilePackageReader:
        with self._lock:
            package = self.packages[index]
            cached = self._viewers.get(index)
            if cached is not None:
                self._viewers.move_to_end(index)
                return cached
        if cached is None:
            start = perf_counter()
            LOG.info("iac_open_start index=%d rel_path=%s path=%s", index, package.rel_path, package.path)
            viewer = AnnotationTilePackageReader(package.path)
            if viewer.payload_type != "image_tiles":
                viewer.close()
                raise ValueError(f"annotation requires image-tile IAC package: {package.path}")
            with self._lock:
                self._viewers[index] = viewer
                stale_viewers = []
                while len(self._viewers) > MAX_OPEN_IAC_VIEWERS:
                    stale_index, stale_viewer = self._viewers.popitem(last=False)
                    stale_viewers.append((stale_index, stale_viewer))
            for stale_index, stale_viewer in stale_viewers:
                LOG.info("iac_close_lru index=%d", stale_index)
                stale_viewer.close()
            LOG.info(
                "iac_open_done index=%d rel_path=%s records=%d tile=%dx%d stride=%dx%d elapsed=%.3fs",
                index,
                package.rel_path,
                len(viewer.records),
                int(viewer.header.get("tile_width", 0)),
                int(viewer.header.get("tile_height", 0)),
                viewer.stride_x,
                viewer.stride_y,
                perf_counter() - start,
            )
            return viewer
        raise RuntimeError("unreachable viewer cache state")

    def annotation_records(self, index: int) -> list[IacRecord]:
        return self.viewer(index).records

    def tissue_fraction(self, index: int, record: IacRecord) -> float:
        package = self.package(index)
        key = (package.rel_path, record.row)
        cached = self._tissue_fraction_cache.get(key)
        if cached is not None:
            return cached
        viewer = self.viewer(index)
        offsets = viewer.record_table.column("offset")
        lengths = viewer.record_table.column("length")
        payload = viewer.reader.read_data_span(
            offsets[record.row].as_py(),
            lengths[record.row].as_py(),
        )
        image = decode_jxl(payload).convert("RGB")
        image.thumbnail((64, 64), Image.Resampling.BILINEAR)
        pixels = list(image.get_flattened_data())
        tissue = sum(
            1
            for red, green, blue in pixels
            if min(red, green, blue) < 220 and max(red, green, blue) > 35
        )
        fraction = tissue / max(1, len(pixels))
        self._tissue_fraction_cache[key] = fraction
        LOG.info(
            "tile_tissue_fraction iac=%s row=%d tile_id=%s fraction=%.4f threshold=%.4f",
            package.rel_path,
            record.row,
            record.tile_id,
            fraction,
            self.min_tissue_fraction,
        )
        return fraction

    def _is_random_candidate(self, index: int, record: IacRecord) -> bool:
        package = self.package(index)
        key = (package.rel_path, record.row)
        if key in self._auto_filtered:
            return False
        if self.min_tissue_fraction <= 0 or self.tissue_fraction(index, record) >= self.min_tissue_fraction:
            return True
        self._auto_filtered.add(key)
        LOG.info("random_tile_auto_filter iac=%s row=%d tile_id=%s", package.rel_path, record.row, record.tile_id)
        return False

    def _first_random_candidate(self, index: int, records: list[IacRecord]) -> IacRecord | None:
        for record in records:
            if self._is_random_candidate(index, record):
                return record
        return None

    def _roi_navigation_candidates(
        self,
        processed_tile_ids: set[str],
        *,
        allowed_tile_ids: set[str] | None = None,
        exclude_tile_id: str | None = None,
    ) -> list[dict]:
        if not self.roi_mode or self.roi_queue is None:
            return []
        return self.roi_queue.weighted_remaining(
            processed_tile_ids=processed_tile_ids,
            roi_positive_counts=self.state.roi_positive_counts(),
            roi_positive_by_tile={
                str(item["tile_id"]): set(item.get("spatial", []))
                for item in self.state.annotations.values()
                if item.get("tile_id")
            },
            priority_attributes=self.roi_priority_attributes,
            allowed_tile_ids=allowed_tile_ids,
            exclude_tile_id=exclude_tile_id,
        )

    def annotation_json(self, index: int, row: int) -> dict:
        package = self.package(index)
        record = self.viewer(index)._by_row[row]
        item = self.state.annotation_for(package, record)
        return {
            "annotation": item,
            "candidate": None,
        }

    def package_json(self) -> list[dict]:
        items = []
        with self._lock:
            packages = list(self.packages)
        for idx, package in enumerate(packages):
            counts = self.state.lightweight_counts_for_package(package)
            auto_filtered = sum(1 for rel_path, _row in self._auto_filtered if rel_path == package.rel_path)
            counts["total"] = max(0, counts["total"] - auto_filtered)
            counts["remaining"] = max(0, counts["remaining"] - auto_filtered)
            items.append(
                {
                    "index": idx,
                    "name": Path(package.rel_path).name,
                    "rel_path": package.rel_path,
                    "dataset": package.dataset,
                    "total": counts["total"],
                    "annotated": counts["annotated"],
                    "remaining": counts["remaining"],
                    "auto_filtered": auto_filtered,
                }
            )
        LOG.info("api_packages count=%d", len(items))
        return items

    def progress(self, index: int) -> dict:
        with self._lock:
            packages = list(self.packages)
            package = self.packages[index]
        counts = self.state.lightweight_counts_for_package(package)
        auto_filtered_by_package = {
            item.rel_path: sum(1 for rel_path, _row in self._auto_filtered if rel_path == item.rel_path)
            for item in packages
        }
        counts["total"] = max(0, counts["total"] - auto_filtered_by_package[package.rel_path])
        counts["remaining"] = max(0, counts["remaining"] - auto_filtered_by_package[package.rel_path])
        lightweight_by_package = {
            item.rel_path: self.state.lightweight_counts_for_package(item)
            for item in packages
        }
        for item in packages:
            filtered = auto_filtered_by_package[item.rel_path]
            lightweight_by_package[item.rel_path]["total"] = max(0, lightweight_by_package[item.rel_path]["total"] - filtered)
            lightweight_by_package[item.rel_path]["remaining"] = max(0, lightweight_by_package[item.rel_path]["remaining"] - filtered)
        overall = {
            "annotated": sum(item["annotated"] for item in lightweight_by_package.values()),
            "total": sum(item["total"] for item in lightweight_by_package.values()),
            "skipped": sum(item["skipped"] for item in lightweight_by_package.values()),
        }
        overall["remaining"] = max(0, overall["total"] - overall["annotated"] - overall["skipped"])
        classification_counts = {name: 0 for name in self.state.classification_prototypes}
        spatial_counts = {name: 0 for name in self.state.spatial_prototypes}
        for key, item in self.state.annotations.items():
            if key in self.state.skipped or not _is_counted_annotation(item):
                continue
            classification_counts[item["classification"]] = classification_counts.get(item["classification"], 0) + 1
            for label in item["spatial"]:
                spatial_counts[label] = spatial_counts.get(label, 0) + 1
        package_classification_counts = {name: 0 for name in self.state.classification_prototypes}
        package_spatial_counts = {name: 0 for name in self.state.spatial_prototypes}
        for item in self.state.counted_annotations_for_package(package):
            package_classification_counts[item["classification"]] = package_classification_counts.get(item["classification"], 0) + 1
            for label in item["spatial"]:
                package_spatial_counts[label] = package_spatial_counts.get(label, 0) + 1
        LOG.info(
            "progress_read iac=%s annotated=%d total=%d remaining=%d overall_annotated=%d overall_total=%d",
            package.rel_path,
            counts["annotated"],
            counts["total"],
            counts["remaining"],
            overall["annotated"],
            overall["total"],
        )
        roi_counts = self.state.roi_positive_counts() if self.roi_mode else {}
        roi_information_policy = (
            self.roi_queue.information_policy()
            if self.roi_mode and self.roi_queue is not None
            else {}
        )
        if self.roi_priority_attributes:
            roi_priority_attributes = list(self.roi_priority_attributes)
        elif roi_information_policy:
            roi_priority_attributes = [
                name
                for name, value in roi_information_policy.items()
                if float(value.get("urgency", 0.0)) > 0.0
            ]
        else:
            roi_priority_attributes = []
        processed_tile_ids = {
            str(item["tile_id"])
            for item in self.state.annotations.values()
            if item.get("tile_id")
        }
        processed_tile_ids.update(
            parts[1]
            for key in self.state.skipped
            if len(parts := str(key).split("::")) >= 2
        )
        allowed_tile_ids = (
            {
                str(item["tile_id"])
                for item in (
                    self.review_queue.candidates
                    if self.review_queue is not None
                    else self.priority_queue.candidates
                )
            }
            if self.review_queue is not None or self.priority_queue is not None
            else None
        )
        return {
            "package": counts,
            "overall": overall,
            "classification": classification_counts,
            "spatial": spatial_counts,
            "package_classification": package_classification_counts,
            "package_spatial": package_spatial_counts,
            "roi_counts": roi_counts,
            "roi_priority_attributes": roi_priority_attributes,
            "roi_information_policy": roi_information_policy,
            "roi_sampling_policy": (
                self.roi_queue.sampling_policy()
                if self.roi_mode and self.roi_queue is not None
                else {}
            ),
            "roi_navigation_status": (
                self.roi_queue.direct_hint_status(
                    processed_tile_ids,
                    allowed_tile_ids=allowed_tile_ids,
                )
                if self.roi_mode and self.roi_queue is not None
                else {}
            ),
            "auto_filtered": sum(auto_filtered_by_package.values()),
            "priority": self.priority_queue.progress(self.state) if self.priority_queue else None,
            "review": self.review_queue.progress(self.state) if self.review_queue else None,
        }

    def _find_package_for_tile(self, tile_id: str, candidate_dict: dict, packages: list[AnnotationPackage]) -> tuple[int, AnnotationPackage] | None:
        pkg_map = {Path(p.rel_path).as_posix(): idx for idx, p in enumerate(packages)}
        target_iac = candidate_dict.get("iac")
        if target_iac:
            p_idx = pkg_map.get(Path(target_iac).as_posix())
            if p_idx is not None:
                return p_idx, packages[p_idx]
            base_name = Path(target_iac).name
            for idx, p in enumerate(packages):
                if Path(p.rel_path).name == base_name:
                    return idx, p

        slide_name = candidate_dict.get("slide")
        if slide_name:
            for idx, p in enumerate(packages):
                if slide_name in p.rel_path:
                    return idx, p

        if "_" in tile_id:
            prefix = tile_id.split("_")[0]
            for idx, p in enumerate(packages):
                if prefix in p.rel_path:
                    return idx, p

        # Check already loaded viewers to avoid re-opening
        for idx, p in enumerate(packages):
            if idx in self._viewers:
                viewer = self._viewers[idx]
                for r in viewer.records:
                    if r.tile_id == tile_id:
                        return idx, p

        # Scan other viewers if not found (typically only happens in test mocks with many packages and no metadata)
        for idx, p in enumerate(packages):
            if idx not in self._viewers:
                viewer = self.viewer(idx)
                for r in viewer.records:
                    if r.tile_id == tile_id:
                        return idx, p
        return None

    def _strict_review_record(
        self,
        packages: list[AnnotationPackage],
        *,
        package_index: int | None,
        exclude_tile_id: str | None,
    ) -> dict:
        if self.review_queue is None:
            raise RuntimeError("strict review queue is not configured")
        reviewed, skipped = self.state.review_completion(
            self.review_queue.review_id
        )
        processed = reviewed | skipped
        for candidate in self.review_queue.candidates:
            tile_id = candidate["tile_id"]
            if tile_id in processed or tile_id == exclude_tile_id:
                continue
            found = self._find_package_for_tile(tile_id, candidate, packages)
            if found is None:
                raise FileNotFoundError(
                    f"strict review tile is absent from the input packages: {tile_id}"
                )
            selected_index, package = found
            if package_index is not None and selected_index != package_index:
                continue
            viewer = self.viewer(selected_index)
            record = viewer._by_row.get(int(candidate["row"]))
            if record is None or record.tile_id != tile_id:
                record = next(
                    (item for item in viewer.records if item.tile_id == tile_id),
                    None,
                )
            if record is None:
                raise FileNotFoundError(
                    f"strict review tile row is absent from its package: {tile_id}"
                )
            return {
                "record": viewer._record_json(record),
                "package_index": selected_index,
                "selection": "strict_review_manifest",
            }
        return {
            "record": None,
            "done": (
                "review_complete"
                if package_index is None
                else "review_package_complete"
            ),
        }

    def random_record(
        self,
        index: int | str,
        exclude_row: int | None = None,
        exclude_tile_id: str | None = None,
        after_row: int | None = None,
    ) -> dict:
        with self._lock:
            packages = list(self.packages)

        global_mode = False
        if str(index).strip().lower() == "all":
            global_mode = True
        else:
            try:
                pkg_idx = int(index)
                target_package = packages[pkg_idx]
            except (ValueError, IndexError):
                global_mode = True

        if self.review_queue is not None:
            return self._strict_review_record(
                packages,
                package_index=None if global_mode else pkg_idx,
                exclude_tile_id=exclude_tile_id,
            )

        if not global_mode:
            viewer = self.viewer(pkg_idx)
            remaining = [record for record in viewer.records if not self.state.is_annotated(target_package, record)]
            if exclude_row is not None:
                remaining = [record for record in remaining if record.row != exclude_row]
            if exclude_tile_id is not None:
                remaining = [record for record in remaining if record.tile_id != exclude_tile_id]
            if not remaining:
                LOG.info("random_tile_empty iac=%s", target_package.rel_path)
                return {"record": None}
            if after_row is not None:
                ordered = sorted(remaining, key=lambda record: record.row)
                record = next((record for record in ordered if record.row > after_row and self._is_random_candidate(pkg_idx, record)), None)
                if record is None:
                    record = next((record for record in ordered if self._is_random_candidate(pkg_idx, record)), None)
                if record is None:
                    return {"record": None, "done": "no_tissue_candidates"}
                return {"record": viewer._record_json(record), "package_index": pkg_idx}
            remaining_by_tile = {record.tile_id: record for record in remaining}
            roi_priority = self._roi_navigation_candidates(
                set(),
                allowed_tile_ids=set(remaining_by_tile),
                exclude_tile_id=exclude_tile_id,
            )
            for candidate in roi_priority:
                record = remaining_by_tile[candidate["tile_id"]]
                if self._is_random_candidate(pkg_idx, record):
                    LOG.info(
                        "random_tile_roi_hint iac=%s row=%d tile_id=%s source_spatial=%s targets=%s",
                        target_package.rel_path,
                        record.row,
                        record.tile_id,
                        ",".join(candidate["source_spatial"]),
                        ",".join(
                            candidate.get(
                                "dynamic_priority_attributes",
                                [],
                            )[:3]
                        ),
                    )
                    return {
                        "record": viewer._record_json(record),
                        "package_index": pkg_idx,
                        "selection": "component_presence_navigation_hint",
                    }
            priority_remaining = (
                [record for record in remaining if self.priority_queue.contains(record.tile_id)]
                if self.priority_queue else []
            )
            random.shuffle(priority_remaining)
            record = self._first_random_candidate(pkg_idx, priority_remaining)
            if record is None:
                random.shuffle(remaining)
                record = self._first_random_candidate(pkg_idx, remaining)
                if record is not None and self.priority_queue is not None:
                    self.priority_queue.add(target_package, record)
            if record is None:
                LOG.info("random_tile_no_tissue iac=%s threshold=%.3f", target_package.rel_path, self.min_tissue_fraction)
                return {"record": None, "done": "no_tissue_candidates"}
            LOG.info("random_tile iac=%s row=%d tile_id=%s remaining=%d", target_package.rel_path, record.row, record.tile_id, len(remaining))
            return {"record": viewer._record_json(record), "package_index": pkg_idx}
        else:
            # 1. Gather all annotated or skipped tile_ids in memory
            annotated_tile_ids = {item["tile_id"] for item in self.state.annotations.values() if "tile_id" in item}
            skipped_tile_ids = set()
            for key in self.state.skipped:
                parts = key.split("::")
                if len(parts) >= 2:
                    skipped_tile_ids.add(parts[1])
            processed_tile_ids = annotated_tile_ids | skipped_tile_ids

            # 2. Within the shared 3000-tile boundary, component-presence
            # records make spatial random navigation purposeful.
            if self.priority_queue is not None:
                shared_remaining_ids = {
                    item["tile_id"] for item in self.priority_queue.candidates
                    if item["tile_id"] not in processed_tile_ids
                }
                roi_priority = self._roi_navigation_candidates(
                    processed_tile_ids,
                    allowed_tile_ids=shared_remaining_ids,
                    exclude_tile_id=exclude_tile_id,
                )
                for chosen_candidate in roi_priority:
                    found = self._find_package_for_tile(chosen_candidate["tile_id"], chosen_candidate, packages)
                    if found is None:
                        continue
                    p_idx, pkg = found
                    viewer = self.viewer(p_idx)
                    record = viewer._by_row.get(int(chosen_candidate.get("row", -1)))
                    if record is None or record.tile_id != chosen_candidate["tile_id"]:
                        record = next(
                            (item for item in viewer.records if item.tile_id == chosen_candidate["tile_id"]),
                            None,
                        )
                    if record is not None and self._is_random_candidate(p_idx, record):
                        LOG.info(
                            "random_tile_global_roi_hint iac=%s row=%d tile_id=%s source_spatial=%s targets=%s",
                            pkg.rel_path,
                            record.row,
                            record.tile_id,
                            ",".join(chosen_candidate["source_spatial"]),
                            ",".join(
                                chosen_candidate.get(
                                    "dynamic_priority_attributes",
                                    [],
                                )[:3]
                            ),
                        )
                        return {
                            "record": viewer._record_json(record),
                            "package_index": p_idx,
                            "selection": "component_presence_navigation_hint",
                        }

            # 3. Both expert tasks exhaust the same shared tile boundary first.
            if self.priority_queue is not None:
                priority_remaining = [
                    item for item in self.priority_queue.candidates
                    if item["tile_id"] not in processed_tile_ids
                    and (exclude_tile_id is None or item["tile_id"] != exclude_tile_id)
                ]
                random.shuffle(priority_remaining)
                for chosen_candidate in priority_remaining:
                    found = self._find_package_for_tile(chosen_candidate["tile_id"], chosen_candidate, packages)
                    if found is None:
                        continue
                    p_idx, pkg = found
                    viewer = self.viewer(p_idx)
                    if chosen_candidate["row"] in viewer._by_row:
                        record = viewer._by_row[chosen_candidate["row"]]
                        if record.tile_id == chosen_candidate["tile_id"] and self._is_random_candidate(p_idx, record):
                            LOG.info("random_tile_global_priority iac=%s row=%d tile_id=%s", pkg.rel_path, record.row, record.tile_id)
                            return {"record": viewer._record_json(record), "package_index": p_idx}
                    for record in viewer.records:
                        if record.tile_id == chosen_candidate["tile_id"] and self._is_random_candidate(p_idx, record):
                            LOG.info("random_tile_global_priority_search iac=%s row=%d tile_id=%s", pkg.rel_path, record.row, record.tile_id)
                            return {"record": viewer._record_json(record), "package_index": p_idx}

            # 4. Fallback: weighted-random choice based on remaining counts per package
            processed_counts = {}
            for key in list(self.state.annotations.keys()) + list(self.state.skipped):
                parts = key.split("::")
                if parts:
                    rel_path = parts[0]
                    processed_counts[rel_path] = processed_counts.get(rel_path, 0) + 1

            available_packages = []
            weights = []
            for p_idx, pkg in enumerate(packages):
                processed = processed_counts.get(pkg.rel_path, 0)
                remaining = pkg.total - processed
                if remaining > 0:
                    available_packages.append((p_idx, pkg))
                    weights.append(remaining)

            if not available_packages:
                return {"record": None}

            package_order = random.choices(range(len(available_packages)), weights=weights, k=len(available_packages) * 2)
            package_order.extend(index for index in range(len(available_packages)) if index not in package_order)
            for chosen_idx in package_order:
                p_idx, pkg = available_packages[chosen_idx]
                viewer = self.viewer(p_idx)
                remaining_records = [
                    record for record in viewer.records
                    if not self.state.is_annotated(pkg, record)
                    and (exclude_tile_id is None or record.tile_id != exclude_tile_id)
                ]
                random.shuffle(remaining_records)
                record = self._first_random_candidate(p_idx, remaining_records)
                if record is not None:
                    if self.priority_queue is not None:
                        self.priority_queue.add(pkg, record)
                    LOG.info("random_tile_global_fallback iac=%s row=%d tile_id=%s", pkg.rel_path, record.row, record.tile_id)
                    return {"record": viewer._record_json(record), "package_index": p_idx}
            return {"record": None, "done": "no_tissue_candidates"}

    def reviewed_record(self, index: int) -> dict:
        package = self.package(index)
        viewer = self.viewer(index)
        reviewed = [
            record for record in self.annotation_records(index)
            if self.state.is_counted_record(package, record)
        ]
        if not reviewed:
            return {"record": None}
        reviewed.sort(key=lambda item: item.row)
        cursor = self._review_cursor_by_package.get(index, 0) % len(reviewed)
        record = reviewed[cursor]
        self._review_cursor_by_package[index] = cursor + 1
        return {"record": viewer._record_json(record)}

    def reviewed_records(self, package_index: int | str = "all") -> dict:
        with self._lock:
            packages = list(self.packages)
        index_by_rel_path = {package.rel_path: index for index, package in enumerate(packages)}
        selected_index = None if package_index == "all" else int(package_index)
        items = []
        for annotation in list(self.state.annotations.values()):
            if not _is_counted_annotation(annotation):
                continue
            index = index_by_rel_path.get(str(annotation.get("iac") or ""))
            if index is None or (selected_index is not None and index != selected_index):
                continue
            roi = list(annotation.get("roi") or [])
            items.append(
                {
                    "package_index": index,
                    "record": {
                        "row": int(annotation["row"]),
                        "tile_id": str(annotation.get("tile_id") or ""),
                        "x": int(annotation.get("x") or 0),
                        "y": int(annotation.get("y") or 0),
                    },
                    "classification": str(annotation.get("classification") or ""),
                    "spatial": list(annotation.get("spatial") or []),
                    "roi_count": sum(1 for entry in roi if entry.get("geometry") is not None),
                }
            )
        items.sort(key=lambda item: (item["package_index"], item["record"]["row"]))
        return {"items": items, "total": len(items)}

    def select_nearest(self, index: int, x: float, y: float) -> dict:
        viewer = self.viewer(index)
        if not self.roi_mode:
            result = viewer.nearest("__all__", x, y)
        else:
            records = self.annotation_records(index)
            if not records:
                return {"record": None}
            nearest = min(
                records,
                key=lambda record: (record.display_x - x) ** 2 + (record.display_y - y) ** 2,
            )
            result = {"record": viewer._record_json(nearest)}
        record = result.get("record")
        if record:
            with self._lock:
                package = self.packages[index]
            LOG.info("nearest_tile iac=%s x=%.1f y=%.1f row=%s tile_id=%s", package.rel_path, x, y, record.get("row"), record.get("tile_id"))
        return result

    def annotated_rows(self, index: int) -> set[int]:
        with self._lock:
            package = self.packages[index]
        return {record.row for record in self.annotation_records(index) if self.state.is_annotated(package, record)}

    def thumbnail_jpg(self, index: int, token: str | None = None, selected_row: int | None = None) -> bytes:
        if token:
            self.activate_thumbnail_token(token)
        with self._lock:
            package = self.packages[index]
        cache_path = self.overview_cache_path(package)
        if cache_path.exists():
            LOG.info("overview_cache_hit iac=%s path=%s", package.rel_path, cache_path)
            return self._overview_with_annotations(index, cache_path, selected_row=selected_row)

        viewer = self.viewer(index)
        start = perf_counter()
        records = viewer.records
        min_grid_x = min(record.grid_x for record in records)
        max_grid_x = max(record.grid_x for record in records)
        min_grid_y = min(record.grid_y for record in records)
        max_grid_y = max(record.grid_y for record in records)
        grid_w = max_grid_x - min_grid_x + 1
        grid_h = max_grid_y - min_grid_y + 1
        canvas_w = max(1, grid_w * OVERVIEW_CELL_PX)
        canvas_h = max(1, grid_h * OVERVIEW_CELL_PX)
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        offsets = viewer.record_table.column("offset")
        lengths = viewer.record_table.column("length")

        def decode_overview_tile(record: IacRecord) -> tuple[int, int, Image.Image] | None:
            if not self.thumbnail_token_active(token):
                return None
            try:
                payload = viewer.reader.read_data_span(
                    offsets[record.row].as_py(),
                    lengths[record.row].as_py(),
                )
                if not self.thumbnail_token_active(token):
                    return None
                tile = decode_jxl(payload).convert("RGB")
                tile = tile.resize((OVERVIEW_CELL_PX, OVERVIEW_CELL_PX), Image.Resampling.BILINEAR)
                x = (record.grid_x - min_grid_x) * OVERVIEW_CELL_PX
                y = (record.grid_y - min_grid_y) * OVERVIEW_CELL_PX
                return x, y, tile
            except Exception:
                LOG.exception("thumbnail_tile_decode_failed iac=%s row=%d", package.rel_path, record.row)
                return None

        workers = min(OVERVIEW_WORKERS, max(1, len(records)))
        executor = ThreadPoolExecutor(max_workers=workers)
        cancelled = False
        try:
            futures = [executor.submit(decode_overview_tile, record) for record in records]
            for future in as_completed(futures):
                if not self.thumbnail_token_active(token):
                    executor.shutdown(wait=False, cancel_futures=True)
                    cancelled = True
                    break
                item = future.result()
                if item is None:
                    continue
                x, y, tile = item
                canvas.paste(tile, (x, y))
        finally:
            if not cancelled:
                executor.shutdown(wait=True, cancel_futures=True)
        if not self.thumbnail_token_active(token):
            LOG.info("overview_build_cancelled iac=%s", package.rel_path)
            buffer = io.BytesIO()
            Image.new("RGB", (1, 1), "white").save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        canvas.save(tmp_path, format="JPEG", quality=85, optimize=True)
        tmp_path.replace(cache_path)
        LOG.info(
            "overview_cache_build iac=%s records=%d workers=%d grid=%dx%d cell=%d canvas=%dx%d path=%s elapsed=%.3fs",
            package.rel_path,
            len(records),
            workers,
            grid_w,
            grid_h,
            OVERVIEW_CELL_PX,
            canvas_w,
            canvas_h,
            cache_path,
            perf_counter() - start,
        )
        return self._overview_with_annotations(index, cache_path, selected_row=selected_row)

    def _overview_with_annotations(self, index: int, cache_path: Path, selected_row: int | None = None) -> bytes:
        package = self.package(index)
        viewer = self.viewer(index)
        records = viewer.records
        min_grid_x = min(record.grid_x for record in records)
        min_grid_y = min(record.grid_y for record in records)
        image = Image.open(cache_path).convert("RGB")
        annotated = self.state.annotations_for_package(package)
        draw = ImageDraw.Draw(image, "RGBA")
        for item in annotated:
            row = viewer._by_row.get(int(item["row"]))
            if row is None:
                continue
            x = (row.grid_x - min_grid_x) * OVERVIEW_CELL_PX
            y = (row.grid_y - min_grid_y) * OVERVIEW_CELL_PX
            draw.rectangle([x, y, x + OVERVIEW_CELL_PX, y + OVERVIEW_CELL_PX], outline=(0, 140, 80, 220), width=1)
        if selected_row is not None and selected_row in viewer._by_row:
            row = viewer._by_row[selected_row]
            x = (row.grid_x - min_grid_x) * OVERVIEW_CELL_PX
            y = (row.grid_y - min_grid_y) * OVERVIEW_CELL_PX
            cx = x + OVERVIEW_CELL_PX // 2
            cy = y + OVERVIEW_CELL_PX // 2
            span = max(12, OVERVIEW_CELL_PX * 3)
            draw.line([cx - span, cy, cx + span, cy], fill=(220, 20, 20, 255), width=2)
            draw.line([cx, cy - span, cx, cy + span], fill=(220, 20, 20, 255), width=2)
            draw.rectangle([x, y, x + OVERVIEW_CELL_PX, y + OVERVIEW_CELL_PX], outline=(220, 20, 20, 255), width=2)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        LOG.info("overview_return iac=%s path=%s annotated=%d bytes=%d", package.rel_path, cache_path, len(annotated), buffer.tell())
        return buffer.getvalue()

    def context_window(
        self,
        index: int,
        row: int,
        *,
        radius: int = CONTEXT_PREFETCH_RADIUS,
    ) -> dict:
        viewer = self.viewer(index)
        center = viewer._by_row[row]
        radius = max(
            CONTEXT_VISIBLE_RADIUS,
            min(CONTEXT_MAX_RADIUS, int(radius)),
        )
        tile_w = max(1, int(viewer.header.get("tile_width", viewer.stride_x))) // 2
        tile_h = max(1, int(viewer.header.get("tile_height", viewer.stride_y))) // 2
        tile_w = max(1, tile_w)
        tile_h = max(1, tile_h)
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                record = viewer._image_lookup.get(
                    (
                        center.slide_key,
                        center.grid_x + dx,
                        center.grid_y + dy,
                    )
                )
                cells.append(
                    {
                        "dx": dx,
                        "dy": dy,
                        "record": (
                            None
                            if record is None
                            else viewer._record_json(record)
                        ),
                    }
                )
        return {
            "center": viewer._record_json(center),
            "radius": radius,
            "visible_radius": CONTEXT_VISIBLE_RADIUS,
            "grid_size": radius * 2 + 1,
            "visible_grid_size": CONTEXT_VISIBLE_RADIUS * 2 + 1,
            "cell_width": tile_w,
            "cell_height": tile_h,
            "cells": cells,
        }

    def context_center(
        self,
        index: int,
        row: int,
        *,
        grid_x: float,
        grid_y: float,
    ) -> dict:
        viewer = self.viewer(index)
        source = viewer._by_row[row]
        exact = viewer._image_lookup.get(
            (
                source.slide_key,
                int(round(grid_x)),
                int(round(grid_y)),
            )
        )
        if exact is not None:
            return {"record": viewer._record_json(exact)}
        records = viewer._by_slide.get(source.slide_key, [])
        if not records:
            raise FileNotFoundError(
                f"slide has no context records: {source.slide_key}"
            )
        nearest = min(
            records,
            key=lambda record: (
                (record.grid_x - grid_x) ** 2
                + (record.grid_y - grid_y) ** 2
            ),
        )
        return {"record": viewer._record_json(nearest)}

    def context_jpg(
        self,
        index: int,
        row: int,
        *,
        radius: int = CONTEXT_VISIBLE_RADIUS,
        selected_row: int | None = None,
    ) -> bytes:
        start = perf_counter()
        viewer = self.viewer(index)
        window = self.context_window(index, row, radius=radius)
        center = viewer._by_row[row]
        radius = int(window["radius"])
        side = int(window["grid_size"])
        tile_w = int(window["cell_width"])
        tile_h = int(window["cell_height"])
        canvas = Image.new(
            "RGB",
            (tile_w * side, tile_h * side),
            (245, 247, 249),
        )
        draw = ImageDraw.Draw(canvas)
        placements = []
        for cell in window["cells"]:
            x = (int(cell["dx"]) + radius) * tile_w
            y = (int(cell["dy"]) + radius) * tile_h
            record_json = cell["record"]
            if record_json is None:
                draw.rectangle(
                    [x, y, x + tile_w - 1, y + tile_h - 1],
                    outline=(210, 215, 220),
                )
                continue
            placements.append(
                (
                    x,
                    y,
                    viewer._by_row[int(record_json["row"])],
                )
            )

        payloads = viewer.reader.read_payloads(
            record.row for _x, _y, record in placements
        )

        def decode_context_tile(
            item: tuple[
                tuple[int, int, IacRecord],
                bytes,
            ],
        ) -> tuple[int, int, IacRecord, Image.Image | None]:
            (x, y, record), payload = item
            try:
                image = decode_jxl(payload).convert("RGB")
                image = image.resize(
                    (tile_w, tile_h),
                    Image.Resampling.BILINEAR,
                )
                return x, y, record, image
            except Exception:
                LOG.exception("context_tile_decode_failed row=%d", record.row)
                return x, y, record, None

        workers = min(CONTEXT_WORKERS, max(1, len(placements)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            decoded = executor.map(
                decode_context_tile,
                zip(placements, payloads),
            )
            for x, y, _record, image in decoded:
                if image is None:
                    draw.rectangle(
                        [x, y, x + tile_w - 1, y + tile_h - 1],
                        outline=(210, 80, 80),
                    )
                else:
                    canvas.paste(image, (x, y))

        draw = ImageDraw.Draw(canvas)
        for offset in range(side + 1):
            x = min(canvas.width - 1, offset * tile_w)
            y = min(canvas.height - 1, offset * tile_h)
            draw.line(
                [x, 0, x, canvas.height - 1],
                fill=(205, 210, 216),
                width=1,
            )
            draw.line(
                [0, y, canvas.width - 1, y],
                fill=(205, 210, 216),
                width=1,
            )

        selected = viewer._by_row.get(
            row if selected_row is None else int(selected_row)
        )
        if selected is not None and selected.slide_key == center.slide_key:
            dx = selected.grid_x - center.grid_x
            dy = selected.grid_y - center.grid_y
            if -radius <= dx <= radius and -radius <= dy <= radius:
                selected_x = (dx + radius) * tile_w
                selected_y = (dy + radius) * tile_h
                draw.rectangle(
                    [
                        selected_x,
                        selected_y,
                        selected_x + tile_w - 1,
                        selected_y + tile_h - 1,
                    ],
                    outline=(220, 20, 20),
                    width=4,
                )
                draw.line(
                    [
                        selected_x,
                        selected_y + tile_h // 2,
                        selected_x + tile_w,
                        selected_y + tile_h // 2,
                    ],
                    fill=(220, 20, 20),
                    width=2,
                )
                draw.line(
                    [
                        selected_x + tile_w // 2,
                        selected_y,
                        selected_x + tile_w // 2,
                        selected_y + tile_h,
                    ],
                    fill=(220, 20, 20),
                    width=2,
                )
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=90)
        LOG.info(
            "context_render index=%d row=%d selected_row=%s radius=%d "
            "records=%d bytes=%d elapsed=%.3fs",
            index,
            row,
            selected_row,
            radius,
            len(placements),
            buffer.tell(),
            perf_counter() - start,
        )
        return buffer.getvalue()


class AnnotationArchive:
    def __init__(
        self,
        initial: AnnotationData,
        *,
        input_path: str | Path,
        roi_candidate_manifest: str | Path | None = None,
        roi_information_report: str | Path | None = None,
        roi_priority_attributes: list[str] | tuple[str, ...] | None = None,
        roi_mode: bool | None = None,
        priority_queue: SharedPriorityQueue | None = None,
        review_queue: StrictReviewQueue | None = None,
        include_packages: list[str] | tuple[str, ...] | None = None,
        min_tissue_fraction: float = 0.30,
    ) -> None:
        self.input_path = str(Path(input_path).resolve())
        self.roi_candidate_manifest = str(roi_candidate_manifest) if roi_candidate_manifest else None
        self.roi_information_report = (
            str(roi_information_report)
            if roi_information_report
            else initial.roi_information_report
        )
        self.roi_priority_attributes = tuple(roi_priority_attributes or initial.roi_priority_attributes)
        self.roi_mode = initial.roi_mode if roi_mode is None else bool(roi_mode)
        self.priority_queue = priority_queue if priority_queue is not None else initial.priority_queue
        self.review_queue = review_queue if review_queue is not None else initial.review_queue
        self.include_packages = tuple(include_packages or initial.include_packages)
        self.min_tissue_fraction = float(min_tissue_fraction)
        self.base_state_path = initial.state.state_path
        self.manifest_path = self.base_state_path.with_name(f"{self.base_state_path.stem}.versions.json")
        self.version_dir = self.base_state_path.with_name(f"{self.base_state_path.stem}.versions")
        self._data: dict[str, AnnotationData] = {"main": initial}
        self._names: dict[str, str] = {"main": "Main"}
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for item in payload.get("versions", []):
                version_id = str(item.get("id") or "").strip()
                if not version_id or version_id == "main" or version_id in self._data:
                    continue
                state_path = Path(str(item.get("state_path") or ""))
                if not state_path.is_absolute():
                    state_path = self.manifest_path.parent / state_path
                self._data[version_id] = AnnotationData(
                    self.input_path,
                    state_path,
                    roi_candidate_manifest=self.roi_candidate_manifest,
                    roi_information_report=self.roi_information_report,
                    roi_priority_attributes=self.roi_priority_attributes,
                    roi_mode=self.roi_mode,
                    priority_queue=self.priority_queue,
                    review_queue=self.review_queue,
                    include_packages=self.include_packages,
                    min_tissue_fraction=self.min_tissue_fraction,
                )
                self._names[version_id] = str(item.get("name") or version_id)

    def data(self, version_id: str | None = None) -> AnnotationData:
        selected = str(version_id or self.default_version)
        if selected not in self._data:
            raise ValueError(f"annotation version is not configured: {selected}")
        return self._data[selected]

    @property
    def default_version(self) -> str:
        return "marking" if "marking" in self._data else "main"

    def versions_json(self) -> dict:
        return {
            "versions": [
                {
                    "id": version_id,
                    "name": self._names[version_id],
                    "state_path": str(item.state.state_path),
                    "annotations": len(item.state.annotations),
                }
                for version_id, item in self._data.items()
            ]
        }

    def labels_json(self, version_id: str | None = None) -> dict:
        return self.data(version_id).state.labels_json()

    def change_label(
        self,
        task: str,
        operation: str,
        *,
        version_id: str | None = None,
        label_id: str = "",
        name: str = "",
    ) -> dict:
        return self.data(version_id).state.change_label(task, operation, label_id=label_id, name=name)

    def create_version(self, name: str, source_version: str = "main") -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("version name is required")
        slug = re.sub(r"[^a-z0-9]+", "-", clean_name.lower()).strip("-") or "version"
        version_id = slug
        suffix = 2
        while version_id in self._data:
            version_id = f"{slug}-{suffix}"
            suffix += 1
        source = self.data(source_version)
        state_path = self.version_dir / f"{version_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "input_path": self.input_path,
            "classification_prototypes": list(source.state.classification_prototypes),
            "spatial_prototypes": list(source.state.spatial_prototypes),
            "label_definitions": source.state.label_definitions,
            "annotations": {},
            "skipped": [],
            "last_iac": "",
            "roi_mode": source.state.require_complete_roi,
            "annotation_version": {"id": version_id, "name": clean_name, "source": source_version},
        }
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=state_path.parent) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(state_path)
        self._data[version_id] = AnnotationData(
            self.input_path,
            state_path,
            roi_candidate_manifest=self.roi_candidate_manifest,
            roi_information_report=self.roi_information_report,
            roi_priority_attributes=self.roi_priority_attributes,
            roi_mode=self.roi_mode,
            priority_queue=self.priority_queue,
            review_queue=self.review_queue,
            include_packages=self.include_packages,
            min_tissue_fraction=self.min_tissue_fraction,
        )
        self._names[version_id] = clean_name
        self._flush_manifest()
        LOG.info("annotation_version_create id=%s name=%s state=%s", version_id, clean_name, state_path)
        result = self.versions_json()
        result["created"] = version_id
        return result

    def _flush_manifest(self) -> None:
        payload = {
            "version": 1,
            "base_state": str(self.base_state_path),
            "versions": [
                {
                    "id": version_id,
                    "name": self._names[version_id],
                    "state_path": os.path.relpath(item.state.state_path, self.manifest_path.parent),
                }
                for version_id, item in self._data.items()
                if version_id != "main"
            ],
        }
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.manifest_path.parent) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.manifest_path)

    def close(self) -> None:
        for item in self._data.values():
            item.close()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HCC-SemPath Prototype Annotation</title>
<style>
:root{color-scheme:light;--bg:#f4f6f8;--panel:#fff;--line:#d7dce2;--text:#202124;--muted:#687381;--blue:#1a73e8;--blue-soft:#e8f0fe;--green-soft:#dff3eb;--danger:#b42318}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text);background:var(--bg);height:100svh;overflow:hidden}
button,input{font:inherit}button{min-height:40px;border:1px solid var(--line);background:var(--panel);color:var(--text);cursor:pointer}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button.ghost{background:transparent}button:disabled{opacity:.55;cursor:default}
.layout{display:grid;grid-template-columns:300px minmax(0,1fr);height:100svh}.layout.queue-collapsed{grid-template-columns:0 minmax(0,1fr)}
.review-mode aside{position:relative;height:100svh;min-height:0}.review-mode #packageHeader,.review-mode #packages{display:none}.review-mode .reviewedPanel{display:block!important;position:absolute;inset:0;width:auto;height:auto;min-height:0;overflow:hidden;border-top:0}.review-mode .reviewedPanel>summary{height:44px;pointer-events:none}.review-mode .reviewedList{display:block;position:absolute;inset:44px 0 0;width:auto;height:auto;max-height:none;overflow-y:auto}
.modeNav{display:flex;align-items:center;gap:2px;padding:2px;border:1px solid var(--line);background:#f5f7f9}.modeNav button{min-height:32px;padding:4px 12px;border:0;background:transparent}.modeNav button.active{background:#202124;color:#fff}.versionSelect{min-height:32px;border:1px solid var(--line);background:#fff;padding:0 8px}.compactButton{min-height:32px;padding:4px 9px}.labelManage{min-height:32px;padding:4px 10px}
aside{overflow:hidden;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column}.layout.queue-collapsed aside{border:0}.layout.queue-collapsed aside>*{display:none}
.panelHead{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px;border-bottom:1px solid var(--line)}.panelTitle{font-weight:650}.panelBody{overflow:auto;padding:12px}.packageBody{min-height:0;flex:1}.reviewedPanel{flex:0 0 auto;border-top:1px solid var(--line);background:#fafbfc}.reviewedPanel[open]{display:block}.reviewedPanel summary{cursor:pointer;padding:11px 12px;font-weight:650}.reviewedList{box-sizing:border-box;width:100%;height:min(36svh,360px);max-height:min(36svh,360px);overflow-x:hidden;overflow-y:scroll;overscroll-behavior-y:contain;scrollbar-gutter:stable;touch-action:pan-y;-webkit-overflow-scrolling:touch;padding:0 10px 10px}.reviewedItem{display:block;width:100%;min-height:0;text-align:left;padding:7px 8px;margin-bottom:5px;background:#fff}.reviewedItem.active{border-color:var(--blue);background:var(--blue-soft)}.reviewedItemMeta{display:block;color:var(--muted);font-size:11px;margin-top:2px}.labelBody{min-height:0;flex:1}.labelActions{border-top:1px solid var(--line);padding:10px 12px;background:var(--panel)}
main{min-width:0;display:grid;grid-template-rows:auto auto minmax(0,1fr);height:100svh}.topbar{display:grid;grid-template-columns:auto auto minmax(160px,1fr) auto;align-items:center;gap:10px;padding:8px 12px;border-bottom:1px solid var(--line);background:var(--panel)}.topbarTitle{min-width:0;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.topTools{display:flex;align-items:center;gap:6px}.topTools button{min-height:32px;padding:4px 10px}.subbar{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:42px;padding:5px 12px;border-bottom:1px solid var(--line);background:#fafbfc}.versionGroup{display:flex;align-items:center;gap:6px;min-width:0}.versionLabel{color:var(--muted);font-size:12px}.topProgress{min-width:300px;text-align:right;font-size:12px;color:var(--muted)}.topProgress b{color:var(--text)}
.workspace{min-height:0;overflow:auto;padding:12px}.primaryWorkspace{display:grid;grid-template-columns:minmax(420px,680px) minmax(300px,1fr);grid-template-rows:auto auto;column-gap:20px;align-items:start}.pkg{position:relative;padding:8px;border:1px solid var(--line);margin-bottom:6px;cursor:pointer;background:#fff;overflow:hidden}.pkg.active{border-color:var(--blue);background:var(--blue-soft)}
.pkg>*{position:relative;z-index:1}.pkg::before{content:"";position:absolute;inset:0 auto 0 0;width:var(--pct,0%);background:#dff3eb;z-index:0}
.muted{color:var(--muted);font-size:12px}.thumbWrap{position:relative;width:100%;height:100%;min-height:0;border:1px solid #c7cbd1;background:white;overflow:hidden;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}.imageControlRow{display:flex;align-items:center;gap:8px;margin:0 0 6px}.tileControlRow{grid-column:1;grid-row:1;width:100%;justify-content:space-between}
.thumb{display:block;width:auto;height:auto;max-width:none;background:white;cursor:crosshair;-webkit-user-drag:none;user-drag:none;touch-action:none;-webkit-user-select:none;user-select:none}.loading{padding:18px;color:#6b7280;font-size:12px}
.tileViewport{grid-column:1;grid-row:2;width:100%;height:520px;overflow:auto;border:1px solid #c7cbd1;background:#f8fafc}.tileViewport,.tileStage,.tile,.roiCanvas{-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}.tileStage{position:relative;width:224px;height:224px;background:#fff;transform-origin:0 0}.tile{position:absolute;inset:0;width:224px;height:224px;object-fit:contain;background:#fff;-webkit-user-drag:none}.roiCanvas{position:absolute;inset:0;width:224px;height:224px;touch-action:none;cursor:crosshair}.roiClassBar{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;max-width:none;overflow:visible;padding-bottom:0}.roiClassButton{min-height:32px;padding:4px 8px;border-left-width:8px;white-space:nowrap;flex:0 0 auto}.roiClassButton.selected{background:#111827;color:#fff}.roiTools{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:8px 0}.roiTools button{min-height:32px;padding:4px 8px}.roiTools button.selected{background:var(--blue);color:#fff}.roiPlan{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0;padding:8px;border:1px solid var(--line);background:#f8fafc}.roiPlanStatus{flex:1 1 260px}.roiPlanDecision{display:flex;flex-wrap:wrap;gap:6px}.roiPlanDecision button{min-height:32px;padding:4px 9px}.rangeControl{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:#fff;padding:4px 8px;min-height:32px}.rangeControl input{width:130px}.rangeValue{min-width:38px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px}.chips{display:grid;grid-template-columns:1fr;gap:6px;margin:8px 0 16px}
.panzoom{touch-action:none;transform-origin:0 0;will-change:transform;cursor:grab}.panzoom.dragging{cursor:grabbing}
button.chip{position:relative;text-align:left;border:1px solid #c7cbd1;background:#fff;padding:8px;cursor:pointer;overflow:hidden}button.chip::before{content:"";position:absolute;inset:0 auto 0 0;width:var(--pct,0%);background:#eef4ff;z-index:0}button.chip.selected{background:#1a73e8;color:white;border-color:#1a73e8}button.chip.selected::before{background:rgba(255,255,255,.18)}button.chip span{position:relative;z-index:1}.chipRow{display:flex;justify-content:space-between;gap:8px}.chipShortcut{display:inline-grid;place-items:center;min-width:20px;height:20px;margin-right:6px;border:1px solid #aab2bd;border-radius:3px;background:#f7f8fa;color:#374151;font:600 12px/1 system-ui}.chipCount{font-variant-numeric:tabular-nums;color:#374151}button.chip.selected .chipShortcut{border-color:rgba(255,255,255,.65);background:rgba(255,255,255,.16);color:#fff}button.chip.selected .chipCount{color:white}
.annotationControls{grid-column:2;grid-row:1/span 2;min-width:0;width:100%;border-left:1px solid var(--line);padding-left:16px}.annotationControls .chips{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.actions button{padding:8px 12px}.bar{height:6px;background:#e5e7eb;margin:4px 0}.bar>div{height:6px;background:#18865b}.statusLine{min-height:20px;margin-top:6px;color:var(--muted);font-size:12px;white-space:pre-wrap}.roiNavigationNotice{margin:7px 0;padding:7px 9px;border:1px solid #f59e0b;background:#fffbeb;color:#92400e;font-size:12px;white-space:pre-wrap}
pre{white-space:pre-wrap;font-size:12px;background:#f1f3f4;padding:8px}
.authGate{position:fixed;inset:0;background:rgba(244,246,248,.96);z-index:10;display:none;align-items:center;justify-content:center;padding:18px}.authBox{width:min(420px,100%);background:#fff;border:1px solid var(--line);padding:16px;box-shadow:0 14px 40px rgba(15,23,42,.18)}.authBox h3{margin:0 0 10px}.authBox input{width:100%;min-height:42px;border:1px solid var(--line);padding:0 10px;margin-bottom:10px}.authBox .status{min-height:18px;color:var(--danger);font-size:12px}
.contextOverlay{position:fixed;inset:0;background:rgba(15,23,42,.86);z-index:9;display:flex;flex-direction:column}.contextBar{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;background:#fff}.contextHint{margin-top:2px;color:var(--muted);font-size:12px}.contextStage{min-height:0;flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center;padding:12px}.contextViewport{--context-cell-w:112px;--context-cell-h:112px;--context-grid-x:0px;--context-grid-y:0px;position:relative;overflow:hidden;background-color:#eef1f4;background-image:linear-gradient(to right,#cfd5dc 1px,transparent 1px),linear-gradient(to bottom,#cfd5dc 1px,transparent 1px),repeating-linear-gradient(135deg,#f2f4f6 0 9px,#e8ecf0 9px 18px);background-size:var(--context-cell-w) var(--context-cell-h),var(--context-cell-w) var(--context-cell-h),auto;background-position:var(--context-grid-x) var(--context-grid-y),var(--context-grid-x) var(--context-grid-y),0 0;border:1px solid rgba(255,255,255,.65);box-shadow:0 12px 38px rgba(0,0,0,.34);touch-action:none;cursor:grab;-webkit-user-select:none;user-select:none}.contextViewport.dragging{cursor:grabbing}.contextImg{position:absolute;left:0;top:0;max-width:none;max-height:none;background:#fff;pointer-events:none;-webkit-user-drag:none;will-change:transform;transform:translate3d(0,0,0)}.contextLoading{position:absolute;right:8px;bottom:8px;padding:5px 8px;background:rgba(15,23,42,.78);color:#fff;font-size:12px;pointer-events:none}.contextLoading.hidden{display:none}
.overviewBarTools{display:flex;align-items:center;gap:8px}.overviewStage{min-height:0;flex:1;overflow:auto;padding:12px}
.labelDialog{width:min(700px,calc(100% - 24px));max-height:80svh;border:1px solid var(--line);padding:0}.dialogHead{display:flex;align-items:center;justify-content:space-between;padding:12px;border-bottom:1px solid var(--line)}.dialogBody{padding:12px;overflow:auto;max-height:60svh}.labelEditorRow{display:grid;grid-template-columns:70px minmax(0,1fr) auto auto auto;gap:6px;align-items:center;margin-bottom:6px}.labelEditorRow input{min-height:36px;border:1px solid var(--line);padding:0 8px}.labelEditorRow button{min-height:36px;padding:4px 8px}.addLabelRow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;margin-top:12px}.addLabelRow input{min-height:38px;border:1px solid var(--line);padding:0 8px}
.hidden{display:none!important}
@media(max-width:1200px){
  .primaryWorkspace{display:block}.tileControlRow{width:min(100%,680px)}.tileViewport{width:min(100%,680px)}.annotationControls{width:min(100%,760px);border-left:0;border-top:1px solid var(--line);margin-top:10px;padding-left:0;padding-top:10px}.roiClassBar{flex-wrap:nowrap;max-width:min(100%,760px);overflow-x:auto;padding-bottom:14px}
}
@media(max-width:760px){
  body{height:100svh;overflow:hidden}.layout,.layout.queue-collapsed{display:block;height:100svh}
  aside{position:fixed;inset:0 0 auto 0;z-index:5;max-height:45svh;border-right:0;border-bottom:1px solid var(--line);box-shadow:0 8px 24px rgba(15,23,42,.16)}.layout.queue-collapsed aside{display:none}
  main{height:100svh;grid-template-rows:auto auto minmax(0,1fr)}.topbar{position:sticky;top:0;z-index:3;grid-template-columns:auto minmax(0,1fr) auto;gap:6px;padding:7px 8px}.topbarTitle{grid-column:1/-1;grid-row:1}.topbar>#toggleQueue{grid-column:1;grid-row:2}.topbar>.modeNav{grid-column:2;grid-row:2;justify-self:start}.topbar>.topTools{grid-column:3;grid-row:2}.subbar{align-items:flex-start;flex-direction:column;gap:5px;padding:6px 8px}.versionGroup{width:100%}.versionSelect{min-width:0;flex:1}.topProgress{width:100%;min-width:0;text-align:left}.workspace{padding:8px}.tileViewport{height:auto;aspect-ratio:1 / 1}.panelBody{padding:10px}.chips{gap:6px;margin-bottom:10px}button.chip{min-height:38px;padding:7px}
  .reviewedList{height:26svh;max-height:26svh}
  .review-mode aside{position:fixed;height:45svh}.review-mode .reviewedList{height:auto;max-height:none}
}
</style>
</head>
<body><div id="authGate" class="authGate"><div class="authBox"><h3>Annotation token</h3><input id="authInput" autocomplete="off"><button id="authSubmit" class="primary" type="button">Open</button><div id="authStatus" class="status"></div></div></div>
<div id="layout" class="layout"><aside><div id="packageHeader" class="panelHead"><div><div class="panelTitle">IAC packages</div><div id="packageSummary" class="muted"></div></div><button id="hideQueue" class="ghost" type="button">Hide</button></div><div id="packages" class="panelBody packageBody"></div><details id="reviewedDetails" class="reviewedPanel"><summary>Marked tiles (<span id="reviewedCount">0</span>)</summary><div id="reviewedList" class="reviewedList"><div class="muted">Expand to load saved annotations.</div></div></details></aside>
<main><div class="topbar"><button id="toggleQueue" type="button">Tiles</button><div id="modeNav" class="modeNav"></div><div id="title" class="topbarTitle">Prototype annotation</div><div class="topTools"><button id="reviewModeBtn" type="button">Review</button><button id="overviewBtn" type="button">Overview</button><button id="contextBtn" type="button">Context</button><button id="manageLabels" class="labelManage" type="button">Labels</button></div></div><div class="subbar"><div class="versionGroup"><span class="versionLabel">Version</span><select id="versionSelect" class="versionSelect" title="Annotation version"></select><button id="newVersion" class="compactButton" type="button">New</button></div><div id="progress" class="topProgress"></div></div><section class="workspace"><div class="primaryWorkspace"><div class="imageControlRow tileControlRow"><div class="muted" id="recordMeta"></div><label class="rangeControl">Tile zoom <input id="tileZoom" type="range" min="1" max="6" step="0.1" value="2"><span id="tileZoomValue" class="rangeValue">2.0×</span></label></div>
<div id="tileViewport" class="tileViewport"><div id="tileStage" class="tileStage"><img id="tile" class="tile" draggable="false"><canvas id="roiCanvas" class="roiCanvas" width="224" height="224"></canvas></div></div><div class="annotationControls"><div id="prototypeLabels"><div id="classificationSection"><div id="classification" class="chips"></div></div><div id="spatialSection"><div id="spatial" class="chips"></div></div></div><div id="roiClassBar" class="roiClassBar"></div><div id="roiStatus" class="muted">Select an ROI class, then draw on the tile.</div><div id="roiPlan" class="roiPlan"><button type="button" id="roiPlanGenerate">Find similar marks</button><label id="roiSimilarityControl" class="rangeControl hidden">Similarity <input id="roiSimilarity" type="range" min="0.20" max="0.99" step="0.01" value="0.70"><span id="roiSimilarityValue" class="rangeValue">70%</span></label><label id="roiExclusionControl" class="rangeControl hidden">Exclusion <input id="roiExclusion" type="range" min="3" max="40" step="1" value="7"><span id="roiExclusionValue" class="rangeValue">7 px</span></label><div id="roiPlanStatus" class="roiPlanStatus muted">Select one class and add 1–3 reliable point or circle seeds.</div><div id="roiPlanDecision" class="roiPlanDecision hidden"><button type="button" id="roiPlanAccept" class="primary">Accept visible matches</button><button type="button" id="roiPlanRestart">Start from scratch</button></div></div><div id="roiNavigationNotice" class="roiNavigationNotice hidden"></div><details id="roiCountDetails" class="muted"><summary>ROI positive tile counts</summary><div id="roiCountProgress"></div></details><div id="roiTools" class="roiTools"><button type="button" data-roi-tool="point">Point</button><button type="button" data-roi-tool="brush">Brush</button><button type="button" data-roi-tool="eraser">Eraser</button><button type="button" data-roi-tool="circle">Circle</button><label class="rangeControl">Brush / eraser width <input id="brushWidth" type="range" min="0.012" max="0.500" step="0.002" value="0.035"><span id="brushWidthValue" class="rangeValue">3.5%</span></label><button type="button" id="roiUndo">Undo</button><button type="button" id="roiRedo">Redo</button><button type="button" id="roiClear">Clear selected class</button><button type="button" id="tileGridToggle">Grid</button></div><div class="actions"><button id="saveNext" onclick="save()" class="primary">Save + next</button><button id="skipNext" onclick="nextRandom(true)">Skip + next</button><button id="previousTile" onclick="reviewed()">Browse saved tiles</button></div><div id="status" class="statusLine"></div></div></div></section></main>
</div>
<div id="overviewOverlay" class="contextOverlay hidden"><div class="contextBar"><div><b>Location overview</b><div class="contextHint">Drag or horizontal-scroll to explore · click a location to open its tile</div></div><div class="overviewBarTools"><label class="rangeControl">Zoom <input id="overviewZoom" type="range" min="0.25" max="12" step="0.25" value="1"><span id="overviewZoomValue" class="rangeValue">1.0×</span></label><button id="overviewClose" type="button">Close</button></div></div><div class="overviewStage"><div id="thumbWrap" class="thumbWrap"><div id="thumbLoading" class="loading">Loading overview...</div><img id="thumb" class="thumb panzoom" draggable="false"></div></div></div>
<div id="contextOverlay" class="contextOverlay hidden"><div class="contextBar"><div><b>5×5 context</b><div id="contextMeta" class="muted"></div><div class="contextHint">Drag or horizontal-scroll to explore · click a tile to annotate it · C to close</div></div><button id="contextClose" type="button">Close</button></div><div id="contextStage" class="contextStage"><div id="contextViewport" class="contextViewport"><img id="contextImg" class="contextImg" draggable="false"><div id="contextLoading" class="contextLoading hidden">Loading nearby tiles…</div></div></div></div>
<dialog id="labelDialog" class="labelDialog"><div class="dialogHead"><b id="labelDialogTitle">Label management</b><button id="closeLabels" type="button">Close</button></div><div class="dialogBody"><div id="labelEditor"></div><div class="addLabelRow"><input id="newLabelName" placeholder="New label name"><button id="addLabel" class="primary" type="button">Add</button><button id="saveLabels" type="button">Save label configuration</button></div><div id="labelStatus" class="muted"></div></div></dialog>
<dialog id="versionDialog" class="labelDialog"><div class="dialogHead"><b>Create annotation version</b><button id="closeVersion" type="button">Close</button></div><div class="dialogBody"><div class="addLabelRow"><input id="newVersionName" placeholder="Version name"><button id="createVersion" class="primary" type="button">Create</button></div><div id="versionStatus" class="muted">The new version starts with no annotations and keeps the current label configuration.</div></div></dialog>
<script>
let packages=[], pkg=0, current=null, classification="", spatial=new Set(), classificationCounts={}, spatialCounts={}, restoredLastIac=false, showGrid=false, navigationMode='global', tileHistory=[], tileForward=[], resumeAfterRow=null;
let roi=[],roiTool='point',roiToolByClass={},roiAttribute='__all__',roiDrawing=null,roiPreview=null,roiCursor=null,roiAllComplete=true,roiPlan=null,roiPlanLoading=false;
let roiClassState={},roiExclusionOverrides={};
let reviewedReturn=null,reviewedItems=[],reviewedIndex=-1;
let pendingLabelAdds=[];
const CONTEXT_RADIUS=5,CONTEXT_RECENTER_DELTA=2;
let contextData=null,contextCellMap=new Map(),contextSelected=null,contextCenterRow=null;
let contextTx=0,contextTy=0,contextCellW=1,contextCellH=1,contextImageW=1,contextImageH=1;
let contextViewportW=1,contextViewportH=1;
let contextDrag=null,contextFrame=0,contextRequest=0,contextPendingWindow=null,contextTrackpadWheel=false,contextWheelTimer=0;
let undoStack=[],redoStack=[];
let tileScale=2,tileZoomUserSet=false;
const ROI_ALL='__all__';
const NUCLEUS_MATCH_CLASSES=new Set(['hepatocellular-parenchyma','inflammatory-cell']);
const AREA_ONLY_CLASSES=new Set(['necrosis','fibrous-stroma']);
const ROI_COLORS=['#e11d48','#f97316','#eab308','#22c55e','#14b8a6','#06b6d4','#3b82f6','#8b5cf6','#d946ef','#84cc16','#64748b'];
const TOKEN_KEYS=['token','auth_token','access_token'];
let AUTH_TOKEN='';
function tokenFromUrl(){const params=new URLSearchParams(location.search); for(const key of TOKEN_KEYS){const value=params.get(key); if(value)return value;} return '';}
function setAuthToken(token){AUTH_TOKEN=token||''; if(AUTH_TOKEN)localStorage.setItem('hcc_sempath_annotation_token',AUTH_TOKEN);}
function authed(path){return path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(AUTH_TOKEN);}
function scoped(path){return path+(path.includes('?')?'&':'?')+'mode='+encodeURIComponent(MODE)+'&version='+encodeURIComponent(VERSION)}
async function api(path, opts){const r=await fetch(authed(scoped(path)), opts); if(!r.ok) throw new Error(await r.text()); return r.headers.get('content-type')?.includes('json')?r.json():r.blob();}
function setQueueOpen(open){document.getElementById('layout').classList.toggle('queue-collapsed',!open);}
function isMobile(){return window.matchMedia('(max-width:760px)').matches;}
function applyModeLayout(){const classificationMode=MODE==='classification';document.getElementById('roiClassBar').style.display=classificationMode?'none':'flex';document.getElementById('roiStatus').style.display=classificationMode?'none':'block';document.getElementById('roiPlan').style.display=classificationMode?'none':'flex';document.getElementById('roiCountDetails').style.display=classificationMode?'none':'block';document.getElementById('roiTools').style.display=classificationMode?'none':'flex';document.getElementById('roiCanvas').style.display=classificationMode?'none':'block';document.getElementById('prototypeLabels').style.display=classificationMode?'block':'none';document.getElementById('classificationSection').style.display=classificationMode?'block':'none';document.getElementById('spatialSection').style.display='none';document.getElementById('tileViewport').style.cursor=classificationMode?'default':'crosshair';}
const AUTH_REQUIRED=%AUTH_REQUIRED_JSON%;
async function ensureAuth(){if(!AUTH_REQUIRED){document.getElementById('authGate').style.display='none';return}setAuthToken(tokenFromUrl()||localStorage.getItem('hcc_sempath_annotation_token')||''); if(!AUTH_TOKEN){document.getElementById('authGate').style.display='flex'; throw new Error('');}}
async function submitAuth(){setAuthToken(document.getElementById('authInput').value.trim()); try{await api('/api/scan-status'); document.getElementById('authGate').style.display='none'; refreshPackage().catch(e=>document.getElementById('status').textContent=e.message||String(e));}catch(e){document.getElementById('authStatus').textContent='Invalid token.';}}
function setupPanZoom(el,onClick,options={}){
    let scale=1,tx=0,ty=0,drag=false,sx=0,sy=0,stx=0,sty=0,moved=0,minTx=0,minTy=0,frame=0,trackpadWheel=false;
    const wheelZoom=options.wheelZoom!==false,horizontalWheelPan=options.horizontalWheelPan===true,doubleClickReset=options.doubleClickReset!==false,onScale=options.onScale||null;
    const clamp=()=>{tx=Math.min(0,Math.max(minTx,tx));ty=Math.min(0,Math.max(minTy,ty))};
    const measure=()=>{const viewport=el.parentElement,baseW=el.naturalWidth||el.offsetWidth||0,baseH=el.naturalHeight||el.offsetHeight||0;minTx=Math.min(0,viewport.clientWidth-baseW*scale);minTy=Math.min(0,viewport.clientHeight-baseH*scale);clamp()};
    const paint=()=>{frame=0;el.style.transform=`translate3d(${tx}px,${ty}px,0) scale(${scale})`;if(onScale)onScale(scale)};
    const apply=()=>{clamp();if(!frame)frame=requestAnimationFrame(paint)};
    const setScale=next=>{scale=Math.min(12,Math.max(.25,next));measure();apply()};
    el.draggable=false;
    el.addEventListener('dragstart',ev=>ev.preventDefault());
    el.addEventListener('load',()=>{measure();apply()});
    el.addEventListener('wheel',ev=>{if(horizontalWheelPan&&Math.abs(ev.deltaX)>=2)trackpadWheel=true;if(horizontalWheelPan&&trackpadWheel){ev.preventDefault();measure();tx-=ev.deltaX;ty-=ev.deltaY;apply();return}if(!wheelZoom)return;ev.preventDefault();setScale(scale*(ev.deltaY<0?1.15:.87))},{passive:false});
    el.addEventListener('pointerdown',ev=>{if(ev.button!==0)return;ev.preventDefault();measure();drag=true;moved=0;sx=ev.clientX;sy=ev.clientY;stx=tx;sty=ty;el.classList.add('dragging');el.setPointerCapture(ev.pointerId)});
    el.addEventListener('pointermove',ev=>{if(!drag)return;ev.preventDefault();const dx=ev.clientX-sx,dy=ev.clientY-sy;moved=Math.max(moved,Math.abs(dx)+Math.abs(dy));tx=stx+dx;ty=sty+dy;apply()});
    el.addEventListener('pointerup',ev=>{if(!drag)return;ev.preventDefault();drag=false;el.classList.remove('dragging');if(el.hasPointerCapture(ev.pointerId))el.releasePointerCapture(ev.pointerId);if(moved<6&&onClick)onClick(ev)});
    el.addEventListener('pointercancel',()=>{drag=false;el.classList.remove('dragging')});
    if(doubleClickReset)el.addEventListener('dblclick',ev=>{ev.preventDefault();tx=0;ty=0;setScale(1)});
    el.setPanZoomScale=setScale;el.resetPanZoom=()=>{tx=0;ty=0;setScale(1)};
    window.addEventListener('resize',()=>{measure();apply()});
}
function labelName(task,id){const item=(LABELS.label_sets[task]||[]).find(x=>x.id===id);return item?item.name:id}
function prototypeButton(task,id,selected,count,maxCount,onClick,shortcut=''){const b=document.createElement('button'); const pct=maxCount?Math.round(count/maxCount*100):0; b.className='chip'+(selected?' selected':''); b.style.setProperty('--pct',pct+'%');const key=shortcut?`<kbd class=chipShortcut>${shortcut}</kbd>`:'';b.innerHTML=`<span class=chipRow><span>${key}${labelName(task,id)}</span><span class=chipCount>${count}</span></span>`; b.onclick=onClick; return b;}
function roiColor(attribute){const i=SPATIAL.indexOf(attribute);return ROI_COLORS[(i<0?0:i)%ROI_COLORS.length]}
function snapshotRoi(){return JSON.stringify({roi,roiClassState})}
function restoreRoi(snapshot){const state=JSON.parse(snapshot);roi=state.roi||[];roiClassState=state.roiClassState||{};syncPositiveLabels();renderRoi()}
function pushUndo(){undoStack.push(snapshotRoi()); if(undoStack.length>100)undoStack.shift(); redoStack=[]}
function undoRoi(){if(!undoStack.length)return;redoStack.push(snapshotRoi());restoreRoi(undoStack.pop());document.getElementById('status').textContent='Undone.'}
function redoRoi(){if(!redoStack.length)return;undoStack.push(snapshotRoi());restoreRoi(redoStack.pop());document.getElementById('status').textContent='Redone.'}
function roiToolAllowed(tool,attribute=roiAttribute){return !(tool==='point'&&AREA_ONLY_CLASSES.has(attribute))}
function selectRoiTool(tool,remember=true){let next=['point','brush','eraser','circle'].includes(tool)?tool:'point';if(!roiToolAllowed(next))next='brush';roiTool=next;if(remember&&roiAttribute!==ROI_ALL)roiToolByClass[roiAttribute]=roiTool;document.querySelectorAll('[data-roi-tool]').forEach(button=>{button.disabled=!roiToolAllowed(button.dataset.roiTool);button.classList.toggle('selected',button.dataset.roiTool===roiTool)})}
function toggleRoiClass(name){pushUndo();roiPlan=null;if(roiAttribute===name){roiClassState[name]=roiClassState[name]==='negative'?'open':'negative'}else{roiAttribute=name;if(!roiClassState[name])roiClassState[name]='open';selectRoiTool(roiToolByClass[name]||(AREA_ONLY_CLASSES.has(name)?'brush':'point'),false)}renderLabels();renderRoi()}
function hasPositiveRoi(attribute){return roi.some(item=>item.attribute===attribute&&item.geometry&&item.state!=='negative')}
function roiClassIcon(attribute,state){return state==='negative'?'−':hasPositiveRoi(attribute)?'●':'○'}
function renderRoiClassBar(){const bar=document.getElementById('roiClassBar'); if(!bar)return; bar.style.display=ROI_MODE?'flex':'none'; bar.innerHTML=''; if(!ROI_MODE)return; const all=document.createElement('button'); all.type='button'; all.className='roiClassButton'+(roiAttribute===ROI_ALL?' selected':''); all.style.borderLeftColor='#111827'; all.textContent='All'; all.onclick=()=>{roiPlan=null;roiAttribute=ROI_ALL;renderLabels();renderRoi()}; bar.appendChild(all); SPATIAL.forEach(x=>{const state=roiClassState[x]||'open';const positive=hasPositiveRoi(x);const b=document.createElement('button'); b.type='button'; b.className='roiClassButton'+(roiAttribute===x?' selected':''); b.style.borderLeftColor=state==='negative'?'#111827':roiColor(x); b.textContent=`${roiClassIcon(x,state)} ${labelName('spatial',x)}`; b.title=state==='negative'?'Explicit negative; click again to return to positive/uncertain.':positive?'Positive ROI present; clear its ROI before marking this class negative.':'Positive or uncertain by default; no positive ROI yet. Click selected class again to mark explicit negative.'; b.onclick=()=>toggleRoiClass(x); bar.appendChild(b)})}
function renderLabels(){renderRoiClassBar();const a=document.getElementById('classification');a.innerHTML='';document.getElementById('spatial').innerHTML='';if(MODE==='classification'){const classificationMax=Math.max(1,...Object.values(classificationCounts));CLASSIFICATION.forEach((x,index)=>a.appendChild(prototypeButton('classification',x,classification===x,classificationCounts[x]||0,classificationMax,()=>{classification=x;renderLabels()},index<9?String(index+1):'')))}document.getElementById('roiStatus').textContent=ROI_MODE?(roiAttribute===ROI_ALL?'All spatial classes visible. Select one class before drawing. Unmarked classes stay unknown/ignored unless explicitly toggled negative.':`Drawing ROI for: ${labelName('spatial',roiAttribute)}. Unmarked classes stay unknown/ignored unless explicitly toggled negative.`):'';applyModeLayout();}
function roiPoint(ev,clamp=false){const r=document.getElementById('roiCanvas').getBoundingClientRect();let x=(ev.clientX-r.left)/r.width,y=(ev.clientY-r.top)/r.height;if(clamp){x=Math.max(0,Math.min(1,x));y=Math.max(0,Math.min(1,y))}return{x,y}}
function applyTileZoom(){const stage=document.getElementById('tileStage'); if(!stage)return; stage.style.transform=`scale(${tileScale})`; stage.style.marginRight=(224*(tileScale-1))+'px'; stage.style.marginBottom=(224*(tileScale-1))+'px'}
function setTileZoom(scale){tileScale=Math.min(6,Math.max(1,scale));applyTileZoom();const input=document.getElementById('tileZoom'),value=document.getElementById('tileZoomValue');if(input)input.value=tileScale;if(value)value.textContent=`${tileScale.toFixed(1)}×`}
function applyDefaultTileZoom(){if(tileZoomUserSet)return;if(!isMobile()){setTileZoom(2);return}const viewport=document.getElementById('tileViewport');if(!viewport)return;const available=viewport.clientWidth;if(available>0)setTileZoom(available/224)}
function resetTileViewportForRecord(){const workspace=document.querySelector('.workspace'),viewport=document.getElementById('tileViewport');if(workspace){workspace.scrollTop=0;workspace.scrollLeft=0}if(viewport){viewport.scrollTop=0;viewport.scrollLeft=0}applyDefaultTileZoom()}
function brushWidth(){return parseFloat(document.getElementById('brushWidth')?.value||'0.035')}
function setBrushWidth(width){const input=document.getElementById('brushWidth');const next=Math.min(parseFloat(input.max),Math.max(parseFloat(input.min),width));input.value=next.toFixed(3);updateBrushWidth()}
function updateBrushWidth(){const width=brushWidth();document.getElementById('brushWidthValue').textContent=`${(width*100).toFixed(1)}%`;renderRoi()}
function seedCentersForSelectedClass(){if(roiAttribute===ROI_ALL)return[];return roi.filter(item=>item.attribute===roiAttribute&&item.geometry&&(item.geometry.type==='point'||item.geometry.type==='circle')).map(item=>item.geometry.type==='point'?item.geometry.point:item.geometry.center)}
function occupiedRoiCenters(){return roi.filter(item=>item.geometry&&(item.geometry.type==='point'||item.geometry.type==='circle')).map(item=>item.geometry.type==='point'?item.geometry.point:item.geometry.center)}
function similarityThreshold(){return parseFloat(document.getElementById('roiSimilarity')?.value||'0.70')}
function setRoiSimilarityThreshold(value){const input=document.getElementById('roiSimilarity');const next=Math.min(parseFloat(input.max),Math.max(parseFloat(input.min),value));input.value=String(next.toFixed(2));document.getElementById('roiSimilarityValue').textContent=`${Math.round(next*100)}%`}
function roiExclusionDistance(){return parseFloat(document.getElementById('roiExclusion')?.value||'7')}
function setRoiExclusionDistance(distance){const input=document.getElementById('roiExclusion');const next=Math.min(parseFloat(input.max),Math.max(parseFloat(input.min),distance));input.value=String(Math.round(next));document.getElementById('roiExclusionValue').textContent=`${Math.round(next)} px`}
function plannedCenter(item){const g=item.geometry;if(g?.type==='point')return g.point;if(g?.type==='circle')return g.center;return null}
function visiblePlannedSuggestions(){const suggestions=roiPlan?.suggestions||[],threshold=similarityThreshold(),minimumSq=(roiExclusionDistance()/224)**2,occupied=occupiedRoiCenters(),target=roiPlan?.summary?.preview_target??suggestions.length,selected=[],selectedCenters=[];for(const item of suggestions){if((item.confidence??0)<threshold)continue;const center=plannedCenter(item);if(!center)continue;if(occupied.some(point=>(center[0]-point[0])**2+(center[1]-point[1])**2<minimumSq))continue;if(selectedCenters.some(point=>(center[0]-point[0])**2+(center[1]-point[1])**2<minimumSq))continue;selected.push(item);selectedCenters.push(center);if(selected.length>=target)break}return selected}
function renderRoiPlan(){const status=document.getElementById('roiPlanStatus'),decision=document.getElementById('roiPlanDecision'),button=document.getElementById('roiPlanGenerate'),controls=[document.getElementById('roiSimilarityControl'),document.getElementById('roiExclusionControl')],supported=NUCLEUS_MATCH_CLASSES.has(roiAttribute);if(!status)return;button.disabled=roiPlanLoading||!current||(!supported&&roiAttribute!==ROI_ALL);if(!supported&&roiAttribute!==ROI_ALL){button.textContent='Find similar marks';status.textContent='Image-only nucleus matching is currently validated for hepatocellular parenchyma and inflammatory cells only.';decision.classList.add('hidden');controls.forEach(control=>control.classList.add('hidden'));return}if(roiPlanLoading){button.textContent='Finding matches…';status.textContent='Matching each selected seed as an independent local image template.';decision.classList.add('hidden');controls.forEach(control=>control.classList.add('hidden'));return}button.textContent=roiPlan?'Rebuild similar marks':'Find similar marks';if(!roiPlan){const seeds=seedCentersForSelectedClass();status.textContent=roiAttribute===ROI_ALL?'Select one ROI class, then add 1–3 reliable point or circle seeds.':`${labelName('spatial',roiAttribute)} · ${seeds.length} seed${seeds.length===1?'':'s'}. Add reliable marks, then find similar centers.`;decision.classList.add('hidden');controls.forEach(control=>control.classList.add('hidden'));return}const visible=visiblePlannedSuggestions(),automatic=roiPlan.summary?.minimum_spacing_pixels,exclusion=roiExclusionDistance(),detected=roiPlan.summary?.candidate_count??roiPlan.suggestions.length,maxScore=roiPlan.summary?.maximum_similarity,target=roiPlan.summary?.preview_target;document.getElementById('roiSimilarityValue').textContent=`${Math.round(similarityThreshold()*100)}%`;document.getElementById('roiExclusionValue').textContent=`${Math.round(exclusion)} px`;status.textContent=`Nucleus-image preview · ${roiPlan.summary?.seed_count||0} seeds${target?` → target ${target}`:''} · detected ${detected} centers${maxScore!=null?` · best ${Math.round(maxScore*100)}%`:''} · ${Math.round(exclusion)} px exclusion${automatic?` (auto ${automatic.toFixed(1)})`:''} · showing ${visible.length}/${roiPlan.suggestions.length} matches. Preview marks are not saved yet.`;decision.classList.remove('hidden');controls.forEach(control=>control.classList.remove('hidden'))}
function drawPointMark(x,point,color,alpha=1){const px=point[0]*224,py=point[1]*224;x.save();x.globalAlpha=alpha;for(const [radius,fill] of [[3.4,'#000'],[2.3,'#fff'],[1.2,color]]){x.fillStyle=fill;x.beginPath();x.arc(px,py,radius,0,Math.PI*2);x.fill()}x.restore()}
function drawPlannedPoint(x,g,color){const px=g.point[0]*224,py=g.point[1]*224;x.save();x.globalAlpha=.5;x.lineCap='round';for(const [width,stroke] of [[2.5,'rgba(15,23,42,.9)'],[1.5,'#fff'],[.7,color]]){x.lineWidth=width;x.strokeStyle=stroke;x.beginPath();x.moveTo(px-3.5,py);x.lineTo(px+3.5,py);x.moveTo(px,py-3.5);x.lineTo(px,py+3.5);x.stroke()}x.restore()}
function drawPlannedCircle(x,g,color){const px=g.center[0]*224,py=g.center[1]*224,radius=g.radius*224;x.save();x.globalAlpha=1;x.setLineDash([]);x.fillStyle=color+'55';x.beginPath();x.arc(px,py,radius,0,Math.PI*2);x.fill();for(const [width,stroke] of [[7,'rgba(15,23,42,.9)'],[5,'#fff'],[2.5,color]]){x.lineWidth=width;x.strokeStyle=stroke;x.beginPath();x.arc(px,py,radius,0,Math.PI*2);x.stroke()}x.restore()}
function renderRoi(){const c=document.getElementById('roiCanvas'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);const tmpCanvas=document.createElement('canvas');tmpCanvas.width=c.width;tmpCanvas.height=c.height;const tx=tmpCanvas.getContext('2d');let hasBrushes=false;const planned=visiblePlannedSuggestions();for(const item of [...roi,...planned,...(roiPreview?[roiPreview]:[])]){if(roiAttribute!==ROI_ALL&&item.attribute!==roiAttribute)continue;const g=item.geometry;if(!g)continue;const color=roiColor(item.attribute),active=roiAttribute===ROI_ALL||item.attribute===roiAttribute,isPlan=planned.includes(item),isDrawing=item===roiPreview;if(isPlan){if(g.type==='point')drawPlannedPoint(x,g,color);else if(g.type==='circle')drawPlannedCircle(x,g,color);continue}if(g.type==='brush'){hasBrushes=true;tx.strokeStyle=color;tx.lineWidth=Math.max(2,(g.width||.035)*224);tx.lineCap='round';tx.lineJoin='round';tx.beginPath();g.points.forEach((p,i)=>i?tx.lineTo(p[0]*224,p[1]*224):tx.moveTo(p[0]*224,p[1]*224));tx.stroke();}else{x.globalAlpha=isDrawing ? .65 : (active?1:.28);x.strokeStyle=color;x.fillStyle=color+'44';x.lineWidth=item.attribute===roiAttribute?4:2;if(g.type==='point'){drawPointMark(x,g.point,color,active?1:.42)}else if(g.type==='circle'){x.beginPath();x.arc(g.center[0]*224,g.center[1]*224,g.radius*224,0,Math.PI*2);x.fill();x.stroke()}}}if(hasBrushes){x.globalAlpha=0.4;x.drawImage(tmpCanvas,0,0);}if(showGrid){x.save();x.strokeStyle='rgba(0,0,0,0.4)';x.lineWidth=1;const cellSize=14;for(let i=1;i<16;i++){const pos=i*cellSize;x.beginPath();x.moveTo(pos,0);x.lineTo(pos,224);x.stroke();x.beginPath();x.moveTo(0,pos);x.lineTo(224,pos);x.stroke();}x.restore();}if(roiCursor){x.globalAlpha=.85;x.strokeStyle='#111827';x.fillStyle='rgba(255,255,255,.18)';x.lineWidth=1.5;x.beginPath();x.arc(roiCursor.x*224,roiCursor.y*224,Math.max(2,brushWidth()*224/2),0,Math.PI*2);x.fill();x.stroke()}x.globalAlpha=1;renderRoiPlan()}
function plannedGeometryDuplicate(item){const g=item.geometry;return roi.some(old=>{if(old.attribute!==item.attribute||old.geometry?.type!==g.type)return false;const og=old.geometry;if(g.type==='point')return Math.hypot(og.point[0]-g.point[0],og.point[1]-g.point[1])<.018;if(g.type==='circle')return Math.hypot(og.center[0]-g.center[0],og.center[1]-g.center[1])<.04;return JSON.stringify(og)===JSON.stringify(g)})}
async function generateRoiPlan(){if(!ROI_MODE||!current||roiPlanLoading)return;if(roiAttribute===ROI_ALL){document.getElementById('status').textContent='Select one ROI class before finding similar marks.';return}if(!NUCLEUS_MATCH_CLASSES.has(roiAttribute)){document.getElementById('status').textContent='Image-only nucleus matching is not validated for this ROI class.';return}const seeds=seedCentersForSelectedClass();if(!seeds.length){document.getElementById('status').textContent='Add at least one reliable point or circle seed for the selected class first.';return}const tileKey=`${pkg}:${current.row}:${current.tile_id}`;roiPlan=null;roiPlanLoading=true;renderRoi();try{const result=await api('/api/roi-similar',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({package:pkg,row:current.row,attribute:roiAttribute,seeds,occupied:occupiedRoiCenters()})});if(!current||tileKey!==`${pkg}:${current.row}:${current.tile_id}`)return;setRoiSimilarityThreshold(result.summary?.recommended_similarity??0.70);setRoiExclusionDistance(roiExclusionOverrides[roiAttribute]??result.summary?.minimum_spacing_pixels??7);roiPlan=result;document.getElementById('status').textContent='Similar local image centers are ready. Adjust similarity and exclusion distance, then accept visible matches or start from scratch.';}catch(e){document.getElementById('status').textContent='Similarity search failed: '+(e.message||String(e))}finally{roiPlanLoading=false;renderRoi()}}
function acceptRoiPlan(){if(!roiPlan)return;pushUndo();let added=0;for(const proposed of visiblePlannedSuggestions()){const item={attribute:proposed.attribute,state:'positive',review_complete:false,geometry:proposed.geometry};if(plannedGeometryDuplicate(item))continue;roi.push(item);delete roiClassState[item.attribute];added++}roiPlan=null;syncPositiveLabels();renderRoi();document.getElementById('status').textContent=`Accepted ${added} similar mark${added===1?'':'s'}. All marks remain editable.`}
function restartRoiFromScratch(){pushUndo();roi=[];roiClassState={};roiExclusionOverrides={};roiDrawing=null;roiPreview=null;roiCursor=null;roiPlan=null;syncPositiveLabels();renderRoi();document.getElementById('status').textContent='Started this tile from scratch.'}
function syncPositiveLabels(){spatial=new Set(roi.filter(item=>item.geometry&&item.state!=='negative').map(item=>item.attribute));renderLabels()}
function pointDistanceSq(a,b){return(a[0]-b[0])**2+(a[1]-b[1])**2}
function densifyPath(points,maxStep){if(!points.length)return[];const dense=[points[0]];for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],distance=Math.sqrt(pointDistanceSq(a,b)),steps=Math.max(1,Math.ceil(distance/maxStep));for(let step=1;step<=steps;step++){const t=step/steps;dense.push([a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t])}}return dense}
function eraseBrushGeometry(item,center,eraserRadius){const geometry=item.geometry,threshold=eraserRadius+(geometry.width||.035)/2,points=densifyPath(geometry.points||[],Math.max(.002,Math.min(.008,eraserRadius/2))),runs=[];let run=[];for(const point of points){if(pointDistanceSq(point,center)>threshold**2)run.push(point);else{if(run.length>1)runs.push(run);run=[]}}if(run.length>1)runs.push(run);return runs.map(points=>({...item,geometry:{...geometry,points}}))}
function eraseCurrentClassAt(center,eraserRadius){const next=[];for(const item of roi){const geometry=item.geometry;if(item.attribute!==roiAttribute||!geometry){next.push(item);continue}if(geometry.type==='point'){if(pointDistanceSq(geometry.point,center)>eraserRadius**2)next.push(item)}else if(geometry.type==='circle'){if(pointDistanceSq(geometry.center,center)>(geometry.radius+eraserRadius)**2)next.push(item)}else if(geometry.type==='brush')next.push(...eraseBrushGeometry(item,center,eraserRadius));else next.push(item)}roi=next}
function eraseCurrentClassAlong(start,end,eraserRadius){for(const point of densifyPath([start,end],Math.max(.002,eraserRadius/2)))eraseCurrentClassAt(point,eraserRadius)}
function finishEraserStroke(ev=null){if(roiDrawing?.tool!=='eraser')return false;if(ev&&roiDrawing.pointerId!==ev.pointerId)return false;const c=document.getElementById('roiCanvas'),pointerId=roiDrawing.pointerId;roiDrawing=null;roiPreview=null;if(c.hasPointerCapture(pointerId))c.releasePointerCapture(pointerId);syncPositiveLabels();renderRoi();document.getElementById('status').textContent=`Erased ${labelName('spatial',roiAttribute)} within the selected range.`;return true}
function setupEraser(){const c=document.getElementById('roiCanvas');c.addEventListener('pointerdown',ev=>{if(ev.button!==0||roiTool!=='eraser')return;ev.preventDefault();ev.stopImmediatePropagation();if(roiAttribute===ROI_ALL){document.getElementById('status').textContent='Select one spatial class before erasing.';return}if(roiClassState[roiAttribute]==='negative'){document.getElementById('status').textContent='This ROI class is marked negative. Toggle it back before erasing.';return}const p=roiPoint(ev);pushUndo();roiPlan=null;roiDrawing={tool:'eraser',pointerId:ev.pointerId,points:[[p.x,p.y]],width:brushWidth()};eraseCurrentClassAt([p.x,p.y],roiDrawing.width/2);c.setPointerCapture(ev.pointerId);renderRoi()},{capture:true});c.addEventListener('pointermove',ev=>{if(roiDrawing?.tool!=='eraser'||roiDrawing.pointerId!==ev.pointerId)return;ev.preventDefault();ev.stopImmediatePropagation();const p=roiPoint(ev),next=[p.x,p.y],previous=roiDrawing.points[roiDrawing.points.length-1];roiCursor=p;roiDrawing.points.push(next);eraseCurrentClassAlong(previous,next,roiDrawing.width/2);renderRoi()},{capture:true});window.addEventListener('pointerup',ev=>finishEraserStroke(ev),true);window.addEventListener('pointercancel',ev=>finishEraserStroke(ev),true);window.addEventListener('mouseup',()=>finishEraserStroke(),true);window.addEventListener('blur',()=>finishEraserStroke());c.addEventListener('lostpointercapture',()=>finishEraserStroke())}
function updateRoiCursor(ev){const p=roiPoint(ev);roiCursor=((roiTool==='brush'||roiTool==='eraser')&&roiAttribute!==ROI_ALL&&roiClassState[roiAttribute]!=='negative')?p:null;renderRoi()}
function setupTileUndo(){document.getElementById('tileStage').addEventListener('contextmenu',ev=>{if(!ROI_MODE)return;ev.preventDefault();ev.stopPropagation();undoRoi()})}
function finishRoiStroke(ev=null){if(!roiDrawing||roiDrawing.tool==='eraser')return false;if(ev&&roiDrawing.pointerId!==ev.pointerId)return false;const c=document.getElementById('roiCanvas'),drawing=roiDrawing,pointerId=drawing.pointerId,p=ev?roiPoint(ev):(drawing.last||drawing.start);roiDrawing=null;roiPreview=null;pushUndo();roiPlan=null;if(drawing.tool==='circle'){const dx=p.x-drawing.start.x,dy=p.y-drawing.start.y;roi.push({attribute:drawing.attribute,state:'positive',review_complete:false,geometry:{type:'circle',coordinate_space:'normalized',center:[drawing.start.x,drawing.start.y],radius:Math.sqrt(dx*dx+dy*dy)}})}else{const finalPoint=[p.x,p.y],lastPoint=drawing.points[drawing.points.length-1];if(!lastPoint||pointDistanceSq(lastPoint,finalPoint)>1e-12)drawing.points.push(finalPoint);roi.push({attribute:drawing.attribute,state:'positive',review_complete:false,geometry:{type:'brush',coordinate_space:'normalized',points:drawing.points,width:drawing.width}})}if(c.hasPointerCapture(pointerId))c.releasePointerCapture(pointerId);syncPositiveLabels();renderRoi();return true}
function setupRoi(){const c=document.getElementById('roiCanvas');c.addEventListener('pointerenter',ev=>updateRoiCursor(ev));c.addEventListener('pointerleave',()=>{roiCursor=null;renderRoi()});c.addEventListener('pointerdown',ev=>{if(ev.button!==0)return;if(ROI_MODE&&roiAttribute===ROI_ALL){document.getElementById('status').textContent='Select one spatial class before drawing.';return}if(!roiAttribute)return;if(roiClassState[roiAttribute]==='negative'){document.getElementById('status').textContent='This ROI class is marked negative. Toggle it back before drawing.';return}if(roiTool==='point'){pushUndo();roiPlan=null;const p=roiPoint(ev);roi.push({attribute:roiAttribute,state:'positive',review_complete:false,geometry:{type:'point',coordinate_space:'normalized',point:[p.x,p.y],radius:.018}});syncPositiveLabels();renderRoi();return}const p=roiPoint(ev);roiDrawing={tool:roiTool,attribute:roiAttribute,pointerId:ev.pointerId,start:p,last:p,points:[[p.x,p.y]],width:brushWidth()};roiPreview=null;c.setPointerCapture(ev.pointerId)});c.addEventListener('pointermove',ev=>{updateRoiCursor(ev);if(!roiDrawing||roiDrawing.tool==='eraser'||roiDrawing.pointerId!==ev.pointerId)return;const p=roiPoint(ev);roiDrawing.last=p;if(roiDrawing.tool==='circle'){const dx=p.x-roiDrawing.start.x,dy=p.y-roiDrawing.start.y;roiPreview={attribute:roiDrawing.attribute,state:'positive',review_complete:false,geometry:{type:'circle',coordinate_space:'normalized',center:[roiDrawing.start.x,roiDrawing.start.y],radius:Math.sqrt(dx*dx+dy*dy)}}}else{roiDrawing.points.push([p.x,p.y]);roiPreview={attribute:roiDrawing.attribute,state:'positive',review_complete:false,geometry:{type:'brush',coordinate_space:'normalized',points:roiDrawing.points,width:roiDrawing.width}}}renderRoi()});window.addEventListener('pointerup',ev=>finishRoiStroke(ev),true);window.addEventListener('pointercancel',ev=>finishRoiStroke(ev),true);window.addEventListener('mouseup',()=>finishRoiStroke(),true);window.addEventListener('blur',()=>finishRoiStroke());c.addEventListener('lostpointercapture',()=>finishRoiStroke())}
function overviewIsOpen(){return !document.getElementById('overviewOverlay').classList.contains('hidden')}
function setThumbnailSrc(){if(!overviewIsOpen())return;const img=document.getElementById('thumb'); const loading=document.getElementById('thumbLoading'); const token=`${Date.now()}-${pkg}`; const row=current?`&row=${current.row}`:''; img.dataset.token=String(token); loading.style.display='block'; loading.textContent='Building overview...'; img.style.display='none'; img.onload=()=>{if(img.dataset.token!==String(token)) return; loading.style.display='none'; img.style.display='block'; if(img.resetPanZoom)img.resetPanZoom();}; img.onerror=()=>{if(img.dataset.token===String(token)) loading.textContent='Overview failed to load.';}; img.src=authed(scoped(`/api/thumbnail?package=${pkg}${row}&thumb_token=${encodeURIComponent(token)}&t=${Date.now()}`));}
function openOverview(){document.getElementById('overviewOverlay').classList.remove('hidden');setThumbnailSrc()}
function closeOverview(){document.getElementById('overviewOverlay').classList.add('hidden');document.getElementById('thumb').removeAttribute('src')}
async function loadPackages(){
    packages=await api('/api/packages');
    const scan=await api('/api/scan-status');
    if(!restoredLastIac&&scan.last_iac){
        const found=packages.find(p=>p.rel_path===scan.last_iac);
        if(found) {
            pkg=found.index;
            navigationMode='global';
            resumeAfterRow=null;
        }
        restoredLastIac=true;
    }
    document.getElementById('packageSummary').textContent=`${packages.length} IAC package${packages.length===1?'':'s'}${scan.done?'':' · scanning...'}`;
    if(scan.error){document.getElementById('status').textContent=scan.error;}
    const box=document.getElementById('packages');
    box.innerHTML='';
    let totalAnnotated = 0, totalTiles = 0;
    packages.forEach(p => {
        totalAnnotated += p.annotated || 0;
        totalTiles += p.total || 0;
    });
    const overallPct = totalTiles ? Math.round(totalAnnotated / totalTiles * 100) : 0;
    const allDiv = document.createElement('div');
    allDiv.className = 'pkg' + (navigationMode === 'global' ? ' active' : '');
    allDiv.style.setProperty('--pct', overallPct + '%');
    allDiv.innerHTML = `<b>[Global Mode] All Packages</b><div class=muted>Prioritize candidates globally · ${totalAnnotated}/${totalTiles} · ${overallPct}%</div>`;
    allDiv.onclick = () => {
        navigationMode = 'global';
        restoredLastIac = true;
        if (isMobile()) setQueueOpen(false);
        refreshPackage();
    };
    box.appendChild(allDiv);
    packages.forEach(p=>{
        const pct=p.total?Math.round(p.annotated/p.total*100):0;
        const d=document.createElement('div');
        d.className='pkg'+(navigationMode === 'local' && p.index===pkg?' active':'');
        d.style.setProperty('--pct',pct+'%');
        d.innerHTML=`<b>Package ${p.index+1}</b><div class=muted>${p.annotated}/${p.total} · ${pct}%</div>`;
        d.onclick=()=>{
            pkg=p.index;
            navigationMode='local';
            restoredLastIac=true;
            if(isMobile())setQueueOpen(false);
            refreshPackage();
        };
        box.appendChild(d);
    });
    return scan;
}
async function refreshPackage(){
    try {
        const scan=await loadPackages();
        if(!packages.length){
            document.getElementById('recordMeta').textContent=scan.done?'No image-tile IAC packages found.':'Scanning IAC packages...';
            setTimeout(refreshPackage,1000);
            return;
        }
        if(pkg>=packages.length) pkg=0;
        setThumbnailSrc();
        await progress();
        if(REVIEW_MODE){
            document.getElementById('reviewedDetails').open=true;
            await loadReviewedList();
            reviewedIndex=reviewedItems.findIndex(item=>item.package_index===pkg&&item.record.row===scan.last_row);
            await reviewStep(0);
        }else await nextRandom();
    } catch(e) {
        document.getElementById('status').textContent = 'Error in refreshPackage: ' + (e.message || String(e));
        console.error(e);
    }
}
async function progress(){const p=await api('/api/progress?package='+pkg);classificationCounts=p.classification;spatialCounts=p.roi_counts&&Object.keys(p.roi_counts).length?p.roi_counts:p.spatial;renderLabels();const filtered=p.auto_filtered?` · ${p.auto_filtered} blank filtered`:'';const priority=p.priority;const review=p.review;const activeProgress=review||priority;const handled=activeProgress?activeProgress.reviewed+activeProgress.skipped:0;const progressPct=activeProgress&&activeProgress.total?Math.round(handled/activeProgress.total*100):0;const progressLine=review?`<b>Review list ${review.reviewed}/${review.total}</b> completed · ${review.remaining} remaining · ${review.skipped} skipped<div class=bar><div style="width:${progressPct}%"></div></div>`:priority?`<b>Priority list ${priority.reviewed}/${priority.total}</b> reviewed · ${priority.remaining} remaining · ${priority.skipped} skipped<div class=bar><div style="width:${progressPct}%"></div></div>`:'';document.getElementById('progress').innerHTML=`${progressLine}<div>All tiles: ${p.overall.annotated}/${p.overall.total} reviewed · ${p.overall.remaining} remaining · ${p.overall.skipped} skipped${filtered}</div>`;document.getElementById('reviewedCount').textContent=p.overall.annotated;const active=new Set(p.roi_priority_attributes||[]);document.getElementById('roiCountProgress').innerHTML=Object.keys(p.roi_counts||{}).map(x=>`${active.has(x)?'●':'○'} ${labelName('spatial',x)}: ${p.roi_counts[x]||0}`).join('<br>');const exhausted=[...active].filter(x=>p.roi_navigation_status&&p.roi_navigation_status[x]&&p.roi_navigation_status[x].exhausted);const notice=document.getElementById('roiNavigationNotice');notice.textContent=exhausted.map(x=>`${labelName('spatial',x)}: all ${p.roi_navigation_status[x].total} candidates reviewed; the information curve needs more coverage.`).join('\n');notice.classList.toggle('hidden',!exhausted.length);}
async function showRecord(rec){current=rec; classification=""; spatial=new Set();roi=[];roiClassState={};roiExclusionOverrides={};undoStack=[];redoStack=[];roiPreview=null;roiCursor=null;roiPlan=null;roiPlanLoading=false;roiAttribute=ROI_MODE?ROI_ALL:'';roiAllComplete=true;resetTileViewportForRecord();renderLabels();renderRoi(); if(!rec){document.getElementById('recordMeta').textContent='No unreviewed tile to show.'; document.getElementById('tile').removeAttribute('src'); return;}const saved=await api(`/api/annotation-state?package=${pkg}&row=${rec.row}`);if(saved.annotation){classification=saved.annotation.classification||'';spatial=new Set(saved.annotation.spatial||[]);roi=(saved.annotation.roi||[]).filter(item=>item.geometry);(saved.annotation.roi||[]).filter(item=>item.review_complete&&item.state==='negative'&&!item.geometry).forEach(item=>{roiClassState[item.attribute]='negative'});const complete=new Set((saved.annotation.roi||[]).filter(item=>item.review_complete).map(item=>item.attribute));roiAllComplete=SPATIAL.every(name=>complete.has(name))}renderLabels();renderRoi();document.getElementById('recordMeta').textContent=`Package ${pkg+1} · tile ${rec.row+1} · x=${rec.x} y=${rec.y}`; const tile=document.getElementById('tile'); tile.src=authed(scoped(`/api/tile?package=${pkg}&row=${rec.row}`)); resetTileViewportForRecord(); setThumbnailSrc();}
async function nextRandom(isSkip=false){
    try {
        if(REVIEW_MODE){await reviewStep(1);return}
        if(current){tileHistory.push({package:pkg,record:current});tileForward=[];}
        if(isSkip&&current){
            await api('/api/skip',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({package:pkg,row:current.row})});
        }
        const exclude=current?`&exclude_row=${current.row}&exclude_tile_id=${current.tile_id}`:'';
        const resume=resumeAfterRow===null?'':`&after_row=${resumeAfterRow}`;
        let r;
        if(navigationMode==='global'){
            r=await api('/api/random?package=all'+exclude+resume);
            if(r.record&&r.package_index!==undefined){
                pkg=r.package_index;
                await progress();
            }
        }else{
            r=await api('/api/random?package='+pkg+exclude+resume);
            if(!r.record&&ROI_MODE&&!r.done){
                const next=packages.find(p=>p.remaining>0&&p.index!==pkg);
                if(next){
                    pkg=next.index;
                    await loadPackages();
                    await progress();
                    setThumbnailSrc();
                    r=await api('/api/random?package='+pkg);
                }
            }
        }
        if(r.done==='no_tissue_candidates')document.getElementById('status').textContent='No remaining tile meets the tissue threshold.';
        if(r.done==='review_complete')document.getElementById('status').textContent='This review list is complete.';
        if(r.done==='review_package_complete')document.getElementById('status').textContent='This package has no remaining review tile. Switch to All Packages to continue.';
        resumeAfterRow=null;
        await showRecord(r.record);
    } catch(e) {
        document.getElementById('status').textContent = 'Error in nextRandom: ' + (e.message || String(e));
        console.error(e);
    }
}
async function reviewed(){if(REVIEW_MODE){await reviewStep(-1);return}const previous=tileHistory.pop();if(!previous){document.getElementById('status').textContent='No previous tile in this session.';return}if(current)tileForward.push({package:pkg,record:current});pkg=previous.package;navigationMode='local';await progress();await showRecord(previous.record)}
async function openReviewed(item){if(!REVIEW_MODE&&!reviewedReturn&&current)reviewedReturn={package:pkg,record:current,navigationMode};tileForward=[];reviewedIndex=reviewedItems.findIndex(candidate=>candidate.package_index===item.package_index&&candidate.record.row===item.record.row);pkg=item.package_index;navigationMode='local';if(!REVIEW_MODE)await loadPackages();await progress();await showRecord(item.record);renderReviewedList();if(isMobile())setQueueOpen(false)}
async function returnFromReviewed(){const target=reviewedReturn;reviewedReturn=null;if(!target)return;pkg=target.package;navigationMode=target.navigationMode;await loadPackages();await progress();await showRecord(target.record)}
function renderReviewedList(){const root=document.getElementById('reviewedList');root.innerHTML='';if(!reviewedItems.length){root.innerHTML='<div class="muted">No saved annotations in this mode and version.</div>';return}reviewedItems.forEach((item,index)=>{const button=document.createElement('button');button.type='button';button.className='reviewedItem'+(current&&pkg===item.package_index&&current.row===item.record.row?' active':'');const title=document.createElement('span');title.textContent=REVIEW_MODE?`${index+1} / ${reviewedItems.length}`:`Package ${item.package_index+1} · tile ${item.record.row+1}`;const meta=document.createElement('span');meta.className='reviewedItemMeta';meta.textContent=MODE==='classification'?labelName('classification',item.classification):`${item.spatial.map(x=>labelName('spatial',x)).join(', ')||'No positive ROI'} · ${item.roi_count} mark${item.roi_count===1?'':'s'}`;button.append(title,meta);button.onclick=()=>openReviewed(item);root.appendChild(button);if(button.classList.contains('active'))button.scrollIntoView({block:'nearest'})})}
async function loadReviewedList(){const root=document.getElementById('reviewedList');root.innerHTML='<div class="muted">Loading...</div>';try{const result=await api('/api/reviewed-list?package=all');reviewedItems=result.items||[];document.getElementById('reviewedCount').textContent=result.total;renderReviewedList();return result}catch(e){reviewedItems=[];root.innerHTML='';const error=document.createElement('div');error.className='muted';error.textContent=e.message||String(e);root.appendChild(error);throw e}}
async function reviewStep(delta){if(!reviewedItems.length){await showRecord(null);document.getElementById('status').textContent='No saved annotations in this mode and version.';return}if(reviewedIndex<0)reviewedIndex=0;else reviewedIndex=Math.min(reviewedItems.length-1,Math.max(0,reviewedIndex+delta));await openReviewed(reviewedItems[reviewedIndex]);document.getElementById('status').textContent=`Review ${reviewedIndex+1}/${reviewedItems.length}.`}
async function forwardTile(){if(REVIEW_MODE){await reviewStep(1);return}const next=tileForward.pop();if(!next){document.getElementById('status').textContent='No next tile in this session.';return}if(current)tileHistory.push({package:pkg,record:current});pkg=next.package;navigationMode='local';await progress();await showRecord(next.record)}
async function save(){
    try {
        if(!current){await nextRandom(); return;}
        if(!ROI_MODE&&!classification){document.getElementById('status').textContent='Select one classification prototype first.'; return;}
        if(ROI_MODE&&roiPlan){document.getElementById('status').textContent='Choose Accept visible matches or Start from scratch before saving.';return;}
        const positives=new Set(roi.filter(item=>item.geometry&&item.state!=='negative').map(item=>item.attribute));
        const negativeAttrs=SPATIAL.filter(attribute=>roiClassState[attribute]==='negative');
        const conflicts=negativeAttrs.filter(attribute=>positives.has(attribute));
        if(conflicts.length){window.alert(`ROI conflict: marked negative but has ROI annotation: ${conflicts.join(', ')}`);document.getElementById('status').textContent='Resolve negative/positive ROI conflict before saving.';return}
        const negativePayload=negativeAttrs.map(attribute=>({attribute,state:'negative',review_complete:true}));
        const roiPayload=ROI_MODE?[...roi,...negativePayload]:roi;
        const saveClassification=ROI_MODE?CLASSIFICATION[0]:classification;
        await api('/api/annotation',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({package:pkg,row:current.row,classification:saveClassification,spatial:[...spatial],roi:roiPayload})});
        document.getElementById('status').textContent='Saved.';
        setThumbnailSrc();
        await progress();
        await loadPackages();
        if(document.getElementById('reviewedDetails').open)await loadReviewedList();
        VERSIONS=(await api('/api/versions')).versions;renderVersions();
        if(REVIEW_MODE){reviewedIndex=Math.min(reviewedIndex+1,reviewedItems.length-1);await reviewStep(0)}else await nextRandom();
    } catch(e) {
        document.getElementById('status').textContent = 'Error in save: ' + (e.message || String(e));
        console.error(e);
    }
}
function clampContextPan(){
    contextTx=Math.min(0,Math.max(contextViewportW-contextImageW,contextTx));
    contextTy=Math.min(0,Math.max(contextViewportH-contextImageH,contextTy));
}
function paintContextPan(){
    contextFrame=0;
    const viewport=document.getElementById('contextViewport');
    document.getElementById('contextImg').style.transform=`translate3d(${contextTx}px,${contextTy}px,0)`;
    const gridX=((contextTx%contextCellW)+contextCellW)%contextCellW,gridY=((contextTy%contextCellH)+contextCellH)%contextCellH;
    viewport.style.setProperty('--context-grid-x',`${gridX}px`);
    viewport.style.setProperty('--context-grid-y',`${gridY}px`);
}
function scheduleContextPan(){
    if(!contextFrame)contextFrame=requestAnimationFrame(paintContextPan);
}
function layoutContext(focus={fx:.5,fy:.5}){
    if(!contextData)return;
    const stage=document.getElementById('contextStage'),viewport=document.getElementById('contextViewport'),img=document.getElementById('contextImg');
    const visible=contextData.visible_grid_size,availableW=Math.max(80,stage.clientWidth-24),availableH=Math.max(80,stage.clientHeight-24);
    const scale=Math.max(.1,Math.min(availableW/(visible*contextData.cell_width),availableH/(visible*contextData.cell_height)));
    contextCellW=contextData.cell_width*scale;contextCellH=contextData.cell_height*scale;
    const viewportW=visible*contextCellW,viewportH=visible*contextCellH;
    contextViewportW=viewportW;contextViewportH=viewportH;
    contextImageW=contextData.grid_size*contextCellW;contextImageH=contextData.grid_size*contextCellH;
    viewport.style.width=`${viewportW}px`;viewport.style.height=`${viewportH}px`;
    viewport.style.setProperty('--context-cell-w',`${contextCellW}px`);
    viewport.style.setProperty('--context-cell-h',`${contextCellH}px`);
    img.style.width=`${contextImageW}px`;img.style.height=`${contextImageH}px`;
    contextTx=viewportW/2-(contextData.radius+focus.fx)*contextCellW;
    contextTy=viewportH/2-(contextData.radius+focus.fy)*contextCellH;
    clampContextPan();scheduleContextPan();
}
function contextCellAtLocal(x,y){
    if(!contextData)return null;
    const column=Math.floor((x-contextTx)/contextCellW),row=Math.floor((y-contextTy)/contextCellH);
    return contextCellMap.get(`${column-contextData.radius}:${row-contextData.radius}`)||null;
}
function contextCenterPosition(){
    const imageX=(contextViewportW/2-contextTx)/contextCellW,imageY=(contextViewportH/2-contextTy)/contextCellH;
    return{imageX,imageY,column:Math.floor(imageX),row:Math.floor(imageY),fx:imageX-Math.floor(imageX),fy:imageY-Math.floor(imageY)};
}
function installContextWindow(data,image,centerRow,focus){
    contextData=data;contextCenterRow=centerRow;
    contextCellMap=new Map(data.cells.map(cell=>[`${cell.dx}:${cell.dy}`,cell]));
    document.getElementById('contextImg').src=image.src;
    layoutContext(focus);
    document.getElementById('contextMeta').textContent=`Package ${contextSelected.package+1} · selected tile ${contextSelected.row+1} · view center ${data.center.row+1}`;
}
async function loadContextWindow(centerRow,focus={fx:.5,fy:.5}){
    if(!contextSelected)return;
    const request=++contextRequest,loading=document.getElementById('contextLoading');
    contextPendingWindow=null;
    loading.classList.remove('hidden');
    const query=`package=${contextSelected.package}&row=${centerRow}&radius=${CONTEXT_RADIUS}`;
    const metadataPromise=api(`/api/context-window?${query}`);
    const preload=new Image();
    const imagePromise=new Promise((resolve,reject)=>{preload.onload=()=>resolve(preload);preload.onerror=()=>reject(new Error('Context image failed to load.'));});
    preload.src=authed(scoped(`/api/context?${query}&selected_row=${contextSelected.row}`));
    try{
        const [data,image]=await Promise.all([metadataPromise,imagePromise]);
        if(request!==contextRequest)return;
        if(contextDrag){
            contextPendingWindow={data,image,centerRow,focus};
        }else{
            installContextWindow(data,image,centerRow,focus);
        }
    }finally{
        if(request===contextRequest)loading.classList.add('hidden');
    }
}
async function replenishContextBuffer(){
    if(!contextData||!contextSelected)return;
    const data=contextData,center=contextCenterPosition();
    const absoluteGridX=data.center.grid_x+center.imageX-data.radius-.5;
    const absoluteGridY=data.center.grid_y+center.imageY-data.radius-.5;
    if(Math.max(Math.abs(absoluteGridX-data.center.grid_x),Math.abs(absoluteGridY-data.center.grid_y))<CONTEXT_RECENTER_DELTA)return;
    const request=contextRequest,loading=document.getElementById('contextLoading');
    loading.textContent='Loading nearby tiles…';loading.classList.remove('hidden');
    try{
        const target=await api(`/api/context-center?package=${contextSelected.package}&row=${data.center.row}&grid_x=${absoluteGridX}&grid_y=${absoluteGridY}`);
        if(contextDrag||contextData!==data||request!==contextRequest)return;
        await loadContextWindow(
            target.record.row,
            {
                fx:.5+absoluteGridX-target.record.grid_x,
                fy:.5+absoluteGridY-target.record.grid_y,
            },
        );
    }finally{
        if(request===contextRequest)loading.classList.add('hidden');
    }
}
async function openContextTile(record){
    if(!record)return;
    if(current)tileHistory.push({package:pkg,record:current});
    tileForward=[];
    closeContext();
    await showRecord(record);
    document.getElementById('status').textContent='Opened tile from context view.';
}
function closeContext(){
    contextRequest++;
    if(contextWheelTimer){clearTimeout(contextWheelTimer);contextWheelTimer=0}
    contextDrag=null;contextPendingWindow=null;
    document.getElementById('contextViewport').classList.remove('dragging');
    document.getElementById('contextLoading').classList.add('hidden');
    document.getElementById('contextOverlay').classList.add('hidden');
}
async function openContext(){
    if(!current){document.getElementById('status').textContent='No tile selected.';return}
    contextSelected={package:pkg,row:current.row};
    contextData=null;contextCellMap=new Map();contextCenterRow=current.row;
    document.getElementById('contextMeta').textContent=`Package ${pkg+1} · selected tile ${current.row+1}`;
    document.getElementById('contextOverlay').classList.remove('hidden');
    try{await loadContextWindow(current.row)}catch(e){document.getElementById('contextMeta').textContent=e.message||String(e)}
}
function setupContextNavigator(){
    const viewport=document.getElementById('contextViewport');
    viewport.addEventListener('wheel',ev=>{
        if(!contextData)return;
        if(Math.abs(ev.deltaX)>=2)contextTrackpadWheel=true;
        if(!contextTrackpadWheel)return;
        ev.preventDefault();
        contextRequest++;contextPendingWindow=null;
        contextTx-=ev.deltaX;contextTy-=ev.deltaY;scheduleContextPan();
        if(contextWheelTimer)clearTimeout(contextWheelTimer);
        contextWheelTimer=setTimeout(()=>{
            contextWheelTimer=0;
            replenishContextBuffer().catch(error=>{document.getElementById('contextMeta').textContent=error.message||String(error)});
        },120);
    },{passive:false});
    viewport.addEventListener('pointerdown',ev=>{
        if(ev.button!==0||!contextData)return;
        ev.preventDefault();
        contextDrag={pointerId:ev.pointerId,startX:ev.clientX,startY:ev.clientY,startTx:contextTx,startTy:contextTy,moved:0};
        viewport.classList.add('dragging');viewport.setPointerCapture(ev.pointerId);
    });
    viewport.addEventListener('pointermove',ev=>{
        if(!contextDrag||contextDrag.pointerId!==ev.pointerId)return;
        const dx=ev.clientX-contextDrag.startX,dy=ev.clientY-contextDrag.startY;
        contextDrag.moved=Math.max(contextDrag.moved,Math.abs(dx)+Math.abs(dy));
        contextTx=contextDrag.startTx+dx;contextTy=contextDrag.startTy+dy;
        scheduleContextPan();
    });
    viewport.addEventListener('pointerup',ev=>{
        if(!contextDrag||contextDrag.pointerId!==ev.pointerId)return;
        const moved=contextDrag.moved;contextDrag=null;viewport.classList.remove('dragging');
        if(moved<6){
            contextPendingWindow=null;
            const rect=viewport.getBoundingClientRect(),cell=contextCellAtLocal(ev.clientX-rect.left,ev.clientY-rect.top);
            if(cell?.record)openContextTile(cell.record);
        }else{
            contextRequest++;contextPendingWindow=null;
            replenishContextBuffer().catch(error=>{document.getElementById('contextMeta').textContent=error.message||String(error)});
        }
    });
    viewport.addEventListener('pointercancel',()=>{contextDrag=null;contextRequest++;contextPendingWindow=null;viewport.classList.remove('dragging')});
}
function renderModeNav(){const nav=document.getElementById('modeNav');nav.innerHTML='';MODES.forEach(mode=>{const b=document.createElement('button');b.type='button';b.textContent=mode==='classification'?'Classification':'Spatial';b.classList.toggle('active',mode===MODE);b.disabled=mode===MODE;b.onclick=()=>{const url=new URL(location.href);url.searchParams.set('mode',mode);url.searchParams.delete('version');if(AUTH_TOKEN)url.searchParams.set('token',AUTH_TOKEN);location.href=url.toString()};nav.appendChild(b)});document.getElementById('title').textContent=MODE==='classification'?'Classification prototype supervision':'Spatial prototype supervision';}
function versionStorageKey(){return `hcc_sempath_annotation_version:${MODE}`}
function renderVersions(){const select=document.getElementById('versionSelect');select.innerHTML='';VERSIONS.forEach(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=`${item.name} (${item.annotations})`;option.selected=item.id===VERSION;select.appendChild(option)});}
async function createVersion(){const name=document.getElementById('newVersionName').value.trim();const status=document.getElementById('versionStatus');if(!name){status.textContent='Enter a version name.';return}try{const result=await api('/api/versions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({mode:MODE,name,source_version:VERSION})});const url=new URL(location.href);url.searchParams.set('version',result.created);localStorage.setItem(versionStorageKey(),result.created);if(AUTH_TOKEN)url.searchParams.set('token',AUTH_TOKEN);location.href=url.toString()}catch(e){status.textContent=e.message||String(e)}}
function labelDrafts(){return Object.fromEntries([...document.querySelectorAll('#labelEditor input[data-label-id]')].map(input=>[input.dataset.labelId,input.value]));}
function renderLabelEditor(drafts={}){const task=MODE;document.getElementById('labelDialogTitle').textContent=`${task.toUpperCase()} label management · ${VERSION}`;const root=document.getElementById('labelEditor');root.innerHTML='';(LABELS.label_sets[task]||[]).forEach(item=>{const row=document.createElement('div');row.className='labelEditorRow';const badge=document.createElement('span');badge.className='muted';badge.textContent=item.active?'Active':'Archived';const input=document.createElement('input');input.value=drafts[item.id]??item.name;input.dataset.labelId=item.id;input.disabled=!item.active;const rename=document.createElement('button');rename.type='button';rename.textContent='Rename';rename.disabled=!item.active;rename.onclick=()=>changeLabel(task,'rename',item.id,input.value);const state=document.createElement('button');state.type='button';state.textContent=item.active?'Archive':'Restore';state.onclick=()=>changeLabel(task,item.active?'archive':'restore',item.id,'');const remove=document.createElement('button');remove.type='button';remove.textContent='Delete';remove.onclick=()=>{if(window.confirm(`Delete label "${item.name}"?`))changeLabel(task,'delete',item.id,'')};row.append(badge,input,rename,state,remove);root.appendChild(row)});pendingLabelAdds.forEach(name=>{const row=document.createElement('div');row.className='labelEditorRow';row.innerHTML=`<span class=muted>Pending new</span><span>${name}</span>`;root.appendChild(row)});}
async function changeLabel(task,operation,labelId,name,preserveDrafts=false){const status=document.getElementById('labelStatus');const drafts=preserveDrafts?labelDrafts():{};try{LABELS=await api('/api/labels',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({mode:MODE,task,operation,label_id:labelId,name})});CLASSIFICATION=LABELS.label_sets.classification.filter(x=>x.active).map(x=>x.id);SPATIAL=LABELS.label_sets.spatial.filter(x=>x.active).map(x=>x.id);renderLabelEditor(drafts);renderLabels();status.textContent='Saved.'}catch(e){status.textContent=e.message||String(e)}}
function clearSelectedRoiClass(){if(roiAttribute===ROI_ALL){document.getElementById('status').textContent='Select one spatial class before clearing.';return}const cleared=roiAttribute;pushUndo();roi=roi.filter(x=>x.attribute!==cleared);delete roiClassState[cleared];delete roiExclusionOverrides[cleared];roiDrawing=null;roiPreview=null;roiCursor=null;roiPlan=null;syncPositiveLabels();renderRoi();document.getElementById('status').textContent=`Cleared ROI state for: ${labelName('spatial',cleared)}.`}
setupEraser();setupRoi();setupTileUndo();document.querySelectorAll('[data-roi-tool]').forEach(button=>button.addEventListener('click',()=>selectRoiTool(button.dataset.roiTool)));document.getElementById('roiUndo').onclick=undoRoi;document.getElementById('roiRedo').onclick=redoRoi;document.getElementById('roiClear').onclick=clearSelectedRoiClass;selectRoiTool('point',false);
document.getElementById('roiPlanGenerate').onclick=generateRoiPlan;document.getElementById('roiPlanAccept').onclick=acceptRoiPlan;document.getElementById('roiPlanRestart').onclick=restartRoiFromScratch;
document.getElementById('roiSimilarity').addEventListener('input',renderRoi);
document.getElementById('roiExclusion').addEventListener('input',()=>{if(roiAttribute!==ROI_ALL)roiExclusionOverrides[roiAttribute]=roiExclusionDistance();renderRoi()});
document.getElementById('brushWidth').addEventListener('input',updateBrushWidth);updateBrushWidth();
document.getElementById('tileZoom').addEventListener('input',ev=>{tileZoomUserSet=true;setTileZoom(parseFloat(ev.target.value))});setTileZoom(tileScale);requestAnimationFrame(applyDefaultTileZoom);document.getElementById('tileGridToggle').onclick=()=>{showGrid=!showGrid;document.getElementById('tileGridToggle').classList.toggle('selected',showGrid);renderRoi();};document.getElementById('roiCanvas').addEventListener('wheel',ev=>{if(!ROI_MODE||roiTool!=='brush')return;ev.preventDefault();setBrushWidth(brushWidth()*(ev.deltaY<0?1.12:1/1.12))},{passive:false});
setupContextNavigator();
const overviewZoom=document.getElementById('overviewZoom'),overviewZoomValue=document.getElementById('overviewZoomValue'),thumb=document.getElementById('thumb');function syncOverviewZoom(scale){overviewZoom.value=scale;overviewZoomValue.textContent=`${scale.toFixed(2).replace(/0$/,'')}×`}setupPanZoom(thumb,async ev=>{const img=ev.target, r=img.getBoundingClientRect(); const x=(ev.clientX-r.left)/r.width, y=(ev.clientY-r.top)/r.height; const rec=await api(`/api/nearest?package=${pkg}&rx=${x}&ry=${y}`); await showRecord(rec.record);closeOverview();},{wheelZoom:false,horizontalWheelPan:true,doubleClickReset:false,onScale:syncOverviewZoom});overviewZoom.addEventListener('input',ev=>thumb.setPanZoomScale(parseFloat(ev.target.value)));
document.getElementById('toggleQueue').addEventListener('click',()=>setQueueOpen(document.getElementById('layout').classList.contains('queue-collapsed')));
document.getElementById('hideQueue').addEventListener('click',()=>setQueueOpen(false));
document.getElementById('reviewedDetails').addEventListener('toggle',ev=>{if(REVIEW_MODE&&!ev.target.open){ev.target.open=true;return}if(ev.target.open)loadReviewedList();else returnFromReviewed()});
document.getElementById('reviewModeBtn').addEventListener('click',()=>{const url=new URL(location.href);if(REVIEW_MODE)url.searchParams.delete('review');else url.searchParams.set('review','1');if(AUTH_TOKEN)url.searchParams.set('token',AUTH_TOKEN);location.href=url.toString()});
document.getElementById('overviewBtn').addEventListener('click',openOverview);
document.getElementById('overviewClose').addEventListener('click',closeOverview);
document.getElementById('contextBtn').addEventListener('click',openContext);
document.getElementById('contextClose').addEventListener('click',closeContext);
window.addEventListener('resize',()=>{applyDefaultTileZoom();if(contextData&&!document.getElementById('contextOverlay').classList.contains('hidden'))layoutContext()});
document.getElementById('versionSelect').addEventListener('change',ev=>{const url=new URL(location.href);url.searchParams.set('version',ev.target.value);localStorage.setItem(versionStorageKey(),ev.target.value);if(AUTH_TOKEN)url.searchParams.set('token',AUTH_TOKEN);location.href=url.toString()});
document.getElementById('newVersion').addEventListener('click',()=>document.getElementById('versionDialog').showModal());
document.getElementById('closeVersion').addEventListener('click',()=>document.getElementById('versionDialog').close());
document.getElementById('createVersion').addEventListener('click',createVersion);
document.getElementById('manageLabels').addEventListener('click',()=>{pendingLabelAdds=[];renderLabelEditor();document.getElementById('labelDialog').showModal()});
document.getElementById('closeLabels').addEventListener('click',()=>{pendingLabelAdds=[];document.getElementById('labelDialog').close()});
async function savePendingLabelNames(){const task=MODE;const pending=[...document.querySelectorAll('#labelEditor input[data-label-id]')].map(input=>({id:input.dataset.labelId,name:input.value.trim()})).filter(item=>{const current=(LABELS.label_sets[task]||[]).find(label=>label.id===item.id);return current&&current.active&&item.name&&item.name!==current.name});for(const item of pending)await changeLabel(task,'rename',item.id,item.name,true);}
document.getElementById('addLabel').addEventListener('click',()=>{const input=document.getElementById('newLabelName');const status=document.getElementById('labelStatus');const name=input.value.trim();const finalNames=[...document.querySelectorAll('#labelEditor input[data-label-id]')].map(item=>item.value.trim().toLocaleLowerCase()).concat(pendingLabelAdds.map(item=>item.toLocaleLowerCase()));if(!name){status.textContent='Enter a label name.';return}if(finalNames.includes(name.toLocaleLowerCase())){status.textContent='A current label or draft already uses this name.';return}pendingLabelAdds.push(name);input.value='';renderLabelEditor(labelDrafts());status.textContent='Added as a draft. Save label configuration to commit.';});
document.getElementById('saveLabels').addEventListener('click',async()=>{const status=document.getElementById('labelStatus');try{await savePendingLabelNames();for(const name of pendingLabelAdds)await changeLabel(MODE,'add','',name,true);pendingLabelAdds=[];renderLabelEditor();status.textContent='Saved to this version.';}catch(e){status.textContent=e.message||String(e);}});
document.getElementById('authSubmit').addEventListener('click',submitAuth);
document.getElementById('authInput').addEventListener('keydown',ev=>{if(ev.key==='Enter')submitAuth();});
document.addEventListener('keydown',ev=>{const mod=ev.metaKey||ev.ctrlKey,target=ev.target,editing=target&&(target.matches?.('input,textarea,select')||target.isContentEditable),dialogOpen=document.querySelector('dialog[open]');if(!mod&&!editing&&!dialogOpen&&ev.key.toLowerCase()==='c'){ev.preventDefault();if(document.getElementById('contextOverlay').classList.contains('hidden')){if(overviewIsOpen())closeOverview();openContext()}else closeContext();return}if(MODE==='classification'&&!mod&&!editing&&!dialogOpen&&ev.key==='Enter'){ev.preventDefault();if(!ev.repeat)nextRandom(true);return}if(MODE==='classification'&&!mod&&!editing&&!dialogOpen&&/^[1-9]$/.test(ev.key)){const index=Number(ev.key)-1;if(index<CLASSIFICATION.length){ev.preventDefault();classification=CLASSIFICATION[index];renderLabels()}return}if(MODE==='classification'&&!mod&&!editing&&!dialogOpen&&ev.code==='Space'){ev.preventDefault();if(!ev.repeat)save();return}if(!mod)return;if(ev.key.toLowerCase()==='z'){ev.preventDefault();if(ev.shiftKey)redoRoi();else undoRoi()}else if(ev.key.toLowerCase()==='y'){ev.preventDefault();redoRoi()}});
let CLASSIFICATION=%CLASSIFICATION_JSON%; let SPATIAL=%SPATIAL_JSON%; let LABELS=%LABELS_JSON%; const ROI_MODE=%ROI_MODE_JSON%; const MODE=%MODE_JSON%; const MODES=%MODES_JSON%; const VERSION=%VERSION_JSON%; const REVIEW_MODE=%REVIEW_MODE_JSON%; let VERSIONS=%VERSIONS_JSON%; localStorage.setItem(`hcc_sempath_annotation_version:${MODE}`,VERSION);document.body.classList.toggle('review-mode',REVIEW_MODE);document.getElementById('reviewModeBtn').classList.toggle('active',REVIEW_MODE);document.getElementById('reviewModeBtn').textContent=REVIEW_MODE?'Reviewing':'Review';const previousButton=document.getElementById('previousTile');previousButton.textContent='Previous tile';const forwardButton=document.createElement('button');forwardButton.type='button';forwardButton.textContent='Next tile';forwardButton.onclick=forwardTile;previousButton.after(forwardButton);if(REVIEW_MODE){document.getElementById('reviewedDetails').open=true;document.getElementById('skipNext').textContent='Next tile';document.getElementById('toggleQueue').textContent='Reviewed tiles'}renderModeNav(); renderVersions(); renderLabels(); if(isMobile())setQueueOpen(false); ensureAuth().then(()=>refreshPackage()).catch(e=>{if(e.message)document.getElementById('status').textContent=e.message||String(e);});
</script></body></html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict | list) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(
    data: AnnotationData | AnnotationArchive | dict[str, AnnotationData | AnnotationArchive],
    auth_token: str,
    roi_plan_generator=None,
):
    raw_datasets = data if isinstance(data, dict) else ({"spatial" if isinstance(data, AnnotationData) and data.roi_mode else "classification": data})
    archives = {
        mode: item if isinstance(item, AnnotationArchive) else AnnotationArchive(
            item,
            input_path=item.input_root,
            roi_candidate_manifest=item.roi_queue.path if item.roi_queue else None,
            roi_information_report=item.roi_information_report,
            roi_priority_attributes=tuple(item.roi_priority_attributes),
            roi_mode=item.roi_mode,
            priority_queue=item.priority_queue,
            review_queue=item.review_queue,
            min_tissue_fraction=item.min_tissue_fraction,
        )
        for mode, item in raw_datasets.items()
    }
    default_mode = next(iter(archives))

    def select_data(mode: str | None, version: str | None = None) -> tuple[str, str, AnnotationData]:
        selected = (mode or default_mode).lower()
        if selected not in archives:
            raise ValueError(f"annotation mode is not configured: {selected}")
        selected_version = str(version or archives[selected].default_version)
        return selected, selected_version, archives[selected].data(selected_version)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    LOG.info("ui_open path=/")
                    mode = qs.get("mode", [default_mode])[0]
                    version = qs.get("version", [None])[0]
                    review_mode = qs.get("review", ["0"])[0].strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    mode, version, selected_data = select_data(mode, version)
                    html = (
                        HTML.replace("%CLASSIFICATION_JSON%", json.dumps(selected_data.state.classification_prototypes))
                        .replace("%SPATIAL_JSON%", json.dumps(selected_data.state.spatial_prototypes))
                        .replace("%LABELS_JSON%", json.dumps(selected_data.state.labels_json()))
                        .replace("%ROI_MODE_JSON%", json.dumps(selected_data.roi_mode))
                        .replace("%MODE_JSON%", json.dumps(mode))
                        .replace("%MODES_JSON%", json.dumps(list(archives)))
                        .replace("%VERSION_JSON%", json.dumps(version))
                        .replace("%REVIEW_MODE_JSON%", json.dumps(review_mode))
                        .replace("%AUTH_REQUIRED_JSON%", json.dumps(bool(auth_token)))
                        .replace("%VERSIONS_JSON%", json.dumps(archives[mode].versions_json()["versions"]))
                    )
                    body = html.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not _auth_ok(_request_auth_token(qs), auth_token):
                    self.send_error(HTTPStatus.FORBIDDEN, "invalid annotation token")
                    return
                mode, version, selected_data = select_data(
                    qs.get("mode", [default_mode])[0], qs.get("version", [None])[0]
                )
                if parsed.path == "/api/packages":
                    _json_response(self, selected_data.package_json())
                    return
                if parsed.path == "/api/scan-status":
                    _json_response(self, selected_data.scan_status())
                    return
                if parsed.path == "/api/labels":
                    _json_response(self, selected_data.state.labels_json())
                    return
                if parsed.path == "/api/versions":
                    _json_response(self, archives[mode].versions_json())
                    return
                pkg_val = qs.get("package", ["0"])[0]
                index = int(pkg_val) if pkg_val != "all" else 0
                if parsed.path == "/api/progress":
                    _json_response(self, selected_data.progress(int(pkg_val)))
                    return
                if parsed.path == "/api/random":
                    exclude_row = int(qs["exclude_row"][0]) if "exclude_row" in qs else None
                    exclude_tile_id = qs["exclude_tile_id"][0] if "exclude_tile_id" in qs else None
                    after_row = int(qs["after_row"][0]) if "after_row" in qs else None
                    _json_response(self, selected_data.random_record(pkg_val, exclude_row=exclude_row, exclude_tile_id=exclude_tile_id, after_row=after_row))
                    return
                if parsed.path == "/api/reviewed":
                    _json_response(self, selected_data.reviewed_record(index))
                    return
                if parsed.path == "/api/reviewed-list":
                    _json_response(self, selected_data.reviewed_records(pkg_val))
                    return
                if parsed.path == "/api/annotation-state":
                    row = int(qs["row"][0])
                    _json_response(self, selected_data.annotation_json(index, row))
                    return
                if parsed.path == "/api/nearest":
                    viewer = selected_data.viewer(index)
                    rx = float(qs.get("rx", ["0"])[0])
                    ry = float(qs.get("ry", ["0"])[0])
                    bounds = viewer._bounds(viewer.records)
                    x = bounds[0] + rx * max(1, bounds[1] - bounds[0] + viewer.stride_x)
                    y = bounds[2] + ry * max(1, bounds[3] - bounds[2] + viewer.stride_y)
                    _json_response(self, selected_data.select_nearest(index, x, y))
                    return
                if parsed.path == "/api/thumbnail":
                    token = qs.get("thumb_token", [""])[0]
                    selected_row = int(qs["row"][0]) if "row" in qs else None
                    body = selected_data.thumbnail_jpg(index, token=token, selected_row=selected_row)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/context-center":
                    row = int(qs["row"][0])
                    grid_x = float(qs["grid_x"][0])
                    grid_y = float(qs["grid_y"][0])
                    _json_response(
                        self,
                        selected_data.context_center(
                            index,
                            row,
                            grid_x=grid_x,
                            grid_y=grid_y,
                        ),
                    )
                    return
                if parsed.path == "/api/context-window":
                    row = int(qs["row"][0])
                    radius = int(
                        qs.get(
                            "radius",
                            [str(CONTEXT_PREFETCH_RADIUS)],
                        )[0]
                    )
                    _json_response(
                        self,
                        selected_data.context_window(
                            index,
                            row,
                            radius=radius,
                        ),
                    )
                    return
                if parsed.path == "/api/context":
                    row = int(qs["row"][0])
                    radius = int(
                        qs.get(
                            "radius",
                            [str(CONTEXT_VISIBLE_RADIUS)],
                        )[0]
                    )
                    selected_row = int(
                        qs.get("selected_row", [str(row)])[0]
                    )
                    body = selected_data.context_jpg(
                        index,
                        row,
                        radius=radius,
                        selected_row=selected_row,
                    )
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/tile":
                    row = int(qs["row"][0])
                    package = selected_data.package(index)
                    LOG.info("tile_read iac=%s row=%d", package.rel_path, row)
                    body = selected_data.viewer(index).read_tile_png(row)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                LOG.exception("api_get_failed path=%s query=%s", parsed.path, parsed.query)
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if not _auth_ok(_request_auth_token(qs), auth_token):
                self.send_error(HTTPStatus.FORBIDDEN, "invalid annotation token")
                return
            if parsed.path == "/api/roi-similar":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    _mode, _version, selected_data = select_data(
                        str(payload.get("mode") or qs.get("mode", [default_mode])[0]),
                        str(payload.get("version") or qs.get("version", [None])[0]) if (payload.get("version") or qs.get("version")) else None,
                    )
                    if not selected_data.roi_mode:
                        raise ValueError("ROI similarity is available only in spatial mode")
                    if roi_plan_generator is None:
                        raise ValueError("ROI similarity generator is not configured")
                    index = int(payload["package"])
                    row = int(payload["row"])
                    attribute = str(payload["attribute"])
                    if attribute not in selected_data.state.spatial_prototypes:
                        raise ValueError(f"unknown spatial attribute: {attribute}")
                    tile_png = selected_data.viewer(index).read_tile_png(row)
                    LOG.info(
                        "roi_similarity_request iac=%s row=%d attribute=%s seeds=%d",
                        selected_data.package(index).rel_path,
                        row,
                        attribute,
                        len(payload.get("seeds") or []),
                    )
                    result = roi_plan_generator.generate_similar(
                        tile_png,
                        attribute=attribute,
                        seeds=list(payload.get("seeds") or []),
                        occupied=list(payload.get("occupied") or []),
                    )
                    summary = result["summary"]
                    LOG.info(
                        "roi_similarity_result iac=%s row=%d attribute=%s candidates=%d suggestions=%d best=%.3f recommended=%.3f target=%d",
                        selected_data.package(index).rel_path,
                        row,
                        attribute,
                        int(summary.get("candidate_count", summary.get("suggestion_count", 0))),
                        int(summary.get("suggestion_count", 0)),
                        float(summary.get("maximum_similarity", 0.0)),
                        float(summary.get("recommended_similarity", 0.0)),
                        int(summary.get("preview_target", 0)),
                    )
                    _json_response(self, result)
                except Exception as exc:
                    LOG.exception("api_roi_similarity_failed")
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/api/annotation":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    mode, version, selected_data = select_data(
                        str(payload.get("mode") or qs.get("mode", [default_mode])[0]),
                        str(payload.get("version") or qs.get("version", [None])[0]) if (payload.get("version") or qs.get("version")) else None,
                    )
                    index = int(payload["package"])
                    row = int(payload["row"])
                    viewer = selected_data.viewer(index)
                    record = viewer._by_row[row]
                    package = selected_data.package(index)
                    LOG.info("annotation_request iac=%s row=%d classification=%s spatial=%s", package.rel_path, row, payload.get("classification"), payload.get("spatial", []))
                    if (
                        selected_data.priority_queue is not None
                        and selected_data.review_queue is None
                    ):
                        selected_data.priority_queue.add(package, record)
                    selected_data.state.save_annotation(
                        package,
                        record,
                        str(payload["classification"]),
                        list(payload.get("spatial", [])),
                        list(payload.get("roi", [])),
                        review_id=(
                            selected_data.review_queue.review_id
                            if selected_data.review_queue
                            else None
                        ),
                    )
                    _json_response(self, {"ok": True})
                except Exception as exc:
                    LOG.exception("api_post_failed path=%s", parsed.path)
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/api/skip":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    mode, version, selected_data = select_data(
                        str(payload.get("mode") or qs.get("mode", [default_mode])[0]),
                        str(payload.get("version") or qs.get("version", [None])[0]) if (payload.get("version") or qs.get("version")) else None,
                    )
                    index = int(payload["package"])
                    row = int(payload["row"])
                    viewer = selected_data.viewer(index)
                    record = viewer._by_row[row]
                    package = selected_data.package(index)
                    LOG.info("skip_request iac=%s row=%d", package.rel_path, row)
                    if (
                        selected_data.priority_queue is not None
                        and selected_data.review_queue is None
                    ):
                        selected_data.priority_queue.add(package, record)
                    selected_data.state.save_skip(
                        package,
                        record,
                        review_id=(
                            selected_data.review_queue.review_id
                            if selected_data.review_queue
                            else None
                        ),
                    )
                    _json_response(self, {"ok": True})
                except Exception as exc:
                    LOG.exception("api_skip_failed path=%s", parsed.path)
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/api/labels":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    _mode, _version, selected_data = select_data(
                        str(payload.get("mode") or qs.get("mode", [default_mode])[0]),
                        str(payload.get("version") or qs.get("version", [None])[0]) if (payload.get("version") or qs.get("version")) else None,
                    )
                    result = selected_data.state.change_label(
                        str(payload["task"]), str(payload["operation"]),
                        label_id=str(payload.get("label_id") or ""),
                        name=str(payload.get("name") or ""),
                    )
                    _json_response(self, result)
                except Exception as exc:
                    LOG.exception("api_labels_failed")
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/api/versions":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    mode = str(payload.get("mode") or qs.get("mode", [default_mode])[0]).lower()
                    if mode not in archives:
                        raise ValueError(f"annotation mode is not configured: {mode}")
                    result = archives[mode].create_version(
                        str(payload.get("name") or ""),
                        str(payload.get("source_version") or "main"),
                    )
                    _json_response(self, result)
                except Exception as exc:
                    LOG.exception("api_versions_failed")
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def _annotation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the classification/spatial prototype annotation UI."
    )
    parser.add_argument("--input", required=True, help="IAC file or directory containing image-tile IAC packages.")
    parser.add_argument("--classification-state", required=True, help="Classification JSON state file.")
    parser.add_argument("--spatial-state", required=True, help="Spatial annotation JSON state file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Use 0 to pick a free port.")
    parser.add_argument("--token", default="", help="Annotation UI auth token; overrides the persisted state .auth-token when set.")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication for a temporary trusted-network session.",
    )
    parser.add_argument(
        "--min-tissue-fraction",
        type=float,
        default=0.30,
        help="Exclude random candidates below this approximate tissue fraction (default: 0.30).",
    )
    parser.add_argument(
        "--priority-manifest",
        default="",
        help=(
            "Optional shared mutable tile-priority manifest. Omit it for "
            "ordinary random annotation."
        ),
    )
    parser.add_argument(
        "--classification-review-manifest",
        default="",
        help=(
            "Optional read-only strict tile list for a classification review pass. "
            "Navigation stops when this list is complete."
        ),
    )
    parser.add_argument(
        "--include-packages",
        default="",
        help=(
            "Optional comma-separated IAC paths or glob patterns relative to the input "
            "directory. All Packages and package navigation are restricted "
            "to this set."
        ),
    )
    parser.add_argument(
        "--spatial-review-manifest",
        default="",
        help=(
            "Optional read-only strict tile list for a spatial review pass. "
            "Navigation stops when this list is complete."
        ),
    )
    parser.add_argument(
        "--roi-candidate-manifest",
        default="",
        help=(
            "Optional component-presence candidate manifest used to prioritize "
            "spatial navigation within the shared tile boundary."
        ),
    )
    parser.add_argument(
        "--roi-priority-attributes",
        default="",
        help=(
            "Optional comma-separated target restriction for component-presence navigation. "
            "Related labels may be used as retrieval hints for "
            "the selected targets. "
            "Normal annotation should leave this empty so the current "
            "information report selects deficient ROI attributes dynamically."
        ),
    )
    parser.add_argument(
        "--roi-information-report",
        default="",
        help=(
            "Optional current spatial information-curve report used to "
            "prioritize deficient components."
        ),
    )
    parser.add_argument(
        "--review-existing",
        action="store_true",
        help=(
            "Open the shared classification/spatial UI in review mode: show only saved "
            "annotations and navigate them in deterministic order."
        ),
    )
    parser.add_argument("--no-open", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _annotation_parser().parse_args()
    from hcc_sempath.modeling.roi_plan import RoiPlanGenerator

    priority_queue = (
        SharedPriorityQueue(args.priority_manifest)
        if args.priority_manifest
        else None
    )
    classification_review_queue = (
        StrictReviewQueue(args.classification_review_manifest)
        if args.classification_review_manifest
        else None
    )
    spatial_review_queue = (
        StrictReviewQueue(args.spatial_review_manifest)
        if args.spatial_review_manifest
        else None
    )
    roi_priority_attributes = [
        value.strip() for value in args.roi_priority_attributes.split(",") if value.strip()
    ]
    include_packages = [
        value.strip() for value in args.include_packages.split(",") if value.strip()
    ]
    roi_information_report = (
        Path(args.roi_information_report)
        if args.roi_information_report
        else None
    )
    classification_initial = AnnotationData(
        args.input,
        args.classification_state,
        priority_queue=priority_queue,
        review_queue=classification_review_queue,
        include_packages=include_packages,
        min_tissue_fraction=args.min_tissue_fraction,
    )
    spatial_initial = AnnotationData(
        args.input,
        args.spatial_state,
        roi_candidate_manifest=args.roi_candidate_manifest or None,
        roi_information_report=roi_information_report,
        roi_priority_attributes=roi_priority_attributes,
        roi_mode=True,
        priority_queue=priority_queue,
        review_queue=spatial_review_queue,
        include_packages=include_packages,
        min_tissue_fraction=args.min_tissue_fraction,
    )
    datasets = {
        "classification": AnnotationArchive(
            classification_initial,
            input_path=args.input,
            priority_queue=priority_queue,
            review_queue=classification_review_queue,
            include_packages=include_packages,
            min_tissue_fraction=args.min_tissue_fraction,
        ),
        "spatial": AnnotationArchive(
            spatial_initial,
            input_path=args.input,
            roi_candidate_manifest=args.roi_candidate_manifest or None,
            roi_information_report=roi_information_report,
            roi_priority_attributes=roi_priority_attributes,
            roi_mode=True,
            priority_queue=priority_queue,
            review_queue=spatial_review_queue,
            include_packages=include_packages,
            min_tissue_fraction=args.min_tissue_fraction,
        ),
    }
    auth_token = (
        ""
        if args.no_auth
        else _load_or_create_auth_token(
            args.classification_state,
            args.token,
        )
    )
    roi_plan_generator = RoiPlanGenerator()
    port = _find_free_port(args.host, args.port)
    server = ThreadingHTTPServer(
        (args.host, port), make_handler(datasets, auth_token, roi_plan_generator)
    )
    base_url = f"http://{args.host}:{port}/"
    query = f"token={auth_token}" if auth_token else ""
    if args.review_existing:
        query += "&mode=spatial&review=1"
    url = f"{base_url}?{query}" if query else base_url
    states = ",".join(f"{mode}:{item.data().state.state_path}" for mode, item in datasets.items())
    LOG.info("server_start url=%s states=%s", base_url, states)
    print(f"annotation_ui url={url} states={states}", flush=True)
    if not args.no_open:
        LOG.info("browser_open url=%s", url)
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for item in datasets.values():
            item.close()
        server.server_close()


if __name__ == "__main__":
    main()
