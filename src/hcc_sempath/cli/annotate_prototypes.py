from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import logging
import os
import random
import secrets
import socket
import tempfile
import threading
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from hcc_sempath.cli.view_iac import IacRecord, IacViewerData
from hcc_sempath.io.iatro_iac import read_header, read_payload
from hcc_sempath.io.tile_package import decode_jxl


LOG = logging.getLogger("hcc_sempath.annotate_prototypes")
OVERVIEW_CELL_PX = 4
OVERVIEW_WORKERS = 8
MAX_OPEN_IAC_VIEWERS = 8

L1_PROTOTYPES = [
    "HCC-tumor",
    "Background-liver",
    "Inflammatory-stromal",
    "Degenerative-material",
]

L2_PROTOTYPES = [
    "hepatocellular-parenchyma-present",
    "necrosis-present",
    "hemorrhage-present",
    "bile-pigment-present",
    "inflammatory-cell-present",
    "fibrous-stroma-present",
    "steatosis-vacuolation-present",
    "hyaline-change-present",
    "vascular-structure-present",
    "ductular-portal-present",
]


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
    return bool(provided) and hmac.compare_digest(provided, expected)


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


def _candidate_iac_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".iac" else []

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


def discover_iac_packages(input_path: str | Path) -> list[AnnotationPackage]:
    root = Path(input_path).resolve()
    LOG.info("discover_iac_start input=%s", root)
    packages = []
    for path in _candidate_iac_paths(root):
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
    return bool(str(item.get("l1") or item.get("level1_label") or "").strip())


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


class AnnotationState:
    def __init__(self, state_path: str | Path, input_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.input_path = str(Path(input_path).resolve())
        self.annotations: dict[str, dict] = {}
        self.skipped: set[str] = set()
        self.extra_payload: dict = {}
        self.last_iac = ""
        self.l1_prototypes = list(L1_PROTOTYPES)
        self.l2_prototypes = list(L2_PROTOTYPES)
        self.revision = 0
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.input_path = str(Path(payload.get("input_path", self.input_path)).resolve())
            self.annotations = dict(payload.get("annotations", {}))
            self.skipped = set(str(item) for item in payload.get("skipped", []))
            self.last_iac = str(payload.get("last_iac") or "")
            self.l1_prototypes = _state_prototypes(payload, "l1_prototypes", L1_PROTOTYPES, "l1")
            self.l2_prototypes = _state_prototypes(payload, "l2_prototypes", L2_PROTOTYPES, "l2")
            known_keys = {"version", "input_path", "l1_prototypes", "l2_prototypes", "annotations", "skipped", "last_iac"}
            self.extra_payload = {key: value for key, value in payload.items() if key not in known_keys}
            LOG.info(
                "state_load path=%s annotations=%d skipped=%d l1=%d l2=%d input=%s",
                self.state_path,
                len(self.annotations),
                len(self.skipped),
                len(self.l1_prototypes),
                len(self.l2_prototypes),
                self.input_path,
            )
        else:
            LOG.info("state_new path=%s input=%s", self.state_path, self.input_path)

    @property
    def csv_path(self) -> Path:
        return self.state_path.with_suffix(".csv")

    def is_annotated(self, package: AnnotationPackage, record: IacRecord) -> bool:
        key = _annotation_key(package, record)
        return key in self.annotations or key in self.skipped

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
        l1: str,
        l2: list[str],
        roi: list[dict] | None = None,
    ) -> None:
        if l1 not in self.l1_prototypes:
            raise ValueError(f"unknown L1 prototype: {l1}")
        unknown_l2 = sorted(set(l2) - set(self.l2_prototypes))
        if unknown_l2:
            raise ValueError(f"unknown L2 prototype(s): {unknown_l2}")
        roi = list(roi or [])
        for item in roi:
            if item.get("attribute") not in self.l2_prototypes:
                raise ValueError(f"unknown ROI L2 attribute: {item.get('attribute')}")
            if item.get("state", "positive") not in {"positive", "negative"}:
                raise ValueError(f"unknown ROI state: {item.get('state')}")
            geometry = item.get("geometry")
            if geometry is None and not bool(item.get("review_complete", False)):
                raise ValueError("ROI item requires geometry or review_complete=true")
            if geometry is not None and geometry.get("type") not in {"point", "brush", "circle", "polygon"}:
                raise ValueError(f"unsupported ROI geometry: {geometry.get('type')}")
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
            "l1": l1,
            "l2": list(l2),
            "roi": roi,
        }
        key = _annotation_key(package, record)
        self.annotations[key] = payload
        self.skipped.discard(key)
        self.last_iac = package.rel_path
        self.revision += 1
        self.flush()
        LOG.info(
            "annotation_save iac=%s row=%d tile_id=%s x=%d y=%d l1=%s l2=%s total_annotations=%d",
            package.rel_path,
            record.row,
            record.tile_id,
            record.display_x,
            record.display_y,
            l1,
            ",".join(l2) if l2 else "-",
            len(self.annotations),
        )

    def flush(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.extra_payload)
        payload.update({
            "version": 1,
            "input_path": self.input_path,
            "l1_prototypes": self.l1_prototypes,
            "l2_prototypes": self.l2_prototypes,
            "last_iac": self.last_iac,
            "annotations": self.annotations,
            "skipped": sorted(self.skipped),
        })
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.state_path.parent) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.state_path)
        self._write_csv()
        LOG.info("state_flush json=%s csv=%s annotations=%d", self.state_path, self.csv_path, len(self.annotations))

    def _write_csv(self) -> None:
        base_fields = ["dataset", "iac", "iac_path", "tile_id", "row", "slide", "split", "x", "y", "l1", "l2"]
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
                row["l2"] = ";".join(row.get("l2", []))
                writer.writerow(row)


class AnnotationData:
    def __init__(self, input_path: str | Path, state_path: str | Path, *, async_scan: bool = False) -> None:
        self.input_root = Path(input_path).resolve()
        self.cache_root = (self.input_root if self.input_root.is_dir() else self.input_root.parent) / ".hcc_sempath_annotation_cache"
        self.packages: list[AnnotationPackage] = []
        self.state = AnnotationState(state_path, input_path)
        self._viewers: OrderedDict[int, IacViewerData] = OrderedDict()
        self._lock = threading.RLock()
        self._scan_done = False
        self._scan_error = ""
        self._scan_thread: threading.Thread | None = None
        self._thumbnail_token = ""
        if async_scan:
            self._scan_thread = threading.Thread(target=self._scan_packages, name="iac-scan", daemon=True)
            self._scan_thread.start()
        else:
            self.packages = discover_iac_packages(self.input_root)
            self._scan_done = True

    def _scan_packages(self) -> None:
        LOG.info("discover_iac_background_start input=%s", self.input_root)
        try:
            count = 0
            for path in _candidate_iac_paths(self.input_root):
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
            return {"done": self._scan_done, "error": self._scan_error, "packages": len(self.packages), "last_iac": self.state.last_iac}

    def package(self, index: int) -> AnnotationPackage:
        with self._lock:
            return self.packages[index]

    def activate_thumbnail_token(self, token: str) -> None:
        with self._lock:
            if token and token != self._thumbnail_token:
                LOG.info("thumbnail_token_activate token=%s", token)
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

    def viewer(self, index: int) -> IacViewerData:
        with self._lock:
            package = self.packages[index]
            cached = self._viewers.get(index)
            if cached is not None:
                self._viewers.move_to_end(index)
                return cached
        if cached is None:
            start = perf_counter()
            LOG.info("iac_open_start index=%d rel_path=%s path=%s", index, package.rel_path, package.path)
            viewer = IacViewerData(package.path)
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

    def package_json(self) -> list[dict]:
        items = []
        with self._lock:
            packages = list(self.packages)
        for idx, package in enumerate(packages):
            counts = self.state.lightweight_counts_for_package(package)
            items.append(
                {
                    "index": idx,
                    "name": Path(package.rel_path).name,
                    "rel_path": package.rel_path,
                    "dataset": package.dataset,
                    "total": counts["total"],
                    "annotated": counts["annotated"],
                    "remaining": counts["remaining"],
                }
            )
        LOG.info("api_packages count=%d", len(items))
        return items

    def progress(self, index: int) -> dict:
        with self._lock:
            packages = list(self.packages)
            package = self.packages[index]
        viewer = self.viewer(index)
        counts = self.state.counts_for_package(package, viewer.records)
        overall = {
            "annotated": sum(len(self.state.counted_annotations_for_package(item)) for item in packages),
            "total": sum(item.total for item in packages),
            "skipped": sum(
                1
                for item in packages
                for key in self.state.skipped
                if key.startswith(f"{item.rel_path}::")
            ),
        }
        overall["remaining"] = max(0, overall["total"] - overall["annotated"] - overall["skipped"])
        l1_counts = {name: 0 for name in self.state.l1_prototypes}
        l2_counts = {name: 0 for name in self.state.l2_prototypes}
        for counted_package in packages:
            for item in self.state.counted_annotations_for_package(counted_package):
                l1_counts[item["l1"]] = l1_counts.get(item["l1"], 0) + 1
                for label in item["l2"]:
                    l2_counts[label] = l2_counts.get(label, 0) + 1
        package_l1_counts = {name: 0 for name in self.state.l1_prototypes}
        package_l2_counts = {name: 0 for name in self.state.l2_prototypes}
        for item in self.state.counted_annotations_for_package(package):
            package_l1_counts[item["l1"]] = package_l1_counts.get(item["l1"], 0) + 1
            for label in item["l2"]:
                package_l2_counts[label] = package_l2_counts.get(label, 0) + 1
        LOG.info(
            "progress_read iac=%s annotated=%d total=%d remaining=%d overall_annotated=%d overall_total=%d",
            package.rel_path,
            counts["annotated"],
            counts["total"],
            counts["remaining"],
            overall["annotated"],
            overall["total"],
        )
        return {"package": counts, "overall": overall, "l1": l1_counts, "l2": l2_counts, "package_l1": package_l1_counts, "package_l2": package_l2_counts}

    def random_record(self, index: int) -> dict:
        with self._lock:
            package = self.packages[index]
        viewer = self.viewer(index)
        remaining = [record for record in viewer.records if not self.state.is_annotated(package, record)]
        if not remaining:
            LOG.info("random_tile_empty iac=%s", package.rel_path)
            return {"record": None}
        record = random.choice(remaining)
        LOG.info("random_tile iac=%s row=%d tile_id=%s remaining=%d", package.rel_path, record.row, record.tile_id, len(remaining))
        return {"record": viewer._record_json(record)}

    def select_nearest(self, index: int, x: float, y: float) -> dict:
        viewer = self.viewer(index)
        result = viewer.nearest("__all__", x, y)
        record = result.get("record")
        if record:
            with self._lock:
                package = self.packages[index]
            LOG.info("nearest_tile iac=%s x=%.1f y=%.1f row=%s tile_id=%s", package.rel_path, x, y, record.get("row"), record.get("tile_id"))
        return result

    def annotated_rows(self, index: int) -> set[int]:
        with self._lock:
            package = self.packages[index]
        viewer = self.viewer(index)
        return {record.row for record in viewer.records if self.state.is_annotated(package, record)}

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
                payload = read_payload(package.path, viewer.header, offsets[record.row].as_py(), lengths[record.row].as_py())
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

    def context_jpg(self, index: int, row: int) -> bytes:
        viewer = self.viewer(index)
        center = viewer._by_row[row]
        tile_w = max(1, int(viewer.header.get("tile_width", viewer.stride_x))) // 2
        tile_h = max(1, int(viewer.header.get("tile_height", viewer.stride_y))) // 2
        tile_w = max(1, tile_w)
        tile_h = max(1, tile_h)
        canvas = Image.new("RGB", (tile_w * 5, tile_h * 5), (245, 247, 249))
        draw = ImageDraw.Draw(canvas)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                record = viewer._image_lookup.get((center.slide_key, center.grid_x + dx, center.grid_y + dy))
                x = (dx + 2) * tile_w
                y = (dy + 2) * tile_h
                if record is None:
                    draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(210, 215, 220))
                    continue
                try:
                    image = Image.open(io.BytesIO(viewer.read_tile_png(record.row))).convert("RGB")
                    image = image.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                    canvas.paste(image, (x, y))
                except Exception:
                    LOG.exception("context_tile_decode_failed row=%d", record.row)
                    draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(210, 80, 80))
        center_x = 2 * tile_w
        center_y = 2 * tile_h
        draw.rectangle([center_x, center_y, center_x + tile_w - 1, center_y + tile_h - 1], outline=(220, 20, 20), width=4)
        draw.line([center_x, center_y + tile_h // 2, center_x + tile_w, center_y + tile_h // 2], fill=(220, 20, 20), width=2)
        draw.line([center_x + tile_w // 2, center_y, center_x + tile_w // 2, center_y + tile_h], fill=(220, 20, 20), width=2)
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=90)
        LOG.info("context_render index=%d row=%d bytes=%d", index, row, buffer.tell())
        return buffer.getvalue()


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
.layout{display:grid;grid-template-columns:300px minmax(0,1fr) 320px;height:100svh}.layout.queue-collapsed{grid-template-columns:0 minmax(0,1fr) 320px}.layout.labels-collapsed{grid-template-columns:300px minmax(0,1fr) 0}.layout.queue-collapsed.labels-collapsed{grid-template-columns:0 minmax(0,1fr) 0}
aside,.right{overflow:hidden;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column}.right{border-left:1px solid var(--line);border-right:0}.layout.queue-collapsed aside,.layout.labels-collapsed .right{border:0}.layout.queue-collapsed aside>*,.layout.labels-collapsed .right>*{display:none}
.panelHead{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px;border-bottom:1px solid var(--line)}.panelTitle{font-weight:650}.panelBody{overflow:auto;padding:12px}.labelBody{min-height:0;flex:1}.labelActions{border-top:1px solid var(--line);padding:10px 12px;background:var(--panel)}
main{min-width:0;display:grid;grid-template-rows:auto minmax(0,1fr);height:100svh}.topbar{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line);background:var(--panel)}.topbarTitle{min-width:0;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.workspace{min-height:0;overflow:auto;padding:12px}.pkg{position:relative;padding:8px;border:1px solid var(--line);margin-bottom:6px;cursor:pointer;background:#fff;overflow:hidden}.pkg.active{border-color:var(--blue);background:var(--blue-soft)}
.pkg>*{position:relative;z-index:1}.pkg::before{content:"";position:absolute;inset:0 auto 0 0;width:var(--pct,0%);background:#dff3eb;z-index:0}
.muted{color:var(--muted);font-size:12px}.thumbWrap{min-height:220px;border:1px solid #c7cbd1;background:white;overflow:auto}
.thumb{display:block;width:auto;height:auto;max-width:none;background:white;cursor:crosshair}.loading{padding:18px;color:#6b7280;font-size:12px}
.tileStage{position:relative;width:224px;height:224px;border:1px solid #c7cbd1;background:#fff}.tile{position:absolute;inset:0;width:224px;height:224px;object-fit:contain;background:#fff}.roiCanvas{position:absolute;inset:0;width:224px;height:224px;touch-action:none;cursor:crosshair}.roiTools{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}.roiTools button{min-height:32px;padding:4px 8px}.roiTools button.selected{background:var(--blue);color:#fff}.chips{display:grid;grid-template-columns:1fr;gap:6px;margin:8px 0 16px}
.panzoom{touch-action:none;transform-origin:0 0;will-change:transform;cursor:grab}.panzoom.dragging{cursor:grabbing}
button.chip{position:relative;text-align:left;border:1px solid #c7cbd1;background:#fff;padding:8px;cursor:pointer;overflow:hidden}button.chip::before{content:"";position:absolute;inset:0 auto 0 0;width:var(--pct,0%);background:#eef4ff;z-index:0}button.chip.selected{background:#1a73e8;color:white;border-color:#1a73e8}button.chip.selected::before{background:rgba(255,255,255,.18)}button.chip span{position:relative;z-index:1}.chipRow{display:flex;justify-content:space-between;gap:8px}.chipCount{font-variant-numeric:tabular-nums;color:#374151}button.chip.selected .chipCount{color:white}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.actions button{padding:8px 10px}.bar{height:8px;background:#e5e7eb;margin:6px 0 10px}.bar>div{height:8px;background:#18865b}
pre{white-space:pre-wrap;font-size:12px;background:#f1f3f4;padding:8px}
.authGate{position:fixed;inset:0;background:rgba(244,246,248,.96);z-index:10;display:none;align-items:center;justify-content:center;padding:18px}.authBox{width:min(420px,100%);background:#fff;border:1px solid var(--line);padding:16px;box-shadow:0 14px 40px rgba(15,23,42,.18)}.authBox h3{margin:0 0 10px}.authBox input{width:100%;min-height:42px;border:1px solid var(--line);padding:0 10px;margin-bottom:10px}.authBox .status{min-height:18px;color:var(--danger);font-size:12px}
.contextOverlay{position:fixed;inset:0;background:rgba(15,23,42,.86);z-index:9;display:flex;flex-direction:column}.contextBar{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;background:#fff}.contextStage{min-height:0;flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center}.contextStage img{max-width:none;max-height:none;background:#fff;border:1px solid rgba(255,255,255,.5)}
.hidden{display:none!important}
@media(max-width:760px){
  body{height:100svh;overflow:hidden}.layout,.layout.queue-collapsed,.layout.labels-collapsed,.layout.queue-collapsed.labels-collapsed{display:block;height:100svh}
  aside{position:fixed;inset:0 0 auto 0;z-index:5;max-height:45svh;border-right:0;border-bottom:1px solid var(--line);box-shadow:0 8px 24px rgba(15,23,42,.16)}.layout.queue-collapsed aside{display:none}
  .right{position:fixed;inset:auto 0 0 0;z-index:4;max-height:48svh;border-left:0;border-top:1px solid var(--line);box-shadow:0 -8px 24px rgba(15,23,42,.12)}.layout.labels-collapsed .right{display:none}
  main{height:100svh;grid-template-rows:auto minmax(0,1fr)}.topbar{position:sticky;top:0;z-index:3}.workspace{padding:8px}.tile{width:192px;height:192px}.thumbWrap{max-height:calc(100svh - 310px);min-height:160px}.actions{grid-template-columns:1fr 1fr}.panelBody{padding:10px}.labelActions{padding:8px 10px}.chips{gap:6px;margin-bottom:10px}button.chip{min-height:38px;padding:7px}
}
</style>
</head>
<body><div id="authGate" class="authGate"><div class="authBox"><h3>Annotation token</h3><input id="authInput" autocomplete="off"><button id="authSubmit" class="primary" type="button">Open</button><div id="authStatus" class="status"></div></div></div>
<div id="layout" class="layout"><aside><div class="panelHead"><div><div class="panelTitle">IAC packages</div><div id="packageSummary" class="muted"></div></div><button id="hideQueue" class="ghost" type="button">Hide</button></div><div id="packages" class="panelBody"></div></aside>
<main><div class="topbar"><button id="toggleQueue" type="button">Tiles</button><button id="toggleLabels" type="button">Labels</button><button id="contextBtn" type="button">Context</button><div id="title" class="topbarTitle">Prototype annotation</div></div><section class="workspace"><div class="muted" id="recordMeta"></div>
<p><div class="tileStage"><img id="tile" class="tile"><canvas id="roiCanvas" class="roiCanvas" width="224" height="224"></canvas></div></p><div class="roiTools"><button type="button" data-roi-tool="point">Point</button><button type="button" data-roi-tool="brush">Brush</button><button type="button" data-roi-tool="circle">Circle</button><button type="button" id="roiUndo">Undo</button><button type="button" id="roiClear">Clear</button><label><input id="roiComplete" type="checkbox"> all 10 L2 attributes completely reviewed</label></div><div id="roiStatus" class="muted">Select an L2 attribute, then annotate every visible focus. Complete review makes every unmarked attribute/token negative.</div><h3>Location overview</h3><div id="thumbWrap" class="thumbWrap"><div id="thumbLoading" class="loading">Loading overview...</div><img id="thumb" class="thumb panzoom"></div></section></main>
<section class="right"><div class="panelHead"><div class="panelTitle">Labels</div><button id="hideLabels" class="ghost" type="button">Hide</button></div><div class="panelBody labelBody"><h3>Progress</h3><div id="progress"></div><h3>L1 primary</h3><div id="l1" class="chips"></div>
<h3>L2 attributes</h3><div id="l2" class="chips"></div></div><div class="labelActions"><div class="actions"><button onclick="save()" class="primary">Save + next</button><button onclick="nextRandom()">Skip / random</button></div><pre id="status"></pre></div></section>
</div>
<div id="contextOverlay" class="contextOverlay hidden"><div class="contextBar"><div><b>5x5 context</b><div id="contextMeta" class="muted"></div></div><button id="contextClose" type="button">Close</button></div><div class="contextStage"><img id="contextImg" class="panzoom"></div></div>
<script>
let packages=[], pkg=0, current=null, l1="", l2=new Set(), l1Counts={}, l2Counts={}, restoredLastIac=false;
let roi=[],roiTool='point',roiAttribute='',roiDrawing=null,roiAllComplete=false;
const TOKEN_KEYS=['token','auth_token','access_token'];
let AUTH_TOKEN='';
function tokenFromUrl(){const params=new URLSearchParams(location.search); for(const key of TOKEN_KEYS){const value=params.get(key); if(value)return value;} return '';}
function setAuthToken(token){AUTH_TOKEN=token||''; if(AUTH_TOKEN)localStorage.setItem('hcc_sempath_annotation_token',AUTH_TOKEN);}
function authed(path){return path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(AUTH_TOKEN);}
async function api(path, opts){const r=await fetch(authed(path), opts); if(!r.ok) throw new Error(await r.text()); return r.headers.get('content-type')?.includes('json')?r.json():r.blob();}
function setQueueOpen(open){document.getElementById('layout').classList.toggle('queue-collapsed',!open);}
function setLabelsOpen(open){document.getElementById('layout').classList.toggle('labels-collapsed',!open);}
function isMobile(){return window.matchMedia('(max-width:760px)').matches;}
async function ensureAuth(){setAuthToken(tokenFromUrl()||localStorage.getItem('hcc_sempath_annotation_token')||''); if(!AUTH_TOKEN){document.getElementById('authGate').style.display='flex'; throw new Error('');}}
async function submitAuth(){setAuthToken(document.getElementById('authInput').value.trim()); try{await api('/api/scan-status'); document.getElementById('authGate').style.display='none'; refreshPackage().catch(e=>document.getElementById('status').textContent=e.message||String(e));}catch(e){document.getElementById('authStatus').textContent='Invalid token.';}}
function setupPanZoom(el,onClick){let scale=1,tx=0,ty=0,drag=false,sx=0,sy=0,stx=0,sty=0,moved=0; const apply=()=>{el.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`;}; el.addEventListener('wheel',ev=>{ev.preventDefault(); const next=Math.min(12,Math.max(.25,scale*(ev.deltaY<0?1.15:.87))); scale=next; apply();},{passive:false}); el.addEventListener('pointerdown',ev=>{drag=true;moved=0;sx=ev.clientX;sy=ev.clientY;stx=tx;sty=ty;el.classList.add('dragging');el.setPointerCapture(ev.pointerId);}); el.addEventListener('pointermove',ev=>{if(!drag)return; const dx=ev.clientX-sx,dy=ev.clientY-sy; moved=Math.max(moved,Math.abs(dx)+Math.abs(dy)); tx=stx+dx; ty=sty+dy; apply();}); el.addEventListener('pointerup',ev=>{if(!drag)return; drag=false; el.classList.remove('dragging'); if(moved<6&&onClick)onClick(ev);}); el.addEventListener('dblclick',ev=>{ev.preventDefault(); scale=1;tx=0;ty=0;apply();}); el.resetPanZoom=()=>{scale=1;tx=0;ty=0;apply();};}
function prototypeButton(name, selected, count, maxCount, onClick){const b=document.createElement('button'); const pct=maxCount?Math.round(count/maxCount*100):0; b.className='chip'+(selected?' selected':''); b.style.setProperty('--pct',pct+'%'); b.innerHTML=`<span class=chipRow><span>${name}</span><span class=chipCount>${count}</span></span>`; b.onclick=onClick; return b;}
function renderLabels(){const a=document.getElementById('l1'); a.innerHTML=''; const l1Max=Math.max(1,...Object.values(l1Counts)); L1.forEach(x=>a.appendChild(prototypeButton(x,l1===x,l1Counts[x]||0,l1Max,()=>{l1=x;renderLabels()}))); const b=document.getElementById('l2'); b.innerHTML=''; const l2Max=Math.max(1,...Object.values(l2Counts)); L2.forEach(x=>b.appendChild(prototypeButton(x,l2.has(x),l2Counts[x]||0,l2Max,()=>{roiAttribute=x;l2.add(x);renderLabels();renderRoi()}))); document.getElementById('roiComplete').checked=roiAllComplete;document.getElementById('roiStatus').textContent=roiAttribute?`ROI attribute: ${roiAttribute}`:'Select an L2 attribute, then annotate every visible focus.';}
function roiPoint(ev){const r=document.getElementById('roiCanvas').getBoundingClientRect();return{x:(ev.clientX-r.left)/r.width,y:(ev.clientY-r.top)/r.height}}
function renderRoi(){const c=document.getElementById('roiCanvas'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);for(const item of roi){const g=item.geometry;x.strokeStyle=item.attribute===roiAttribute?'#ff1f1f':'#ffb000';x.fillStyle='rgba(255,31,31,.25)';x.lineWidth=3;if(g.type==='point'){x.beginPath();x.arc(g.point[0]*224,g.point[1]*224,5,0,Math.PI*2);x.fill();x.stroke()}else if(g.type==='circle'){x.beginPath();x.arc(g.center[0]*224,g.center[1]*224,g.radius*224,0,Math.PI*2);x.fill();x.stroke()}else if(g.type==='brush'){x.beginPath();g.points.forEach((p,i)=>i?x.lineTo(p[0]*224,p[1]*224):x.moveTo(p[0]*224,p[1]*224));x.stroke()}}}
function setupRoi(){const c=document.getElementById('roiCanvas');c.addEventListener('pointerdown',ev=>{if(!roiAttribute)return;if(roiTool==='point'){const p=roiPoint(ev);roi.push({attribute:roiAttribute,state:'positive',review_complete:false,geometry:{type:'point',coordinate_space:'normalized',point:[p.x,p.y],radius:.035}});renderRoi();return}const p=roiPoint(ev);roiDrawing={start:p,points:[[p.x,p.y]]};c.setPointerCapture(ev.pointerId)});c.addEventListener('pointermove',ev=>{if(!roiDrawing)return;const p=roiPoint(ev);roiDrawing.points.push([p.x,p.y])});c.addEventListener('pointerup',ev=>{if(!roiDrawing)return;const p=roiPoint(ev);if(roiTool==='circle'){const dx=p.x-roiDrawing.start.x,dy=p.y-roiDrawing.start.y;roi.push({attribute:roiAttribute,state:'positive',review_complete:false,geometry:{type:'circle',coordinate_space:'normalized',center:[roiDrawing.start.x,roiDrawing.start.y],radius:Math.sqrt(dx*dx+dy*dy)}})}else{roi.push({attribute:roiAttribute,state:'positive',review_complete:false,geometry:{type:'brush',coordinate_space:'normalized',points:roiDrawing.points,width:.035}})}roiDrawing=null;renderRoi()})}
function setThumbnailSrc(){const img=document.getElementById('thumb'); const loading=document.getElementById('thumbLoading'); const token=`${Date.now()}-${pkg}`; const row=current?`&row=${current.row}`:''; img.dataset.token=String(token); loading.style.display='block'; loading.textContent='Building overview...'; img.style.display='none'; img.onload=()=>{if(img.dataset.token!==String(token)) return; loading.style.display='none'; img.style.display='block'; if(img.resetPanZoom)img.resetPanZoom();}; img.onerror=()=>{if(img.dataset.token===String(token)) loading.textContent='Overview failed to load.';}; img.src=authed(`/api/thumbnail?package=${pkg}${row}&thumb_token=${encodeURIComponent(token)}&t=${Date.now()}`);}
async function loadPackages(){packages=await api('/api/packages'); const scan=await api('/api/scan-status'); if(!restoredLastIac&&scan.last_iac){const found=packages.find(p=>p.rel_path===scan.last_iac); if(found)pkg=found.index; restoredLastIac=true;} document.getElementById('packageSummary').textContent=`${packages.length} IAC package${packages.length===1?'':'s'}${scan.done?'':' · scanning...'}`; if(scan.error){document.getElementById('status').textContent=scan.error;} const box=document.getElementById('packages'); box.innerHTML=''; packages.forEach(p=>{const pct=p.total?Math.round(p.annotated/p.total*100):0; const d=document.createElement('div'); d.className='pkg'+(p.index===pkg?' active':''); d.style.setProperty('--pct',pct+'%'); d.innerHTML=`<b>${p.name}</b><div class=muted>${p.dataset||'no dataset'} · ${p.annotated}/${p.total} · ${pct}%</div>`; d.onclick=()=>{pkg=p.index; restoredLastIac=true; if(isMobile())setQueueOpen(false); refreshPackage();}; box.appendChild(d)}); return scan;}
async function refreshPackage(){const scan=await loadPackages(); if(!packages.length){document.getElementById('recordMeta').textContent=scan.done?'No image-tile IAC packages found.':'Scanning IAC packages...'; setTimeout(refreshPackage,1000); return;} if(pkg>=packages.length) pkg=0; setThumbnailSrc(); await progress(); await nextRandom();}
async function progress(){const p=await api('/api/progress?package='+pkg); l1Counts=p.l1; l2Counts=p.l2; renderLabels(); const overallPct=p.overall.total?Math.round(p.overall.annotated/p.overall.total*100):0; const pkgPct=p.package.total?Math.round(p.package.annotated/p.package.total*100):0; document.getElementById('progress').innerHTML=`<div><b>${p.overall.annotated}/${p.overall.total}</b> tiles marked (${overallPct}%)</div><div class=bar><div style="width:${overallPct}%"></div></div><div class="muted">Skipped: ${p.overall.skipped} · Remaining: ${p.overall.remaining}</div><div class="muted">Current IAC: ${p.package.annotated}/${p.package.total} (${pkgPct}%) · skipped ${p.package.skipped}</div>`;}
async function showRecord(rec){current=rec; l1=""; l2=new Set();roi=[];roiAttribute='';roiAllComplete=false;document.getElementById('roiComplete').checked=false;renderLabels();renderRoi(); if(!rec){document.getElementById('recordMeta').textContent='All tiles in this IAC are annotated.'; document.getElementById('tile').removeAttribute('src'); return;} document.getElementById('recordMeta').textContent=`${packages[pkg].rel_path} · ${rec.tile_id} · x=${rec.x} y=${rec.y} row=${rec.row}`; const tile=document.getElementById('tile'); tile.src=authed(`/api/tile?package=${pkg}&row=${rec.row}`); setThumbnailSrc();}
async function nextRandom(){const r=await api('/api/random?package='+pkg); await showRecord(r.record);}
async function save(){if(!current){await nextRandom(); return;} if(!l1){document.getElementById('status').textContent='Select one L1 primary prototype first.'; return;}const roiPayload=[...roi,...(roiAllComplete?L2.map(attribute=>({attribute,state:'negative',review_complete:true})):[])]; await api('/api/annotation',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({package:pkg,row:current.row,l1,l2:[...l2],roi:roiPayload})}); document.getElementById('status').textContent='Saved.'; setThumbnailSrc(); await progress(); await loadPackages(); await nextRandom();}
async function openContext(){if(!current){document.getElementById('status').textContent='No tile selected.'; return;} const overlay=document.getElementById('contextOverlay'); const img=document.getElementById('contextImg'); document.getElementById('contextMeta').textContent=`${packages[pkg].rel_path} · ${current.tile_id}`; overlay.classList.remove('hidden'); img.onload=()=>{if(img.resetPanZoom)img.resetPanZoom();}; img.src=authed(`/api/context?package=${pkg}&row=${current.row}&t=${Date.now()}`);}
setupRoi();document.querySelectorAll('[data-roi-tool]').forEach(b=>b.addEventListener('click',()=>{roiTool=b.dataset.roiTool;document.querySelectorAll('[data-roi-tool]').forEach(x=>x.classList.toggle('selected',x===b))}));document.getElementById('roiUndo').onclick=()=>{roi.pop();renderRoi()};document.getElementById('roiClear').onclick=()=>{roi=roi.filter(x=>x.attribute!==roiAttribute);renderRoi()};document.getElementById('roiComplete').onchange=ev=>{roiAllComplete=ev.target.checked};document.querySelector('[data-roi-tool="point"]').classList.add('selected');
setupPanZoom(document.getElementById('contextImg'));
setupPanZoom(document.getElementById('thumb'),async ev=>{const img=ev.target, r=img.getBoundingClientRect(); const x=(ev.clientX-r.left)/r.width, y=(ev.clientY-r.top)/r.height; const rec=await api(`/api/nearest?package=${pkg}&rx=${x}&ry=${y}`); await showRecord(rec.record);});
document.getElementById('toggleQueue').addEventListener('click',()=>setQueueOpen(document.getElementById('layout').classList.contains('queue-collapsed')));
document.getElementById('hideQueue').addEventListener('click',()=>setQueueOpen(false));
document.getElementById('toggleLabels').addEventListener('click',()=>setLabelsOpen(document.getElementById('layout').classList.contains('labels-collapsed')));
document.getElementById('hideLabels').addEventListener('click',()=>setLabelsOpen(false));
document.getElementById('contextBtn').addEventListener('click',openContext);
document.getElementById('contextClose').addEventListener('click',()=>document.getElementById('contextOverlay').classList.add('hidden'));
document.getElementById('authSubmit').addEventListener('click',submitAuth);
document.getElementById('authInput').addEventListener('keydown',ev=>{if(ev.key==='Enter')submitAuth();});
const L1=%L1_JSON%; const L2=%L2_JSON%; renderLabels(); if(isMobile()){setQueueOpen(false); setLabelsOpen(false);} ensureAuth().then(()=>refreshPackage()).catch(e=>{if(e.message)document.getElementById('status').textContent=e.message||String(e);});
</script></body></html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict | list) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(data: AnnotationData, auth_token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    LOG.info("ui_open path=/")
                    html = HTML.replace("%L1_JSON%", json.dumps(data.state.l1_prototypes)).replace("%L2_JSON%", json.dumps(data.state.l2_prototypes))
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
                if parsed.path == "/api/packages":
                    _json_response(self, data.package_json())
                    return
                if parsed.path == "/api/scan-status":
                    _json_response(self, data.scan_status())
                    return
                index = int(qs.get("package", ["0"])[0])
                if parsed.path == "/api/progress":
                    _json_response(self, data.progress(index))
                    return
                if parsed.path == "/api/random":
                    _json_response(self, data.random_record(index))
                    return
                if parsed.path == "/api/nearest":
                    viewer = data.viewer(index)
                    rx = float(qs.get("rx", ["0"])[0])
                    ry = float(qs.get("ry", ["0"])[0])
                    bounds = viewer._bounds(viewer.records)
                    x = bounds[0] + rx * max(1, bounds[1] - bounds[0] + viewer.stride_x)
                    y = bounds[2] + ry * max(1, bounds[3] - bounds[2] + viewer.stride_y)
                    _json_response(self, data.select_nearest(index, x, y))
                    return
                if parsed.path == "/api/thumbnail":
                    token = qs.get("thumb_token", [""])[0]
                    selected_row = int(qs["row"][0]) if "row" in qs else None
                    body = data.thumbnail_jpg(index, token=token, selected_row=selected_row)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/context":
                    row = int(qs["row"][0])
                    body = data.context_jpg(index, row)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/tile":
                    row = int(qs["row"][0])
                    package = data.package(index)
                    LOG.info("tile_read iac=%s row=%d", package.rel_path, row)
                    body = data.viewer(index).read_tile_png(row)
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
            if parsed.path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                index = int(payload["package"])
                row = int(payload["row"])
                viewer = data.viewer(index)
                record = viewer._by_row[row]
                package = data.package(index)
                LOG.info("annotation_request iac=%s row=%d l1=%s l2=%s", package.rel_path, row, payload.get("l1"), payload.get("l2", []))
                data.state.save_annotation(
                    package,
                    record,
                    str(payload["l1"]),
                    list(payload.get("l2", [])),
                    list(payload.get("roi", [])),
                )
                _json_response(self, {"ok": True})
            except Exception as exc:
                LOG.exception("api_post_failed path=%s", parsed.path)
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Open a browser UI for L1/L2 prototype tile annotation.")
    parser.add_argument("--input", required=True, help="IAC file or directory containing image-tile IAC packages.")
    parser.add_argument("--state", required=True, help="JSON annotation state file; CSV is exported next to it.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Use 0 to pick a free port.")
    parser.add_argument("--token", default="", help="Annotation UI auth token; overrides the persisted state .auth-token when set.")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    data = AnnotationData(args.input, args.state)
    auth_token = _load_or_create_auth_token(args.state, args.token)
    port = _find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(data, auth_token))
    base_url = f"http://{args.host}:{port}/"
    url = f"{base_url}?token={auth_token}"
    LOG.info("server_start url=%s state=%s csv=%s", base_url, args.state, data.state.csv_path)
    print(f"annotation_ui url={url} state={args.state} csv={data.state.csv_path}", flush=True)
    if not args.no_open:
        LOG.info("browser_open url=%s", url)
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        data.close()
        server.server_close()


if __name__ == "__main__":
    main()
