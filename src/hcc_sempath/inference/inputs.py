"""Resolve raster, WSI, and IAC inputs into the SemPath tile-package contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from iatro.iac import read_header
from iatro.iac.adapters.manifests import TileRecord
from iatro.iac.adapters.tiles import build_tile_package_from_records, encode_jxl_array

from hcc_sempath.build.tiles import WSI_SUFFIXES
from hcc_sempath.build.wsi import build_wsi_iac
from hcc_sempath.iac_naming import (
    is_pathology_tile_name,
    pathology_prediction_path,
    pathology_tile_path,
    pathology_tile_stem,
)
from hcc_sempath.modeling.models import STUDENT_IMAGE_SIZE


RASTER_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SUPPORTED_RASTER_SIZES = {STUDENT_IMAGE_SIZE}


@dataclass(frozen=True)
class InputPlan:
    source_path: Path
    kind: str
    name: str
    tile_package: Path
    prediction_package: Path

    @property
    def creates_tile_package(self) -> bool:
        return self.kind != "iac"


@dataclass(frozen=True)
class MaterializationOptions:
    split: str
    target_mpp: float
    native_mpp: float | None
    native_mpp_y: float | None
    min_tissue_fraction: float
    prefilter_tissue_fraction: float
    white_threshold: int
    black_threshold: int
    mask_max_pixels: int
    max_tiles: int | None
    workers: int
    lossless: bool
    distance: float
    effort: int
    overwrite: bool
    show_progress: bool


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "pathology"


def _kind(path: Path, *, direct: bool) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".iac":
        if direct or is_pathology_tile_name(path):
            return "iac"
        return None
    if suffix in WSI_SUFFIXES:
        return "wsi"
    if suffix in RASTER_SUFFIXES:
        return "raster"
    return None


def _discover_sources(values: list[str]) -> list[tuple[Path, str]]:
    discovered: list[tuple[Path, str]] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            kind = _kind(path, direct=True)
            if kind is None:
                raise ValueError(f"unsupported inference input: {path}")
            discovered.append((path, kind))
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
            kind = _kind(candidate, direct=False)
            if kind is not None:
                discovered.append((candidate.resolve(), kind))
    unique = list(dict.fromkeys(discovered))
    if not unique:
        supported = ", ".join(sorted({".iac", *WSI_SUFFIXES, *RASTER_SUFFIXES}))
        raise ValueError(f"input did not resolve to a supported pathology source ({supported})")
    return unique


def _validate_tile_package(path: Path) -> None:
    header = read_header(path)
    if header.get("payload_type") != "image_tiles":
        raise ValueError(f"input IAC is not an image-tile package: {path}")
    size = (int(header.get("tile_height", 0)), int(header.get("tile_width", 0)))
    if size != (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE):
        raise ValueError(
            f"SemPath requires {STUDENT_IMAGE_SIZE}px tile payloads, got {size} from {path}"
        )


def plan_inputs(values: list[str], output_dir: str | Path) -> list[InputPlan]:
    output_dir = Path(output_dir).expanduser().resolve()
    plans: list[InputPlan] = []
    for source, kind in _discover_sources(values):
        if kind == "iac":
            _validate_tile_package(source)
            try:
                name = pathology_tile_stem(source)
            except ValueError:
                name = _safe_name(source.stem)
            tile_package = source
        else:
            name = _safe_name(source.stem)
            tile_package = pathology_tile_path(output_dir, name)
        plans.append(
            InputPlan(
                source_path=source,
                kind=kind,
                name=name,
                tile_package=tile_package,
                prediction_package=pathology_prediction_path(output_dir, name),
            )
        )

    collisions: dict[Path, list[Path]] = {}
    for plan in plans:
        collisions.setdefault(plan.prediction_package, []).append(plan.source_path)
    duplicates = {path: sources for path, sources in collisions.items() if len(sources) > 1}
    if duplicates:
        path, sources = next(iter(duplicates.items()))
        raise ValueError(
            f"multiple inputs resolve to {path.name}: "
            + ", ".join(str(source) for source in sources[:4])
        )
    return plans


def _matching_generated_package(plan: InputPlan) -> bool:
    try:
        header = read_header(plan.tile_package)
        _validate_tile_package(plan.tile_package)
    except Exception:
        return False
    source_path = str((header.get("source") or {}).get("path") or "")
    if not source_path:
        return False
    try:
        return Path(source_path).expanduser().resolve() == plan.source_path
    except OSError:
        return False


def _build_raster_package(plan: InputPlan, options: MaterializationOptions) -> None:
    with Image.open(plan.source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if width != height or width not in SUPPORTED_RASTER_SIZES:
            sizes = ", ".join(f"{size}x{size}" for size in sorted(SUPPORTED_RASTER_SIZES))
            raise ValueError(
                f"single-image inference requires a square {sizes} raster, got {width}x{height}: "
                f"{plan.source_path}"
            )
        crop_offset = (width - STUDENT_IMAGE_SIZE) // 2
        crop_box = (
            crop_offset,
            crop_offset,
            crop_offset + STUDENT_IMAGE_SIZE,
            crop_offset + STUDENT_IMAGE_SIZE,
        )
        model_image = image.crop(crop_box)

    array = np.asarray(model_image, dtype=np.uint8).copy()
    tile_id = f"{plan.name}_0000000"
    record = TileRecord(
        tile_id=tile_id,
        patient_id=plan.name,
        slide_id=plan.name,
        tile_path=Path(f"tiles/{tile_id}.jxl"),
        x=crop_offset,
        y=crop_offset,
        split=options.split,
    )
    payload = encode_jxl_array(array, lossless=True, distance=None, effort=options.effort)
    source_metadata: dict[str, object] = {
        "path": str(plan.source_path),
        "bytes": plan.source_path.stat().st_size,
        "width": width,
        "height": height,
        "input_kind": "single_raster",
    }
    if options.native_mpp is not None:
        source_metadata["native_mpp_x"] = options.native_mpp
        source_metadata["native_mpp_y"] = options.native_mpp_y or options.native_mpp
        source_metadata["native_mpp_source"] = "argument"
    build_tile_package_from_records(
        records=[record],
        payloads=[payload],
        output_path=plan.tile_package,
        tile_width=STUDENT_IMAGE_SIZE,
        tile_height=STUDENT_IMAGE_SIZE,
        stride_x=1,
        stride_y=1,
        lossless=True,
        effort=options.effort,
        overwrite=options.overwrite,
        extra_header={
            "source": source_metadata,
            "tiling": {
                "input_kind": "single_raster",
                "original_width": width,
                "original_height": height,
                "crop_box": list(crop_box),
                "level_read_width": STUDENT_IMAGE_SIZE,
                "level_read_height": STUDENT_IMAGE_SIZE,
                "level_downsample": 1.0,
                "retained_tiles": 1,
            },
        },
    )


def materialize_input(plan: InputPlan, options: MaterializationOptions) -> Path:
    if plan.kind == "iac":
        return plan.tile_package
    plan.tile_package.parent.mkdir(parents=True, exist_ok=True)
    if plan.tile_package.exists() and not options.overwrite:
        if _matching_generated_package(plan):
            return plan.tile_package
        raise FileExistsError(
            f"tile output already exists and does not match the source; pass --overwrite: "
            f"{plan.tile_package}"
        )
    if plan.kind == "raster":
        _build_raster_package(plan, options)
    elif plan.kind == "wsi":
        build_wsi_iac(
            wsi_path=plan.source_path,
            output_path=plan.tile_package,
            patient_id=plan.name,
            slide_id=plan.name,
            split=options.split,
            target_mpp=options.target_mpp,
            native_mpp=options.native_mpp,
            native_mpp_y=options.native_mpp_y,
            tile_size=STUDENT_IMAGE_SIZE,
            min_tissue_fraction=options.min_tissue_fraction,
            max_tiles=options.max_tiles,
            lossless=options.lossless,
            distance=options.distance,
            effort=options.effort,
            workers=options.workers,
            white_threshold=options.white_threshold,
            black_threshold=options.black_threshold,
            prefilter_tissue_fraction=options.prefilter_tissue_fraction,
            mask_max_pixels=options.mask_max_pixels,
            overwrite=options.overwrite,
            show_progress=options.show_progress,
        )
    else:  # pragma: no cover - InputPlan construction owns this invariant.
        raise ValueError(f"unsupported planned input kind: {plan.kind}")
    _validate_tile_package(plan.tile_package)
    return plan.tile_package
