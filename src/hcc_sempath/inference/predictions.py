from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa

from iatro.iac import Codec, VariableRecordPack, read_tables


SCHEMA_VERSION = 1
PAYLOAD_TYPE = "hcc_sempath_tile_predictions"
_PAYLOAD_MAGIC = b"HSP1"
_IAC_MANAGED_HEADER_FIELDS = frozenset(
    {
        "format",
        "version",
        "header_bytes",
        "payload_type",
        "codec",
        "codec_params",
        "checksum",
        "num_slides",
        "num_records",
        "slide_table_offset",
        "slide_table_length",
        "index_table_offset",
        "index_table_length",
        "data_offset",
        "data_length",
    }
)


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def source_index_sha256(header: dict, slide_table: pa.Table, index_table: pa.Table) -> str:
    """Digest stable source identity without rereading all image payload bytes."""

    semantic_header = {
        key: header.get(key)
        for key in (
            "coordinate_mode",
            "origin",
            "tile_width",
            "tile_height",
            "stride_x",
            "stride_y",
            "source",
            "tiling",
            "num_records",
        )
    }
    digest = hashlib.sha256(
        json.dumps(
            semantic_header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for table in (slide_table, index_table):
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        digest.update(sink.getvalue().to_pybytes())
    return digest.hexdigest()


def _probability_dtype(name: str) -> np.dtype:
    if name not in {"uint8", "uint16", "float16"}:
        raise ValueError(f"unsupported spatial probability dtype: {name}")
    return np.dtype("<f2" if name == "float16" else name)


def _encode_probability(array: np.ndarray, dtype_name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("prediction probabilities must be finite and in [0, 1]")
    if dtype_name == "float16":
        return array.astype("<f2")
    dtype = _probability_dtype(dtype_name)
    maximum = np.iinfo(dtype).max
    return np.rint(array * maximum).astype(dtype)


def _decode_probability(array: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "float16":
        return array.astype(np.float32)
    maximum = np.iinfo(_probability_dtype(dtype_name)).max
    return array.astype(np.float32) / float(maximum)


def encode_prediction_payload(
    classification: np.ndarray,
    instance: np.ndarray,
    abundance: np.ndarray,
    *,
    spatial_dtype: str,
) -> bytes:
    classification = _encode_probability(classification, "float16")
    instance = _encode_probability(instance, spatial_dtype)
    abundance = _encode_probability(abundance, spatial_dtype)
    raw = b"".join(
        (
            _PAYLOAD_MAGIC,
            struct.pack("<III", classification.size, instance.size, abundance.size),
            classification.tobytes(order="C"),
            instance.tobytes(order="C"),
            abundance.tobytes(order="C"),
        )
    )
    return raw


def decode_prediction_payload(payload: bytes, header: dict) -> dict[str, np.ndarray]:
    raw = payload
    if raw[:4] != _PAYLOAD_MAGIC:
        raise ValueError("invalid SemPath prediction payload magic")
    classification_size, instance_size, abundance_size = struct.unpack_from("<III", raw, 4)
    class_count = len(header["classification_class_names"])
    component_count = len(header["spatial_component_names"])
    grid_h, grid_w = (int(value) for value in header["spatial_grid_shape"])
    expected_spatial = component_count * grid_h * grid_w
    if classification_size != class_count or instance_size != expected_spatial or abundance_size != expected_spatial:
        raise ValueError("prediction payload shape does not match package header")
    offset = 16
    classification_bytes = classification_size * 2
    classification = np.frombuffer(
        raw, dtype="<f2", count=classification_size, offset=offset
    ).astype(np.float32)
    offset += classification_bytes
    spatial_dtype_name = str(header["spatial_probability_encoding"]["dtype"])
    spatial_dtype = _probability_dtype(spatial_dtype_name)
    spatial_bytes = expected_spatial * spatial_dtype.itemsize
    instance = np.frombuffer(raw, dtype=spatial_dtype, count=expected_spatial, offset=offset)
    offset += spatial_bytes
    abundance = np.frombuffer(raw, dtype=spatial_dtype, count=expected_spatial, offset=offset)
    offset += spatial_bytes
    if offset != len(raw):
        raise ValueError("prediction payload has trailing or missing bytes")
    shape = (component_count, grid_h, grid_w)
    return {
        "classification_probabilities": classification,
        "spatial_instance_probabilities": _decode_probability(instance, spatial_dtype_name).reshape(shape),
        "spatial_abundance_probabilities": _decode_probability(abundance, spatial_dtype_name).reshape(shape),
    }


def prediction_index_table(
    source_index: pa.Table,
    rows: list[int],
    *,
    split: str | None = None,
) -> pa.Table:
    selected = source_index.take(pa.array(rows, type=pa.int64()))
    required = ["slide_idx", "tile_x", "tile_y", "tile_id", "split"]
    missing = [name for name in required if name not in selected.column_names]
    if missing:
        raise ValueError(f"source tile index is missing fields: {missing}")
    columns = [selected.column(name) for name in required[:-1]]
    columns.append(
        selected.column("split")
        if split is None
        else pa.array([str(split)] * len(rows), type=pa.string())
    )
    columns.append(selected.column("split"))
    columns.append(pa.array(rows, type=pa.uint32()))
    if "crc32" in selected.column_names:
        columns.append(selected.column("crc32"))
        names = [*required, "source_split", "source_row", "source_payload_crc32"]
    else:
        names = [*required, "source_split", "source_row"]
    return pa.Table.from_arrays(columns, names=names)


def prediction_header(
    *,
    source_path: str | Path,
    source_header: dict,
    source_index_digest: str,
    checkpoint_path: str | Path,
    checkpoint_file_digest: str,
    checkpoint_model_digest: str,
    classification_names: list[str],
    component_names: list[str],
    grid_shape: tuple[int, int],
    spatial_stride: int,
    patch_size: int,
    patch_padding: int,
    spatial_dtype: str,
    dataset_split: str,
) -> dict:
    tiling = dict(source_header.get("tiling") or {})
    source = dict(source_header.get("source") or {})
    source.pop("path", None)
    level_downsample = float(tiling.get("level_downsample", 1.0))
    tile_width = int(source_header["tile_width"])
    tile_height = int(source_header["tile_height"])
    read_width = int(tiling.get("level_read_width", tile_width))
    read_height = int(tiling.get("level_read_height", tile_height))
    dtype = _probability_dtype(spatial_dtype)
    quantization_error = (
        None
        if spatial_dtype == "float16"
        else 0.5 / float(np.iinfo(dtype).max)
    )
    return {
        "payload_type": PAYLOAD_TYPE,
        "schema_version": SCHEMA_VERSION,
        "created_by": "hcc-sempath",
        "source_package_name": Path(source_path).name,
        "source_dataset": Path(source_path).parent.name,
        "dataset_split": str(dataset_split),
        "source_iac_index_sha256": source_index_digest,
        "source": source,
        "source_tiling": tiling,
        "coordinate_mode": source_header.get("coordinate_mode", "pixel"),
        "origin": source_header.get("origin", "top_left"),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "stride_x": int(source_header.get("stride_x", tile_width)),
        "stride_y": int(source_header.get("stride_y", tile_height)),
        "coordinate_transform": {
            "orientation": "identity_top_left_x_right_y_down",
            "tile_origin_units": "wsi_level0_pixels",
            "model_pixel_to_level0_scale_x": level_downsample * read_width / tile_width,
            "model_pixel_to_level0_scale_y": level_downsample * read_height / tile_height,
            "grid_cell_center_model_pixel": {
                "x": f"column*{spatial_stride} - {patch_padding} + {patch_size}/2",
                "y": f"row*{spatial_stride} - {patch_padding} + {patch_size}/2",
            },
        },
        "classification_class_names": classification_names,
        "classification_probability_dtype": "float16",
        "spatial_component_names": component_names,
        "spatial_heads": ["instance", "abundance"],
        "spatial_grid_shape": list(grid_shape),
        "spatial_output_stride": int(spatial_stride),
        "spatial_patch_size": int(patch_size),
        "spatial_patch_padding": int(patch_padding),
        "spatial_probability_encoding": {
            "dtype": spatial_dtype,
            "range": [0.0, 1.0],
            "decode": "value / dtype_max" if spatial_dtype != "float16" else "IEEE-754 float16",
            "maximum_absolute_quantization_error": quantization_error,
        },
        "checkpoint_name": Path(checkpoint_path).name,
        "checkpoint_sha256": checkpoint_file_digest,
        "checkpoint_model_sha256": checkpoint_model_digest,
    }


def write_prediction_package(
    output_path: str | Path,
    *,
    header: dict,
    slide_table: pa.Table,
    index_table: pa.Table,
    payloads: Iterable[bytes],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        extra_header = dict(header)
        payload_type = str(extra_header.get("payload_type", PAYLOAD_TYPE))
        if payload_type != PAYLOAD_TYPE:
            raise ValueError(f"unexpected prediction payload type: {payload_type}")
        managed = sorted(_IAC_MANAGED_HEADER_FIELDS & extra_header.keys())
        if managed != ["payload_type"] and managed:
            raise ValueError(
                "prediction header must not define IAC-managed fields: "
                + ", ".join(managed)
            )
        extra_header.pop("payload_type", None)
        VariableRecordPack.build(
            temporary,
            payload_type=PAYLOAD_TYPE,
            slide_table=slide_table,
            index_table=index_table,
            objects=payloads,
            codec=Codec.create(Codec.ZSTD, level=6),
            extra_header=extra_header,
            overwrite=False,
            streaming=True,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(output_path)


class PredictionPackageReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader = VariableRecordPack(self.path)
        if self._reader.header.get("payload_type") != PAYLOAD_TYPE:
            raise ValueError(f"not a SemPath prediction package: {path}")
        if int(self._reader.header.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported SemPath prediction schema")

    @property
    def header(self) -> dict:
        return self._reader.header

    @property
    def slide_table(self) -> pa.Table:
        return self._reader.slide_table

    @property
    def index_table(self) -> pa.Table:
        return self._reader.index_table

    @property
    def record_count(self) -> int:
        return len(self._reader.index_table)

    def read_at(self, row: int) -> dict[str, np.ndarray]:
        return decode_prediction_payload(self._reader.read(row), self.header)

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> "PredictionPackageReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def grid_cell_center_level0(
    header: dict,
    *,
    tile_x: int,
    tile_y: int,
    row: int,
    column: int,
) -> tuple[float, float]:
    grid_h, grid_w = (int(value) for value in header["spatial_grid_shape"])
    if not (0 <= row < grid_h and 0 <= column < grid_w):
        raise IndexError(f"grid cell outside {grid_h}x{grid_w}: row={row} column={column}")
    origin_x = tile_x * int(header["stride_x"]) if header["coordinate_mode"] == "tile_grid" else tile_x
    origin_y = tile_y * int(header["stride_y"]) if header["coordinate_mode"] == "tile_grid" else tile_y
    stride = int(header["spatial_output_stride"])
    padding = int(header["spatial_patch_padding"])
    patch_size = int(header["spatial_patch_size"])
    model_x = column * stride - padding + patch_size / 2.0
    model_y = row * stride - padding + patch_size / 2.0
    transform = header["coordinate_transform"]
    return (
        origin_x + model_x * float(transform["model_pixel_to_level0_scale_x"]),
        origin_y + model_y * float(transform["model_pixel_to_level0_scale_y"]),
    )
