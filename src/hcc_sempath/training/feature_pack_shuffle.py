from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow as pa

from ..io.iatrocache import PackReader, build_pack, build_pack_data_segment_from_file, read_header, read_tables
from .feature_pack_merge import MERGED_FEATURE_PAYLOAD_TYPE


ROW_ORDER_SEED_KEY = "row_order_seed"
ROW_ORDER_MODE_KEY = "row_order_mode"
ROW_ORDER_SOURCE_KEY = "row_order_source"


def maybe_prepare_shuffled_iac_packages(
    *,
    cfg: dict,
    split: str,
    tile_packages: list[str],
    teacher_package_paths: dict[str, list[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    data_cfg = cfg.get("data", {})
    if not bool(data_cfg.get("auto_shuffle_iac_rows", False)):
        return tile_packages, teacher_package_paths
    seed = int(data_cfg.get("iac_row_order_seed", cfg.get("runtime", {}).get("seed", 13)))
    if seed <= 0:
        raise ValueError(f"data.iac_row_order_seed must be positive when auto_shuffle_iac_rows is enabled: {seed}")
    if not teacher_package_paths:
        return tile_packages, teacher_package_paths

    teacher_names = list(teacher_package_paths)
    worker_count = _prepare_worker_count(data_cfg, len(tile_packages))

    def prepare_one(package_idx: int) -> tuple[int, str, str, str]:
        tile_path = Path(tile_packages[package_idx])
        feature_paths = {name: Path(teacher_package_paths[name][package_idx]) for name in teacher_names}
        unique_feature_paths = sorted(set(feature_paths.values()))
        if len(unique_feature_paths) != 1:
            raise ValueError(
                "auto_shuffle_iac_rows requires merged teacher feature packages before row shuffling: "
                f"split={split} tile={tile_path}"
            )
        feature_path = unique_feature_paths[0]
        _ensure_merged_feature_path(feature_path)
        _prepare_tile_feature_pair(tile_path=tile_path, feature_path=feature_path, seed=seed)
        return (
            package_idx,
            str(tile_path),
            str(feature_path),
            f"iac_row_order_ready split={split} seed={seed} tile={tile_path.name} feature={feature_path.name}",
        )

    if worker_count == 1:
        results = [prepare_one(package_idx) for package_idx in range(len(tile_packages))]
    else:
        print(f"iac_row_order_prepare split={split} seed={seed} workers={worker_count} packages={len(tile_packages)}")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(prepare_one, range(len(tile_packages))))

    shuffled_tile_packages: list[str] = []
    shuffled_teacher_packages = {name: [] for name in teacher_names}
    for _, tile_path, feature_path, message in sorted(results, key=lambda item: item[0]):
        print(message)
        shuffled_tile_packages.append(tile_path)
        for name in teacher_names:
            shuffled_teacher_packages[name].append(feature_path)
    return shuffled_tile_packages, shuffled_teacher_packages


def _prepare_worker_count(data_cfg: dict, package_count: int) -> int:
    workers = int(data_cfg.get("auto_iac_prepare_workers", 1) or 1)
    return max(1, min(int(package_count), workers))


def _ensure_merged_feature_path(path: Path) -> None:
    header = read_header(path)
    if header.get("payload_type") != MERGED_FEATURE_PAYLOAD_TYPE:
        raise ValueError(f"not a merged teacher feature package for row shuffling: {path}")


def _header_seed(path: Path) -> int:
    header = read_header(path)
    return int(header.get(ROW_ORDER_SEED_KEY, 0) or 0)


def _target_permutation(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(count).astype(np.int64, copy=False)


def _tile_ids(path: Path) -> list[str]:
    _, _, table = read_tables(path)
    return [str(value) for value in table.column("tile_id").to_pylist()]


def _prepare_tile_feature_pair(*, tile_path: Path, feature_path: Path, seed: int) -> None:
    tile_header = read_header(tile_path)
    feature_header = read_header(feature_path)
    tile_seed = int(tile_header.get(ROW_ORDER_SEED_KEY, 0) or 0)
    feature_seed = int(feature_header.get(ROW_ORDER_SEED_KEY, 0) or 0)
    nonzero = {value for value in (tile_seed, feature_seed) if value != 0}
    if any(value != seed for value in nonzero):
        raise ValueError(
            f"existing row_order_seed conflicts with requested seed={seed}: "
            f"tile={tile_seed} feature={feature_seed} tile_path={tile_path} feature_path={feature_path}"
        )
    if tile_seed == seed and feature_seed == seed:
        _validate_pair_order(tile_path, feature_path)
        return

    tile_ids = _tile_ids(tile_path)
    feature_ids = _tile_ids(feature_path)
    if tile_seed == 0 and feature_seed == 0:
        if tile_ids != feature_ids:
            raise ValueError(f"unshuffled tile/feature tile_id order mismatch: tile={tile_path} feature={feature_path}")
        permutation = _target_permutation(len(tile_ids), seed)
    elif tile_seed == 0 and feature_seed == seed:
        permutation = _permutation_to_match(tile_ids, feature_ids, tile_path, feature_path)
    elif tile_seed == seed and feature_seed == 0:
        permutation = _permutation_to_match(feature_ids, tile_ids, feature_path, tile_path)
    else:
        raise ValueError(
            f"inconsistent row_order_seed state: tile={tile_seed} feature={feature_seed} "
            f"tile_path={tile_path} feature_path={feature_path}"
        )

    tile_tmp = None
    feature_tmp = None
    try:
        if tile_seed == 0:
            tile_tmp = _rewrite_tile_pack(tile_path, permutation, seed)
        if feature_seed == 0:
            feature_tmp = _rewrite_feature_pack(feature_path, permutation, seed)
        candidate_tile = tile_tmp or tile_path
        candidate_feature = feature_tmp or feature_path
        _validate_pair_order(candidate_tile, candidate_feature)
        if tile_tmp is not None:
            tile_tmp.replace(tile_path)
        if feature_tmp is not None:
            feature_tmp.replace(feature_path)
        _validate_pair_order(tile_path, feature_path)
    finally:
        for path in (tile_tmp, feature_tmp):
            if path is not None and path.exists():
                path.unlink()


def _permutation_to_match(source_ids: list[str], target_ids: list[str], source_path: Path, target_path: Path) -> np.ndarray:
    index = {tile_id: idx for idx, tile_id in enumerate(source_ids)}
    if len(index) != len(source_ids):
        raise ValueError(f"duplicate tile_id values in source package: {source_path}")
    try:
        return np.asarray([index[tile_id] for tile_id in target_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"tile_id sets differ: source={source_path} target={target_path}") from exc


def _temporary_output_path(path: Path, suffix: str) -> Path:
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    tmp_path.unlink()
    return tmp_path


def _shuffle_table(table: pa.Table, permutation: np.ndarray) -> pa.Table:
    return table.take(pa.array(permutation, type=pa.int64()))


def _rewrite_tile_pack(path: Path, permutation: np.ndarray, seed: int) -> Path:
    header, slide_table, record_table = read_tables(path)
    if header.get("payload_type") != "image_tiles":
        raise ValueError(f"not an image tile package: {path}")
    reader = PackReader(path)
    try:
        payloads = [reader.read_payload(int(row)) for row in permutation]
    finally:
        reader.close()
    shuffled_table = _shuffle_table(record_table, permutation)
    tmp_path = _temporary_output_path(path, ".rowshuffle.tiles.iac")
    build_pack(
        tmp_path,
        _row_order_header(header, seed),
        slide_table,
        shuffled_table,
        payloads,
        overwrite=False,
    )
    _validate_seed(tmp_path, seed)
    return tmp_path


def _rewrite_feature_pack(path: Path, permutation: np.ndarray, seed: int) -> Path:
    header, slide_table, record_table = read_tables(path)
    if header.get("payload_type") != MERGED_FEATURE_PAYLOAD_TYPE:
        raise ValueError(f"not a merged teacher feature package: {path}")
    record_bytes = int(header["merged_record_bytes"])
    reader = PackReader(path)
    data_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.data.", delete=False) as data_tmp:
            data_path = Path(data_tmp.name)
            for row in permutation:
                payload = reader.read_data_span(int(row) * record_bytes, record_bytes)
                data_tmp.write(payload)
        expected_length = len(permutation) * record_bytes
        if data_path.stat().st_size != expected_length:
            raise ValueError(f"shuffled feature data size mismatch: expected={expected_length} got={data_path.stat().st_size}")
        tmp_path = _temporary_output_path(path, ".rowshuffle.features.iac")
        build_pack_data_segment_from_file(
            tmp_path,
            _row_order_header(header, seed),
            slide_table,
            _shuffle_table(record_table, permutation),
            data_path,
            data_length=expected_length,
            overwrite=False,
        )
        _validate_seed(tmp_path, seed)
        return tmp_path
    finally:
        reader.close()
        if data_path is not None and data_path.exists():
            data_path.unlink()


def _row_order_header(header: dict, seed: int) -> dict:
    result = dict(header)
    result[ROW_ORDER_SEED_KEY] = int(seed)
    result[ROW_ORDER_MODE_KEY] = "shuffled"
    result[ROW_ORDER_SOURCE_KEY] = "hcc-sempath-auto-shuffle"
    return result


def _validate_seed(path: Path, seed: int) -> None:
    got = _header_seed(path)
    if got != int(seed):
        raise ValueError(f"row_order_seed validation failed: path={path} expected={seed} got={got}")


def _validate_pair_order(tile_path: Path, feature_path: Path) -> None:
    tile_header, _, tile_table = read_tables(tile_path)
    feature_header, _, feature_table = read_tables(feature_path)
    if tile_header.get("payload_type") != "image_tiles":
        raise ValueError(f"not an image tile package: {tile_path}")
    if feature_header.get("payload_type") != MERGED_FEATURE_PAYLOAD_TYPE:
        raise ValueError(f"not a merged teacher feature package: {feature_path}")
    if int(tile_header["num_records"]) != int(feature_header["num_records"]):
        raise ValueError(
            f"tile/feature record count mismatch: tile={tile_header['num_records']} "
            f"feature={feature_header['num_records']} tile_path={tile_path} feature_path={feature_path}"
        )
    tile_ids = [str(value) for value in tile_table.column("tile_id").to_pylist()]
    feature_ids = [str(value) for value in feature_table.column("tile_id").to_pylist()]
    if tile_ids != feature_ids:
        raise ValueError(f"tile/feature row order mismatch: tile={tile_path} feature={feature_path}")
    tile_seed = int(tile_header.get(ROW_ORDER_SEED_KEY, 0) or 0)
    feature_seed = int(feature_header.get(ROW_ORDER_SEED_KEY, 0) or 0)
    if tile_seed != feature_seed:
        raise ValueError(
            f"tile/feature row_order_seed mismatch: tile_seed={tile_seed} feature_seed={feature_seed} "
            f"tile={tile_path} feature={feature_path}"
        )
