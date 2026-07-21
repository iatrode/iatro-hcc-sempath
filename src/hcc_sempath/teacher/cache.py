from __future__ import annotations

import argparse
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import csv
import gc
import json
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file as load_safetensors_file
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset
from tqdm import tqdm
from iatro.iac.adapters.features import build_teacher_feature_package, build_teacher_feature_package_from_tile_package
from iatro.iac import read_tables
from iatro.iac.adapters.manifests import TileRecord
from iatro.iac.adapters.tiles import TilePackageReader, read_package_manifest, read_package_metadata
from iatro.iac.adapters.validate import validate_package


_THREAD_LOCAL = threading.local()


TEACHER_MODEL_PRESETS: dict[str, dict] = {
    "gigapath": {
        "model_name": "hf_hub:prov-gigapath/prov-gigapath",
        "teacher_name": "gigapath",
        "description": "Planned supported Prov-GigaPath teacher.",
    },
    "uni2_h": {
        "model_name": "hf-hub:MahmoodLab/UNI2-h",
        "teacher_name": "uni2_h",
        "description": "Supported UNI2-h teacher.",
        "model_kwargs": {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
    },
    "virchow2": {
        "model_name": "hf-hub:paige-ai/Virchow2",
        "teacher_name": "virchow2",
        "description": "Supported Virchow2 teacher using class-token plus mean patch-token features.",
        "feature_mode": "virchow2",
        "model_kwargs": {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        },
    },
    "h_optimus_1": {
        "model_name": "hf-hub:bioptimus/H-optimus-1",
        "teacher_name": "h_optimus_1",
        "description": "Supported H-optimus-1 teacher.",
    },
}


def _preset_help() -> str:
    lines = [
        "Planned supported presets for --teacher:",
    ]
    for name, item in TEACHER_MODEL_PRESETS.items():
        lines.append(
            f"  {name}: model_name={item['model_name']} teacher_name={item['teacher_name']} - {item['description']}"
        )
    lines.append("")
    lines.append(
        "Advanced: custom timm names, hf_hub:* names, and local model directories are accepted, "
        "but are not part of the planned supported/tested teacher set."
    )
    return "\n".join(lines)


def _resolve_model_spec(model_name: str) -> dict:
    preset = TEACHER_MODEL_PRESETS.get(model_name)
    if preset:
        return {
            "model_name": preset["model_name"],
            "model_kwargs": dict(preset.get("model_kwargs", {})),
            "feature_mode": preset.get("feature_mode", "default"),
        }
    model_path = Path(model_name)
    local_preset = TEACHER_MODEL_PRESETS.get(model_path.name)
    if model_path.is_dir() and local_preset:
        return {
            "model_name": model_name,
            "model_kwargs": dict(local_preset.get("model_kwargs", {})),
            "feature_mode": local_preset.get("feature_mode", "default"),
        }
    return {
        "model_name": model_name,
        "model_kwargs": {},
        "feature_mode": "default",
    }


def _pool_virchow2_features(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 3 or output.shape[1] < 6:
        raise ValueError(f"expected Virchow2 token output with shape [batch, tokens, dim], got {tuple(output.shape)}")
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def _decode_model_args(model_args: dict) -> dict:
    decoded = dict(model_args)
    named_objects = {
        "timm.layers.SwiGLUPacked": timm.layers.SwiGLUPacked,
        "torch.nn.SiLU": torch.nn.SiLU,
    }
    for key, value in list(decoded.items()):
        if isinstance(value, str) and value in named_objects:
            decoded[key] = named_objects[value]
    return decoded


def _local_config_path(model_path: Path) -> Path:
    local_config_path = model_path / "hcc_sempath_model.json"
    if local_config_path.exists():
        return local_config_path
    return model_path / "config.json"


def _local_weight_path(model_path: Path, config: dict) -> Path:
    configured = config.get("weight_path")
    candidates = [configured] if configured else []
    candidates.extend(["pytorch_model.bin", "model.safetensors"])
    candidates.extend(path.name for path in sorted(model_path.glob("*.safetensors")))
    for candidate in candidates:
        if not candidate:
            continue
        weight_path = model_path / candidate
        if weight_path.exists():
            return weight_path
    raise FileNotFoundError(f"missing local teacher weights under {model_path}")


def _load_state_dict(weight_path: Path) -> dict:
    if weight_path.suffix == ".safetensors":
        return load_safetensors_file(str(weight_path), device="cpu")
    return torch.load(weight_path, map_location="cpu", weights_only=False)


class TimmTeacherEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        model_kwargs: dict | None = None,
        feature_mode: str = "default",
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        model_kwargs = dict(model_kwargs or {})
        model_path = Path(model_name)
        if model_path.is_dir():
            config_path = _local_config_path(model_path)
            if not config_path.exists():
                raise FileNotFoundError(f"missing local teacher config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architecture = config.get("architecture")
            if not architecture:
                raise ValueError(f"local teacher config missing architecture: {config_path}")
            model_args = _decode_model_args(config.get("model_args", {}))
            for key in ("num_classes", "global_pool", "img_size", "patch_size"):
                if key in config and key not in model_args:
                    model_args[key] = config[key]
            pretrained_cfg = config.get("pretrained_cfg")
            if "img_size" not in model_args and isinstance(pretrained_cfg, dict):
                input_size = pretrained_cfg.get("input_size")
                if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
                    model_args["img_size"] = int(input_size[1])
            model_args.update(model_kwargs)
            model_args.setdefault("num_classes", 0)
            self.model = timm.create_model(architecture, pretrained=False, **model_args)
            if isinstance(pretrained_cfg, dict):
                self.model.pretrained_cfg = {**getattr(self.model, "pretrained_cfg", {}), **config["pretrained_cfg"]}
            if pretrained:
                weight_path = _local_weight_path(model_path, config)
                state_dict = _load_state_dict(weight_path)
                self.model.load_state_dict(state_dict, strict=False)
            self.feature_mode = config.get("feature_mode", self.feature_mode)
        else:
            if not model_kwargs:
                model_kwargs = {"num_classes": 0, "global_pool": "avg"}
            self.model = timm.create_model(model_name, pretrained=pretrained, **model_kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if self.feature_mode == "virchow2":
            return _pool_virchow2_features(output)
        if isinstance(output, tuple):
            output = output[0]
        return output


class PackageTeacherTileDataset(Dataset):
    def __init__(self, package_path: str | Path, tile_ids: list[str], transform) -> None:
        self.package_path = Path(package_path)
        self.tile_ids = tile_ids
        self.transform = transform
        self._reader: TilePackageReader | None = None

    def __len__(self) -> int:
        return len(self.tile_ids)

    def _ensure_reader(self) -> TilePackageReader:
        if self._reader is None:
            self._reader = TilePackageReader(self.package_path)
        return self._reader

    def __getitem__(self, index: int) -> dict:
        return self._read_item(index)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        images = self._ensure_reader().read_images_at(indices)
        return [
            {
                "tile_id": self.tile_ids[index],
                "image": self.transform(image.convert("RGB")),
            }
            for index, image in zip(indices, images)
        ]

    def _read_item(self, index: int) -> dict:
        image = self._ensure_reader().read_image_at(index)
        return {
            "tile_id": self.tile_ids[index],
            "image": self.transform(image.convert("RGB")),
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self) -> None:
        self.close()


def _release_inference_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collate_teacher_samples(samples: list[dict]) -> dict:
    return {
        "tile_id": [sample["tile_id"] for sample in samples],
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
    }


def _batch_count(total: int, batch_size: int) -> int:
    return (total + batch_size - 1) // batch_size


def _thread_tile_reader(package_path: Path) -> TilePackageReader:
    readers = getattr(_THREAD_LOCAL, "tile_readers", None)
    if readers is None:
        readers = {}
        _THREAD_LOCAL.tile_readers = readers
    key = str(package_path)
    reader = readers.get(key)
    if reader is None:
        reader = TilePackageReader(package_path)
        readers[key] = reader
    return reader


def _read_teacher_tile_sample(package_path: Path, tile_ids: list[str], transform, index: int) -> dict:
    image = _thread_tile_reader(package_path).read_image_at(index)
    return {
        "tile_id": tile_ids[index],
        "image": transform(image.convert("RGB")),
    }


def _read_teacher_tile_chunk(package_path: Path, tile_ids: list[str], transform, indices: list[int]) -> list[dict]:
    reader = _thread_tile_reader(package_path)
    images = reader.read_images_at(indices)
    return [
        {
            "tile_id": tile_ids[index],
            "image": transform(image.convert("RGB")),
        }
        for index, image in zip(indices, images)
    ]


def _chunk_indices(indices: list[int], num_workers: int) -> list[list[int]]:
    if not indices:
        return []
    chunk_size = max(1, min(64, (len(indices) + max(1, num_workers) - 1) // max(1, num_workers)))
    return [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]


def _build_teacher_batch(
    executor: ThreadPoolExecutor,
    package_path: Path,
    tile_ids: list[str],
    transform,
    indices: list[int],
    num_workers: int,
) -> dict:
    futures = [
        executor.submit(_read_teacher_tile_chunk, package_path, tile_ids, transform, chunk)
        for chunk in _chunk_indices(indices, num_workers)
    ]
    samples = []
    for future in futures:
        samples.extend(future.result())
    return _collate_teacher_samples(samples)


class BoundedTeacherBatchIterator:
    def __init__(
        self,
        package_path: Path,
        tile_ids: list[str],
        transform,
        *,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        initial_prefetch: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.package_path = package_path
        self.tile_ids = tile_ids
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        self.total = len(tile_ids)
        self._next_start = 0
        self._closed = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._max_prefetch = max(0, int(prefetch_factor))
        self._capacity = self._max_prefetch if initial_prefetch is None else max(0, min(int(initial_prefetch), self._max_prefetch))
        self._ready_batches: queue.Queue[dict | BaseException | None] | None = None
        self._slots: threading.Semaphore | None = None

        if self.total == 0 or self.num_workers <= 0 or self._max_prefetch <= 0:
            return

        self._ready_batches = queue.Queue(maxsize=self._max_prefetch)
        self._slots = threading.Semaphore(self._capacity)
        self._thread = threading.Thread(target=self._producer, name="teacher-batch-producer", daemon=True)
        self._thread.start()

    def promote(self) -> None:
        if self._slots is None:
            return
        extra_slots = self._max_prefetch - self._capacity
        if extra_slots <= 0:
            return
        for _ in range(extra_slots):
            self._slots.release()
        self._capacity = self._max_prefetch

    def _producer(self) -> None:
        assert self._ready_batches is not None
        assert self._slots is not None
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for start in range(0, self.total, self.batch_size):
                    if self._stop_event.is_set():
                        return
                    self._slots.acquire()
                    if self._stop_event.is_set():
                        return
                    end = min(start + self.batch_size, self.total)
                    batch = _build_teacher_batch(
                        executor,
                        self.package_path,
                        self.tile_ids,
                        self.transform,
                        list(range(start, end)),
                        self.num_workers,
                    )
                    self._ready_batches.put(batch)
            self._ready_batches.put(None)
        except BaseException as exc:
            self._ready_batches.put(exc)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._closed or self.total == 0:
            raise StopIteration

        if self._ready_batches is None:
            if self._next_start >= self.total:
                self.close()
                raise StopIteration
            start = self._next_start
            end = min(start + self.batch_size, self.total)
            self._next_start = end
            samples = [
                _read_teacher_tile_sample(self.package_path, self.tile_ids, self.transform, index)
                for index in range(start, end)
            ]
            return _collate_teacher_samples(samples)

        item = self._ready_batches.get()
        if item is None:
            self.close()
            raise StopIteration
        if self._slots is not None:
            self._slots.release()
        if isinstance(item, BaseException):
            self.close()
            raise item
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._slots is not None:
            self._slots.release()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _iter_bounded_teacher_batches(
    package_path: Path,
    tile_ids: list[str],
    transform,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    initial_prefetch: int | None = None,
) -> BoundedTeacherBatchIterator:
    return BoundedTeacherBatchIterator(
        package_path,
        tile_ids,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        initial_prefetch=initial_prefetch,
    )


def _read_dataset_sample(dataset: Dataset, index: int) -> dict:
    return dataset[index]


def _iter_bounded_dataset_batches(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    total = len(dataset)
    if total == 0:
        return
    max_inflight = batch_size * (max(0, int(prefetch_factor)) + 1)
    max_inflight = max(batch_size, max_inflight)

    if num_workers <= 0:
        for start in range(0, total, batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, total))]
            yield _collate_teacher_samples(samples)
        return

    with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
        pending: dict[int, Future] = {}
        next_submit = 0
        next_yield = 0

        def fill_window(current_batch_items: int = 0) -> None:
            nonlocal next_submit
            while next_submit < total and len(pending) + current_batch_items < max_inflight:
                pending[next_submit] = executor.submit(_read_dataset_sample, dataset, next_submit)
                next_submit += 1

        fill_window()
        while next_yield < total:
            end = min(next_yield + batch_size, total)
            samples = []
            for index in range(next_yield, end):
                future = pending.pop(index)
                samples.append(future.result())
                fill_window(len(samples))
            next_yield = end
            yield _collate_teacher_samples(samples)


def _resolve_precision(precision: str, device: str) -> str:
    if precision != "auto":
        return precision
    if not device.startswith("cuda"):
        return "fp32"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def _autocast_context(device: str, precision: str):
    precision = _resolve_precision(precision, device)
    if not device.startswith("cuda") or precision == "fp32":
        return nullcontext()
    dtype_by_precision = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return torch.autocast(device_type="cuda", dtype=dtype_by_precision[precision])


def _resolve_feature_dtype(feature_dtype: str, precision: str, device: str) -> str:
    if feature_dtype != "auto":
        return feature_dtype
    return "float16" if _resolve_precision(precision, device) in {"fp16", "bf16"} else "float32"


def _torch_feature_dtype(feature_dtype: str) -> torch.dtype:
    if feature_dtype == "float16":
        return torch.float16
    if feature_dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported feature dtype: {feature_dtype}")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "teacher"


def _teacher_name(model: str, output_teacher_name: str) -> str:
    if output_teacher_name:
        return _safe_name(output_teacher_name)
    preset = TEACHER_MODEL_PRESETS.get(model)
    if preset:
        return _safe_name(preset["teacher_name"])
    return _safe_name(Path(model).name.replace(".features", ""))


def _discover_tile_packages(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        metadata = read_package_metadata(root)
        if metadata.get("payload_type") != "image_tiles":
            raise ValueError(f"not an image tile .iac package: {root}")
        return [root]
    if root.is_dir():
        packages = []
        invalid = []
        for candidate in sorted(root.rglob("*.iac")):
            try:
                metadata = read_package_metadata(candidate)
            except Exception as exc:
                invalid.append(f"{candidate}: {exc}")
                continue
            if metadata.get("payload_type") == "image_tiles":
                packages.append(candidate)
        if invalid:
            sample = "; ".join(invalid[:5])
            raise ValueError(f"invalid .iac package(s) under {root}: count={len(invalid)} sample={sample}")
        if packages:
            return packages
        raise FileNotFoundError(f"no image tile .iac packages found under {root}")
    raise FileNotFoundError(f"input path does not exist: {root}")


def _tile_size(package_path: str | Path) -> tuple[int, int]:
    metadata = read_package_metadata(package_path)
    if metadata.get("payload_type") != "image_tiles":
        raise ValueError(f"not an image tile package: {package_path}")
    width = int(metadata["tile_width"])
    height = int(metadata["tile_height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid tile size in {package_path}: {width}x{height}")
    return width, height


def _validate_common_tile_size(package_paths: list[Path]) -> tuple[int, int]:
    sizes = {path: _tile_size(path) for path in package_paths}
    unique = set(sizes.values())
    if len(unique) != 1:
        detail = ", ".join(f"{path}:{width}x{height}" for path, (width, height) in list(sizes.items())[:5])
        raise ValueError(f"inconsistent tile sizes across input IAC packages: {detail}")
    return next(iter(unique))


def _default_output_path(package_path: Path, output_dir: Path, teacher_name: str) -> Path:
    stem = package_path.stem
    if stem.endswith(".tiles"):
        stem = stem[:-len(".tiles")]
    return output_dir / f"{stem}.{teacher_name}.features.iac"


def _validate_feature_output(package_path: str | Path, expected_teacher: str = "", full: bool = False) -> dict:
    path = Path(package_path)
    header, _, _ = read_tables(path)
    validate_package(path, max_decode=0 if full else 1)
    if expected_teacher and header.get("teacher") != expected_teacher:
        raise ValueError(f"teacher mismatch: got={header.get('teacher')} expected={expected_teacher}")
    return {
        "records": int(header["num_records"]),
        "teacher": str(header["teacher"]),
        "feature_dim": int(header["feature_dim"]),
        "package_bytes": path.stat().st_size,
    }


def _progress_path(output_path: Path, progress_manifest: str | Path | None) -> Path:
    if progress_manifest is not None:
        return Path(progress_manifest)
    if output_path.suffix == ".iac":
        return output_path.with_suffix(".teacher_cache_progress.csv")
    return output_path / "teacher_cache_progress.csv"


def _write_progress(progress_path: Path, rows: list[dict], total: int, started: float) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_package",
        "output",
        "status",
        "teacher",
        "records",
        "feature_dim",
        "package_bytes",
        "elapsed_sec",
        "error",
    ]
    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    summary = {
        "total": total,
        "processed": len(rows),
        "ok": sum(1 for row in rows if row.get("status") in {"ok", "skipped"}),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "elapsed_sec": round(time.time() - started, 3),
        "progress_manifest": str(progress_path),
    }
    progress_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class TeacherPackageWork:
    package_path: Path
    metadata: dict
    records: list[TileRecord]
    tile_ids: list[str]
    batches: BoundedTeacherBatchIterator

    def promote(self) -> None:
        self.batches.promote()

    def close(self) -> None:
        self.batches.close()


def _prepare_teacher_package_work(
    model: torch.nn.Module,
    package_path: str | Path,
    tile_size: tuple[int, int] | None,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    data_config: dict | None,
    initial_prefetch: int | None = None,
) -> TeacherPackageWork:
    package_path = Path(package_path)
    metadata = read_package_metadata(package_path)
    if metadata.get("payload_type") != "image_tiles":
        raise ValueError(f"not an image tile package: {package_path}")
    if tile_size is None:
        tile_size = (int(metadata["tile_width"]), int(metadata["tile_height"]))
    tile_width, tile_height = tile_size
    resolved_data_config = dict(data_config) if data_config is not None else resolve_model_data_config(model.model)
    resolved_data_config["input_size"] = (3, tile_height, tile_width)
    transform = create_transform(**resolved_data_config, is_training=False)
    records = read_package_manifest(package_path)
    tile_ids = [record.tile_id for record in records]
    batches = _iter_bounded_teacher_batches(
        package_path,
        tile_ids,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        initial_prefetch=initial_prefetch,
    )
    return TeacherPackageWork(
        package_path=package_path,
        metadata=metadata,
        records=records,
        tile_ids=tile_ids,
        batches=batches,
    )


@torch.inference_mode()
def cache_teacher_features_from_records(
    model: torch.nn.Module,
    records: list[TileRecord],
    dataset: Dataset,
    output_path: str | Path,
    tile_size: tuple[int, int],
    batch_size: int,
    device: str,
    teacher_name: str,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    overwrite: bool = False,
    precision: str = "fp32",
    feature_dtype: str = "float32",
) -> None:
    model = model.to(device).eval()

    def features():
        try:
            batches = _iter_bounded_dataset_batches(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
            for batch in tqdm(batches, total=_batch_count(len(dataset), batch_size), desc="teacher batches"):
                images = batch["image"].to(device, non_blocking=device.startswith("cuda"))
                with _autocast_context(device, precision):
                    batch_features = model(images).detach().to(dtype=_torch_feature_dtype(feature_dtype))
                    batch_features = batch_features.cpu().numpy()
                for feature in batch_features:
                    yield feature
                del batch, images, batch_features
        finally:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()
            _release_inference_memory(device)

    build_teacher_feature_package(
        records,
        features(),
        output_path,
        teacher_name=teacher_name,
        dtype=feature_dtype,
        overwrite=overwrite,
    )


@torch.inference_mode()
def cache_teacher_features_from_package(
    model: torch.nn.Module,
    package_path: str | Path,
    output_path: str | Path,
    tile_size: tuple[int, int] | None,
    batch_size: int,
    device: str,
    teacher_name: str,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    overwrite: bool = False,
    precision: str = "fp32",
    feature_dtype: str = "float32",
    data_config: dict | None = None,
    work: TeacherPackageWork | None = None,
) -> None:
    model = model.to(device).eval()
    prepared = work
    if prepared is None:
        prepared = _prepare_teacher_package_work(
            model,
            package_path,
            tile_size,
            batch_size,
            num_workers,
            prefetch_factor,
            data_config,
        )
    else:
        prepared.promote()
    metadata = prepared.metadata
    tile_width = int(metadata["tile_width"])
    tile_height = int(metadata["tile_height"])

    def features():
        try:
            for batch in tqdm(
                prepared.batches,
                total=_batch_count(len(prepared.tile_ids), batch_size),
                desc="teacher batches",
            ):
                images = batch["image"].to(device, non_blocking=device.startswith("cuda"))
                with _autocast_context(device, precision):
                    batch_features_tensor = model(images)
                batch_features_tensor = batch_features_tensor.detach().to(dtype=_torch_feature_dtype(feature_dtype))
                batch_features = batch_features_tensor.cpu().numpy()
                for feature in batch_features:
                    yield feature
                del batch, images, batch_features, batch_features_tensor
        finally:
            prepared.close()
            _release_inference_memory(device)

    build_teacher_feature_package(
        prepared.records,
        features(),
        output_path,
        teacher_name=teacher_name,
        dtype=feature_dtype,
        tile_width=tile_width,
        tile_height=tile_height,
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
        overwrite=overwrite,
    )


def cache_teacher_features_from_packages(
    model: torch.nn.Module,
    package_paths: list[Path],
    output: str | Path,
    batch_size: int,
    device: str,
    teacher_name: str,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    overwrite: bool = False,
    progress_manifest: str | Path | None = None,
    continue_on_error: bool = False,
    precision: str = "fp32",
    feature_dtype: str = "float32",
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    validate_output: bool = False,
) -> None:
    tile_size = _validate_common_tile_size(package_paths)
    output_path = Path(output)
    output_is_file = output_path.suffix == ".iac"
    if len(package_paths) > 1 and output_is_file:
        raise ValueError("--output must be a directory when --input resolves to multiple packages")
    if not output_is_file:
        output_path.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows: list[dict] = []
    progress_path = _progress_path(output_path, progress_manifest)
    data_config = resolve_model_data_config(model.model) if hasattr(model, "model") else None
    runtime_model: torch.nn.Module | None = None
    resolved_precision = _resolve_precision(precision, device)
    resolved_feature_dtype = _resolve_feature_dtype(feature_dtype, precision, device)
    runtime_config = {
        "packages": len(package_paths),
        "output": str(output_path),
        "output_is_file": output_is_file,
        "tile_size": list(tile_size),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "device": device,
        "precision": precision,
        "resolved_precision": resolved_precision,
        "feature_dtype": feature_dtype,
        "resolved_feature_dtype": resolved_feature_dtype,
        "compile": compile_model,
        "compile_mode": compile_mode if compile_model else "",
        "overwrite": overwrite,
        "validate_output": validate_output,
        "continue_on_error": continue_on_error,
        "progress_manifest": str(progress_path),
    }
    print(f"teacher_cache_config {json.dumps(runtime_config, sort_keys=True)}", flush=True)

    def get_runtime_model() -> torch.nn.Module:
        nonlocal runtime_model
        if runtime_model is None:
            runtime_model = model.to(device).eval()
            if compile_model:
                print(f"teacher_model_compiling mode={compile_mode}", flush=True)
                runtime_model = torch.compile(runtime_model, mode=compile_mode)
                print("teacher_model_compiled", flush=True)
        return runtime_model

    package_outputs = [
        output_path if output_is_file else _default_output_path(package_path, output_path, teacher_name)
        for package_path in package_paths
    ]
    warmed_index: int | None = None
    warmed_work: TeacherPackageWork | None = None

    def next_build_index(start: int) -> int | None:
        for candidate_index in range(start, len(package_paths)):
            candidate_output = package_outputs[candidate_index]
            if overwrite or not candidate_output.exists():
                return candidate_index
        return None

    def warm_next(start: int) -> None:
        nonlocal warmed_index, warmed_work
        if warmed_work is not None:
            return
        candidate_index = next_build_index(start)
        if candidate_index is None:
            return
        warmed_index = candidate_index
        warmed_work = _prepare_teacher_package_work(
            get_runtime_model(),
            package_paths[candidate_index],
            tile_size,
            batch_size,
            num_workers,
            prefetch_factor,
            data_config,
            initial_prefetch=1,
        )

    try:
        for index, package_path in enumerate(package_paths):
            package_output = package_outputs[index]
            print(f"feature_package_start tile_package={package_path} output={package_output} teacher={teacher_name}", flush=True)
            row = {
                "tile_package": str(package_path),
                "output": str(package_output),
                "teacher": teacher_name,
            }
            item_started = time.time()
            try:
                if package_output.exists() and not overwrite:
                    metadata = _validate_feature_output(package_output, expected_teacher=teacher_name, full=validate_output)
                    row.update(metadata)
                    row["status"] = "skipped"
                    row["elapsed_sec"] = round(time.time() - item_started, 3)
                    rows.append(row)
                    _write_progress(progress_path, rows, len(package_paths), started)
                    print(f"feature_package_skipped existing_valid tile_package={package_path} output={package_output}", flush=True)
                    continue
                if warmed_index == index and warmed_work is not None:
                    work = warmed_work
                    warmed_index = None
                    warmed_work = None
                    work.promote()
                else:
                    work = _prepare_teacher_package_work(
                        get_runtime_model(),
                        package_path,
                        tile_size,
                        batch_size,
                        num_workers,
                        prefetch_factor,
                        data_config,
                    )
                warm_next(index + 1)
                cache_teacher_features_from_package(
                    model=get_runtime_model(),
                    package_path=package_path,
                    output_path=package_output,
                    tile_size=tile_size,
                    batch_size=batch_size,
                    device=device,
                    teacher_name=teacher_name,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    overwrite=overwrite,
                    precision=precision,
                    feature_dtype=resolved_feature_dtype,
                    data_config=data_config,
                    work=work,
                )
                metadata = _validate_feature_output(package_output, expected_teacher=teacher_name, full=validate_output)
                row.update(metadata)
                row["status"] = "ok"
                row["elapsed_sec"] = round(time.time() - item_started, 3)
                rows.append(row)
                _write_progress(progress_path, rows, len(package_paths), started)
                print(f"feature_package_ok tile_package={package_path} output={package_output}", flush=True)
            except Exception as exc:
                row["status"] = "failed"
                row["elapsed_sec"] = round(time.time() - item_started, 3)
                row["error"] = str(exc)
                rows.append(row)
                _write_progress(progress_path, rows, len(package_paths), started)
                if not continue_on_error:
                    raise
                print(f"feature_package_failed tile_package={package_path} output={package_output} error={exc}", flush=True)
    finally:
        if warmed_work is not None:
            warmed_work.close()
    print(f"teacher_cache_progress manifest={progress_path} summary={progress_path.with_suffix('.json')}", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s --input INPUT --output OUTPUT --teacher TEACHER [options]",
        description="Run a teacher model and write a teacher feature IatroCache package.",
        epilog=_preset_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", help="Input image-tile .iac file or directory recursively scanned for image-tile .iac packages.")
    parser.add_argument("--output", required=True, help="Output .features.iac file, or output directory for multiple input packages.")
    parser.add_argument(
        "--teacher",
        help="Teacher preset, timm model name, hf_hub:* model name, or local model directory.",
    )
    parser.add_argument(
        "--teacher-name",
        default="",
        help="Teacher name recorded in output metadata and filenames. Defaults to the preset teacher name.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8, help="Tile read/decode worker threads for --input reads.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Global prefetched batches beyond the current batch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="bf16",
        help="Teacher inference precision.",
    )
    parser.add_argument(
        "--feature-dtype",
        choices=("auto", "float32", "float16"),
        default="auto",
        help="Feature matrix dtype written to IAC. auto writes float16 for fp16/bf16 inference and float32 otherwise.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the teacher with torch.compile before generating features.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="torch.compile mode used with --compile.",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress-manifest",
        default=None,
        help="CSV progress manifest path. Defaults to teacher_cache_progress.csv under the output directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue directory batch processing after a package fails and record the failure in the progress manifest.",
    )
    parser.add_argument(
        "--validate-output",
        action="store_true",
        help="Fully validate generated or skipped feature IAC packages by checking fixed-length feature records.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not args.input:
        parser.error("--input is required")
    if not args.teacher:
        parser.error("--teacher is required, for example --teacher gigapath")
    teacher_name = _teacher_name(args.teacher, args.teacher_name)
    print(
        f"teacher_cache_start input={args.input} output={args.output} teacher={args.teacher} "
        f"teacher_name={teacher_name} device={args.device} precision={args.precision} "
        f"feature_dtype={args.feature_dtype} compile={args.compile_model}",
        flush=True,
    )
    print(f"teacher_cache_scanning_input path={args.input}", flush=True)
    package_paths = _discover_tile_packages(args.input)
    print(f"teacher_cache_input_discovered packages={len(package_paths)}", flush=True)
    model_spec = _resolve_model_spec(args.teacher)
    print(
        f"teacher_model_loading teacher={args.teacher} model_name={model_spec['model_name']} "
        f"pretrained={args.pretrained}",
        flush=True,
    )
    model = TimmTeacherEncoder(
        model_spec["model_name"],
        pretrained=args.pretrained,
        model_kwargs=model_spec["model_kwargs"],
        feature_mode=model_spec["feature_mode"],
    )
    print(f"teacher_model_loaded teacher={args.teacher} feature_mode={model_spec['feature_mode']}", flush=True)
    cache_teacher_features_from_packages(
        model=model,
        package_paths=package_paths,
        output=args.output,
        batch_size=args.batch_size,
        device=args.device,
        teacher_name=teacher_name,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        overwrite=args.overwrite,
        progress_manifest=args.progress_manifest,
        continue_on_error=args.continue_on_error,
        precision=args.precision,
        feature_dtype=args.feature_dtype,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        validate_output=args.validate_output,
    )


if __name__ == "__main__":
    main()
