from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import csv
from dataclasses import dataclass
import json
import re
import time
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file as load_safetensors_file
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ..io.feature_cache import build_teacher_feature_package, build_teacher_feature_package_from_tile_package
from ..io.iatrocache import read_tables
from ..io.manifests import TileRecord
from ..io.tile_package import TilePackageReader, read_package_manifest, read_package_metadata
from ..io.validate_package import _validate_common, _validate_teacher_features


TEACHER_MODEL_PRESETS: dict[str, dict] = {
    "h_optimus_1": {
        "model_name": "hf_hub:bioptimus/H-optimus-1",
        "teacher_name": "h_optimus_1",
        "description": "Planned supported H-optimus-1 teacher.",
    },
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


def _resolve_model_name(model_name: str) -> str:
    preset = TEACHER_MODEL_PRESETS.get(model_name)
    return preset["model_name"] if preset else model_name


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
            model_args.update(model_kwargs)
            model_args.setdefault("num_classes", 0)
            self.model = timm.create_model(architecture, pretrained=False, **model_args)
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
    def __init__(self, package_path: str | Path, records: list[TileRecord], transform) -> None:
        self.package_path = Path(package_path)
        self.records = records
        self.transform = transform
        self._reader: TilePackageReader | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_reader(self) -> TilePackageReader:
        if self._reader is None:
            self._reader = TilePackageReader(self.package_path)
        return self._reader

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = self._ensure_reader().read_image(record.tile_id)
        return {
            "tile_id": record.tile_id,
            "image": self.transform(image.convert("RGB")),
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __del__(self) -> None:
        if self._reader is not None:
            self._reader.close()


@dataclass
class PreparedTeacherPackage:
    package_path: Path
    records: list[TileRecord]
    loader: DataLoader
    iterator: object
    total_batches: int
    tile_width: int
    tile_height: int
    stride_x: int
    stride_y: int


def _loader_kwargs(num_workers: int, prefetch_factor: int, pin_memory: bool) -> dict:
    kwargs = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        kwargs["persistent_workers"] = True
    return kwargs


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


def _prepare_teacher_package(
    package_path: str | Path,
    tile_size: tuple[int, int],
    data_config: dict,
    batch_size: int,
    device: str,
    num_workers: int,
    prefetch_factor: int,
) -> PreparedTeacherPackage:
    package_path = Path(package_path)
    metadata = read_package_metadata(package_path)
    if metadata.get("payload_type") != "image_tiles":
        raise ValueError(f"not an image tile package: {package_path}")
    tile_width, tile_height = tile_size
    records = read_package_manifest(package_path)
    package_data_config = dict(data_config)
    package_data_config["input_size"] = (3, tile_height, tile_width)
    transform = create_transform(**package_data_config, is_training=False)
    dataset = PackageTeacherTileDataset(package_path, records, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=device.startswith("cuda"),
        ),
    )
    return PreparedTeacherPackage(
        package_path=package_path,
        records=records,
        loader=loader,
        iterator=iter(loader),
        total_batches=len(loader),
        tile_width=int(metadata["tile_width"]),
        tile_height=int(metadata["tile_height"]),
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
    )


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
    header, slide_table, record_table = read_tables(path)
    _validate_common(header, slide_table, record_table)
    if full:
        _validate_teacher_features(str(path), header, record_table, max_payload=0)
    else:
        if header.get("payload_type") != "teacher_features":
            raise ValueError(f"not a teacher feature package: {path}")
        if not header.get("teacher"):
            raise ValueError("teacher_features header requires non-empty teacher")
        if int(header.get("feature_dim", 0)) <= 0:
            raise ValueError(f"invalid feature_dim: {header.get('feature_dim')}")
        np.dtype(header["dtype"])
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=device.startswith("cuda"),
        ),
    )

    def features():
        for batch in tqdm(loader, total=len(loader), desc="teacher batches"):
            images = batch["image"].to(device, non_blocking=device.startswith("cuda"))
            with _autocast_context(device, precision):
                batch_features = model(images).detach().to(dtype=_torch_feature_dtype(feature_dtype))
                batch_features = batch_features.cpu().numpy()
            for feature in batch_features:
                yield feature

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
) -> None:
    if tile_size is None:
        tile_size = _tile_size(package_path)
    tile_width, tile_height = tile_size
    data_config = dict(data_config) if data_config is not None else resolve_model_data_config(model.model)
    data_config["input_size"] = (3, tile_height, tile_width)
    transform = create_transform(**data_config, is_training=False)
    records = read_package_manifest(package_path)
    dataset = PackageTeacherTileDataset(package_path, records, transform)
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=device.startswith("cuda"),
        ),
    )

    def features():
        for batch in tqdm(loader, total=len(loader), desc="teacher batches"):
            images = batch["image"].to(device, non_blocking=device.startswith("cuda"))
            with _autocast_context(device, precision):
                batch_features = model(images).detach().to(dtype=_torch_feature_dtype(feature_dtype))
                batch_features = batch_features.cpu().numpy()
            for feature in batch_features:
                yield feature

    build_teacher_feature_package_from_tile_package(
        package_path,
        features(),
        output_path,
        teacher_name=teacher_name,
        dtype=feature_dtype,
        overwrite=overwrite,
    )


@torch.inference_mode()
def cache_teacher_features_from_prepared_package(
    model: torch.nn.Module,
    prepared: PreparedTeacherPackage,
    output_path: str | Path,
    device: str,
    teacher_name: str,
    overwrite: bool = False,
    precision: str = "fp32",
    feature_dtype: str = "float32",
) -> None:
    model = model.to(device).eval()

    def features():
        for batch in tqdm(prepared.iterator, total=prepared.total_batches, desc="teacher batches"):
            images = batch["image"].to(device, non_blocking=device.startswith("cuda"))
            with _autocast_context(device, precision):
                batch_features = model(images).detach().to(dtype=_torch_feature_dtype(feature_dtype))
                batch_features = batch_features.cpu().numpy()
            for feature in batch_features:
                yield feature

    build_teacher_feature_package(
        prepared.records,
        features(),
        output_path,
        teacher_name=teacher_name,
        dtype=feature_dtype,
        tile_width=prepared.tile_width,
        tile_height=prepared.tile_height,
        stride_x=prepared.stride_x,
        stride_y=prepared.stride_y,
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
    prefetch_packages: bool = True,
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
        "prefetch_packages": prefetch_packages,
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
    prepare_executor = ThreadPoolExecutor(max_workers=1) if prefetch_packages and len(package_paths) > 1 else None
    prepare_future: Future | None = None
    prepare_future_path: Path | None = None

    def submit_prepare(package_path: Path) -> Future:
        print(f"feature_package_prefetch_start tile_package={package_path}", flush=True)
        assert prepare_executor is not None
        return prepare_executor.submit(
            _prepare_teacher_package,
            package_path,
            tile_size,
            data_config or {},
            batch_size,
            device,
            num_workers,
            prefetch_factor,
        )

    def schedule_next_prepare(current_index: int) -> None:
        nonlocal prepare_future, prepare_future_path
        if prepare_executor is None or prepare_future is not None:
            return
        for next_index in range(current_index + 1, len(package_paths)):
            next_output = package_outputs[next_index]
            if next_output.exists() and not overwrite:
                continue
            prepare_future_path = package_paths[next_index]
            prepare_future = submit_prepare(prepare_future_path)
            return

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
                    if prepare_future is not None and prepare_future_path == package_path:
                        prepare_future.cancel()
                        prepare_future = None
                        prepare_future_path = None
                    metadata = _validate_feature_output(package_output, expected_teacher=teacher_name, full=validate_output)
                    row.update(metadata)
                    row["status"] = "skipped"
                    row["elapsed_sec"] = round(time.time() - item_started, 3)
                    rows.append(row)
                    _write_progress(progress_path, rows, len(package_paths), started)
                    print(f"feature_package_skipped existing_valid tile_package={package_path} output={package_output}", flush=True)
                    schedule_next_prepare(index)
                    continue
                if prepare_future is not None and prepare_future_path == package_path:
                    prepared = prepare_future.result()
                    prepare_future = None
                    prepare_future_path = None
                    print(f"feature_package_prefetch_ready tile_package={package_path}", flush=True)
                else:
                    prepared = _prepare_teacher_package(
                        package_path,
                        tile_size,
                        data_config or {},
                        batch_size,
                        device,
                        num_workers,
                        prefetch_factor,
                    )
                schedule_next_prepare(index)
                cache_teacher_features_from_prepared_package(
                    model=get_runtime_model(),
                    prepared=prepared,
                    output_path=package_output,
                    device=device,
                    teacher_name=teacher_name,
                    overwrite=overwrite,
                    precision=precision,
                    feature_dtype=resolved_feature_dtype,
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
        if prepare_executor is not None:
            prepare_executor.shutdown(wait=False, cancel_futures=True)
    print(f"teacher_cache_progress manifest={progress_path} summary={progress_path.with_suffix('.json')}", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s --input INPUT --output OUTPUT --teacher TEACHER [options]",
        description="Run a teacher model and write a teacher feature IatroCache package.",
        epilog=_preset_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", help="Input image-tile .iac file or directory recursively scanned for image-tile .iac packages.")
    parser.add_argument("--tile-package", dest="input", help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True, help="Output .features.iac file, or output directory for multiple input packages.")
    parser.add_argument(
        "--teacher",
        help="Teacher preset, timm model name, hf_hub:* model name, or local model directory.",
    )
    parser.add_argument("--model", dest="teacher", help=argparse.SUPPRESS)
    parser.add_argument("--model-name", dest="teacher", help=argparse.SUPPRESS)
    parser.add_argument(
        "--teacher-name",
        default="",
        help="Teacher name recorded in output metadata and filenames. Defaults to the preset teacher name.",
    )
    parser.add_argument("--output-teacher-name", dest="teacher_name", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for --input reads.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker for --input reads.")
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
    parser.add_argument(
        "--prefetch-packages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare the next input IAC package while the current package is running teacher inference.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not args.input:
        parser.error("--input is required")
    if not args.teacher:
        parser.error("--teacher is required, for example --teacher h_optimus_1")
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
        prefetch_packages=args.prefetch_packages,
    )


if __name__ == "__main__":
    main()
