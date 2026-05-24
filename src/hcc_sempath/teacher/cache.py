from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file as load_safetensors_file
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ..io.feature_cache import build_teacher_feature_package
from ..io.manifests import TileRecord
from ..io.tile_package import TilePackageReader, read_package_manifest, read_package_metadata


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
        "Planned supported presets for --model:",
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


def _loader_kwargs(num_workers: int, prefetch_factor: int, pin_memory: bool) -> dict:
    kwargs = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        kwargs["persistent_workers"] = True
    return kwargs


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
        return [root]
    if root.is_dir():
        packages = []
        for candidate in sorted(root.rglob("*.iac")):
            try:
                metadata = read_package_metadata(candidate)
            except Exception:
                continue
            if metadata.get("payload_type") == "image_tiles":
                packages.append(candidate)
        if packages:
            return packages
        raise FileNotFoundError(f"no image tile .iac packages found under {root}")
    raise FileNotFoundError(f"tile package path does not exist: {root}")


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


@torch.no_grad()
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
            batch_features = model(images).detach().cpu().numpy().astype(np.float32)
            for feature in batch_features:
                yield feature

    build_teacher_feature_package(
        records,
        features(),
        output_path,
        teacher_name=teacher_name,
        dtype="float32",
        overwrite=overwrite,
    )


@torch.no_grad()
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
) -> None:
    if tile_size is None:
        tile_size = _tile_size(package_path)
    tile_width, tile_height = tile_size
    data_config = resolve_model_data_config(model.model)
    data_config["input_size"] = (3, tile_height, tile_width)
    transform = create_transform(**data_config, is_training=False)
    records = read_package_manifest(package_path)
    dataset = PackageTeacherTileDataset(package_path, records, transform)
    cache_teacher_features_from_records(
        model=model,
        records=records,
        dataset=dataset,
        output_path=output_path,
        tile_size=tile_size,
        batch_size=batch_size,
        device=device,
        teacher_name=teacher_name,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
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
) -> None:
    tile_size = _validate_common_tile_size(package_paths)
    output_path = Path(output)
    output_is_file = output_path.suffix == ".iac"
    if len(package_paths) > 1 and output_is_file:
        raise ValueError("--output must be a directory when --tile-package points to multiple packages")
    if not output_is_file:
        output_path.mkdir(parents=True, exist_ok=True)

    for package_path in package_paths:
        package_output = output_path if output_is_file else _default_output_path(package_path, output_path, teacher_name)
        cache_teacher_features_from_package(
            model=model,
            package_path=package_path,
            output_path=package_output,
            tile_size=tile_size,
            batch_size=batch_size,
            device=device,
            teacher_name=teacher_name,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            overwrite=overwrite,
        )
        print(f"feature_package_ok tile_package={package_path} output={package_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a teacher model and write a teacher feature IatroCache package.",
        epilog=_preset_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tile-package", required=True, help="Input image-tile .iac file or directory of image-tile .iac packages.")
    parser.add_argument("--output", required=True, help="Output .features.iac file, or output directory for multiple input packages.")
    parser.add_argument(
        "--model",
        default="h_optimus_1",
        help="Teacher preset, timm model name, hf_hub:* model name, or local model directory.",
    )
    parser.add_argument("--model-name", dest="model", help=argparse.SUPPRESS)
    parser.add_argument("--output-teacher-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for --tile-package reads.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker for --tile-package reads.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    teacher_name = _teacher_name(args.model, args.output_teacher_name)
    model_spec = _resolve_model_spec(args.model)
    package_paths = _discover_tile_packages(args.tile_package)
    model = TimmTeacherEncoder(
        model_spec["model_name"],
        pretrained=args.pretrained,
        model_kwargs=model_spec["model_kwargs"],
        feature_mode=model_spec["feature_mode"],
    )
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
    )


if __name__ == "__main__":
    main()
