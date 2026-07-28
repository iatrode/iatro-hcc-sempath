from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class PrototypeRegistry:
    """Runtime prototype package with tensor values and public metadata."""

    prototypes: torch.Tensor
    names: list[str]
    thresholds: torch.Tensor | None = None
    source: dict[str, Any] | None = None
    version: int = 1

    @property
    def dim(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def count(self) -> int:
        return int(self.prototypes.shape[0])

    def to(self, device: torch.device | str) -> "PrototypeRegistry":
        return PrototypeRegistry(
            prototypes=self.prototypes.to(device),
            names=list(self.names),
            thresholds=self.thresholds.to(device) if self.thresholds is not None else None,
            source=dict(self.source or {}),
            version=self.version,
        )


def _load_payload(path: Path) -> dict[str, Any]:
    if path.is_dir():
        manifest_path = path / "prototype_manifest.yaml"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = yaml.safe_load(handle) or {}
            prototype_file = manifest.get("prototype_file", "prototypes.pt")
            payload = torch.load(path / prototype_file, map_location="cpu")
            if not isinstance(payload, dict):
                payload = {"prototypes": payload}
            return {**manifest, **payload}
        package_path = path / "prototypes.pt"
        if package_path.exists():
            payload = torch.load(package_path, map_location="cpu")
            return payload if isinstance(payload, dict) else {"prototypes": payload}
        raise FileNotFoundError(f"prototype directory missing prototype_manifest.yaml or prototypes.pt: {path}")
    payload = torch.load(path, map_location="cpu")
    return payload if isinstance(payload, dict) else {"prototypes": payload}


def _validate_registry(registry: PrototypeRegistry, expected_dim: int | None = None) -> None:
    if registry.version != 1:
        raise ValueError(f"unsupported prototype package version: {registry.version}")
    if registry.prototypes.ndim != 2:
        raise ValueError(f"prototypes must be 2D, got shape={tuple(registry.prototypes.shape)}")
    if expected_dim is not None and registry.prototypes.shape[1] != expected_dim:
        raise ValueError(f"prototype dim mismatch: got {registry.prototypes.shape[1]}, expected {expected_dim}")
    if len(registry.names) != registry.count:
        raise ValueError(f"prototype name count mismatch: names={len(registry.names)} prototypes={registry.count}")
    if len(set(registry.names)) != len(registry.names):
        raise ValueError("prototype names must be unique")
    if registry.thresholds is not None and registry.thresholds.shape != (registry.count,):
        raise ValueError(
            f"prototype thresholds must have shape=({registry.count},), got {tuple(registry.thresholds.shape)}"
        )
    if registry.count < 2:
        raise ValueError("prototype package must contain at least two classification prototypes")


def load_prototype_registry(path: str | Path, expected_dim: int | None = None) -> PrototypeRegistry:
    path = Path(path)
    payload = _load_payload(path)
    prototypes = torch.as_tensor(payload["prototypes"], dtype=torch.float32)
    if "names" not in payload:
        raise ValueError("prototype package missing required field: names")
    names = [str(name) for name in payload["names"]]
    thresholds_payload = payload.get("thresholds")
    thresholds = torch.as_tensor(thresholds_payload, dtype=torch.float32) if thresholds_payload is not None else None
    source = payload.get("source")
    if source is None:
        source = {"path": str(path)}
    elif not isinstance(source, dict):
        source = {"description": str(source), "path": str(path)}
    elif "path" not in source:
        source = {**source, "path": str(path)}
    registry = PrototypeRegistry(
        prototypes=prototypes,
        names=names,
        thresholds=thresholds,
        source=source,
        version=int(payload.get("version", 1)),
    )
    _validate_registry(registry, expected_dim=expected_dim)
    return registry


def load_prototypes(path: str | Path, expected_dim: int | None = None) -> torch.Tensor:
    return load_prototype_registry(path, expected_dim=expected_dim).prototypes
