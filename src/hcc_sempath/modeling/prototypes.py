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
    groups: list[str | None]
    levels: list[int]
    exclusive: list[bool]
    thresholds: torch.Tensor | None = None
    source: dict[str, Any] | None = None
    version: int = 1

    @property
    def dim(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def count(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def primary_indices(self) -> list[int]:
        return [idx for idx, level in enumerate(self.levels) if level == 1]

    @property
    def attribute_indices(self) -> list[int]:
        return [idx for idx, level in enumerate(self.levels) if level == 2]

    @property
    def primary_prototypes(self) -> torch.Tensor:
        return self.prototypes[self.primary_indices]

    @property
    def attribute_prototypes(self) -> torch.Tensor:
        return self.prototypes[self.attribute_indices]

    def to(self, device: torch.device | str) -> "PrototypeRegistry":
        return PrototypeRegistry(
            prototypes=self.prototypes.to(device),
            names=list(self.names),
            groups=list(self.groups),
            levels=list(self.levels),
            exclusive=list(self.exclusive),
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
    if len(registry.groups) != registry.count:
        raise ValueError(f"prototype group count mismatch: groups={len(registry.groups)} prototypes={registry.count}")
    if len(registry.levels) != registry.count:
        raise ValueError(f"prototype level count mismatch: levels={len(registry.levels)} prototypes={registry.count}")
    if any(level not in {1, 2} for level in registry.levels):
        raise ValueError("prototype levels must be 1 or 2")
    if len(registry.exclusive) != registry.count:
        raise ValueError(
            f"prototype exclusivity count mismatch: exclusive={len(registry.exclusive)} prototypes={registry.count}"
        )
    for name, level, exclusive in zip(registry.names, registry.levels, registry.exclusive):
        if level == 1 and not exclusive:
            raise ValueError(f"level-1 prototype must be exclusive: {name}")
        if level == 2 and exclusive:
            raise ValueError(f"level-2 prototype must be non-exclusive: {name}")
    if registry.thresholds is not None and registry.thresholds.shape != (registry.count,):
        raise ValueError(
            f"prototype thresholds must have shape=({registry.count},), got {tuple(registry.thresholds.shape)}"
        )
    if len(registry.primary_indices) < 2:
        raise ValueError("prototype package must contain at least two level-1 primary prototypes")


def load_prototype_registry(path: str | Path, expected_dim: int | None = None) -> PrototypeRegistry:
    path = Path(path)
    payload = _load_payload(path)
    prototypes = torch.as_tensor(payload["prototypes"], dtype=torch.float32)
    if "names" not in payload:
        raise ValueError("prototype package missing required field: names")
    if "levels" not in payload:
        raise ValueError("prototype package missing required field: levels")
    if "exclusive" not in payload:
        raise ValueError("prototype package missing required field: exclusive")
    names = [str(name) for name in payload["names"]]
    groups_payload = payload.get("groups")
    if groups_payload is None:
        groups: list[str | None] = [None] * prototypes.shape[0]
    else:
        groups = [None if group is None else str(group) for group in groups_payload]
    levels = [int(level) for level in payload["levels"]]
    exclusive = [bool(value) for value in payload["exclusive"]]
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
        groups=groups,
        levels=levels,
        exclusive=exclusive,
        thresholds=thresholds,
        source=source,
        version=int(payload.get("version", 1)),
    )
    _validate_registry(registry, expected_dim=expected_dim)
    return registry


def load_prototypes(path: str | Path, expected_dim: int | None = None) -> torch.Tensor:
    return load_prototype_registry(path, expected_dim=expected_dim).prototypes
