from __future__ import annotations

from pathlib import Path

import torch


def load_prototypes(path: str | Path, expected_dim: int | None = None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    prototypes = payload["prototypes"] if isinstance(payload, dict) else payload
    prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must be 2D, got shape={tuple(prototypes.shape)}")
    if expected_dim is not None and prototypes.shape[1] != expected_dim:
        raise ValueError(f"prototype dim mismatch: got {prototypes.shape[1]}, expected {expected_dim}")
    return prototypes
