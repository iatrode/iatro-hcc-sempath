from __future__ import annotations

from pathlib import Path

import torch


def load_anchors(path: str | Path, expected_dim: int | None = None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    anchors = payload["anchors"] if isinstance(payload, dict) else payload
    anchors = torch.as_tensor(anchors, dtype=torch.float32)
    if anchors.ndim != 2:
        raise ValueError(f"anchors must be 2D, got shape={tuple(anchors.shape)}")
    if expected_dim is not None and anchors.shape[1] != expected_dim:
        raise ValueError(f"anchor dim mismatch: got {anchors.shape[1]}, expected {expected_dim}")
    return anchors
