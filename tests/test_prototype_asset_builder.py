from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _builder_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_prototype_assets_from_annotations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_prototype_assets_from_annotations",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_bank_facility_order_uses_fig1_margin_secondary() -> None:
    module = _builder_module()
    similarity = np.asarray(
        [
            [1.0, 0.7, 0.6],
            [0.7, 1.0, 0.6],
            [0.6, 0.6, 1.0],
        ],
        dtype=np.float32,
    )

    order = module._facility_order(
        similarity,
        3,
        margin_rank=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )

    assert module.SEPARATION_WEIGHT == 32.0
    assert order[0] == 2
