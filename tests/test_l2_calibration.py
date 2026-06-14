from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from hcc_sempath.modeling.models import calibrated_attribute_scores


def _load_export_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_disagreement_l2.py"
    spec = importlib.util.spec_from_file_location("export_disagreement_l2", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_robust_l2_calibration_centers_each_attribute_without_changing_rank() -> None:
    module = _load_export_module()
    cosine_scores = np.array(
        [
            [0.20, 0.70],
            [0.30, 0.72],
            [0.40, 0.74],
            [0.50, 0.76],
            [0.60, 0.78],
        ],
        dtype=np.float32,
    )

    biases, temperatures = module._robust_calibration(cosine_scores)
    calibrated = calibrated_attribute_scores(
        torch.from_numpy(cosine_scores),
        torch.from_numpy(biases),
        torch.from_numpy(temperatures),
    ).numpy()

    np.testing.assert_allclose(np.median(calibrated, axis=0), np.full(2, 0.5), atol=1e-6)
    np.testing.assert_array_equal(np.argsort(calibrated, axis=0), np.argsort(cosine_scores, axis=0))
    assert np.all(temperatures > 0)
