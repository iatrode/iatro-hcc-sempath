from __future__ import annotations

import pytest
import torch

from hcc_sempath.training.zhcc_metrics import _macro_auc


def test_macro_auc_counts_ties_as_half_credit() -> None:
    logits = torch.tensor([[0.5], [0.5], [0.2]])
    targets = torch.tensor([[1.0], [0.0], [0.0]])

    assert _macro_auc(logits, targets) == pytest.approx(0.75)
