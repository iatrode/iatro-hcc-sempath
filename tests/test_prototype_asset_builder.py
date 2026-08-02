from __future__ import annotations

import numpy as np

from hcc_sempath.build import supervision


def test_training_bank_facility_order_uses_fig1_margin_secondary() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.7, 0.6],
            [0.7, 1.0, 0.6],
            [0.6, 0.6, 1.0],
        ],
        dtype=np.float32,
    )

    order = supervision._facility_order(
        similarity,
        3,
        margin_rank=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )

    assert supervision.SEPARATION_WEIGHT == 32.0
    assert order[0] == 2
