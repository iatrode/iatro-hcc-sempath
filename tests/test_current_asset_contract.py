from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from hcc_sempath.spatial_schema import DEFAULT_SPATIAL_COMPONENTS
from hcc_sempath.training.prototype_labels import DEFAULT_L1_CLASSES


REPO = Path(__file__).resolve().parents[1]


def test_frozen_l1_bank_matches_training_contract() -> None:
    payload = json.loads(
        (REPO / "annotations/hcc_prototype_bank.fixed.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(payload["l1_prototypes"]) == DEFAULT_L1_CLASSES
    counts = Counter(
        item["l1"] for item in payload["annotations"].values()
    )
    assert counts == Counter({name: 400 for name in DEFAULT_L1_CLASSES})

    with (
        REPO / "annotations/hcc_prototype_bank.fixed.csv"
    ).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert Counter(row["level1_label"] for row in rows) == counts
    assert all(row["adjudicated"].lower() == "true" for row in rows)


def test_current_l2_annotation_matches_spatial_contract() -> None:
    payload = json.loads(
        (REPO / "annotations/hcc_l2_roi_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(payload["l2_prototypes"]) == DEFAULT_SPATIAL_COMPONENTS
    assert set(payload["annotations"])

