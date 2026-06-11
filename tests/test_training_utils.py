from __future__ import annotations

import csv

from hcc_sempath.training.utils import append_csv


def test_append_csv_extends_header_when_metrics_expand(tmp_path):
    path = tmp_path / "metrics.csv"

    append_csv(path, {"epoch": 1, "loss": 0.5})
    append_csv(path, {"epoch": 2, "loss": 0.4, "teacher_alignment_score": 0.6})

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["epoch", "loss", "teacher_alignment_score"]
    assert rows == [
        {"epoch": "1", "loss": "0.5", "teacher_alignment_score": ""},
        {"epoch": "2", "loss": "0.4", "teacher_alignment_score": "0.6"},
    ]
