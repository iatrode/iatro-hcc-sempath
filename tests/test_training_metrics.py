from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hcc_sempath.training.metrics import (
    evaluate_teacher_outputs,
    retrieval_overlap,
)


def _reference_retrieval_overlap(
    student: torch.Tensor,
    teacher: torch.Tensor,
    topk: int,
) -> float:
    k = min(topk + 1, student.shape[0])
    if k <= 1:
        return 0.0
    student_sim = F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).T
    teacher_sim = F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).T
    student_idx = student_sim.topk(k=k, dim=1).indices[:, 1:]
    teacher_idx = teacher_sim.topk(k=k, dim=1).indices[:, 1:]
    values = [
        len(set(student_row.tolist()) & set(teacher_row.tolist())) / (k - 1)
        for student_row, teacher_row in zip(student_idx, teacher_idx)
    ]
    return sum(values) / len(values)


@pytest.mark.parametrize("sample_count", [1, 4, 17])
def test_vectorized_retrieval_overlap_matches_reference(sample_count: int) -> None:
    generator = torch.Generator().manual_seed(13 + sample_count)
    student = torch.randn(sample_count, 8, generator=generator)
    teacher = torch.randn(sample_count, 8, generator=generator)

    observed = retrieval_overlap(student, teacher, topk=3)
    expected = _reference_retrieval_overlap(student, teacher, topk=3)

    assert observed == pytest.approx(expected)


def test_teacher_evaluation_device_preserves_metrics() -> None:
    generator = torch.Generator().manual_seed(13)
    students = {"teacher": torch.randn(19, 8, generator=generator)}
    teachers = {"teacher": torch.randn(19, 8, generator=generator)}

    baseline = evaluate_teacher_outputs(
        students,
        teachers,
        None,
        topk=3,
        max_pairwise_samples=11,
    )
    explicit_cpu = evaluate_teacher_outputs(
        students,
        teachers,
        None,
        topk=3,
        max_pairwise_samples=11,
        evaluation_device="cpu",
    )

    assert explicit_cpu == pytest.approx(baseline)
