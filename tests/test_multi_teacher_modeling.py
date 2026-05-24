from __future__ import annotations

import torch

from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.losses import multi_teacher_distillation_loss


def test_hcc_sempath_model_returns_shared_embedding_and_teacher_outputs() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={"teacher_a": 5, "teacher_b": 7},
        pretrained=False,
    )
    outputs = model(torch.randn(2, 3, 224, 224))

    assert outputs["embedding"].shape == (2, 11)
    assert outputs["teacher_outputs"]["teacher_a"].shape == (2, 5)
    assert outputs["teacher_outputs"]["teacher_b"].shape == (2, 7)


def test_multi_teacher_distillation_loss_aggregates_named_heads() -> None:
    student_by_teacher = {
        "teacher_a": torch.randn(4, 5, requires_grad=True),
        "teacher_b": torch.randn(4, 7, requires_grad=True),
    }
    teacher_by_name = {
        "teacher_a": torch.randn(4, 5),
        "teacher_b": torch.randn(4, 7),
    }
    anchors_by_teacher = {
        "teacher_a": torch.randn(3, 5),
        "teacher_b": torch.randn(3, 7),
    }

    loss, parts = multi_teacher_distillation_loss(
        student_by_teacher=student_by_teacher,
        teacher_by_name=teacher_by_name,
        anchors_by_teacher=anchors_by_teacher,
        relation_weight=0.25,
        semantic_weight=0.25,
        semantic_temperature=1.0,
    )

    assert loss.ndim == 0
    assert set(parts) == {"feature", "relation", "semantic"}
    loss.backward()
    assert student_by_teacher["teacher_a"].grad is not None
    assert student_by_teacher["teacher_b"].grad is not None
