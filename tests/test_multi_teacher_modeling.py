from __future__ import annotations

import torch

from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.modeling.prototypes import PrototypeRegistry
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
    assert outputs["embedding_norm"].shape == (2, 11)
    torch.testing.assert_close(outputs["embedding_norm"].norm(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-5)
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
    prototypes_by_teacher = {
        "teacher_a": PrototypeRegistry(
            prototypes=torch.randn(4, 5),
            names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich", "fibrotic_stroma"],
            groups=["primary_state", "primary_state", "immune", "stroma"],
            levels=[1, 1, 2, 2],
            exclusive=[True, True, False, False],
        ),
        "teacher_b": PrototypeRegistry(
            prototypes=torch.randn(4, 7),
            names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich", "fibrotic_stroma"],
            groups=["primary_state", "primary_state", "immune", "stroma"],
            levels=[1, 1, 2, 2],
            exclusive=[True, True, False, False],
        ),
    }

    loss, parts = multi_teacher_distillation_loss(
        student_by_teacher=student_by_teacher,
        teacher_by_name=teacher_by_name,
        prototypes_by_teacher=prototypes_by_teacher,
        relation_weight=0.25,
        semantic_weight=0.25,
        semantic_temperature=1.0,
    )

    assert loss.ndim == 0
    assert set(parts) == {"feature", "relation", "semantic", "reliability"}
    assert float(parts["reliability"]) == 1.0
    loss.backward()
    assert student_by_teacher["teacher_a"].grad is not None
    assert student_by_teacher["teacher_b"].grad is not None


def test_multi_teacher_distillation_loss_accepts_per_sample_teacher_weights() -> None:
    student_by_teacher = {"teacher": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)}
    teacher_by_name = {"teacher": torch.tensor([[1.0, 0.0], [1.0, 0.0]])}

    loss, parts = multi_teacher_distillation_loss(
        student_by_teacher=student_by_teacher,
        teacher_by_name=teacher_by_name,
        prototypes_by_teacher=None,
        relation_weight=0.0,
        semantic_weight=0.0,
        semantic_temperature=1.0,
        teacher_sample_weights={"teacher": torch.tensor([1.0, 0.25])},
    )

    assert abs(parts["feature"].item() - 0.125) < 1e-6
    assert abs(parts["reliability"].item() - 0.625) < 1e-6
    loss.backward()
    assert student_by_teacher["teacher"].grad is not None
