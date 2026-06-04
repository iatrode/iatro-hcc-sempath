from __future__ import annotations

from pathlib import Path

import torch

from hcc_sempath.training.prototype_images import (
    PrototypeImageBank,
    build_student_prototype_registry,
    load_prototype_image_bank,
)


class _ToyStudent(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict:
        flat = images.flatten(start_dim=1)
        embedding = torch.stack([flat.mean(dim=1), flat.std(dim=1)], dim=1)
        return {"embedding_norm": torch.nn.functional.normalize(embedding, dim=1), "teacher_outputs": {}}


def _bank() -> PrototypeImageBank:
    images = torch.zeros((4, 3, 4, 4), dtype=torch.uint8)
    images[1:] = 255
    return PrototypeImageBank(
        images=images,
        tile_ids=["a", "b", "c", "d"],
        level1=torch.tensor([0, 0, 1, 1]),
        level2=torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        names=["tumor", "non_tumor", "fibrosis"],
        groups=["primary_state", "primary_state", "attribute_presence"],
        levels=[1, 1, 2],
        exclusive=[True, True, False],
    )


def test_prototype_image_bank_round_trip(tmp_path: Path) -> None:
    bank = _bank()
    path = tmp_path / "zhcc_hcc_prototype_images.pt"
    torch.save(
        {
            "version": 1,
            "images": bank.images,
            "tile_ids": bank.tile_ids,
            "level1": bank.level1,
            "level2": bank.level2,
            "names": bank.names,
            "groups": bank.groups,
            "levels": bank.levels,
            "exclusive": bank.exclusive,
        },
        path,
    )

    loaded = load_prototype_image_bank(path)

    assert loaded.count == 4
    assert loaded.label_contract(2).names == ["tumor", "non_tumor", "fibrosis"]


def test_build_student_prototypes_from_current_model() -> None:
    registry = build_student_prototype_registry(
        model=_ToyStudent(),
        image_bank=_bank(),
        cfg={"data": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}},
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert registry.prototypes.shape == (3, 2)
    assert registry.names == ["tumor", "non_tumor", "fibrosis"]
    assert torch.allclose(registry.prototypes.norm(dim=1), torch.ones(3), atol=1e-6)
