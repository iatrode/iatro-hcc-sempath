from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..modeling.prototypes import PrototypeRegistry


@dataclass(frozen=True)
class PrototypeImageBank:
    images: torch.Tensor
    tile_ids: list[str]
    level1: torch.Tensor
    level2: torch.Tensor
    names: list[str]
    groups: list[str | None]
    levels: list[int]
    exclusive: list[bool]
    source: dict[str, Any] | None = None
    version: int = 1

    @property
    def count(self) -> int:
        return int(self.images.shape[0])

    @property
    def primary_count(self) -> int:
        return sum(1 for level in self.levels if int(level) == 1)

    @property
    def attribute_count(self) -> int:
        return sum(1 for level in self.levels if int(level) == 2)

    def label_contract(self, embedding_dim: int) -> PrototypeRegistry:
        return PrototypeRegistry(
            prototypes=torch.zeros((len(self.names), int(embedding_dim)), dtype=torch.float32),
            names=list(self.names),
            groups=list(self.groups),
            levels=list(self.levels),
            exclusive=list(self.exclusive),
            source={"kind": "prototype_image_label_contract", **dict(self.source or {})},
        )

    def sample_batch(self, *, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = self.count
        take = min(max(1, int(batch_size)), count)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(count, generator=generator)[:take]
        return (
            self.images.index_select(0, indices),
            self.level1.index_select(0, indices),
            self.level2.index_select(0, indices),
        )


def load_prototype_image_bank(path: str | Path) -> PrototypeImageBank:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"prototype image bank must be a dict payload: {path}")
    images = torch.as_tensor(payload["images"], dtype=torch.uint8)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"prototype image bank images must have shape (N, 3, H, W), got {tuple(images.shape)}")
    tile_ids = [str(tile_id) for tile_id in payload["tile_ids"]]
    level1 = torch.as_tensor(payload["level1"], dtype=torch.long)
    level2 = torch.as_tensor(payload["level2"], dtype=torch.float32)
    names = [str(name) for name in payload["names"]]
    levels = [int(level) for level in payload["levels"]]
    exclusive = [bool(value) for value in payload["exclusive"]]
    groups_payload = payload.get("groups")
    groups = [None if group is None else str(group) for group in groups_payload] if groups_payload is not None else [None] * len(names)
    if len(tile_ids) != images.shape[0] or level1.shape != (images.shape[0],):
        raise ValueError("prototype image bank tile_ids/level1 length does not match images")
    if level2.shape != (images.shape[0], sum(1 for level in levels if level == 2)):
        raise ValueError("prototype image bank level2 shape does not match level-2 prototype count")
    if len(names) != len(levels) or len(names) != len(exclusive) or len(names) != len(groups):
        raise ValueError("prototype image bank prototype metadata lengths differ")
    return PrototypeImageBank(
        images=images.contiguous(),
        tile_ids=tile_ids,
        level1=level1,
        level2=level2,
        names=names,
        groups=groups,
        levels=levels,
        exclusive=exclusive,
        source=payload.get("source") if isinstance(payload.get("source"), dict) else None,
        version=int(payload.get("version", 1)),
    )


@torch.no_grad()
def collect_student_prototype_image_embeddings(
    *,
    model,
    image_bank: PrototypeImageBank,
    cfg: dict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    from .engine import _prepare_images

    was_training = bool(model.training)
    model.eval()
    embeddings = []
    try:
        for start in range(0, image_bank.count, max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), image_bank.count)
            batch = {
                "images": image_bank.images[start:end],
                "images_uint8": True,
            }
            images = _prepare_images(batch, cfg, device)
            outputs = model(images)
            embeddings.append(outputs["embedding_norm"].detach())
    finally:
        model.train(was_training)

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def build_student_prototype_registry(
    *,
    model,
    image_bank: PrototypeImageBank,
    cfg: dict,
    device: torch.device,
    batch_size: int,
) -> PrototypeRegistry:
    embedding = collect_student_prototype_image_embeddings(
        model=model,
        image_bank=image_bank,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
    )
    level1 = image_bank.level1.to(device)
    level2 = image_bank.level2.to(device)
    prototypes = []
    counts = []
    for idx, name in enumerate(name for name, level in zip(image_bank.names, image_bank.levels) if level == 1):
        mask = level1 == idx
        if not bool(mask.any()):
            raise ValueError(f"prototype image bank has no level-1 samples for {name}")
        vector = F.normalize(embedding[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
        prototypes.append(vector)
        counts.append(int(mask.sum().item()))
    for idx, name in enumerate(name for name, level in zip(image_bank.names, image_bank.levels) if level == 2):
        mask = level2[:, idx] > 0.5
        if not bool(mask.any()):
            raise ValueError(f"prototype image bank has no level-2 samples for {name}")
        vector = F.normalize(embedding[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
        prototypes.append(vector)
        counts.append(int(mask.sum().item()))
    return PrototypeRegistry(
        prototypes=torch.stack(prototypes, dim=0),
        names=list(image_bank.names),
        groups=list(image_bank.groups),
        levels=list(image_bank.levels),
        exclusive=list(image_bank.exclusive),
        source={
            "kind": "dynamic_student_prototypes",
            "image_bank_count": image_bank.count,
            "counts": counts,
            **dict(image_bank.source or {}),
        },
    )
