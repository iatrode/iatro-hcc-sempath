from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class StudentEncoder(nn.Module):
    """Lightweight tile encoder with a projection head into teacher space."""

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        teacher_dim: int = 256,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        student_dim = int(self.backbone.num_features)
        self.projector = nn.Sequential(
            nn.LayerNorm(student_dim),
            nn.Linear(student_dim, teacher_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projector(features)


class ToyTeacherEncoder(nn.Module):
    """Small frozen teacher used only for local smoke tests."""

    def __init__(self, teacher_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, teacher_dim)
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images).flatten(1)
        return self.head(x)


def normalized_anchor_logits(features: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    anchors = F.normalize(anchors, dim=-1)
    return features @ anchors.transpose(0, 1)

