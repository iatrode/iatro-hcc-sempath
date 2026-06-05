from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class StudentEncoder(nn.Module):
    """Lightweight tile encoder that produces the reusable HCC embedding."""

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        embedding_dim: int = 256,
        pretrained: bool = False,
        projector_type: str = "linear",
        projector_hidden_dim: int = 2048,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        student_dim = int(self.backbone.num_features)
        if projector_type == "linear":
            self.projector = nn.Sequential(
                nn.LayerNorm(student_dim),
                nn.Linear(student_dim, embedding_dim),
            )
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.LayerNorm(student_dim),
                nn.Linear(student_dim, int(projector_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(projector_hidden_dim), embedding_dim),
            )
        else:
            raise ValueError(f"unsupported projector_type: {projector_type}")
        self.embedding_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projector(features)


class TeacherProjectionHead(nn.Module):
    """Teacher-specific alignment head used during distillation."""

    def __init__(
        self,
        embedding_dim: int,
        teacher_dim: int,
        head_type: str = "linear",
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if head_type == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, teacher_dim),
            )
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), teacher_dim),
            )
        else:
            raise ValueError(f"unsupported teacher_head_type: {head_type}")

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class HCCSemPathModel(nn.Module):
    """Shared HCC encoder with teacher-specific training heads."""

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        embedding_dim: int = 256,
        teacher_dims: dict[str, int] | None = None,
        pretrained: bool = False,
        projector_type: str = "linear",
        projector_hidden_dim: int = 2048,
        teacher_head_type: str = "linear",
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not teacher_dims:
            teacher_dims = {"teacher": embedding_dim}
        self.encoder = StudentEncoder(
            backbone_name=backbone_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            projector_type=projector_type,
            projector_hidden_dim=projector_hidden_dim,
            grad_checkpointing=grad_checkpointing,
        )
        self.teacher_heads = nn.ModuleDict(
            {
                name: TeacherProjectionHead(
                    embedding_dim,
                    int(dim),
                    head_type=teacher_head_type,
                    hidden_dim=projector_hidden_dim,
                )
                for name, dim in teacher_dims.items()
            }
        )

    @property
    def teacher_names(self) -> list[str]:
        return list(self.teacher_heads.keys())

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def project_teachers(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(embedding) for name, head in self.teacher_heads.items()}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        embedding = self.encode(images)
        embedding_norm = F.normalize(embedding, dim=-1)
        return {
            "embedding": embedding,
            "embedding_norm": embedding_norm,
            "teacher_outputs": self.project_teachers(embedding),
        }


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


def normalized_prototype_logits(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return features @ prototypes.transpose(0, 1)
