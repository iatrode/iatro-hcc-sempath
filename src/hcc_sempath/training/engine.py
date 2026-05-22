from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import total_distillation_loss
from .metrics import evaluate_embeddings
from .utils import append_csv, ensure_dir, write_json


def run_epoch(model, loader, anchors, optimizer, device, cfg, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "feature": 0.0, "relation": 0.0, "semantic": 0.0}
    n_batches = 0
    for batch in loader:
        images = batch["images"].to(device)
        teacher = batch["teacher_features"].to(device)
        with torch.set_grad_enabled(train):
            student = model(images)
            loss, parts = total_distillation_loss(
                student=student,
                teacher=teacher,
                anchors=anchors,
                relation_weight=float(cfg["loss"]["relation_weight"]),
                semantic_weight=float(cfg["loss"]["semantic_weight"]),
                semantic_temperature=float(cfg["loss"]["semantic_temperature"]),
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        for key in ("feature", "relation", "semantic"):
            totals[key] += float(parts[key].cpu())
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


@torch.no_grad()
def collect_embeddings(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    students = []
    teachers = []
    for batch in loader:
        images = batch["images"].to(device)
        students.append(model(images).cpu())
        teachers.append(batch["teacher_features"].cpu())
    return torch.cat(students), torch.cat(teachers)


def fit(model, train_loader, val_loader, anchors, optimizer, device, cfg) -> dict:
    output_dir = ensure_dir(cfg["runtime"]["output_dir"])
    checkpoints = ensure_dir(output_dir / "checkpoints")
    write_json(output_dir / "resolved_config.json", cfg)
    best_loss = float("inf")
    best_metrics = {}
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, anchors, optimizer, device, cfg, train=True)
        val_metrics = run_epoch(model, val_loader, anchors, optimizer, device, cfg, train=False)
        student, teacher = collect_embeddings(model, val_loader, device)
        embedding_metrics = evaluate_embeddings(student, teacher, anchors.cpu(), int(cfg["train"]["topk"]))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, **embedding_metrics}
        append_csv(output_dir / "metrics.csv", row)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": cfg},
            checkpoints / "last.pt",
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = row
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": cfg},
                checkpoints / "best.pt",
            )
    write_json(output_dir / "summary.json", best_metrics)
    return best_metrics
