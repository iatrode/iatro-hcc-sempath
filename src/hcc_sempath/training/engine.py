from __future__ import annotations

import torch

from .losses import multi_teacher_distillation_loss
from .metrics import evaluate_teacher_outputs
from .utils import append_csv, ensure_dir, write_json


def _move_teachers(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch["teacher_features"].items()}


def _amp_enabled(device: torch.device, cfg: dict, train: bool) -> bool:
    return bool(train and cfg["train"].get("amp", False) and device.type == "cuda")


def run_epoch(model, loader, prototypes, optimizer, device, cfg, train: bool, scaler=None) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "feature": 0.0, "relation": 0.0, "semantic": 0.0, "reliability": 0.0}
    n_batches = 0
    teacher_weights = cfg["loss"].get("teacher_weights")
    for batch in loader:
        images = batch["images"].to(device)
        teachers = _move_teachers(batch, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=_amp_enabled(device, cfg, train)):
                outputs = model(images)
                student_by_teacher = outputs["teacher_outputs"]
                loss, parts = multi_teacher_distillation_loss(
                    student_by_teacher=student_by_teacher,
                    teacher_by_name=teachers,
                    prototypes_by_teacher=prototypes,
                    relation_weight=float(cfg["loss"]["relation_weight"]),
                    semantic_weight=float(cfg["loss"]["semantic_weight"]),
                    semantic_temperature=float(cfg["loss"]["semantic_temperature"]),
                    teacher_weights=teacher_weights,
                    prototype_filter_weight=float(cfg["loss"].get("prototype_filter_weight", 0.0)),
                    prototype_filter_alpha_min=float(cfg["loss"].get("prototype_filter_alpha_min", 0.25)),
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        for key in ("feature", "relation", "semantic", "reliability"):
            totals[key] += float(parts[key].cpu())
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


@torch.no_grad()
def collect_embeddings(
    model,
    loader,
    device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    embeddings = []
    students_by_teacher: dict[str, list[torch.Tensor]] = {}
    teachers_by_name: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        images = batch["images"].to(device)
        outputs = model(images)
        embeddings.append(outputs["embedding"].cpu())
        for name, tensor in outputs["teacher_outputs"].items():
            students_by_teacher.setdefault(name, []).append(tensor.cpu())
        for name, tensor in batch["teacher_features"].items():
            teachers_by_name.setdefault(name, []).append(tensor.cpu())
    return (
        torch.cat(embeddings),
        {name: torch.cat(values) for name, values in students_by_teacher.items()},
        {name: torch.cat(values) for name, values in teachers_by_name.items()},
    )


def fit(model, train_loader, val_loader, prototypes, optimizer, device, cfg) -> dict:
    output_dir = ensure_dir(cfg["runtime"]["output_dir"])
    checkpoints = ensure_dir(output_dir / "checkpoints")
    write_json(output_dir / "resolved_config.json", cfg)
    best_loss = float("inf")
    best_metrics = {}
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", False) and device.type == "cuda"))
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, prototypes, optimizer, device, cfg, train=True, scaler=scaler)
        val_metrics = run_epoch(model, val_loader, prototypes, optimizer, device, cfg, train=False)
        _, student_by_teacher, teacher_by_name = collect_embeddings(model, val_loader, device)
        cpu_prototypes = {name: registry.to("cpu") for name, registry in prototypes.items()} if prototypes else None
        embedding_metrics = evaluate_teacher_outputs(
            student_by_teacher,
            teacher_by_name,
            cpu_prototypes,
            int(cfg["train"]["topk"]),
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, **embedding_metrics}
        append_csv(output_dir / "metrics.csv", row)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "config": cfg},
            checkpoints / "last.pt",
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = row
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "config": cfg},
                checkpoints / "best.pt",
            )
    write_json(output_dir / "summary.json", best_metrics)
    return best_metrics
