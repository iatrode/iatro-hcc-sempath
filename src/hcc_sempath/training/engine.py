from __future__ import annotations

import time

import torch

from .losses import multi_teacher_distillation_loss
from .metrics import evaluate_teacher_outputs
from .utils import append_csv, ensure_dir, write_json


def _move_teachers(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch["teacher_features"].items()}


def _amp_enabled(device: torch.device, cfg: dict, train: bool) -> bool:
    return bool(train and cfg["train"].get("amp", False) and device.type == "cuda")


def _linear_warmup(base_value: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return base_value
    return base_value * min(1.0, max(0.0, epoch / warmup_epochs))


def _log(message: str) -> None:
    print(message, flush=True)


def _cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / (1024 * 1024))


def scheduled_loss_config(cfg: dict, epoch: int) -> dict[str, float | dict]:
    loss_cfg = cfg["loss"]
    return {
        "teacher_weights": loss_cfg.get("teacher_weights"),
        "relation_weight": float(loss_cfg["relation_weight"]),
        "semantic_weight": _linear_warmup(
            float(loss_cfg.get("semantic_weight", 0.0)),
            epoch,
            int(loss_cfg.get("semantic_warmup_epochs", 0)),
        ),
        "semantic_temperature": float(loss_cfg["semantic_temperature"]),
        "prototype_filter_weight": _linear_warmup(
            float(loss_cfg.get("prototype_filter_weight", 0.0)),
            epoch,
            int(loss_cfg.get("prototype_filter_warmup_epochs", 0)),
        ),
        "prototype_filter_alpha_min": float(loss_cfg.get("prototype_filter_alpha_min", 0.25)),
    }


def run_epoch(
    model,
    loader,
    prototypes,
    optimizer,
    device,
    cfg,
    train: bool,
    scaler=None,
    loss_cfg: dict | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "feature": 0.0, "relation": 0.0, "semantic": 0.0, "reliability": 0.0}
    n_batches = 0
    n_tiles = 0
    start = time.perf_counter()
    interval_start = start
    interval_tiles = 0
    phase = "train" if train else "val"
    log_interval = int(cfg["train"].get("log_interval", 0) or 0)
    loss_cfg = loss_cfg or scheduled_loss_config(cfg, epoch=1)
    teacher_weights = loss_cfg.get("teacher_weights")
    iterator = iter(loader)
    while True:
        data_wait_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait = time.perf_counter() - data_wait_start
        if max_batches is not None and n_batches >= max_batches:
            break
        batch_start = time.perf_counter()
        images = batch["images"].to(device)
        n_tiles += int(images.shape[0])
        interval_tiles += int(images.shape[0])
        teachers = _move_teachers(batch, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=_amp_enabled(device, cfg, train)):
                outputs = model(images)
                student_by_teacher = outputs["teacher_outputs"]
                loss, parts = multi_teacher_distillation_loss(
                    student_by_teacher=student_by_teacher,
                    teacher_by_name=teachers,
                    prototypes_by_teacher=prototypes,
                    relation_weight=float(loss_cfg["relation_weight"]),
                    semantic_weight=float(loss_cfg["semantic_weight"]),
                    semantic_temperature=float(loss_cfg["semantic_temperature"]),
                    teacher_weights=teacher_weights,
                    prototype_filter_weight=float(loss_cfg["prototype_filter_weight"]),
                    prototype_filter_alpha_min=float(loss_cfg["prototype_filter_alpha_min"]),
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
        if log_interval > 0 and (n_batches == 1 or n_batches % log_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            interval_elapsed = max(now - interval_start, 1e-9)
            total_elapsed = max(now - start, 1e-9)
            batch_elapsed = max(now - batch_start, 1e-9)
            _log(
                f"{phase}_progress "
                f"batch={n_batches} tiles={n_tiles} "
                f"interval_tiles_per_sec={interval_tiles / interval_elapsed:.2f} "
                f"total_tiles_per_sec={n_tiles / total_elapsed:.2f} "
                f"last_data_wait_sec={data_wait:.3f} "
                f"last_batch_sec={batch_elapsed:.3f} "
                f"loss={float(loss.detach().cpu()):.6f} "
                f"cuda_mem_mb={_cuda_memory_mb(device):.1f}"
            )
            interval_start = now
            interval_tiles = 0
    elapsed = max(time.perf_counter() - start, 1e-9)
    result = {key: value / max(1, n_batches) for key, value in totals.items()}
    result["tiles_per_sec"] = n_tiles / elapsed
    result["tiles"] = float(n_tiles)
    result["seconds"] = elapsed
    return result


@torch.no_grad()
def collect_embeddings(
    model,
    loader,
    device,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    embeddings = []
    students_by_teacher: dict[str, list[torch.Tensor]] = {}
    teachers_by_name: dict[str, list[torch.Tensor]] = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
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
        loss_cfg = scheduled_loss_config(cfg, epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            prototypes,
            optimizer,
            device,
            cfg,
            train=True,
            scaler=scaler,
            loss_cfg=loss_cfg,
            max_batches=cfg["train"].get("max_train_batches"),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            prototypes,
            optimizer,
            device,
            cfg,
            train=False,
            loss_cfg=loss_cfg,
            max_batches=cfg["train"].get("max_val_batches"),
        )
        _, student_by_teacher, teacher_by_name = collect_embeddings(
            model,
            val_loader,
            device,
            max_batches=cfg["train"].get("max_eval_batches", cfg["train"].get("max_val_batches")),
        )
        cpu_prototypes = {name: registry.to("cpu") for name, registry in prototypes.items()} if prototypes else None
        embedding_metrics = evaluate_teacher_outputs(
            student_by_teacher,
            teacher_by_name,
            cpu_prototypes,
            int(cfg["train"]["topk"]),
        )
        row = {
            "epoch": epoch,
            "scheduled_semantic_weight": float(loss_cfg["semantic_weight"]),
            "scheduled_prototype_filter_weight": float(loss_cfg["prototype_filter_weight"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **embedding_metrics,
        }
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
