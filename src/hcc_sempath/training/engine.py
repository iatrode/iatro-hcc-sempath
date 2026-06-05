from __future__ import annotations

import math
from numbers import Number
import random
import time

import numpy as np
import torch

from ..modeling.prototypes import PrototypeRegistry
from .adjudication import prototype_adjudicated_teacher_weights
from .losses import multi_teacher_distillation_loss
from .metrics import evaluate_teacher_outputs
from .prototype_images import (
    PrototypeImageBank,
    build_student_prototype_registry,
    collect_student_prototype_image_embeddings,
)
from .utils import append_csv, ensure_dir, write_json
from .zhcc_losses import teacher_semantic_response_target, zhcc_response_distillation_loss
from .zhcc_metrics import evaluate_zhcc_prototypes


def _move_teachers(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch["teacher_features"].items()
    }


def _prepare_images(batch: dict, cfg: dict, device: torch.device) -> torch.Tensor:
    images = batch["images"].to(device, non_blocking=device.type == "cuda")
    if not bool(batch.get("images_uint8", False)):
        return images
    images = images.to(torch.float32).div_(255.0)
    mean = torch.tensor(cfg["data"].get("mean", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"].get("std", [1.0, 1.0, 1.0]), dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return images.sub_(mean).div_(std)


def _move_prototype_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch.get("prototype_mask", torch.zeros(len(batch["tile_id"]), dtype=torch.bool)).to(device),
        batch.get("prototype_level1", torch.full((len(batch["tile_id"]),), -1, dtype=torch.long)).to(device),
        batch.get("prototype_level2", torch.zeros((len(batch["tile_id"]), 0), dtype=torch.float32)).to(device),
    )


def _amp_enabled(device: torch.device, cfg: dict, train: bool) -> bool:
    return bool(train and cfg["train"].get("amp", False) and device.type == "cuda")


def _linear_warmup(base_value: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return base_value
    return base_value * min(1.0, max(0.0, epoch / warmup_epochs))


def _step_ramp(target: float, global_step: int, start_step: int | None, ramp_steps: int) -> float:
    if start_step is None or global_step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return float(target)
    progress = (global_step - start_step) / float(ramp_steps)
    return float(target) * max(0.0, min(1.0, progress))


def default_schedule_state() -> dict:
    return {
        "teacher_prior_loss_ema": None,
        "teacher_prior_prev_window_ema": None,
        "teacher_prior_window_loss_sum": 0.0,
        "teacher_prior_window_count": 0,
        "teacher_prior_relative_improvement": None,
        "teacher_prior_plateau_count": 0,
        "prototype_start_step": None,
        "filter_start_step": None,
    }


def _ensure_schedule_state(schedule_state: dict | None) -> dict:
    defaults = default_schedule_state()
    if schedule_state is None:
        return defaults
    for key, value in defaults.items():
        schedule_state.setdefault(key, value)
    return schedule_state


def _set_intervention_steps(cfg: dict, schedule_state: dict, global_step: int) -> None:
    if schedule_state.get("prototype_start_step") is not None:
        return
    delay = int(cfg["loss"].get("proto_to_filter_delay_steps", 1000))
    schedule_state["prototype_start_step"] = int(global_step)
    schedule_state["filter_start_step"] = int(global_step) + delay


def update_plateau_schedule_state(
    cfg: dict,
    schedule_state: dict,
    *,
    global_step: int,
    teacher_prior_loss: torch.Tensor | float,
) -> dict:
    loss_cfg = cfg["loss"]
    schedule_state = _ensure_schedule_state(schedule_state)
    if schedule_state.get("prototype_start_step") is not None:
        return schedule_state

    max_warmup = int(loss_cfg.get("max_teacher_warmup_steps", 10000))
    if global_step >= max_warmup:
        _set_intervention_steps(cfg, schedule_state, global_step)
        return schedule_state

    current_sum = schedule_state.get("teacher_prior_window_loss_sum")
    if isinstance(teacher_prior_loss, torch.Tensor):
        if current_sum is None or isinstance(current_sum, float):
            current_sum = torch.zeros((), device=teacher_prior_loss.device)
        elif current_sum.device != teacher_prior_loss.device:
            current_sum = current_sum.to(teacher_prior_loss.device)
        schedule_state["teacher_prior_window_loss_sum"] = current_sum + teacher_prior_loss.detach()
    else:
        schedule_state["teacher_prior_window_loss_sum"] = float(current_sum or 0.0) + float(teacher_prior_loss)

    schedule_state["teacher_prior_window_count"] = int(schedule_state.get("teacher_prior_window_count") or 0) + 1
    window_steps = max(1, int(loss_cfg.get("teacher_prior_plateau_window_steps", 1000)))
    if int(schedule_state["teacher_prior_window_count"]) < window_steps:
        return schedule_state

    window_sum = schedule_state["teacher_prior_window_loss_sum"]
    if isinstance(window_sum, torch.Tensor):
        window_sum_val = float(window_sum.cpu())
    else:
        window_sum_val = float(window_sum)

    window_mean = window_sum_val / float(schedule_state["teacher_prior_window_count"])
    beta = float(loss_cfg.get("teacher_prior_ema_beta", 0.9))
    current_ema = schedule_state.get("teacher_prior_loss_ema")
    if current_ema is None:
        current_ema = window_mean
    else:
        current_ema = beta * float(current_ema) + (1.0 - beta) * window_mean
    prev_ema = schedule_state.get("teacher_prior_prev_window_ema")
    schedule_state["teacher_prior_loss_ema"] = float(current_ema)
    
    if isinstance(teacher_prior_loss, torch.Tensor):
        schedule_state["teacher_prior_window_loss_sum"] = torch.zeros((), device=teacher_prior_loss.device)
    else:
        schedule_state["teacher_prior_window_loss_sum"] = 0.0
    schedule_state["teacher_prior_window_count"] = 0

    if prev_ema is not None:
        relative_improvement = (float(prev_ema) - float(current_ema)) / max(float(prev_ema), 1e-8)
        schedule_state["teacher_prior_relative_improvement"] = float(relative_improvement)
        min_warmup = int(loss_cfg.get("min_teacher_warmup_steps", 2000))
        threshold = float(loss_cfg.get("teacher_prior_plateau_threshold", 0.01))
        if global_step >= min_warmup and relative_improvement < threshold:
            schedule_state["teacher_prior_plateau_count"] = int(schedule_state.get("teacher_prior_plateau_count") or 0) + 1
        else:
            schedule_state["teacher_prior_plateau_count"] = 0
        patience = int(loss_cfg.get("teacher_prior_plateau_patience", 2))
        if int(schedule_state["teacher_prior_plateau_count"]) >= patience:
            _set_intervention_steps(cfg, schedule_state, global_step)
    schedule_state["teacher_prior_prev_window_ema"] = float(current_ema)
    return schedule_state


def intervention_stage(cfg: dict, global_step: int, schedule_state: dict | None) -> str:
    state = _ensure_schedule_state(schedule_state)
    prototype_start = state.get("prototype_start_step")
    filter_start = state.get("filter_start_step")
    if prototype_start is None or global_step < int(prototype_start):
        return "teacher_prior"
    prototype_ramp_steps = int(cfg["loss"].get("prototype_ramp_steps", 1000))
    if global_step < int(prototype_start) + max(0, prototype_ramp_steps):
        return "prototype_ramp"
    if filter_start is None or global_step < int(filter_start):
        return "prototype_active"
    filter_ramp_steps = int(cfg["loss"].get("filter_ramp_steps", 1000))
    if global_step < int(filter_start) + max(0, filter_ramp_steps):
        return "filter_ramp"
    return "pamtd_active"


def _log(message: str) -> None:
    try:
        from tqdm.auto import tqdm
        tqdm.write(message)
    except ImportError:
        print(message, flush=True)


def _cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / (1024 * 1024))





def _build_summary_writer(cfg: dict, output_dir):
    if not bool(cfg["train"].get("tensorboard", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        _log("tensorboard_unavailable reason=torch.utils.tensorboard import failed")
        return None
    log_dir = cfg["train"].get("tensorboard_log_dir") or str(output_dir / "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    _log(f"tensorboard_log_dir={log_dir}")
    return writer


def _build_progress_bar(cfg: dict, *, phase: str, epoch: int, total: int):
    progress = str(cfg["train"].get("progress", "none")).lower()
    if progress not in {"1", "true", "yes", "tqdm"}:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        _log("tqdm_unavailable reason=tqdm import failed")
        return None
    return tqdm(
        total=total,
        desc=f"{phase} epoch {epoch}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        mininterval=float(cfg["train"].get("progress_interval_sec", 5.0)),
    )


def _write_tensorboard_scalars(writer, row: dict, epoch: int) -> None:
    if writer is None:
        return
    for key, value in row.items():
        if isinstance(value, bool) or not isinstance(value, Number):
            continue
        scalar = float(value)
        if not math.isfinite(scalar):
            continue
        if key.startswith("train_"):
            tag = "train/" + key.removeprefix("train_")
        elif key.startswith("val_"):
            tag = "val/" + key.removeprefix("val_")
        elif key in {"lr", "global_step"}:
            tag = "train/" + key
        elif key.startswith("scheduled_"):
            tag = "schedule/" + key.removeprefix("scheduled_")
        else:
            tag = "metrics/" + key
        writer.add_scalar(tag, scalar, epoch)
    writer.flush()


def _write_tensorboard_batch(
    writer,
    *,
    phase: str,
    global_step: int,
    loss: torch.Tensor,
    parts: dict[str, torch.Tensor],
    lr: float,
) -> None:
    if writer is None:
        return
    writer.add_scalar(f"{phase}_batch/loss", float(loss.detach().cpu()), global_step)
    for key in ("feature", "relation", "semantic", "zhcc_proto", "zhcc_response"):
        if key in parts:
            writer.add_scalar(f"{phase}_batch/{key}", float(parts[key].detach().cpu()), global_step)
    writer.add_scalar("train/lr_step", float(lr), global_step)


def _maybe_refresh_dynamic_prototypes(
    *,
    model,
    cfg: dict,
    device: torch.device,
    zhcc_image_bank: PrototypeImageBank | None,
    prototype_state: dict | None,
    global_step: int,
    needed: bool,
) -> PrototypeRegistry | None:
    if prototype_state is None:
        return None
    current = prototype_state.get("zhcc")
    if not needed:
        return current
    if zhcc_image_bank is None:
        return current
    interval = int(cfg["train"].get("dynamic_prototype_refresh_steps", 500))
    last_step = prototype_state.get("last_refresh_step")
    should_refresh = current is None or last_step is None or int(global_step) - int(last_step) >= interval
    if not should_refresh:
        return current
    batch_size = int(cfg["train"].get("dynamic_prototype_batch_size", cfg["train"].get("batch_size", 512)))
    refreshed = build_student_prototype_registry(
        model=model,
        image_bank=zhcc_image_bank,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
    )
    prototype_state["zhcc"] = refreshed
    prototype_state["last_refresh_step"] = int(global_step)
    _log(
        "dynamic_prototypes_refreshed "
        f"global_step={global_step} image_count={zhcc_image_bank.count} "
        f"batch_size={batch_size} prototypes={refreshed.count}"
    )
    return refreshed


def scheduled_loss_config(
    cfg: dict,
    *,
    epoch: int,
    global_step: int,
    schedule_state: dict | None = None,
) -> dict[str, float | dict | bool | str | None]:
    loss_cfg = cfg["loss"]
    semantic_temperature = float(loss_cfg.get("semantic_temperature", 1.0))
    state = _ensure_schedule_state(schedule_state)
    prototype_start = state.get("prototype_start_step")
    filter_start = state.get("filter_start_step")
    zhcc_proto_weight = _step_ramp(
        float(loss_cfg.get("zhcc_proto_weight", 0.0)),
        int(global_step),
        int(prototype_start) if prototype_start is not None else None,
        int(loss_cfg.get("prototype_ramp_steps", 1000)),
    )
    prototype_filter_weight = _step_ramp(
        float(loss_cfg.get("prototype_filter_weight", 0.0)),
        int(global_step),
        int(filter_start) if filter_start is not None else None,
        int(loss_cfg.get("filter_ramp_steps", 1000)),
    )
    zhcc_response_weight = _step_ramp(
        float(loss_cfg.get("zhcc_response_weight", 0.0)),
        int(global_step),
        int(filter_start) if filter_start is not None else None,
        int(loss_cfg.get("filter_ramp_steps", 1000)),
    )
    return {
        "teacher_weights": loss_cfg.get("teacher_weights"),
        "relation_weight": float(loss_cfg["relation_weight"]),
        "semantic_weight": _linear_warmup(
            float(loss_cfg.get("semantic_weight", 0.0)),
            epoch,
            int(loss_cfg.get("semantic_warmup_epochs", 0)),
        ),
        "semantic_temperature": semantic_temperature,
        "primary_temperature": float(loss_cfg.get("primary_temperature", semantic_temperature)),
        "attribute_temperature": float(loss_cfg.get("attribute_temperature", 1.0)),
        "prototype_filter_weight": prototype_filter_weight,
        "prototype_filter_alpha_min": float(loss_cfg.get("prototype_filter_alpha_min", 0.25)),
        "feature_loss_type": str(loss_cfg.get("feature_loss_type", "cosine")),
        "zhcc_proto_weight": zhcc_proto_weight,
        "zhcc_level2_weight": float(loss_cfg.get("zhcc_level2_weight", 0.5)),
        "zhcc_primary_temperature": float(loss_cfg.get("zhcc_primary_temperature", 0.1)),
        "zhcc_attribute_temperature": float(loss_cfg.get("zhcc_attribute_temperature", 0.1)),
        "consensus_weight": float(loss_cfg.get("consensus_weight", 0.4)),
        "prototype_label_weight": float(loss_cfg.get("prototype_label_weight", 0.4)),
        "prototype_l1_agreement_weight": float(loss_cfg.get("prototype_l1_agreement_weight", 0.5)),
        "prototype_l2_agreement_weight": float(loss_cfg.get("prototype_l2_agreement_weight", 0.5)),
        "zhcc_response_weight": zhcc_response_weight,
        "scale_relation_by_alpha": bool(loss_cfg.get("scale_relation_by_alpha", False)),
        "prototype_start_step": prototype_start,
        "filter_start_step": filter_start,
        "teacher_prior_loss_ema": state.get("teacher_prior_loss_ema"),
        "teacher_prior_relative_improvement": state.get("teacher_prior_relative_improvement"),
        "teacher_prior_plateau_count": int(state.get("teacher_prior_plateau_count") or 0),
        "intervention_stage": intervention_stage(cfg, int(global_step), state),
    }


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    if str(cfg["train"].get("scheduler", "none")).lower() != "cosine":
        return None
    total_steps = max(1, int(cfg["train"]["epochs"]) * max(1, int(steps_per_epoch)))
    warmup_steps = max(0, int(cfg["train"].get("warmup_epochs", 0)) * max(1, int(steps_per_epoch)))
    base_lr = float(cfg["train"]["lr"])
    min_factor = float(cfg["train"].get("min_lr", 0.0)) / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_factor, float(step + 1) / float(warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def run_epoch(
    model,
    loader,
    prototypes,
    optimizer,
    device,
    cfg,
    train: bool,
    scaler=None,
    scheduler=None,
    max_batches: int | None = None,
    zhcc_prototypes=None,
    zhcc_image_bank: PrototypeImageBank | None = None,
    zhcc_prototype_state: dict | None = None,
    epoch: int = 1,
    global_step: int = 0,
    schedule_state: dict | None = None,
    summary_writer=None,
    collect_embeddings: bool = False,
    max_eval_batches: int | None = None,
) -> dict[str, float] | tuple[dict[str, float], tuple]:
    model.train(train)
    schedule_state = _ensure_schedule_state(schedule_state)
    totals = {
        "loss": 0.0,
        "feature": 0.0,
        "relation": 0.0,
        "semantic": 0.0,
        "reliability": 0.0,
        "relation_scale": 0.0,
        "zhcc_proto": 0.0,
        "zhcc_response": 0.0,
        "zhcc_l1": 0.0,
        "zhcc_l2": 0.0,
    }
    n_batches = 0
    n_tiles = 0
    start = time.perf_counter()
    interval_start = start
    interval_tiles = 0
    phase = "train" if train else "val"
    log_interval = int(cfg["train"].get("log_interval", 0) or 0)
    tensorboard_batch_interval = int(cfg["train"].get("tensorboard_batch_interval", 0) or 0)
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0) or 0.0)
    last_loss_cfg = scheduled_loss_config(
        cfg,
        epoch=epoch,
        global_step=global_step,
        schedule_state=schedule_state,
    )
    if zhcc_prototype_state is None:
        zhcc_prototype_state = {"zhcc": zhcc_prototypes, "last_refresh_step": None}
    elif "zhcc" not in zhcc_prototype_state:
        zhcc_prototype_state["zhcc"] = zhcc_prototypes
    embeddings_data = None
    if collect_embeddings:
        embeddings_data = {
            "embeddings": [],
            "prototype_masks": [],
            "prototype_level1": [],
            "prototype_level2": [],
            "students_by_teacher": {},
            "teachers_by_name": {},
        }
    iterator = iter(loader)
    progress_total = len(loader)
    if max_batches is not None:
        progress_total = min(progress_total, int(max_batches))
    progress_bar = _build_progress_bar(cfg, phase=phase, epoch=epoch, total=progress_total)
    try:
        while True:
            data_wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait = time.perf_counter() - data_wait_start
            if max_batches is not None and n_batches >= max_batches:
                break
            will_log = (
                progress_bar is None
                and log_interval > 0
                and (n_batches + 1 == 1 or (n_batches + 1) % log_interval == 0)
            )
            batch_start = time.perf_counter()
            image_prepare_start = time.perf_counter()
            images = _prepare_images(batch, cfg, device)
            if will_log and device.type == "cuda":
                torch.cuda.synchronize(device)
            image_prepare = time.perf_counter() - image_prepare_start
            n_tiles += int(images.shape[0])
            interval_tiles += int(images.shape[0])
            teachers = _move_teachers(batch, device)
            prototype_mask, prototype_level1, prototype_level2 = _move_prototype_batch(batch, device)
            with torch.set_grad_enabled(train):
                loss_cfg = scheduled_loss_config(
                    cfg,
                    epoch=epoch,
                    global_step=global_step,
                    schedule_state=schedule_state,
                )
                last_loss_cfg = loss_cfg
                active_zhcc_prototypes = _maybe_refresh_dynamic_prototypes(
                    model=model,
                    cfg=cfg,
                    device=device,
                    zhcc_image_bank=zhcc_image_bank,
                    prototype_state=zhcc_prototype_state,
                    global_step=global_step,
                    needed=bool(
                        train
                        and (
                            float(loss_cfg["zhcc_proto_weight"]) > 0
                            or float(loss_cfg["prototype_filter_weight"]) > 0
                            or float(loss_cfg["zhcc_response_weight"]) > 0
                        )
                    ),
                )
                with torch.autocast(device_type=device.type, enabled=_amp_enabled(device, cfg, train)):
                    outputs = model(images)
                    student_by_teacher = outputs["teacher_outputs"]
                    if collect_embeddings and (max_eval_batches is None or n_batches < max_eval_batches):
                        embeddings_data["embeddings"].append(outputs["embedding_norm"].detach().cpu())
                        p_mask = batch.get("prototype_mask", torch.zeros(len(batch["tile_id"]), dtype=torch.bool))
                        p_lvl1 = batch.get("prototype_level1", torch.full((len(batch["tile_id"]),), -1, dtype=torch.long))
                        p_lvl2 = batch.get("prototype_level2", torch.zeros((len(batch["tile_id"]), 0), dtype=torch.float32))
                        embeddings_data["prototype_masks"].append(p_mask.detach().cpu())
                        embeddings_data["prototype_level1"].append(p_lvl1.detach().cpu())
                        embeddings_data["prototype_level2"].append(p_lvl2.detach().cpu())
                        for name, tensor in student_by_teacher.items():
                            embeddings_data["students_by_teacher"].setdefault(name, []).append(tensor.detach().cpu())
                        for name, tensor in batch["teacher_features"].items():
                            feat_tensor = tensor
                            if isinstance(feat_tensor, np.ndarray):
                                feat_tensor = torch.from_numpy(feat_tensor)
                            embeddings_data["teachers_by_name"].setdefault(name, []).append(feat_tensor.detach().cpu())
                    if float(loss_cfg["prototype_filter_weight"]) > 0:
                        if prototypes is None:
                            raise ValueError(
                                "prototype adjudication requires data.prototype_paths"
                            )
                        alpha_by_teacher, alpha_diag = prototype_adjudicated_teacher_weights(
                            teacher_by_name=teachers,
                            prototypes_by_teacher=prototypes,
                            zhcc_embedding_norm=outputs["embedding_norm"].detach(),
                            zhcc_prototypes=active_zhcc_prototypes,
                            prototype_mask=prototype_mask,
                            prototype_level1=prototype_level1,
                            prototype_level2=prototype_level2,
                            alpha_min=float(loss_cfg["prototype_filter_alpha_min"]),
                            consensus_weight=float(loss_cfg["consensus_weight"]),
                            prototype_label_weight=float(loss_cfg["prototype_label_weight"]),
                            l1_agreement_weight=float(loss_cfg["prototype_l1_agreement_weight"]),
                            l2_agreement_weight=float(loss_cfg["prototype_l2_agreement_weight"]),
                            zhcc_response_weight=float(loss_cfg["zhcc_response_weight"]),
                            filter_strength=float(loss_cfg["prototype_filter_weight"]),
                            primary_temperature=float(loss_cfg["zhcc_primary_temperature"]),
                            attribute_temperature=float(loss_cfg["zhcc_attribute_temperature"]),
                        )
                    else:
                        alpha_by_teacher = None
                        alpha_diag = {}
                    loss, parts = multi_teacher_distillation_loss(
                        student_by_teacher=student_by_teacher,
                        teacher_by_name=teachers,
                        prototypes_by_teacher=prototypes,
                        relation_weight=float(loss_cfg["relation_weight"]),
                        semantic_weight=float(loss_cfg["semantic_weight"]),
                        semantic_temperature=float(loss_cfg["semantic_temperature"]),
                        teacher_weights=loss_cfg.get("teacher_weights"),
                        teacher_sample_weights=alpha_by_teacher,
                        feature_loss_type=str(loss_cfg["feature_loss_type"]),
                        primary_temperature=float(loss_cfg["primary_temperature"]),
                        attribute_temperature=float(loss_cfg["attribute_temperature"]),
                        scale_relation_by_alpha=bool(loss_cfg["scale_relation_by_alpha"]),
                    )
                    if active_zhcc_prototypes is not None and float(loss_cfg["zhcc_proto_weight"]) > 0:
                        if prototypes is None:
                            raise ValueError("zhcc response distillation requires data.prototype_paths")
                        target_primary, target_attributes = teacher_semantic_response_target(
                            teacher_by_name=teachers,
                            prototypes_by_teacher=prototypes,
                            target_registry=active_zhcc_prototypes,
                            teacher_weights=loss_cfg.get("teacher_weights"),
                            teacher_sample_weights=alpha_by_teacher,
                            primary_temperature=float(loss_cfg["zhcc_primary_temperature"]),
                            attribute_temperature=float(loss_cfg["zhcc_attribute_temperature"]),
                        )
                        zhcc_loss, zhcc_parts = zhcc_response_distillation_loss(
                            embedding_norm=outputs["embedding_norm"],
                            prototypes=active_zhcc_prototypes,
                            target_primary=target_primary,
                            target_attributes=target_attributes,
                            level2_weight=float(loss_cfg["zhcc_level2_weight"]),
                            primary_temperature=float(loss_cfg["zhcc_primary_temperature"]),
                            attribute_temperature=float(loss_cfg["zhcc_attribute_temperature"]),
                        )
                    else:
                        zhcc_loss = loss.new_zeros(())
                        zhcc_parts = {
                            "zhcc_proto": loss.new_zeros(()),
                            "zhcc_response": loss.new_zeros(()),
                            "zhcc_l1": loss.new_zeros(()),
                            "zhcc_l2": loss.new_zeros(()),
                        }
                    loss = loss + float(loss_cfg["zhcc_proto_weight"]) * zhcc_loss
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                        if max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
                    teacher_prior_loss = parts["feature"].detach() + float(loss_cfg["relation_weight"]) * parts["relation"].detach()
                    update_plateau_schedule_state(
                        cfg,
                        schedule_state,
                        global_step=global_step,
                        teacher_prior_loss=teacher_prior_loss,
                    )
            totals["loss"] = totals["loss"] + loss.detach()
            for key in ("feature", "relation", "semantic", "reliability", "relation_scale"):
                totals[key] = totals[key] + parts[key].detach()
            for key in ("zhcc_proto", "zhcc_response", "zhcc_l1", "zhcc_l2"):
                totals[key] = totals[key] + zhcc_parts[key].detach()
            for key, value in alpha_diag.items():
                totals.setdefault(key, 0.0)
                totals[key] = totals[key] + value.detach()
            n_batches += 1
            if (
                summary_writer is not None
                and tensorboard_batch_interval > 0
                and train
                and n_batches % tensorboard_batch_interval == 0
            ):
                _write_tensorboard_batch(
                    summary_writer,
                    phase=phase,
                    global_step=global_step,
                    loss=loss,
                    parts={**parts, **zhcc_parts},
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            if progress_bar is not None:
                if n_batches % 10 == 0 or n_batches == progress_total:
                    elapsed = max(time.perf_counter() - start, 1e-9)
                    progress_bar.set_postfix(
                        loss=f"{float(loss.detach().cpu()):.4f}",
                        tiles_s=f"{n_tiles / elapsed:.0f}",
                        refresh=False,
                    )
                progress_bar.update(1)
            if will_log:
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
                    f"last_image_prepare_sec={image_prepare:.3f} "
                    f"last_batch_sec={batch_elapsed:.3f} "
                    f"image_path={'uint8_device_prepare' if batch.get('images_uint8', False) else 'preprocessed_cpu'} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"cuda_mem_mb={_cuda_memory_mb(device):.1f}"
                )
                interval_start = now
                interval_tiles = 0
    finally:
        if progress_bar is not None:
            progress_bar.close()
    elapsed = max(time.perf_counter() - start, 1e-9)
    result = {}
    for key, value in totals.items():
        mean_value = value / max(1, n_batches)
        if isinstance(mean_value, torch.Tensor):
            result[key] = float(mean_value.detach().cpu())
        else:
            result[key] = float(mean_value)
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    result["tiles_per_sec"] = n_tiles / elapsed
    result["tiles"] = float(n_tiles)
    result["seconds"] = elapsed
    result["global_step_end"] = float(global_step)
    result["scheduled_zhcc_proto_weight"] = float(last_loss_cfg["zhcc_proto_weight"])
    result["scheduled_prototype_filter_weight"] = float(last_loss_cfg["prototype_filter_weight"])
    result["scheduled_zhcc_response_weight"] = float(last_loss_cfg["zhcc_response_weight"])
    result["prototype_start_step"] = float(schedule_state["prototype_start_step"] or -1)
    result["filter_start_step"] = float(schedule_state["filter_start_step"] or -1)
    result["teacher_prior_loss_ema"] = float(schedule_state["teacher_prior_loss_ema"] or 0.0)
    result["teacher_prior_relative_improvement"] = float(schedule_state["teacher_prior_relative_improvement"] or 0.0)
    result["teacher_prior_plateau_count"] = float(schedule_state["teacher_prior_plateau_count"] or 0)
    if collect_embeddings:
        collated_embeddings = (
            torch.cat(embeddings_data["embeddings"]),
            {name: torch.cat(values) for name, values in embeddings_data["students_by_teacher"].items()},
            {name: torch.cat(values) for name, values in embeddings_data["teachers_by_name"].items()},
            {
                "prototype_mask": torch.cat(embeddings_data["prototype_masks"]),
                "prototype_level1": torch.cat(embeddings_data["prototype_level1"]),
                "prototype_level2": torch.cat(embeddings_data["prototype_level2"]),
            },
        )
        return result, collated_embeddings
    return result


@torch.no_grad()
def collect_embeddings(
    model,
    loader,
    device,
    cfg: dict | None = None,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    embeddings = []
    prototype_masks = []
    prototype_level1 = []
    prototype_level2 = []
    students_by_teacher: dict[str, list[torch.Tensor]] = {}
    teachers_by_name: dict[str, list[torch.Tensor]] = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if cfg is not None:
            images = _prepare_images(batch, cfg, device)
        else:
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            if bool(batch.get("images_uint8", False)):
                images = images.to(torch.float32).div_(255.0)
        outputs = model(images)
        embeddings.append(outputs["embedding_norm"].cpu())
        prototype_masks.append(batch.get("prototype_mask", torch.zeros(len(batch["tile_id"]), dtype=torch.bool)).cpu())
        prototype_level1.append(batch.get("prototype_level1", torch.full((len(batch["tile_id"]),), -1, dtype=torch.long)).cpu())
        prototype_level2.append(batch.get("prototype_level2", torch.zeros((len(batch["tile_id"]), 0), dtype=torch.float32)).cpu())
        for name, tensor in outputs["teacher_outputs"].items():
            students_by_teacher.setdefault(name, []).append(tensor.cpu())
        for name, tensor in batch["teacher_features"].items():
            teachers_by_name.setdefault(name, []).append(tensor.cpu())
    return (
        torch.cat(embeddings),
        {name: torch.cat(values) for name, values in students_by_teacher.items()},
        {name: torch.cat(values) for name, values in teachers_by_name.items()},
        {
            "prototype_mask": torch.cat(prototype_masks),
            "prototype_level1": torch.cat(prototype_level1),
            "prototype_level2": torch.cat(prototype_level2),
        },
    )


def fit(
    model,
    train_loader,
    val_loader,
    prototypes,
    optimizer,
    device,
    cfg,
    *,
    scheduler=None,
    zhcc_prototypes=None,
    zhcc_image_bank: PrototypeImageBank | None = None,
    resume_state: dict | None = None,
) -> dict:
    output_dir = ensure_dir(cfg["runtime"]["output_dir"])
    checkpoints = ensure_dir(output_dir / "checkpoints")
    write_json(output_dir / "resolved_config.json", cfg)
    best_loss = float((resume_state or {}).get("best_loss", float("inf")))
    best_teacher_alignment = float((resume_state or {}).get("best_teacher_alignment", float("-inf")))
    best_scientific_score = float((resume_state or {}).get("best_scientific_score", float("-inf")))
    best_metrics = dict((resume_state or {}).get("best_metrics", {}))
    start_epoch = int((resume_state or {}).get("epoch", 0)) + 1
    global_step = int((resume_state or {}).get("global_step", 0))
    schedule_state = _ensure_schedule_state((resume_state or {}).get("schedule_state"))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", False) and device.type == "cuda"))
    if resume_state and "scaler" in resume_state:
        scaler.load_state_dict(resume_state["scaler"])
    _restore_rng_state((resume_state or {}).get("rng_state"))
    writer = _build_summary_writer(cfg, output_dir)
    zhcc_prototype_state = {
        "zhcc": zhcc_prototypes,
        "last_refresh_step": (resume_state or {}).get("zhcc_dynamic_prototype_step"),
    }
    try:
        for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                prototypes,
                optimizer,
                device,
                cfg,
                train=True,
                scaler=scaler,
                scheduler=scheduler,
                max_batches=cfg["train"].get("max_train_batches"),
                zhcc_prototypes=zhcc_prototypes,
                zhcc_image_bank=zhcc_image_bank,
                zhcc_prototype_state=zhcc_prototype_state,
                epoch=epoch,
                global_step=global_step,
                schedule_state=schedule_state,
                summary_writer=writer,
            )
            global_step = int(train_metrics["global_step_end"])
            _maybe_refresh_dynamic_prototypes(
                model=model,
                cfg=cfg,
                device=device,
                zhcc_image_bank=zhcc_image_bank,
                prototype_state=zhcc_prototype_state,
                global_step=global_step,
                needed=zhcc_image_bank is not None,
            )
            val_metrics, val_embeddings = run_epoch(
                model,
                val_loader,
                prototypes,
                optimizer,
                device,
                cfg,
                train=False,
                max_batches=cfg["train"].get("max_val_batches"),
                zhcc_prototypes=zhcc_prototype_state.get("zhcc", zhcc_prototypes),
                zhcc_image_bank=zhcc_image_bank,
                zhcc_prototype_state=zhcc_prototype_state,
                epoch=epoch,
                global_step=global_step,
                schedule_state=schedule_state,
                summary_writer=writer,
                collect_embeddings=True,
                max_eval_batches=cfg["train"].get("max_eval_batches", cfg["train"].get("max_val_batches")),
            )
            current_zhcc_prototypes = zhcc_prototype_state.get("zhcc", zhcc_prototypes)
            embeddings, student_by_teacher, teacher_by_name, prototype_labels = val_embeddings
            cpu_prototypes = {name: registry.to("cpu") for name, registry in prototypes.items()} if prototypes else None
            eval_pairwise_max_samples = int(cfg["train"].get("eval_pairwise_max_samples", 4096))
            embedding_metrics = evaluate_teacher_outputs(
                student_by_teacher,
                teacher_by_name,
                cpu_prototypes,
                int(cfg["train"]["topk"]),
                max_pairwise_samples=eval_pairwise_max_samples,
            )
            zhcc_metrics = evaluate_zhcc_prototypes(
                embeddings,
                prototype_labels["prototype_mask"],
                prototype_labels["prototype_level1"],
                prototype_labels["prototype_level2"],
                current_zhcc_prototypes.to("cpu") if current_zhcc_prototypes is not None else None,
                topk=int(cfg["train"]["topk"]),
                max_pairwise_samples=eval_pairwise_max_samples,
            )
            del val_embeddings, embeddings, student_by_teacher, teacher_by_name, prototype_labels
            prototype_bank_metrics = {}
            if zhcc_image_bank is not None and current_zhcc_prototypes is not None:
                bank_batch_size = int(cfg["train"].get("dynamic_prototype_batch_size", cfg["train"].get("batch_size", 512)))
                bank_embeddings = collect_student_prototype_image_embeddings(
                    model=model,
                    image_bank=zhcc_image_bank,
                    cfg=cfg,
                    device=device,
                    batch_size=bank_batch_size,
                ).cpu()
                bank_metrics = evaluate_zhcc_prototypes(
                    bank_embeddings,
                    torch.ones(zhcc_image_bank.count, dtype=torch.bool),
                    zhcc_image_bank.level1,
                    zhcc_image_bank.level2,
                    current_zhcc_prototypes.to("cpu"),
                    topk=int(cfg["train"]["topk"]),
                    max_pairwise_samples=eval_pairwise_max_samples,
                )
                prototype_bank_metrics = {f"prototype_bank_{key}": value for key, value in bank_metrics.items()}
            teacher_alignment_values = [
                float(value) for key, value in embedding_metrics.items() if key.endswith("_feature_cosine")
            ]
            teacher_alignment = (
                float(sum(teacher_alignment_values) / len(teacher_alignment_values))
                if teacher_alignment_values
                else float("-inf")
            )
            scientific_score = (
                (0.0 if not math.isfinite(teacher_alignment) else teacher_alignment)
                - float(cfg["train"].get("scientific_score_zhcc_response_weight", 0.25)) * float(val_metrics["zhcc_proto"])
            )
            loss_cfg = scheduled_loss_config(
                cfg,
                epoch=epoch,
                global_step=global_step,
                schedule_state=schedule_state,
            )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "feature_loss_type": str(loss_cfg["feature_loss_type"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "scheduled_semantic_weight": float(loss_cfg["semantic_weight"]),
                "scheduled_prototype_filter_weight": float(loss_cfg["prototype_filter_weight"]),
                "scheduled_zhcc_proto_weight": float(loss_cfg["zhcc_proto_weight"]),
                "scheduled_zhcc_response_weight": float(loss_cfg["zhcc_response_weight"]),
                "prototype_start_step": -1 if loss_cfg["prototype_start_step"] is None else int(loss_cfg["prototype_start_step"]),
                "filter_start_step": -1 if loss_cfg["filter_start_step"] is None else int(loss_cfg["filter_start_step"]),
                "teacher_prior_loss_ema": 0.0
                if loss_cfg["teacher_prior_loss_ema"] is None
                else float(loss_cfg["teacher_prior_loss_ema"]),
                "teacher_prior_relative_improvement": 0.0
                if loss_cfg["teacher_prior_relative_improvement"] is None
                else float(loss_cfg["teacher_prior_relative_improvement"]),
                "teacher_prior_plateau_count": int(loss_cfg["teacher_prior_plateau_count"]),
                "intervention_stage": str(loss_cfg["intervention_stage"]),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "teacher_alignment_score": 0.0 if not math.isfinite(teacher_alignment) else teacher_alignment,
                "scientific_score": scientific_score,
                **embedding_metrics,
                **zhcc_metrics,
                **prototype_bank_metrics,
            }
            append_csv(output_dir / "metrics.csv", row)
            _write_tensorboard_scalars(writer, row, epoch)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "best_teacher_alignment": best_teacher_alignment,
                "best_scientific_score": best_scientific_score,
                "best_metrics": best_metrics,
                "rng_state": _rng_state(),
                "schedule_state": schedule_state,
                "zhcc_dynamic_prototype_step": zhcc_prototype_state.get("last_refresh_step"),
                "config": cfg,
            }
            torch.save(
                checkpoint,
                checkpoints / "last.pt",
            )
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                checkpoint["best_loss"] = best_loss
                torch.save(
                    checkpoint,
                    checkpoints / "best.pt",
                )
            if teacher_alignment > best_teacher_alignment:
                best_teacher_alignment = teacher_alignment
                checkpoint["best_teacher_alignment"] = best_teacher_alignment
                torch.save(
                    checkpoint,
                    checkpoints / "best_teacher_alignment.pt",
                )
            if scientific_score > best_scientific_score:
                best_scientific_score = scientific_score
                best_metrics = row
                checkpoint["best_scientific_score"] = best_scientific_score
                checkpoint["best_metrics"] = best_metrics
                torch.save(
                    checkpoint,
                    checkpoints / "best_scientific_score.pt",
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if writer is not None:
            writer.close()
    write_json(output_dir / "summary.json", best_metrics)
    return best_metrics
