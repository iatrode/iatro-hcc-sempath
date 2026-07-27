from __future__ import annotations

import csv
import ctypes
from dataclasses import dataclass
import gc
import math
from numbers import Number
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..modeling.models import bounded_logits
from .losses import multi_teacher_distillation_loss
from .metrics import evaluate_teacher_outputs
from .pamtd import (
    prototype_adjudicated_teacher_target,
    prototype_response_distillation_loss,
)
from .prototype_labels import DEFAULT_L1_CLASSES
from .spatial_losses import l1_classification_loss, spatial_morphometry_loss
from .utils import append_csv, ensure_dir, write_json


GRADIENT_DIAGNOSTIC_INTERVAL_STEPS = 1000
EARLY_STOP_MIN_EPOCH = 60
EARLY_STOP_CHECK_INTERVAL_EPOCHS = 10
EARLY_STOP_WINDOW_EPOCHS = 20
EARLY_STOP_ALIGNMENT_GAIN = 0.002
EARLY_STOP_PER_TEACHER_GAIN = 0.003

STEP_METRIC_PARTS = (
    "feature",
    "relation",
    "semantic",
    "pamtd_response",
    "teacher_alpha_mean",
    "l1",
    "l1_accuracy",
    "l1_supervised_tiles",
    "l2_spatial",
    "l2_instance_point",
    "l2_abundance_point",
    "l2_brush_bag",
    "l2_area_positive",
    "l2_explicit_negative",
    "l2_implicit_negative",
    "l2_point_supervised_pairs",
    "l2_brush_supervised_pairs",
    "l2_area_supervised_pairs",
)
STEP_METRIC_FIELDS = (
    "epoch",
    "global_step",
    "l2_supervised_step",
    "tiles_seen_in_epoch",
    "lr",
    "scheduled_semantic_weight",
    "scheduled_l1_weight",
    "scheduled_spatial_weight",
    "scheduled_filter_weight",
    "scheduled_response_weight",
    "l1_active",
    "l2_active",
    "loss",
    *STEP_METRIC_PARTS,
)


class StepMetricsWriter:
    """Buffer per-step scalars and transfer them to one append-only CSV in groups."""

    def __init__(self, path: str | Path, flush_steps: int = 50) -> None:
        self.path = Path(path)
        self.flush_steps = max(1, int(flush_steps))
        self.rows: list[dict[str, float | int]] = []
        self.tensor_buffer: torch.Tensor | None = None

    def append(
        self,
        *,
        epoch: int,
        global_step: int,
        l2_supervised_step: int,
        tiles_seen_in_epoch: int,
        lr: float,
        loss_cfg: dict,
        l1_active: bool,
        l2_active: bool,
        loss: torch.Tensor,
        parts: dict[str, torch.Tensor],
    ) -> None:
        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "l2_supervised_step": int(l2_supervised_step),
            "tiles_seen_in_epoch": int(tiles_seen_in_epoch),
            "lr": float(lr),
            "scheduled_semantic_weight": float(loss_cfg["semantic_weight"]),
            "scheduled_l1_weight": float(loss_cfg["l1_weight"]),
            "scheduled_spatial_weight": float(loss_cfg["spatial_weight"]),
            "scheduled_filter_weight": float(
                loss_cfg["prototype_filter_weight"]
            ),
            "scheduled_response_weight": float(
                loss_cfg["zhcc_response_weight"]
            ),
            "l1_active": int(l1_active),
            "l2_active": int(l2_active),
        }
        reference = loss.detach()
        tensor_values = torch.stack(
            [
                reference,
                *[
                    parts.get(key, reference.new_zeros(())).detach()
                    for key in STEP_METRIC_PARTS
                ],
            ]
        ).float()
        if self.tensor_buffer is None:
            self.tensor_buffer = torch.empty(
                (self.flush_steps, len(tensor_values)),
                device=tensor_values.device,
                dtype=tensor_values.dtype,
            )
        elif self.tensor_buffer.device != tensor_values.device:
            raise RuntimeError("step metrics device changed within one run")
        self.tensor_buffer[len(self.rows)].copy_(tensor_values)
        self.rows.append(row)
        if len(self.rows) >= self.flush_steps:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        tensor_fields = ("loss", *STEP_METRIC_PARTS)
        assert self.tensor_buffer is not None
        tensor_rows = self.tensor_buffer[: len(self.rows)].cpu().tolist()
        materialized: list[dict[str, float | int]] = []
        for index, row in enumerate(self.rows):
            materialized.append(
                {
                    **row,
                    **dict(zip(tensor_fields, tensor_rows[index], strict=True)),
                }
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=STEP_METRIC_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(materialized)
        self.rows.clear()


@dataclass
class PrototypeRefreshState:
    """Fixed expert banks and their last exact student-space refresh."""

    global_loader: object | None
    spatial_loader: object | None
    last_global_step: int | None = None
    last_spatial_global_step: int | None = None


def _objective_gradient_diagnostics(
    global_objective: torch.Tensor,
    spatial_objective: torch.Tensor,
    shared_parameters: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    """Measure global/spatial interaction on the final shared Transformer block."""

    global_grads = torch.autograd.grad(
        global_objective,
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    spatial_grads = torch.autograd.grad(
        spatial_objective,
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    device = shared_parameters[0].device
    global_sq = torch.zeros((), device=device)
    spatial_sq = torch.zeros((), device=device)
    dot = torch.zeros((), device=device)
    for global_grad, spatial_grad in zip(global_grads, spatial_grads, strict=True):
        if global_grad is not None:
            global_grad = global_grad.detach().float()
            global_sq = global_sq + global_grad.square().sum()
        if spatial_grad is not None:
            spatial_grad = spatial_grad.detach().float()
            spatial_sq = spatial_sq + spatial_grad.square().sum()
        if global_grad is not None and spatial_grad is not None:
            dot = dot + (global_grad * spatial_grad).sum()
    global_norm = global_sq.sqrt()
    spatial_norm = spatial_sq.sqrt()
    if not torch.isfinite(global_norm) or not torch.isfinite(spatial_norm):
        raise FloatingPointError("non-finite objective gradient diagnostic")
    denominator = global_norm + spatial_norm
    cosine = torch.tensor(0.0, device=device)
    if global_norm > 0 and spatial_norm > 0:
        cosine = dot / (global_norm * spatial_norm)
    return {
        "gradient_global_norm": float(global_norm.cpu()),
        "gradient_spatial_norm": float(spatial_norm.cpu()),
        "gradient_spatial_share": float((spatial_norm / denominator.clamp_min(1e-12)).cpu()),
        "gradient_spatial_global_cosine": float(cosine.cpu()),
    }


def _should_stop_for_alignment(history: list[dict[str, float]]) -> bool:
    if not history or int(history[-1]["epoch"]) < EARLY_STOP_MIN_EPOCH:
        return False
    if int(history[-1]["epoch"]) % EARLY_STOP_CHECK_INTERVAL_EPOCHS != 0:
        return False
    current = history[-1]
    baseline_epoch = int(current["epoch"]) - EARLY_STOP_WINDOW_EPOCHS
    baseline = next(
        (row for row in reversed(history[:-1]) if int(row["epoch"]) <= baseline_epoch),
        None,
    )
    if baseline is None:
        return False
    if current["teacher_alignment_score"] - baseline["teacher_alignment_score"] >= EARLY_STOP_ALIGNMENT_GAIN:
        return False
    teacher_keys = sorted(key for key in current if key.endswith("_feature_cosine"))
    return bool(teacher_keys) and all(
        current[key] - baseline.get(key, float("-inf")) < EARLY_STOP_PER_TEACHER_GAIN
        for key in teacher_keys
    )


def _set_loader_epoch(loader, epoch: int) -> None:
    """Select the deterministic data order for one training epoch."""

    setter = getattr(loader, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
        return
    for name in ("batch_sampler", "sampler"):
        candidate = getattr(loader, name, None)
        setter = getattr(candidate, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))
            return


def _optimizer_step(
    *,
    loss: torch.Tensor,
    model,
    optimizer,
    scaler,
    max_grad_norm: float,
) -> bool:
    """Backpropagate once and report whether the optimizer actually stepped."""

    if loss.numel() != 1:
        raise FloatingPointError(
            "training loss must be one finite scalar before backward"
        )
    finite = torch.isfinite(loss.detach())
    assert_async = getattr(torch, "_assert_async", None)
    if finite.device.type == "cuda" and assert_async is not None:
        assert_async(
            finite,
            "training loss must be one finite scalar before backward",
        )
    elif not bool(finite):
        raise FloatingPointError(
            "training loss must be one finite scalar before backward"
        )
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and scaler.is_enabled():
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        return float(scaler.get_scale()) >= scale_before

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm > 0 else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()
    return True


def _move_teachers(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in batch["teacher_features"].items():
        finite = torch.isfinite(value).all()
        assert_async = getattr(torch, "_assert_async", None)
        if finite.device.type == "cuda" and assert_async is not None:
            assert_async(
                finite,
                f"teacher cache contains non-finite values: teacher={name}",
            )
        elif not bool(finite):
            raise FloatingPointError(
                f"teacher cache contains non-finite values: teacher={name}"
            )
        feature = value.to(
            device,
            non_blocking=device.type == "cuda",
        )
        result[name] = feature
    return result


def _image_normalization(
    cfg: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(
        cfg["data"].get("mean", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        cfg["data"].get("std", [1.0, 1.0, 1.0]),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    return mean, std


def _normalize_uint8_images_fp16(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Fuse uint8 normalization and the AMP input cast into one CUDA kernel."""

    return ((images.to(torch.float32) / 255.0 - mean) / std).to(torch.float16)


_compiled_normalize_uint8_images_fp16 = torch.compile(
    _normalize_uint8_images_fp16,
    fullgraph=True,
    dynamic=True,
)


def _prepare_images(
    batch: dict,
    cfg: dict,
    device: torch.device,
    *,
    normalization: tuple[torch.Tensor, torch.Tensor] | None = None,
    amp_input: bool = False,
) -> torch.Tensor:
    images = batch["images"].to(device, non_blocking=device.type == "cuda")
    if not bool(batch.get("images_uint8", False)):
        return images
    if bool(batch.get("images_hwc", False)):
        images = images.permute(0, 3, 1, 2)
    mean, std = (
        _image_normalization(cfg, device)
        if normalization is None
        else normalization
    )
    if bool(
        cfg.get("train", {}).get("fused_image_prepare", False)
        and device.type == "cuda"
        and amp_input
    ):
        return _compiled_normalize_uint8_images_fp16(images, mean, std)
    images = images.to(torch.float32).div_(255.0)
    return images.sub_(mean).div_(std)


def _l2_positive_from_batch(batch: dict) -> torch.Tensor:
    return (
        (batch["l2_point_centers"] > 0)
        | (batch["l2_brush_bag_ids"] > 0)
        | batch["l2_area_positive"].to(dtype=torch.bool)
    ).flatten(2).any(dim=2)


def _l2_global_targets_from_spatial(
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize local ROI evidence without promoting local negatives."""

    positive = _l2_positive_from_batch(batch)
    complete_negative = batch["l2_explicit_negative"].to(
        dtype=torch.bool
    ).flatten(2).all(dim=2)
    return positive, positive | complete_negative


@torch.inference_mode()
def _refresh_global_prototypes(
    model,
    loader,
    cfg: dict,
    device: torch.device,
) -> dict[str, int | float]:
    """Recompute exact L1 and global L2 prototypes from the complete bank."""

    raw_model = getattr(model, "_orig_mod", model)
    if raw_model.l1_prototypes is None:
        raise RuntimeError("dynamic prototype refresh requires an L1 readout")
    embedding_dim = int(raw_model.l1_prototypes.shape[1])
    l1_sums = torch.zeros(
        raw_model.l1_num_classes,
        embedding_dim,
        device=device,
        dtype=torch.float32,
    )
    l1_counts = torch.zeros(
        raw_model.l1_num_classes,
        device=device,
        dtype=torch.float32,
    )
    component_count = int(raw_model.spatial_num_components)
    l2_sums = torch.zeros(
        component_count,
        embedding_dim,
        device=device,
        dtype=torch.float32,
    )
    l2_counts = torch.zeros(
        component_count,
        device=device,
        dtype=torch.float32,
    )
    teacher_sums = {
        name: torch.zeros_like(state.prototypes, dtype=torch.float32)
        for name, state in raw_model.teacher_l2_prototypes.items()
    }
    normalization = _image_normalization(cfg, device)
    was_training = bool(raw_model.training)
    raw_model.eval()
    tile_count = 0
    started = time.perf_counter()
    try:
        for batch in loader:
            images = _prepare_images(
                batch,
                cfg,
                device,
                normalization=normalization,
            )
            embedding_norm = F.normalize(
                raw_model.encode(images),
                dim=-1,
            ).float()
            mask, targets, _ = _move_l1_batch(batch, device)
            selected = F.one_hot(
                targets.clamp(0, raw_model.l1_num_classes - 1),
                num_classes=raw_model.l1_num_classes,
            ).to(dtype=embedding_norm.dtype)
            selected = selected * mask.to(
                dtype=embedding_norm.dtype
            ).unsqueeze(1)
            l1_sums.add_(selected.transpose(0, 1) @ embedding_norm)
            l1_counts.add_(selected.sum(dim=0))

            if component_count > 0:
                positive = _l2_positive_from_batch(batch).to(
                    device=device,
                    dtype=embedding_norm.dtype,
                    non_blocking=device.type == "cuda",
                )
                l2_sums.add_(positive.transpose(0, 1) @ embedding_norm)
                l2_counts.add_(positive.sum(dim=0))
                teachers = _move_teachers(batch, device)
                for name, features in teachers.items():
                    normalized = F.normalize(
                        features.detach().float(),
                        dim=-1,
                    )
                    teacher_sums[name].add_(
                        positive.transpose(0, 1) @ normalized
                    )
            tile_count += len(batch["tile_id"])
    finally:
        raw_model.train(was_training)
    raw_model.replace_l1_prototypes(l1_sums, l1_counts)
    if component_count > 0:
        raw_model.replace_global_l2_prototypes(
            l2_sums,
            l2_counts,
            teacher_sums,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "tiles": tile_count,
        "l1_observations": int(l1_counts.sum().item()),
        "l2_positive_observations": int(l2_counts.sum().item()),
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def _refresh_spatial_prototypes(
    model,
    loader,
    cfg: dict,
    device: torch.device,
) -> dict[str, int | float]:
    """Recompute exact local prototypes from all spatially annotated tiles."""

    raw_model = getattr(model, "_orig_mod", model)
    head = raw_model.spatial_head
    if head is None:
        raise RuntimeError("spatial prototype refresh requires the spatial head")
    accumulated = {
        name: (
            torch.zeros_like(getattr(head, f"{name}_prototypes")),
            torch.zeros_like(getattr(head, f"{name}_prototype_counts")),
        )
        for name in (
            "instance",
            "measurement",
            "instance_negative",
            "measurement_negative",
            "instance_implicit_negative",
            "measurement_implicit_negative",
        )
    }
    normalization = _image_normalization(cfg, device)
    was_training = bool(raw_model.training)
    raw_model.eval()
    tile_count = 0
    started = time.perf_counter()
    try:
        for batch in loader:
            spatial_sample_mask = batch[
                "l2_spatial_supervised"
            ].any(dim=1)
            if not bool(spatial_sample_mask.any()):
                continue
            images = _prepare_images(
                batch,
                cfg,
                device,
                normalization=normalization,
            )
            outputs = raw_model(
                images,
                spatial_detach_backbone=True,
                return_spatial_features=True,
                run_spatial=True,
                spatial_sample_mask=spatial_sample_mask,
            )
            active_host = {
                key: batch[key][spatial_sample_mask]
                for key in (
                    "l2_point_centers",
                    "l2_brush_bag_ids",
                    "l2_area_positive",
                    "l2_explicit_negative",
                    "l2_implicit_negative",
                )
            }
            active = _move_spatial_batch(active_host, device)
            observations = head.prototype_observation_sums(
                outputs["l2_spatial_features"],
                point_centers=active["l2_point_centers"],
                brush_bag_ids=active["l2_brush_bag_ids"],
                area_positive=active["l2_area_positive"],
                explicit_negative=active["l2_explicit_negative"],
                implicit_negative=active["l2_implicit_negative"],
            )
            for name, (sums, counts) in observations.items():
                accumulated[name][0].add_(sums)
                accumulated[name][1].add_(counts)
            tile_count += int(spatial_sample_mask.sum())
    finally:
        raw_model.train(was_training)
    head.replace_prototypes(accumulated)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "tiles": tile_count,
        "positive_observations": int(
            accumulated["measurement"][1].sum().item()
        ),
        "seconds": time.perf_counter() - started,
    }


def _maybe_refresh_prototypes(
    *,
    model,
    cfg: dict,
    device: torch.device,
    state: PrototypeRefreshState | None,
    global_step: int,
) -> None:
    if state is None:
        return
    global_interval = int(
        cfg["train"].get("dynamic_prototype_refresh_steps", 500)
    )
    refresh_global = state.global_loader is not None and (
        state.last_global_step is None
        or (
            global_interval > 0
            and global_step - state.last_global_step >= global_interval
        )
    )
    if refresh_global:
        metrics = _refresh_global_prototypes(
            model,
            state.global_loader,
            cfg,
            device,
        )
        state.last_global_step = int(global_step)
        _log(
            "dynamic_global_prototypes_refreshed "
            f"global_step={global_step} tiles={metrics['tiles']} "
            f"l1_observations={metrics['l1_observations']} "
            f"l2_positive_observations={metrics['l2_positive_observations']} "
            f"seconds={metrics['seconds']:.2f}"
        )

    if state.spatial_loader is None:
        return
    spatial_interval = int(
        cfg["train"].get(
            "dynamic_spatial_prototype_refresh_steps",
            500,
        )
    )
    refresh_spatial = (
        state.last_spatial_global_step is None
        or (
            spatial_interval > 0
            and global_step - state.last_spatial_global_step
            >= spatial_interval
        )
    )
    if refresh_spatial:
        metrics = _refresh_spatial_prototypes(
            model,
            state.spatial_loader,
            cfg,
            device,
        )
        state.last_spatial_global_step = int(global_step)
        _log(
            "dynamic_spatial_prototypes_refreshed "
            f"global_step={global_step} "
            f"tiles={metrics['tiles']} "
            f"positive_observations={metrics['positive_observations']} "
            f"seconds={metrics['seconds']:.2f}"
        )


def _move_l1_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    size = len(batch["tile_id"])
    mask = batch.get(
        "prototype_mask",
        torch.zeros(size, dtype=torch.bool),
    )
    target = batch.get(
        "prototype_level1",
        torch.full((size,), -1, dtype=torch.long),
    )
    return (
        mask.to(
            device,
            non_blocking=device.type == "cuda",
        ),
        target.to(
            device,
            non_blocking=device.type == "cuda",
        ),
        bool(mask.any()),
    )


def _move_spatial_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "l2_point_centers",
        "l2_brush_bag_ids",
        "l2_area_positive",
        "l2_explicit_negative",
        "l2_implicit_negative",
    )
    result = {
        key: batch[key].to(
            device,
            non_blocking=device.type == "cuda",
        )
        for key in keys
    }
    exclusion = batch.get(
        "l2_instance_exclusion_support",
        torch.zeros_like(
            batch["l2_area_positive"],
            dtype=torch.bool,
        ),
    )
    result["l2_instance_exclusion_support"] = exclusion.to(
        device,
        non_blocking=device.type == "cuda",
    )
    return result


def _amp_enabled(device: torch.device, cfg: dict, train: bool) -> bool:
    return bool(train and cfg["train"].get("amp", False) and device.type == "cuda")


def _step_ramp(target: float, global_step: int, start_step: int, ramp_steps: int) -> float:
    if global_step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return float(target)
    progress = (global_step - start_step) / float(ramp_steps)
    return float(target) * max(0.0, min(1.0, progress))


def scheduled_loss_config(
    cfg: dict,
    *,
    epoch: int,
    global_step: int,
) -> dict[str, float | dict | bool]:
    """Resolve the teacher-prior and parallel L1/L2 objective schedule."""

    loss_cfg = cfg["loss"]
    del epoch
    semantic_temperature = float(loss_cfg.get("semantic_temperature", 1.0))
    expert_start = int(loss_cfg.get("expert_supervision_start_step", 0))
    expert_ramp = int(loss_cfg.get("expert_supervision_ramp_steps", 0))
    filter_start = int(loss_cfg.get("prototype_filter_start_step", 0))
    filter_ramp = int(loss_cfg.get("prototype_filter_ramp_steps", 1000))
    return {
        "teacher_weights": loss_cfg.get("teacher_weights"),
        "relation_weight": float(loss_cfg.get("relation_weight", 0.0)),
        "semantic_weight": _step_ramp(
            float(loss_cfg.get("semantic_weight", 0.0)),
            int(global_step),
            expert_start,
            expert_ramp,
        ),
        "semantic_temperature": semantic_temperature,
        "primary_temperature": float(
            loss_cfg.get("primary_temperature", semantic_temperature)
        ),
        "pamtd_primary_temperature": float(
            loss_cfg.get("pamtd_primary_temperature", 0.1)
        ),
        "feature_loss_type": str(loss_cfg.get("feature_loss_type", "cosine")),
        "prototype_filter_weight": _step_ramp(
            float(loss_cfg.get("prototype_filter_weight", 0.3)),
            int(global_step),
            filter_start,
            filter_ramp,
        ),
        "prototype_filter_alpha_min": float(
            loss_cfg.get("prototype_filter_alpha_min", 0.25)
        ),
        "prototype_consensus_weight": float(
            loss_cfg.get("prototype_consensus_weight", 1.0)
        ),
        "prototype_label_weight": float(
            loss_cfg.get("prototype_label_weight", 1.0)
        ),
        "prototype_student_weight": float(
            loss_cfg.get("prototype_student_weight", 1.0)
        ),
        "zhcc_response_weight": _step_ramp(
            float(loss_cfg.get("zhcc_response_weight", 0.1)),
            int(global_step),
            int(loss_cfg.get("zhcc_response_start_step", 0)),
            int(loss_cfg.get("zhcc_response_ramp_steps", 1000)),
        ),
        "l2_global_temperature": float(
            loss_cfg.get("l2_global_temperature", 0.1)
        ),
        "l1_weight": _step_ramp(
            float(loss_cfg.get("l1_weight", 1.0)),
            int(global_step),
            expert_start,
            expert_ramp,
        ),
        "spatial_weight": _step_ramp(
            float(loss_cfg.get("spatial_weight", 0.1)),
            int(global_step),
            expert_start,
            expert_ramp,
        ),
        "spatial_point_tolerance_cells": int(
            loss_cfg.get("spatial_point_tolerance_cells", 1)
        ),
        "spatial_abundance_point_weight": float(
            loss_cfg.get("spatial_abundance_point_weight", 0.5)
        ),
        "spatial_brush_weight": float(
            loss_cfg.get("spatial_brush_weight", 1.0)
        ),
        "spatial_brush_top_fraction": float(
            loss_cfg.get("spatial_brush_top_fraction", 0.25)
        ),
        "spatial_explicit_negative_weight": float(
            loss_cfg.get("spatial_explicit_negative_weight", 1.0)
        ),
        "spatial_implicit_negative_weight": float(
            loss_cfg.get("spatial_implicit_negative_weight", 0.05)
        ),
        "spatial_detach_backbone": bool(
            loss_cfg.get("spatial_detach_shared_encoder", False)
        ),
    }


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
):
    if str(cfg["train"].get("scheduler", "none")).lower() != "cosine":
        return None
    total_steps = max(1, int(cfg["train"]["epochs"]) * max(1, int(steps_per_epoch)))
    warmup_steps = max(0, int(cfg["train"].get("lr_warmup_steps", 0)))
    base_lr = float(cfg["train"]["lr"])
    min_factor = float(cfg["train"].get("min_lr", 0.0)) / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_factor, float(step + 1) / float(warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(
            1.0,
            max(0.0, float(step - warmup_steps) / float(decay_steps)),
        )
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


def _release_host_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        return


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
    for key in (
        "feature",
        "relation",
        "semantic",
        "l1",
        "l2_spatial",
        "l2_instance_point",
        "l2_abundance_point",
        "l2_brush_bag",
        "l2_area_positive",
        "l2_explicit_negative",
        "l2_implicit_negative",
    ):
        if key in parts:
            writer.add_scalar(
                f"{phase}_batch/{key}",
                float(parts[key].detach().cpu()),
                global_step,
            )
    writer.add_scalar("train/lr_step", float(lr), global_step)


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
    epoch: int = 1,
    global_step: int = 0,
    l2_supervised_step: int = 0,
    summary_writer=None,
    collect_embeddings: bool = False,
    max_eval_batches: int | None = None,
    prefetched_iterator=None,
    prototype_refresh_state: PrototypeRefreshState | None = None,
    step_metrics_writer: StepMetricsWriter | None = None,
    development_probe: Callable[[int, int, int], None] | None = None,
) -> dict[str, float] | tuple[dict[str, float], tuple]:
    model.train(train)
    totals: dict[str, torch.Tensor | float] = {
        "loss": 0.0,
        "feature": 0.0,
        "relation": 0.0,
        "semantic": 0.0,
        "pamtd_response": 0.0,
        "teacher_alpha_mean": 0.0,
        "l1": 0.0,
        "l1_accuracy": 0.0,
        "l1_supervised_tiles": 0.0,
        "l2_spatial": 0.0,
        "l2_instance_point": 0.0,
        "l2_abundance_point": 0.0,
        "l2_brush_bag": 0.0,
        "l2_area_positive": 0.0,
        "l2_explicit_negative": 0.0,
        "l2_implicit_negative": 0.0,
        "l2_point_supervised_pairs": 0.0,
        "l2_point_count": 0.0,
        "l2_brush_supervised_pairs": 0.0,
        "l2_brush_bag_count": 0.0,
        "l2_area_supervised_pairs": 0.0,
        "l2_explicit_negative_pairs": 0.0,
        "l2_implicit_negative_pairs": 0.0,
    }
    gradient_totals: dict[str, float] = {}
    gradient_count = 0
    last_gradient_step = global_step - GRADIENT_DIAGNOSTIC_INTERVAL_STEPS
    n_batches = 0
    n_tiles = 0
    start = time.perf_counter()
    interval_start = start
    interval_tiles = 0
    progress_detail_interval = float(
        cfg["train"].get("progress_interval_sec", 5.0)
    )
    last_progress_detail = start - progress_detail_interval
    phase = "train" if train else "val"
    log_interval = int(cfg["train"].get("log_interval", 0) or 0)
    tensorboard_batch_interval = int(
        cfg["train"].get("tensorboard_batch_interval", 0) or 0
    )
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0) or 0.0)
    last_loss_cfg = scheduled_loss_config(
        cfg,
        epoch=epoch,
        global_step=global_step,
    )
    image_normalization = _image_normalization(cfg, device)
    embeddings_data = None
    if collect_embeddings:
        embeddings_data = {
            "embeddings": [],
            "prototype_masks": [],
            "prototype_level1": [],
            "l1_logits": [],
            "students_by_teacher": {},
            "teachers_by_name": {},
        }

    iterator = prefetched_iterator if prefetched_iterator is not None else iter(loader)
    progress_total = len(loader)
    if max_batches is not None:
        progress_total = min(progress_total, int(max_batches))
    progress_bar = _build_progress_bar(
        cfg,
        phase=phase,
        epoch=epoch,
        total=progress_total,
    )
    try:
        while True:
            if max_batches is not None and n_batches >= int(max_batches):
                break
            data_wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait = time.perf_counter() - data_wait_start
            will_log = (
                progress_bar is None
                and log_interval > 0
                and (n_batches == 0 or (n_batches + 1) % log_interval == 0)
            )
            batch_start = time.perf_counter()
            image_prepare_start = time.perf_counter()
            images = _prepare_images(
                batch,
                cfg,
                device,
                normalization=image_normalization,
                amp_input=bool(
                    train
                    and cfg["train"].get("amp", False)
                    and device.type == "cuda"
                ),
            )
            if will_log and device.type == "cuda":
                torch.cuda.synchronize(device)
            image_prepare = time.perf_counter() - image_prepare_start
            n_tiles += int(images.shape[0])
            interval_tiles += int(images.shape[0])
            teachers = _move_teachers(batch, device)
            l1_mask, l1_target, l1_is_active = _move_l1_batch(
                batch,
                device,
            )
            spatial_sample_mask = batch["l2_spatial_supervised"].any(dim=1)
            spatial_is_active = bool(spatial_sample_mask.any())
            spatial_host = {
                key: batch[key]
                for key in (
                    "l2_point_centers",
                    "l2_brush_bag_ids",
                    "l2_area_positive",
                    "l2_explicit_negative",
                    "l2_implicit_negative",
                )
            }
            spatial_host["l2_instance_exclusion_support"] = batch.get(
                "l2_instance_exclusion_support",
                torch.zeros_like(
                    batch["l2_area_positive"],
                    dtype=torch.bool,
                ),
            )
            active_spatial_host = (
                {
                    key: value[spatial_sample_mask]
                    for key, value in spatial_host.items()
                }
                if spatial_is_active
                else None
            )
            optimizer_stepped = False

            with torch.set_grad_enabled(train):
                loss_cfg = scheduled_loss_config(
                    cfg,
                    epoch=epoch,
                    global_step=global_step,
                )
                last_loss_cfg = loss_cfg
                l1_objective_active = bool(
                    l1_is_active and float(loss_cfg["l1_weight"]) > 0
                )
                spatial_objective_active = bool(
                    spatial_is_active
                    and float(loss_cfg["spatial_weight"]) > 0
                )
                if train:
                    _maybe_refresh_prototypes(
                        model=model,
                        cfg=cfg,
                        device=device,
                        state=prototype_refresh_state,
                        global_step=global_step,
                    )
                with torch.autocast(
                    device_type=device.type,
                    enabled=_amp_enabled(device, cfg, train),
                ):
                    outputs = model(
                        images,
                        spatial_detach_backbone=bool(
                            loss_cfg["spatial_detach_backbone"]
                        ),
                        return_spatial_features=bool(
                            train and spatial_objective_active
                        ),
                        run_spatial=spatial_objective_active,
                        spatial_sample_mask=(
                            spatial_sample_mask
                            if spatial_objective_active
                            else None
                        ),
                    )
                    raw_model = getattr(model, "_orig_mod", model)
                    l2_positive = None
                    l2_known = None
                    l2_target = None
                    if spatial_is_active:
                        if active_spatial_host is None:  # pragma: no cover
                            raise RuntimeError(
                                "active spatial targets were not prepared"
                            )
                        (
                            active_l2_positive,
                            active_l2_known,
                        ) = _l2_global_targets_from_spatial(
                            active_spatial_host
                        )
                        summary_shape = (
                            spatial_sample_mask.shape[0],
                            active_l2_positive.shape[1],
                        )
                        l2_positive_host = torch.zeros(
                            summary_shape,
                            dtype=torch.bool,
                        )
                        l2_known_host = torch.zeros_like(
                            l2_positive_host
                        )
                        l2_positive_host[spatial_sample_mask] = (
                            active_l2_positive
                        )
                        l2_known_host[spatial_sample_mask] = (
                            active_l2_known
                        )
                        l2_positive = l2_positive_host.to(
                            device,
                            non_blocking=device.type == "cuda",
                        )
                        l2_known = l2_known_host.to(
                            device,
                            non_blocking=device.type == "cuda",
                        )
                        l2_target = l2_positive.to(dtype=torch.float32)
                    student_by_teacher = outputs["teacher_outputs"]
                    teacher_l2_prototypes = {
                        name: (
                            state.prototypes,
                            state.counts,
                        )
                        for name, state in raw_model.teacher_l2_prototypes.items()
                    }
                    pamtd_temperature = float(
                        loss_cfg["pamtd_primary_temperature"]
                    )
                    pamtd_student_logits = (
                        bounded_logits(
                            outputs["l1_similarity"] / pamtd_temperature
                        )
                        if "l1_similarity" in outputs
                        else None
                    )
                    adjudication = (
                        prototype_adjudicated_teacher_target(
                            teacher_by_name=teachers,
                            prototypes_by_teacher=prototypes,
                            student_primary_response=torch.softmax(
                                pamtd_student_logits,
                                dim=-1,
                            ),
                            class_names=cfg["model"].get(
                                "l1_class_names",
                                DEFAULT_L1_CLASSES,
                            ),
                            teacher_l2_prototypes=teacher_l2_prototypes,
                            student_l2_response=raw_model.global_l2_response(
                                outputs["embedding_norm"],
                                temperature=float(
                                    loss_cfg["l2_global_temperature"]
                                ),
                            ),
                            l1_mask=l1_mask,
                            l1_target=l1_target,
                            l2_target=l2_target,
                            l2_known=l2_known,
                            teacher_weights=loss_cfg.get(
                                "teacher_weights"
                            ),
                            filter_strength=float(
                                loss_cfg["prototype_filter_weight"]
                            ),
                            alpha_min=float(
                                loss_cfg["prototype_filter_alpha_min"]
                            ),
                            consensus_weight=float(
                                loss_cfg["prototype_consensus_weight"]
                            ),
                            prototype_label_weight=float(
                                loss_cfg["prototype_label_weight"]
                            ),
                            student_agreement_weight=float(
                                loss_cfg["prototype_student_weight"]
                            ),
                            primary_temperature=pamtd_temperature,
                            l2_temperature=float(
                                loss_cfg["l2_global_temperature"]
                            ),
                        )
                        if (
                            prototypes
                            and pamtd_student_logits is not None
                            and (
                                float(loss_cfg["prototype_filter_weight"]) > 0
                                or float(loss_cfg["zhcc_response_weight"]) > 0
                            )
                        )
                        else None
                    )
                    distillation_loss, distillation_parts = multi_teacher_distillation_loss(
                        student_by_teacher=student_by_teacher,
                        teacher_by_name=teachers,
                        prototypes_by_teacher=prototypes,
                        relation_weight=float(loss_cfg["relation_weight"]),
                        semantic_weight=float(loss_cfg["semantic_weight"]),
                        semantic_temperature=float(loss_cfg["semantic_temperature"]),
                        teacher_weights=loss_cfg.get("teacher_weights"),
                        feature_loss_type=str(loss_cfg["feature_loss_type"]),
                        primary_temperature=float(loss_cfg["primary_temperature"]),
                        teacher_sample_weights=(
                            adjudication.teacher_sample_weights
                            if adjudication is not None
                            else None
                        ),
                    )
                    if adjudication is not None:
                        assert pamtd_student_logits is not None
                        response_loss = prototype_response_distillation_loss(
                            pamtd_student_logits,
                            adjudication.primary_target,
                            temperature=pamtd_temperature,
                            sample_weight=(
                                adjudication.response_sample_weight
                            ),
                        )
                        alpha_mean = adjudication.diagnostics[
                            "teacher_alpha_mean"
                        ]
                    else:
                        response_loss = distillation_loss.new_zeros(())
                        alpha_mean = distillation_loss.new_ones(())
                    if "l1_logits" in outputs and l1_is_active:
                        l1_loss, l1_parts = l1_classification_loss(
                            outputs["l1_logits"],
                            l1_mask,
                            l1_target,
                        )
                    else:
                        l1_loss = distillation_loss.new_zeros(())
                        l1_parts = {
                            "l1": l1_loss.detach(),
                            "l1_accuracy": l1_loss.detach(),
                            "l1_supervised_tiles": l1_loss.detach(),
                        }
                    if (
                        "l2_instance_logits" in outputs
                        and spatial_host["l2_point_centers"].numel() > 0
                    ):
                        if active_spatial_host is None:  # pragma: no cover
                            raise RuntimeError(
                                "active spatial targets were not prepared"
                            )
                        active_spatial = _move_spatial_batch(
                            active_spatial_host,
                            device,
                        )
                        spatial_loss, spatial_parts = spatial_morphometry_loss(
                            instance_logits=outputs["l2_instance_logits"],
                            abundance_logits=outputs["l2_abundance_logits"],
                            point_centers=active_spatial["l2_point_centers"],
                            brush_bag_ids=active_spatial["l2_brush_bag_ids"],
                            area_positive=active_spatial["l2_area_positive"],
                            explicit_negative=active_spatial["l2_explicit_negative"],
                            implicit_negative=active_spatial["l2_implicit_negative"],
                            instance_exclusion_support=active_spatial[
                                "l2_instance_exclusion_support"
                            ],
                            point_centers_host=active_spatial_host[
                                "l2_point_centers"
                            ],
                            brush_bag_ids_host=active_spatial_host[
                                "l2_brush_bag_ids"
                            ],
                            area_positive_host=active_spatial_host[
                                "l2_area_positive"
                            ],
                            instance_exclusion_support_host=(
                                active_spatial_host[
                                    "l2_instance_exclusion_support"
                                ]
                            ),
                            component_names=cfg["data"].get(
                                "spatial_component_names"
                            ),
                            point_tolerance_cells=int(
                                loss_cfg["spatial_point_tolerance_cells"]
                            ),
                            abundance_point_weight=float(
                                loss_cfg["spatial_abundance_point_weight"]
                            ),
                            brush_weight=float(
                                loss_cfg["spatial_brush_weight"]
                            ),
                            brush_top_fraction=float(
                                loss_cfg["spatial_brush_top_fraction"]
                            ),
                            explicit_negative_weight=float(
                                loss_cfg["spatial_explicit_negative_weight"]
                            ),
                            implicit_negative_weight=float(
                                loss_cfg["spatial_implicit_negative_weight"]
                            ),
                        )
                    else:
                        spatial_loss = distillation_loss.new_zeros(())
                        spatial_parts = {
                            key: spatial_loss.detach()
                            for key in (
                                "l2_spatial",
                                "l2_instance_point",
                                "l2_abundance_point",
                                "l2_brush_bag",
                                "l2_area_positive",
                                "l2_explicit_negative",
                                "l2_implicit_negative",
                                "l2_point_supervised_pairs",
                                "l2_point_count",
                                "l2_brush_supervised_pairs",
                                "l2_brush_bag_count",
                                "l2_area_supervised_pairs",
                                "l2_explicit_negative_pairs",
                                "l2_implicit_negative_pairs",
                            )
                        }
                    global_objective = (
                        distillation_loss
                        + float(loss_cfg["l1_weight"]) * l1_loss
                        + float(loss_cfg["zhcc_response_weight"])
                        * response_loss
                    )
                    spatial_objective = (
                        float(loss_cfg["spatial_weight"]) * spatial_loss
                    )
                    loss = global_objective + spatial_objective

                if (
                    train
                    and spatial_is_active
                    and float(loss_cfg["spatial_weight"]) > 0
                    and not bool(loss_cfg["spatial_detach_backbone"])
                    and global_step - last_gradient_step
                    >= GRADIENT_DIAGNOSTIC_INTERVAL_STEPS
                ):
                    shared_parameters = tuple(
                        raw_model.encoder.backbone.blocks[-1].parameters()
                    )
                    diagnostics = _objective_gradient_diagnostics(
                        global_objective,
                        spatial_objective,
                        shared_parameters,
                    )
                    for key, value in diagnostics.items():
                        gradient_totals[key] = gradient_totals.get(key, 0.0) + value
                    gradient_count += 1
                    last_gradient_step = global_step

                if train:
                    optimizer_stepped = _optimizer_step(
                        loss=loss,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        max_grad_norm=max_grad_norm,
                    )
                    if scheduler is not None and optimizer_stepped:
                        scheduler.step()
                    if optimizer_stepped:
                        global_step += 1
                        if spatial_objective_active:
                            l2_supervised_step += 1

            parts = {
                **distillation_parts,
                **l1_parts,
                **spatial_parts,
                "pamtd_response": response_loss.detach(),
                "teacher_alpha_mean": alpha_mean.detach(),
            }
            totals["loss"] = totals["loss"] + loss.detach()
            for key, value in parts.items():
                if key.endswith("_feature_cosine") and key not in totals:
                    totals[key] = value.detach().new_zeros(())
            for key in totals:
                if key == "loss":
                    continue
                if key in parts:
                    totals[key] = totals[key] + parts[key].detach()

            if train and optimizer_stepped and step_metrics_writer is not None:
                step_metrics_writer.append(
                    epoch=epoch,
                    global_step=global_step,
                    l2_supervised_step=l2_supervised_step,
                    tiles_seen_in_epoch=n_tiles,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    loss_cfg=loss_cfg,
                    l1_active=l1_objective_active,
                    l2_active=spatial_objective_active,
                    loss=loss,
                    parts=parts,
                )
            if train and optimizer_stepped and development_probe is not None:
                development_probe(global_step, l2_supervised_step, epoch)

            if collect_embeddings and (
                max_eval_batches is None or n_batches < max_eval_batches
            ):
                assert embeddings_data is not None
                embeddings_data["embeddings"].append(
                    outputs["embedding_norm"].detach().cpu()
                )
                embeddings_data["prototype_masks"].append(l1_mask.detach().cpu())
                embeddings_data["prototype_level1"].append(l1_target.detach().cpu())
                if "l1_logits" in outputs:
                    embeddings_data["l1_logits"].append(
                        outputs["l1_logits"].detach().cpu()
                    )
                for name, tensor in student_by_teacher.items():
                    embeddings_data["students_by_teacher"].setdefault(name, []).append(
                        tensor.detach().cpu()
                    )
                for name, tensor in batch["teacher_features"].items():
                    if isinstance(tensor, np.ndarray):
                        tensor = torch.from_numpy(tensor)
                    embeddings_data["teachers_by_name"].setdefault(name, []).append(
                        tensor.detach().cpu()
                    )

            n_batches += 1
            if (
                summary_writer is not None
                and tensorboard_batch_interval > 0
                and n_batches % tensorboard_batch_interval == 0
            ):
                _write_tensorboard_batch(
                    summary_writer,
                    phase=phase,
                    global_step=global_step,
                    loss=loss,
                    parts=parts,
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            if progress_bar is not None:
                elapsed = max(time.perf_counter() - start, 1e-9)
                now = time.perf_counter()
                if (
                    now - last_progress_detail
                    >= progress_detail_interval
                ):
                    progress_bar.set_postfix(
                        loss=f"{float(loss.detach().cpu()):.4f}",
                        tiles_s=f"{n_tiles / elapsed:.0f}",
                        refresh=False,
                    )
                    last_progress_detail = now
                progress_bar.update(1)
            if will_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_elapsed = max(now - interval_start, 1e-9)
                batch_elapsed = max(now - batch_start, 1e-9)
                _log(
                    f"{phase}_progress "
                    f"batch={n_batches} tiles={n_tiles} "
                    f"interval_tiles_per_sec={interval_tiles / interval_elapsed:.2f} "
                    f"last_data_wait_sec={data_wait:.3f} "
                    f"last_image_prepare_sec={image_prepare:.3f} "
                    f"last_batch_sec={batch_elapsed:.3f} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"cuda_mem_mb={_cuda_memory_mb(device):.1f}"
                )
                interval_start = now
                interval_tiles = 0
    finally:
        if step_metrics_writer is not None:
            step_metrics_writer.flush()
        close_iterator = getattr(iterator, "close", None)
        if callable(close_iterator):
            close_iterator()
        if progress_bar is not None:
            progress_bar.close()

    if n_batches == 0:
        raise ValueError(f"{phase} loader produced no batches")
    elapsed = max(time.perf_counter() - start, 1e-9)
    result: dict[str, float] = {}
    for key, value in totals.items():
        mean_value = value / n_batches
        result[key] = (
            float(mean_value.detach().cpu())
            if isinstance(mean_value, torch.Tensor)
            else float(mean_value)
        )
    for key, value in gradient_totals.items():
        result[key] = value / max(1, gradient_count)
    result["gradient_diagnostic_count"] = float(gradient_count)
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    result["tiles_per_sec"] = n_tiles / elapsed
    result["tiles"] = float(n_tiles)
    result["seconds"] = elapsed
    result["global_step_end"] = float(global_step)
    result["l2_supervised_step_end"] = float(l2_supervised_step)
    result["scheduled_l1_weight"] = float(last_loss_cfg["l1_weight"])
    result["scheduled_spatial_weight"] = float(last_loss_cfg["spatial_weight"])

    if not collect_embeddings:
        return result
    assert embeddings_data is not None
    collated = (
        torch.cat(embeddings_data["embeddings"]),
        {
            name: torch.cat(values)
            for name, values in embeddings_data["students_by_teacher"].items()
        },
        {
            name: torch.cat(values)
            for name, values in embeddings_data["teachers_by_name"].items()
        },
        {
            "prototype_mask": torch.cat(embeddings_data["prototype_masks"]),
            "prototype_level1": torch.cat(embeddings_data["prototype_level1"]),
            "l1_logits": (
                torch.cat(embeddings_data["l1_logits"])
                if embeddings_data["l1_logits"]
                else torch.zeros((0, 0))
            ),
        },
    )
    return result, collated


@torch.no_grad()
def collect_embeddings(
    model,
    loader,
    device,
    cfg: dict | None = None,
    max_batches: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    model.eval()
    embeddings = []
    masks = []
    level1 = []
    l1_logits = []
    students_by_teacher: dict[str, list[torch.Tensor]] = {}
    teachers_by_name: dict[str, list[torch.Tensor]] = {}
    image_normalization = (
        None
        if cfg is None
        else _image_normalization(cfg, device)
    )
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if cfg is not None:
            images = _prepare_images(
                batch,
                cfg,
                device,
                normalization=image_normalization,
            )
        else:
            images = batch["images"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            if bool(batch.get("images_uint8", False)):
                if bool(batch.get("images_hwc", False)):
                    images = images.permute(0, 3, 1, 2)
                images = images.to(torch.float32).div_(255.0)
        outputs = model(images, run_spatial=False)
        embeddings.append(outputs["embedding_norm"].cpu())
        mask, target, _ = _move_l1_batch(batch, torch.device("cpu"))
        masks.append(mask)
        level1.append(target)
        if "l1_logits" in outputs:
            l1_logits.append(outputs["l1_logits"].cpu())
        for name, tensor in outputs["teacher_outputs"].items():
            students_by_teacher.setdefault(name, []).append(tensor.cpu())
        for name, tensor in batch["teacher_features"].items():
            teachers_by_name.setdefault(name, []).append(tensor.cpu())
    if not embeddings:
        raise ValueError("loader produced no batches")
    return (
        torch.cat(embeddings),
        {name: torch.cat(values) for name, values in students_by_teacher.items()},
        {name: torch.cat(values) for name, values in teachers_by_name.items()},
        {
            "prototype_mask": torch.cat(masks),
            "prototype_level1": torch.cat(level1),
            "l1_logits": torch.cat(l1_logits) if l1_logits else torch.zeros((0, 0)),
        },
    )


def _l1_eval_metrics(supervised: dict[str, torch.Tensor]) -> dict[str, float]:
    mask = supervised["prototype_mask"].bool()
    logits = supervised["l1_logits"]
    if logits.numel() == 0 or not bool(mask.any()):
        return {"l1_accuracy": 0.0, "l1_evaluated_tiles": 0.0}
    target = supervised["prototype_level1"][mask]
    prediction = logits[mask].argmax(dim=1)
    return {
        "l1_accuracy": float((prediction == target).float().mean()),
        "l1_evaluated_tiles": float(mask.sum()),
    }


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
    resume_state: dict | None = None,
    prototype_refresh_loader=None,
    spatial_prototype_refresh_loader=None,
) -> dict:
    output_dir = ensure_dir(cfg["runtime"]["output_dir"])
    checkpoints = ensure_dir(output_dir / "checkpoints")
    write_json(output_dir / "resolved_config.json", cfg)
    best_loss = float((resume_state or {}).get("best_loss", float("inf")))
    best_teacher_alignment = float(
        (resume_state or {}).get("best_teacher_alignment", float("-inf"))
    )
    best_l1_accuracy = float(
        (resume_state or {}).get("best_l1_accuracy", float("-inf"))
    )
    best_metrics = dict((resume_state or {}).get("best_metrics", {}))
    last_metrics = dict(
        (resume_state or {}).get(
            "last_metrics",
            (resume_state or {}).get("best_metrics", {}),
        )
    )
    alignment_history = list((resume_state or {}).get("alignment_history", []))
    start_epoch = int((resume_state or {}).get("epoch", 0)) + 1
    global_step = int((resume_state or {}).get("global_step", 0))
    l2_supervised_step = int(
        (resume_state or {}).get("l2_supervised_step", 0)
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg["train"].get("amp", False) and device.type == "cuda"),
    )
    if resume_state and "scaler" in resume_state:
        scaler.load_state_dict(resume_state["scaler"])
    _restore_rng_state((resume_state or {}).get("rng_state"))
    prototype_refresh_state = (
        PrototypeRefreshState(
            global_loader=prototype_refresh_loader,
            spatial_loader=spatial_prototype_refresh_loader,
            last_global_step=(resume_state or {}).get(
                "dynamic_prototype_step"
            ),
            last_spatial_global_step=(resume_state or {}).get(
                "dynamic_spatial_prototype_global_step"
            ),
        )
        if (
            prototype_refresh_loader is not None
            or spatial_prototype_refresh_loader is not None
        )
        else None
    )
    writer = _build_summary_writer(cfg, output_dir)
    step_metrics_writer = StepMetricsWriter(
        output_dir / "step_metrics.csv",
        flush_steps=int(cfg["train"].get("step_metrics_flush_steps", 50)),
    )
    development_probe_interval = int(
        cfg["train"].get("development_probe_interval_steps", 0) or 0
    )
    development_probe_batches = int(
        cfg["train"].get("development_probe_batches", 64)
    )
    development_probe_cfg = {
        **cfg,
        "train": {
            **cfg["train"],
            "progress": False,
            "log_interval": 0,
            "tensorboard_batch_interval": 0,
        },
    }
    _set_loader_epoch(val_loader, 0)

    def run_development_probe(
        step: int,
        spatial_step: int,
        current_epoch: int,
    ) -> None:
        if (
            development_probe_interval <= 0
            or step % development_probe_interval != 0
        ):
            return
        was_training = bool(model.training)
        try:
            probe_metrics = run_epoch(
                model,
                val_loader,
                prototypes,
                optimizer,
                device,
                development_probe_cfg,
                train=False,
                max_batches=development_probe_batches,
                epoch=current_epoch,
                global_step=step,
                l2_supervised_step=spatial_step,
            )
        finally:
            model.train(was_training)
        probe_loss_cfg = scheduled_loss_config(
            cfg,
            epoch=current_epoch,
            global_step=step,
        )
        append_csv(
            output_dir / "development_metrics.csv",
            {
                "epoch": current_epoch,
                "global_step": step,
                "l2_supervised_step": spatial_step,
                "probe_batches": development_probe_batches,
                "scheduled_semantic_weight": float(
                    probe_loss_cfg["semantic_weight"]
                ),
                "scheduled_l1_weight": float(
                    probe_loss_cfg["l1_weight"]
                ),
                "scheduled_spatial_weight": float(
                    probe_loss_cfg["spatial_weight"]
                ),
                **probe_metrics,
            },
        )

    try:
        for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
            _set_loader_epoch(train_loader, epoch - 1)
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
                epoch=epoch,
                global_step=global_step,
                l2_supervised_step=l2_supervised_step,
                summary_writer=writer,
                prototype_refresh_state=prototype_refresh_state,
                step_metrics_writer=step_metrics_writer,
                development_probe=run_development_probe,
            )
            global_step = int(train_metrics["global_step_end"])
            l2_supervised_step = int(
                train_metrics["l2_supervised_step_end"]
            )
            val_iterator = iter(val_loader)
            try:
                val_metrics, val_embeddings = run_epoch(
                    model,
                    val_loader,
                    prototypes,
                    optimizer,
                    device,
                    cfg,
                    train=False,
                    max_batches=cfg["train"].get("max_val_batches"),
                    epoch=epoch,
                    global_step=global_step,
                    l2_supervised_step=l2_supervised_step,
                    summary_writer=writer,
                    collect_embeddings=True,
                    max_eval_batches=cfg["train"].get(
                        "max_eval_batches",
                        cfg["train"].get("max_val_batches"),
                    ),
                    prefetched_iterator=val_iterator,
                )
                val_iterator = None
            finally:
                close_val_iterator = getattr(val_iterator, "close", None)
                if callable(close_val_iterator):
                    close_val_iterator()
            _, student_by_teacher, teacher_by_name, supervised = val_embeddings
            cpu_prototypes = (
                {name: registry.to("cpu") for name, registry in prototypes.items()}
                if prototypes
                else None
            )
            embedding_metrics = evaluate_teacher_outputs(
                student_by_teacher,
                teacher_by_name,
                cpu_prototypes,
                int(cfg["train"]["topk"]),
                max_pairwise_samples=int(
                    cfg["train"].get("eval_pairwise_max_samples", 4096)
                ),
            )
            l1_metrics = _l1_eval_metrics(supervised)
            del val_embeddings, student_by_teacher, teacher_by_name, supervised
            _release_host_memory()

            teacher_alignment_values = [
                float(value)
                for key, value in embedding_metrics.items()
                if key.endswith("_feature_cosine")
            ]
            teacher_alignment = (
                float(sum(teacher_alignment_values) / len(teacher_alignment_values))
                if teacher_alignment_values
                else float("-inf")
            )
            loss_cfg = scheduled_loss_config(
                cfg,
                epoch=epoch,
                global_step=global_step,
            )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "l2_supervised_step": l2_supervised_step,
                "dynamic_prototype_step": (
                    prototype_refresh_state.last_global_step
                    if prototype_refresh_state is not None
                    else None
                ),
                "dynamic_spatial_prototype_global_step": (
                    prototype_refresh_state.last_spatial_global_step
                    if prototype_refresh_state is not None
                    else None
                ),
                "feature_loss_type": str(loss_cfg["feature_loss_type"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "scheduled_semantic_weight": float(loss_cfg["semantic_weight"]),
                "scheduled_l1_weight": float(loss_cfg["l1_weight"]),
                "scheduled_spatial_weight": float(loss_cfg["spatial_weight"]),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "teacher_alignment_score": (
                    0.0 if not math.isfinite(teacher_alignment) else teacher_alignment
                ),
                **embedding_metrics,
                **l1_metrics,
            }
            alignment_history.append(
                {
                    "epoch": float(epoch),
                    "teacher_alignment_score": float(row["teacher_alignment_score"]),
                    **{
                        key: float(value)
                        for key, value in embedding_metrics.items()
                        if key.endswith("_feature_cosine")
                    },
                }
            )
            append_csv(output_dir / "metrics.csv", row)
            _write_tensorboard_scalars(writer, row, epoch)
            last_metrics = row
            improved_loss = val_metrics["loss"] < best_loss
            improved_alignment = teacher_alignment > best_teacher_alignment
            improved_l1 = (
                l1_metrics["l1_evaluated_tiles"] > 0
                and l1_metrics["l1_accuracy"] > best_l1_accuracy
            )
            if improved_loss:
                best_loss = val_metrics["loss"]
                best_metrics = row
            if improved_alignment:
                best_teacher_alignment = teacher_alignment
            if improved_l1:
                best_l1_accuracy = l1_metrics["l1_accuracy"]

            raw_model = getattr(model, "_orig_mod", model)
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "l2_supervised_step": l2_supervised_step,
                "dynamic_prototype_step": (
                    prototype_refresh_state.last_global_step
                    if prototype_refresh_state is not None
                    else None
                ),
                "dynamic_spatial_prototype_global_step": (
                    prototype_refresh_state.last_spatial_global_step
                    if prototype_refresh_state is not None
                    else None
                ),
                "best_loss": best_loss,
                "best_teacher_alignment": best_teacher_alignment,
                "best_l1_accuracy": best_l1_accuracy,
                "best_metrics": best_metrics,
                "last_metrics": last_metrics,
                "alignment_history": alignment_history,
                "rng_state": _rng_state(),
                "config": cfg,
                "training_complete": (
                    epoch >= int(cfg["train"]["epochs"])
                ),
                "expected_epochs": int(cfg["train"]["epochs"]),
            }
            torch.save(checkpoint, checkpoints / "last.pt")
            if improved_loss:
                torch.save(checkpoint, checkpoints / "best.pt")
            if improved_alignment:
                torch.save(checkpoint, checkpoints / "best_teacher_alignment.pt")
            if improved_l1:
                torch.save(checkpoint, checkpoints / "best_l1_accuracy.pt")
            if bool(
                cfg["train"].get(
                    "early_stop_teacher_alignment",
                    not bool(cfg["data"].get("spatial_manifest_path")),
                )
            ) and _should_stop_for_alignment(alignment_history):
                _log(
                    "early_stop "
                    f"epoch={epoch} window={EARLY_STOP_WINDOW_EPOCHS} "
                    f"alignment_gain_threshold={EARLY_STOP_ALIGNMENT_GAIN}"
                )
                break
            _release_host_memory()
    finally:
        step_metrics_writer.flush()
        if writer is not None:
            writer.close()
    spatial_route = bool(cfg["data"].get("spatial_manifest_path"))
    summary_metrics = last_metrics if spatial_route else best_metrics
    if not summary_metrics:
        raise RuntimeError("training produced no summary metrics")
    write_json(output_dir / "summary.json", summary_metrics)
    return summary_metrics
