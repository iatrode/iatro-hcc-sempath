from __future__ import annotations

import csv
import ctypes
from dataclasses import dataclass
import gc
import math
from numbers import Number
from pathlib import Path
import random
import shutil
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import _pipeline_probe as _probe
from ..modeling.models import bounded_logits
from .losses import multi_teacher_distillation_loss
from .metrics import evaluate_teacher_outputs
from .pamtd import (
    prototype_adjudicated_teacher_target,
    prototype_response_distillation_loss,
)
from .prototype_labels import DEFAULT_CLASSIFICATION_CLASSES
from .spatial_losses import classification_objective_loss, spatial_morphometry_loss
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
    "classification",
    "classification_accuracy",
    "classification_supervised_tiles",
    "spatial",
    "spatial_instance_point",
    "spatial_abundance_point",
    "spatial_brush_bag",
    "spatial_area_positive",
    "spatial_measurement_positive",
    "spatial_explicit_negative",
    "spatial_implicit_negative",
    "spatial_point_supervised_pairs",
    "spatial_brush_supervised_pairs",
    "spatial_area_supervised_pairs",
)
STEP_METRIC_FIELDS = (
    "epoch",
    "global_step",
    "spatial_supervised_step",
    "tiles_seen_in_epoch",
    "lr",
    "scheduled_semantic_weight",
    "scheduled_classification_weight",
    "scheduled_spatial_weight",
    "scheduled_filter_weight",
    "scheduled_response_weight",
    "classification_active",
    "spatial_active",
    "loss",
    *STEP_METRIC_PARTS,
)

SPATIAL_EVAL_STAT_PREFIX = "_spatial_eval_"
SPATIAL_EVAL_OBJECTIVES = (
    "instance_point",
    "measurement_positive",
    "explicit_instance",
    "explicit_abundance",
    "implicit",
)


def _component_balanced_mean_from_statistics(
    numerator: torch.Tensor,
    count: torch.Tensor,
) -> float:
    """Reduce one complete-bank objective once across active components."""

    numerator = numerator.to(dtype=torch.float64, device="cpu")
    count = count.to(dtype=torch.float64, device="cpu")
    if numerator.ndim != 1 or count.shape != numerator.shape:
        raise ValueError(
            "spatial evaluation statistics must be component vectors: "
            f"numerator={tuple(numerator.shape)} count={tuple(count.shape)}"
        )
    active = count > 0
    if not bool(active.any()):
        return 0.0
    return float((numerator[active] / count[active]).mean())


def _complete_bank_spatial_metrics(
    statistics: dict[str, torch.Tensor],
    *,
    explicit_negative_weight: float,
    implicit_negative_weight: float,
) -> dict[str, float]:
    """Compute the batch/permutation-invariant spatial validation objective."""

    reduced: dict[str, float] = {}
    for objective in SPATIAL_EVAL_OBJECTIVES:
        sum_key = f"{SPATIAL_EVAL_STAT_PREFIX}{objective}_sum"
        count_key = f"{SPATIAL_EVAL_STAT_PREFIX}{objective}_count"
        if sum_key not in statistics or count_key not in statistics:
            raise ValueError(
                "incomplete spatial evaluation statistics: "
                f"missing={sum_key if sum_key not in statistics else count_key}"
            )
        reduced[objective] = _component_balanced_mean_from_statistics(
            statistics[sum_key],
            statistics[count_key],
        )
    explicit = (
        reduced["explicit_instance"]
        + reduced["explicit_abundance"]
    )
    total = (
        reduced["instance_point"]
        + reduced["measurement_positive"]
        + float(explicit_negative_weight) * explicit
        + float(implicit_negative_weight) * reduced["implicit"]
    )
    return {
        "spatial": total,
        "spatial_instance_point": reduced["instance_point"],
        "spatial_measurement_positive": reduced[
            "measurement_positive"
        ],
        "spatial_explicit_negative": explicit,
        "spatial_implicit_negative": reduced["implicit"],
    }


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
        spatial_supervised_step: int,
        tiles_seen_in_epoch: int,
        lr: float,
        loss_cfg: dict,
        classification_active: bool,
        spatial_active: bool,
        loss: torch.Tensor,
        parts: dict[str, torch.Tensor],
    ) -> None:
        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "spatial_supervised_step": int(spatial_supervised_step),
            "tiles_seen_in_epoch": int(tiles_seen_in_epoch),
            "lr": float(lr),
            "scheduled_semantic_weight": float(loss_cfg["semantic_weight"]),
            "scheduled_classification_weight": float(loss_cfg["classification_weight"]),
            "scheduled_spatial_weight": float(loss_cfg["spatial_weight"]),
            "scheduled_filter_weight": float(
                loss_cfg["prototype_filter_weight"]
            ),
            "scheduled_response_weight": float(
                loss_cfg["zhcc_response_weight"]
            ),
            "classification_active": int(classification_active),
            "spatial_active": int(spatial_active),
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


def _set_loader_batch_cursor(loader, batch: int) -> None:
    batch = int(batch)
    if batch <= 0:
        return
    setter = getattr(loader, "set_batch_cursor", None)
    if callable(setter):
        setter(batch)
        return
    candidate = getattr(loader, "batch_sampler", None)
    setter = getattr(candidate, "set_batch_cursor", None)
    if callable(setter):
        setter(batch)
        return
    raise RuntimeError(
        "training loader cannot resume from a mid-epoch batch cursor"
    )


def _scalar_epoch_metrics(metrics: dict) -> dict[str, float]:
    """Keep public metric rows separate from internal continuation state."""

    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, Number)
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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


def _spatial_positive_from_batch(batch: dict) -> torch.Tensor:
    return (
        (batch["spatial_point_centers"] > 0)
        | (batch["spatial_brush_bag_ids"] > 0)
        | batch["spatial_area_positive"].to(dtype=torch.bool)
    ).flatten(2).any(dim=2)


def _spatial_global_targets_from_spatial(
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize local ROI evidence without promoting local negatives."""

    positive = _spatial_positive_from_batch(batch)
    complete_negative = batch["spatial_explicit_negative"].to(
        dtype=torch.bool
    ).flatten(2).all(dim=2)
    return positive, positive | complete_negative


def _bucket_spatial_sample_mask(
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    """Pad active spatial rows to a bounded set of compiled batch shapes.

    The input mask is produced by the CPU dataloader. Added rows have no
    spatial supervision, so they only stabilize the spatial-head compute
    shape and remain ignored by every spatial loss.
    """

    if sample_mask.ndim != 1 or sample_mask.dtype is not torch.bool:
        raise ValueError("spatial sample mask must be a one-dimensional bool tensor")
    active_count = int(sample_mask.sum().item())
    if active_count == 0:
        return sample_mask
    bucket_size = min(
        1 << (active_count - 1).bit_length(),
        int(sample_mask.numel()),
    )
    if bucket_size == active_count:
        return sample_mask
    compute_mask = sample_mask.clone()
    padding_indices = (~sample_mask).nonzero(as_tuple=False).flatten()
    compute_mask[padding_indices[: bucket_size - active_count]] = True
    return compute_mask


def _spatial_bucket_sizes(batch_size: int) -> tuple[int, ...]:
    """Return every ROI shape reachable by the power-of-two bucketer."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("spatial bucket batch size must be positive")
    sizes: list[int] = []
    bucket = 1
    while bucket < batch_size:
        sizes.append(bucket)
        bucket *= 2
    sizes.append(batch_size)
    return tuple(sizes)


def _warmup_compiled_spatial_buckets(
    model,
    batch: dict,
    cfg: dict,
    device: torch.device,
    *,
    batch_sizes: tuple[int, ...],
) -> dict[str, int | float | list[int]]:
    """Compile every reachable spatial-head forward/backward ROI shape."""

    raw_model = getattr(model, "_orig_mod", model)
    normalized_batch_sizes = tuple(
        sorted({int(value) for value in batch_sizes})
    )
    if (
        getattr(raw_model, "spatial_head", None) is None
        or not normalized_batch_sizes
    ):
        return {
            "graphs": 0,
            "seconds": 0.0,
            "batch_sizes": [],
            "roi_sizes": [],
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
    if normalized_batch_sizes[0] <= 0:
        raise ValueError("spatial warmup requires positive batch sizes")
    available = int(batch["images"].shape[0])
    if normalized_batch_sizes[-1] > available:
        raise ValueError(
            "spatial warmup batch is too small: "
            f"available={available} required={normalized_batch_sizes[-1]}"
        )

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None
    )
    was_training = bool(raw_model.training)
    raw_model.train()
    raw_model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph_count = 0
    roi_sizes: set[int] = set()
    peak_allocated = 0.0
    peak_reserved = 0.0
    try:
        prepared = _prepare_images(
            batch,
            cfg,
            device,
            normalization=_image_normalization(cfg, device),
            amp_input=_amp_enabled(device, cfg),
        )
        for batch_size in normalized_batch_sizes:
            images = prepared[:batch_size]
            for roi_size in _spatial_bucket_sizes(batch_size):
                spatial_sample_mask = torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                )
                spatial_sample_mask[:roi_size] = True
                with torch.autocast(
                    device_type=device.type,
                    enabled=_amp_enabled(device, cfg),
                ):
                    outputs = model(
                        images,
                        spatial_detach_backbone=bool(
                            cfg["loss"].get(
                                "spatial_detach_backbone",
                                False,
                            )
                        ),
                        return_spatial_features=True,
                        run_spatial=True,
                        spatial_sample_mask=spatial_sample_mask,
                    )
                    differentiable = [
                        outputs["embedding_norm"],
                        outputs["classification_logits"],
                        outputs["spatial_instance_logits"],
                        outputs["spatial_abundance_logits"],
                        *outputs["teacher_outputs"].values(),
                    ]
                    surrogate = sum(
                        value.float().square().mean()
                        for value in differentiable
                    )
                surrogate.backward()
                raw_model.zero_grad(set_to_none=True)
                graph_count += 1
                roi_sizes.add(roi_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = float(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            )
            peak_reserved = float(
                torch.cuda.max_memory_reserved(device) / (1024 * 1024)
            )
    finally:
        raw_model.zero_grad(set_to_none=True)
        raw_model.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
    result: dict[str, int | float | list[int]] = {
        "graphs": graph_count,
        "seconds": time.perf_counter() - started,
        "batch_sizes": list(normalized_batch_sizes),
        "roi_sizes": sorted(roi_sizes),
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
    }
    print(
        "compiled_spatial_bucket_warmup "
        f"graphs={result['graphs']} "
        f"batch_sizes={result['batch_sizes']} "
        f"roi_sizes={result['roi_sizes']} "
        f"seconds={result['seconds']:.2f} "
        f"cuda_peak_alloc/resv_mb="
        f"{result['peak_allocated_mb']:.0f}/"
        f"{result['peak_reserved_mb']:.0f}",
        flush=True,
    )
    return result


@torch.inference_mode()
def _refresh_global_prototypes(
    model,
    loader,
    cfg: dict,
    device: torch.device,
) -> dict[str, int | float]:
    """Recompute exact classification and global spatial prototypes from the complete bank."""

    raw_model = getattr(model, "_orig_mod", model)
    if raw_model.classification_prototypes is None:
        raise RuntimeError("dynamic prototype refresh requires a classification readout")
    embedding_dim = int(raw_model.classification_prototypes.shape[1])
    classification_sums = torch.zeros(
        raw_model.classification_num_classes,
        embedding_dim,
        device=device,
        dtype=torch.float32,
    )
    classification_counts = torch.zeros(
        raw_model.classification_num_classes,
        device=device,
        dtype=torch.float32,
    )
    component_count = int(raw_model.spatial_num_components)
    spatial_sums = torch.zeros(
        component_count,
        embedding_dim,
        device=device,
        dtype=torch.float32,
    )
    spatial_counts = torch.zeros(
        component_count,
        device=device,
        dtype=torch.float32,
    )
    teacher_sums = {
        name: torch.zeros_like(state.prototypes, dtype=torch.float32)
        for name, state in raw_model.teacher_spatial_prototypes.items()
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
            mask, targets, _ = _move_classification_batch(batch, device)
            selected = F.one_hot(
                targets.clamp(0, raw_model.classification_num_classes - 1),
                num_classes=raw_model.classification_num_classes,
            ).to(dtype=embedding_norm.dtype)
            selected = selected * mask.to(
                dtype=embedding_norm.dtype
            ).unsqueeze(1)
            classification_sums.add_(selected.transpose(0, 1) @ embedding_norm)
            classification_counts.add_(selected.sum(dim=0))

            if component_count > 0:
                positive = _spatial_positive_from_batch(batch).to(
                    device=device,
                    dtype=embedding_norm.dtype,
                    non_blocking=device.type == "cuda",
                )
                spatial_sums.add_(positive.transpose(0, 1) @ embedding_norm)
                spatial_counts.add_(positive.sum(dim=0))
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
    raw_model.replace_classification_prototypes(classification_sums, classification_counts)
    if component_count > 0:
        raw_model.replace_global_spatial_prototypes(
            spatial_sums,
            spatial_counts,
            teacher_sums,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "tiles": tile_count,
        "classification_observations": int(classification_counts.sum().item()),
        "spatial_positive_observations": int(spatial_counts.sum().item()),
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
                "spatial_supervised"
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
                    "spatial_point_centers",
                    "spatial_instance_exclusion_support",
                    "spatial_brush_bag_ids",
                    "spatial_area_positive",
                    "spatial_explicit_negative",
                    "spatial_implicit_negative",
                )
            }
            active = _move_spatial_batch(active_host, device)
            observations = head.prototype_observation_sums(
                outputs["spatial_features"],
                point_centers=active["spatial_point_centers"],
                instance_exclusion_support=active[
                    "spatial_instance_exclusion_support"
                ],
                brush_bag_ids=active["spatial_brush_bag_ids"],
                area_positive=active["spatial_area_positive"],
                explicit_negative=active["spatial_explicit_negative"],
                implicit_negative=active["spatial_implicit_negative"],
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
            f"classification_observations={metrics['classification_observations']} "
            f"spatial_positive_observations={metrics['spatial_positive_observations']} "
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


def _move_classification_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    size = len(batch["tile_id"])
    mask = batch.get(
        "prototype_mask",
        torch.zeros(size, dtype=torch.bool),
    )
    target = batch.get(
        "prototype_classification",
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
        "spatial_point_centers",
        "spatial_brush_bag_ids",
        "spatial_area_positive",
        "spatial_explicit_negative",
        "spatial_implicit_negative",
    )
    result = {
        key: batch[key].to(
            device,
            non_blocking=device.type == "cuda",
        )
        for key in keys
    }
    exclusion = batch.get(
        "spatial_instance_exclusion_support",
        torch.zeros_like(
            batch["spatial_area_positive"],
            dtype=torch.bool,
        ),
    )
    result["spatial_instance_exclusion_support"] = exclusion.to(
        device,
        non_blocking=device.type == "cuda",
    )
    return result


def _amp_enabled(device: torch.device, cfg: dict) -> bool:
    """Use the configured CUDA precision consistently for training and evaluation."""

    return bool(cfg["train"].get("amp", False) and device.type == "cuda")


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
    """Resolve the teacher-prior and parallel classification/spatial objective schedule."""

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
        "classification_temperature": float(
            loss_cfg.get("classification_temperature", semantic_temperature)
        ),
        "pamtd_classification_temperature": float(
            loss_cfg.get("pamtd_classification_temperature", 0.1)
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
        "spatial_global_temperature": float(
            loss_cfg.get("spatial_global_temperature", 0.1)
        ),
        "classification_weight": _step_ramp(
            float(loss_cfg.get("classification_weight", 1.0)),
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
            loss_cfg.get("spatial_brush_top_fraction", 1.0)
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


def _optimizer_hyperparameters(
    optimizer: torch.optim.Optimizer,
) -> list[dict]:
    """Return the complete restart-relevant optimizer settings without parameters."""

    return [
        {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        for group in optimizer.state_dict()["param_groups"]
    ]


def _scheduler_contract(cfg: dict, steps_per_epoch: int) -> dict:
    """Describe the LR trajectory independently of the scheduler pickle state."""

    scheduler = str(cfg["train"].get("scheduler", "none")).lower()
    planned_epochs = int(cfg["train"]["epochs"])
    return {
        "name": scheduler,
        "base_lr": float(cfg["train"].get("lr", 0.0)),
        "min_lr": float(cfg["train"].get("min_lr", 0.0)),
        "warmup_steps": int(cfg["train"].get("lr_warmup_steps", 0)),
        "steps_per_epoch": int(steps_per_epoch),
        "planned_epochs": planned_epochs,
        "planned_total_steps": planned_epochs * int(steps_per_epoch),
        "terminal_behavior": (
            "clamp_at_min_lr" if scheduler == "cosine" else "unchanged"
        ),
    }


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
        torch.set_rng_state(
            torch.as_tensor(
                state["torch"],
                dtype=torch.uint8,
                device="cpu",
            ).contiguous()
        )
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(
                    value,
                    dtype=torch.uint8,
                    device="cpu",
                ).contiguous()
                for value in state["cuda"]
            ]
        )


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


def _cuda_reserved_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_reserved(device) / (1024 * 1024))


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
        "classification",
        "spatial",
        "spatial_instance_point",
        "spatial_abundance_point",
        "spatial_brush_bag",
        "spatial_area_positive",
        "spatial_measurement_positive",
        "spatial_explicit_negative",
        "spatial_implicit_negative",
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
    spatial_supervised_step: int = 0,
    summary_writer=None,
    collect_embeddings: bool = False,
    max_eval_batches: int | None = None,
    prefetched_iterator=None,
    prototype_refresh_state: PrototypeRefreshState | None = None,
    step_metrics_writer: StepMetricsWriter | None = None,
    development_probe: Callable[[int, int, int], bool] | None = None,
    step_checkpoint: Callable[[int, int, int, int, dict], None] | None = None,
    resume_epoch_accumulator: dict | None = None,
) -> dict[str, float] | tuple[dict[str, float], tuple]:
    model.train(train)
    default_totals: dict[str, torch.Tensor | float] = {
        "loss": 0.0,
        "feature": 0.0,
        "relation": 0.0,
        "semantic": 0.0,
        "pamtd_response": 0.0,
        "teacher_alpha_mean": 0.0,
        "classification": 0.0,
        "classification_accuracy": 0.0,
        "classification_supervised_tiles": 0.0,
        "spatial": 0.0,
        "spatial_instance_point": 0.0,
        "spatial_abundance_point": 0.0,
        "spatial_brush_bag": 0.0,
        "spatial_area_positive": 0.0,
        "spatial_measurement_positive": 0.0,
        "spatial_explicit_negative": 0.0,
        "spatial_implicit_negative": 0.0,
        "spatial_point_supervised_pairs": 0.0,
        "spatial_point_count": 0.0,
        "spatial_brush_supervised_pairs": 0.0,
        "spatial_brush_bag_count": 0.0,
        "spatial_area_supervised_pairs": 0.0,
        "spatial_explicit_negative_pairs": 0.0,
        "spatial_implicit_negative_pairs": 0.0,
    }
    resumed = resume_epoch_accumulator if train else None
    totals = {
        **default_totals,
        **dict((resumed or {}).get("totals", {})),
    }
    spatial_eval_statistics: dict[str, torch.Tensor] = {}
    gradient_totals = dict((resumed or {}).get("gradient_totals", {}))
    gradient_count = int((resumed or {}).get("gradient_count", 0))
    gradient_diagnostic_interval = int(
        cfg["train"].get(
            "gradient_diagnostic_interval_steps",
            GRADIENT_DIAGNOSTIC_INTERVAL_STEPS,
        )
        or 0
    )
    last_gradient_step = int(
        (resumed or {}).get(
            "last_gradient_step",
            global_step - max(1, gradient_diagnostic_interval),
        )
    )
    n_batches = int((resumed or {}).get("batches", 0))
    n_tiles = int((resumed or {}).get("tiles", 0))
    prior_elapsed = float((resumed or {}).get("seconds", 0.0))
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
            "prototype_classification": [],
            "classification_logits": [],
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
            _probe.timeline_event(
                "batch.data_wait",
                seconds=data_wait,
                epoch=epoch,
                batch=n_batches,
                global_step=global_step,
                phase=phase,
            )
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
                amp_input=_amp_enabled(device, cfg),
            )
            if will_log and device.type == "cuda":
                torch.cuda.synchronize(device)
            image_prepare = time.perf_counter() - image_prepare_start
            n_tiles += int(images.shape[0])
            interval_tiles += int(images.shape[0])
            teachers = _move_teachers(batch, device)
            classification_mask, classification_target, classification_is_active = _move_classification_batch(
                batch,
                device,
            )
            spatial_sample_mask = batch["spatial_supervised"].any(dim=1)
            spatial_is_active = bool(spatial_sample_mask.any())
            spatial_compute_mask = (
                _bucket_spatial_sample_mask(spatial_sample_mask)
                if spatial_is_active
                else spatial_sample_mask
            )
            spatial_host = {
                key: batch[key]
                for key in (
                    "spatial_point_centers",
                    "spatial_brush_bag_ids",
                    "spatial_area_positive",
                    "spatial_explicit_negative",
                    "spatial_implicit_negative",
                )
            }
            spatial_host["spatial_instance_exclusion_support"] = batch.get(
                "spatial_instance_exclusion_support",
                torch.zeros_like(
                    batch["spatial_area_positive"],
                    dtype=torch.bool,
                ),
            )
            active_spatial_host = (
                {
                    key: value[spatial_compute_mask]
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
                classification_objective_active = bool(
                    classification_is_active and float(loss_cfg["classification_weight"]) > 0
                )
                spatial_objective_active = bool(
                    spatial_is_active
                    and float(loss_cfg["spatial_weight"]) > 0
                )
                if train:
                    refresh_start = time.perf_counter()
                    _maybe_refresh_prototypes(
                        model=model,
                        cfg=cfg,
                        device=device,
                        state=prototype_refresh_state,
                        global_step=global_step,
                    )
                    _probe.timeline_event(
                        "batch.prototype_refresh",
                        seconds=time.perf_counter() - refresh_start,
                        epoch=epoch,
                        batch=n_batches,
                        global_step=global_step,
                    )
                forward_loss_start = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    enabled=_amp_enabled(device, cfg),
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
                            spatial_compute_mask
                            if spatial_objective_active
                            else None
                        ),
                    )
                    raw_model = getattr(model, "_orig_mod", model)
                    spatial_positive = None
                    spatial_known = None
                    spatial_target = None
                    if spatial_is_active:
                        if active_spatial_host is None:  # pragma: no cover
                            raise RuntimeError(
                                "active spatial targets were not prepared"
                            )
                        (
                            active_spatial_positive,
                            active_spatial_known,
                        ) = _spatial_global_targets_from_spatial(
                            active_spatial_host
                        )
                        summary_shape = (
                            spatial_sample_mask.shape[0],
                            active_spatial_positive.shape[1],
                        )
                        spatial_positive_host = torch.zeros(
                            summary_shape,
                            dtype=torch.bool,
                        )
                        spatial_known_host = torch.zeros_like(
                            spatial_positive_host
                        )
                        spatial_positive_host[spatial_compute_mask] = (
                            active_spatial_positive
                        )
                        spatial_known_host[spatial_compute_mask] = (
                            active_spatial_known
                        )
                        spatial_positive = spatial_positive_host.to(
                            device,
                            non_blocking=device.type == "cuda",
                        )
                        spatial_known = spatial_known_host.to(
                            device,
                            non_blocking=device.type == "cuda",
                        )
                        spatial_target = spatial_positive.to(dtype=torch.float32)
                    student_by_teacher = outputs["teacher_outputs"]
                    teacher_spatial_prototypes = {
                        name: (
                            state.prototypes,
                            state.counts,
                        )
                        for name, state in raw_model.teacher_spatial_prototypes.items()
                    }
                    pamtd_temperature = float(
                        loss_cfg["pamtd_classification_temperature"]
                    )
                    pamtd_student_logits = (
                        bounded_logits(
                            outputs["classification_similarity"] / pamtd_temperature
                        )
                        if "classification_similarity" in outputs
                        else None
                    )
                    adjudication = (
                        prototype_adjudicated_teacher_target(
                            teacher_by_name=teachers,
                            prototypes_by_teacher=prototypes,
                            student_classification_response=torch.softmax(
                                pamtd_student_logits,
                                dim=-1,
                            ),
                            class_names=cfg["model"].get(
                                "classification_class_names",
                                DEFAULT_CLASSIFICATION_CLASSES,
                            ),
                            teacher_spatial_prototypes=teacher_spatial_prototypes,
                            student_spatial_response=raw_model.global_spatial_response(
                                outputs["embedding_norm"],
                                temperature=float(
                                    loss_cfg["spatial_global_temperature"]
                                ),
                            ),
                            classification_mask=classification_mask,
                            classification_target=classification_target,
                            spatial_target=spatial_target,
                            spatial_known=spatial_known,
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
                            classification_temperature=pamtd_temperature,
                            spatial_temperature=float(
                                loss_cfg["spatial_global_temperature"]
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
                        classification_temperature=float(loss_cfg["classification_temperature"]),
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
                            adjudication.classification_target,
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
                    if "classification_logits" in outputs and classification_is_active:
                        classification_loss, classification_parts = classification_objective_loss(
                            outputs["classification_logits"],
                            classification_mask,
                            classification_target,
                        )
                    else:
                        classification_loss = distillation_loss.new_zeros(())
                        classification_parts = {
                            "classification": classification_loss.detach(),
                            "classification_accuracy": classification_loss.detach(),
                            "classification_supervised_tiles": classification_loss.detach(),
                        }
                    if (
                        "spatial_instance_logits" in outputs
                        and spatial_host["spatial_point_centers"].numel() > 0
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
                            instance_logits=outputs["spatial_instance_logits"],
                            abundance_logits=outputs["spatial_abundance_logits"],
                            point_centers=active_spatial["spatial_point_centers"],
                            brush_bag_ids=active_spatial["spatial_brush_bag_ids"],
                            area_positive=active_spatial["spatial_area_positive"],
                            explicit_negative=active_spatial["spatial_explicit_negative"],
                            implicit_negative=active_spatial["spatial_implicit_negative"],
                            instance_exclusion_support=active_spatial[
                                "spatial_instance_exclusion_support"
                            ],
                            point_centers_host=active_spatial_host[
                                "spatial_point_centers"
                            ],
                            brush_bag_ids_host=active_spatial_host[
                                "spatial_brush_bag_ids"
                            ],
                            area_positive_host=active_spatial_host[
                                "spatial_area_positive"
                            ],
                            instance_exclusion_support_host=(
                                active_spatial_host[
                                    "spatial_instance_exclusion_support"
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
                                "spatial",
                                "spatial_instance_point",
                                "spatial_abundance_point",
                                "spatial_brush_bag",
                                "spatial_area_positive",
                                "spatial_measurement_positive",
                                "spatial_explicit_negative",
                                "spatial_implicit_negative",
                                "spatial_point_supervised_pairs",
                                "spatial_point_count",
                                "spatial_brush_supervised_pairs",
                                "spatial_brush_bag_count",
                                "spatial_area_supervised_pairs",
                                "spatial_explicit_negative_pairs",
                                "spatial_implicit_negative_pairs",
                            )
                        }
                    global_objective = (
                        distillation_loss
                        + float(loss_cfg["classification_weight"]) * classification_loss
                        + float(loss_cfg["zhcc_response_weight"])
                        * response_loss
                    )
                    spatial_objective = (
                        float(loss_cfg["spatial_weight"]) * spatial_loss
                    )
                    loss = global_objective + spatial_objective
                _probe.timeline_event(
                    "batch.forward_and_loss",
                    seconds=time.perf_counter() - forward_loss_start,
                    epoch=epoch,
                    batch=n_batches,
                    global_step=global_step,
                    classification_active=classification_objective_active,
                    spatial_active=spatial_objective_active,
                    spatial_sample_count=int(spatial_sample_mask.sum().item()),
                    spatial_compute_count=int(spatial_compute_mask.sum().item()),
                )

                if (
                    train
                    and gradient_diagnostic_interval > 0
                    and spatial_is_active
                    and float(loss_cfg["spatial_weight"]) > 0
                    and not bool(loss_cfg["spatial_detach_backbone"])
                    and global_step - last_gradient_step
                    >= gradient_diagnostic_interval
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
                    optimizer_start = time.perf_counter()
                    optimizer_stepped = _optimizer_step(
                        loss=loss,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        max_grad_norm=max_grad_norm,
                    )
                    _probe.timeline_event(
                        "batch.backward_optimizer",
                        seconds=time.perf_counter() - optimizer_start,
                        epoch=epoch,
                        batch=n_batches,
                        global_step=global_step,
                        spatial_active=spatial_objective_active,
                    )
                    if scheduler is not None and optimizer_stepped:
                        scheduler.step()
                    if optimizer_stepped:
                        global_step += 1
                        if spatial_objective_active:
                            spatial_supervised_step += 1

            parts = {
                **distillation_parts,
                **classification_parts,
                **spatial_parts,
                "pamtd_response": response_loss.detach(),
                "teacher_alpha_mean": alpha_mean.detach(),
            }
            if not train:
                for key, value in parts.items():
                    if not key.startswith(SPATIAL_EVAL_STAT_PREFIX):
                        continue
                    current = value.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    if key in spatial_eval_statistics:
                        spatial_eval_statistics[key] = (
                            spatial_eval_statistics[key] + current
                        )
                    else:
                        spatial_eval_statistics[key] = current.clone()
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
                    spatial_supervised_step=spatial_supervised_step,
                    tiles_seen_in_epoch=n_tiles,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    loss_cfg=loss_cfg,
                    classification_active=classification_objective_active,
                    spatial_active=spatial_objective_active,
                    loss=loss,
                    parts=parts,
                )
            if collect_embeddings and (
                max_eval_batches is None or n_batches < max_eval_batches
            ):
                assert embeddings_data is not None
                embeddings_data["embeddings"].append(
                    outputs["embedding_norm"].detach().cpu()
                )
                embeddings_data["prototype_masks"].append(classification_mask.detach().cpu())
                embeddings_data["prototype_classification"].append(classification_target.detach().cpu())
                if "classification_logits" in outputs:
                    embeddings_data["classification_logits"].append(
                        outputs["classification_logits"].detach().cpu()
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
            stop_requested = False
            if train and optimizer_stepped and development_probe is not None:
                probe_start = time.perf_counter()
                stop_requested = bool(
                    development_probe(
                        global_step,
                        spatial_supervised_step,
                        epoch,
                    )
                )
                _probe.timeline_event(
                    "batch.development_probe",
                    seconds=time.perf_counter() - probe_start,
                    epoch=epoch,
                    batch=n_batches,
                    global_step=global_step,
                )
            if train and optimizer_stepped and step_checkpoint is not None:
                checkpoint_totals = {
                    key: (
                        float(value.detach().cpu())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    for key, value in totals.items()
                }
                step_checkpoint(
                    global_step,
                    spatial_supervised_step,
                    epoch,
                    n_batches,
                    {
                        "totals": checkpoint_totals,
                        "gradient_totals": dict(gradient_totals),
                        "gradient_count": gradient_count,
                        "last_gradient_step": last_gradient_step,
                        "batches": n_batches,
                        "tiles": n_tiles,
                        "seconds": (
                            prior_elapsed + time.perf_counter() - start
                        ),
                    },
                )
            _probe.timeline_event(
                "batch.total",
                seconds=time.perf_counter() - batch_start,
                epoch=epoch,
                batch=n_batches,
                global_step=global_step,
                phase=phase,
            )
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
                    detail = {
                        "loss": f"{float(loss.detach().cpu()):.4f}",
                        "tiles_s": f"{n_tiles / elapsed:.0f}",
                    }
                    if device.type == "cuda":
                        detail["cuda_alloc/resv_mb"] = (
                            f"{_cuda_memory_mb(device):.0f}/"
                            f"{_cuda_reserved_memory_mb(device):.0f}"
                        )
                    progress_bar.set_postfix(detail, refresh=False)
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
                    f"cuda_mem_mb={_cuda_memory_mb(device):.1f} "
                    "cuda_reserved_mb="
                    f"{_cuda_reserved_memory_mb(device):.1f}"
                )
                interval_start = now
                interval_tiles = 0
            if stop_requested:
                break
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
    elapsed = max(
        prior_elapsed + time.perf_counter() - start,
        1e-9,
    )
    result: dict[str, float] = {}
    for key, value in totals.items():
        mean_value = value / n_batches
        result[key] = (
            float(mean_value.detach().cpu())
            if isinstance(mean_value, torch.Tensor)
            else float(mean_value)
        )
    if spatial_eval_statistics:
        result.update(
            _complete_bank_spatial_metrics(
                spatial_eval_statistics,
                explicit_negative_weight=float(
                    last_loss_cfg[
                        "spatial_explicit_negative_weight"
                    ]
                ),
                implicit_negative_weight=float(
                    last_loss_cfg[
                        "spatial_implicit_negative_weight"
                    ]
                ),
            )
        )
    for key, value in gradient_totals.items():
        result[key] = value / max(1, gradient_count)
    result["gradient_diagnostic_count"] = float(gradient_count)
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    result["tiles_per_sec"] = n_tiles / elapsed
    result["tiles"] = float(n_tiles)
    result["seconds"] = elapsed
    result["cuda_memory_allocated_mb"] = _cuda_memory_mb(device)
    result["cuda_memory_reserved_mb"] = _cuda_reserved_memory_mb(device)
    result["global_step_end"] = float(global_step)
    result["spatial_supervised_step_end"] = float(spatial_supervised_step)
    result["batch_in_epoch_end"] = float(n_batches)
    result["epoch_accumulator_end"] = {
        "totals": {
            key: (
                float(value.detach().cpu())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
            for key, value in totals.items()
        },
        "gradient_totals": dict(gradient_totals),
        "gradient_count": gradient_count,
        "last_gradient_step": last_gradient_step,
        "batches": n_batches,
        "tiles": n_tiles,
        "seconds": elapsed,
    }
    result["scheduled_classification_weight"] = float(last_loss_cfg["classification_weight"])
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
            "prototype_classification": torch.cat(embeddings_data["prototype_classification"]),
            "classification_logits": (
                torch.cat(embeddings_data["classification_logits"])
                if embeddings_data["classification_logits"]
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
    classification = []
    classification_logits = []
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
        mask, target, _ = _move_classification_batch(batch, torch.device("cpu"))
        masks.append(mask)
        classification.append(target)
        if "classification_logits" in outputs:
            classification_logits.append(outputs["classification_logits"].cpu())
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
            "prototype_classification": torch.cat(classification),
            "classification_logits": torch.cat(classification_logits) if classification_logits else torch.zeros((0, 0)),
        },
    )


def _classification_eval_metrics(supervised: dict[str, torch.Tensor]) -> dict[str, float]:
    mask = supervised["prototype_mask"].bool()
    logits = supervised["classification_logits"]
    if logits.numel() == 0 or not bool(mask.any()):
        return {
            "classification_accuracy": 0.0,
            "classification_balanced_accuracy": 0.0,
            "classification_macro_f1": 0.0,
            "classification_cross_entropy": float("inf"),
            "classification_balanced_cross_entropy": float("inf"),
            "classification_evaluated_tiles": 0.0,
            "classification_evaluated_classes": 0.0,
            "classification_total_classes": float(
                logits.shape[1]
                if logits.ndim == 2
                else 0
            ),
        }
    target = supervised["prototype_classification"][mask]
    selected_logits = logits[mask].float()
    prediction = selected_logits.argmax(dim=1)
    class_count = int(selected_logits.shape[1])
    recalls: list[float] = []
    f1_values: list[float] = []
    per_sample_cross_entropy = F.cross_entropy(
        selected_logits,
        target,
        reduction="none",
    )
    class_cross_entropies: list[float] = []
    for class_index in range(class_count):
        target_positive = target == class_index
        if not bool(target_positive.any()):
            continue
        predicted_positive = prediction == class_index
        true_positive = int(
            (target_positive & predicted_positive).sum()
        )
        false_positive = int(
            ((~target_positive) & predicted_positive).sum()
        )
        false_negative = int(
            (target_positive & (~predicted_positive)).sum()
        )
        recalls.append(
            true_positive / max(1, true_positive + false_negative)
        )
        f1_values.append(
            (2 * true_positive)
            / max(
                1,
                2 * true_positive + false_positive + false_negative,
            )
        )
        class_cross_entropies.append(
            float(per_sample_cross_entropy[target_positive].mean())
        )
    return {
        "classification_accuracy": float((prediction == target).float().mean()),
        "classification_balanced_accuracy": float(np.mean(recalls)),
        "classification_macro_f1": float(np.mean(f1_values)),
        "classification_cross_entropy": float(
            per_sample_cross_entropy.mean().detach()
        ),
        "classification_balanced_cross_entropy": float(
            np.mean(class_cross_entropies)
        ),
        "classification_evaluated_tiles": float(mask.sum()),
        "classification_evaluated_classes": float(len(recalls)),
        "classification_total_classes": float(class_count),
    }


SELECTION_COMPONENTS = (
    "teacher",
    "classification",
    "spatial",
)
DEFAULT_SELECTION_WEIGHTS = {
    "teacher": 0.50,
    "classification": 0.25,
    "spatial": 0.25,
}


def _selection_weights(cfg: dict) -> dict[str, float]:
    configured = cfg.get("train", {}).get(
        "selection_metric_weights",
        DEFAULT_SELECTION_WEIGHTS,
    )
    if not isinstance(configured, dict):
        raise ValueError("train.selection_metric_weights must be a mapping")
    unknown = sorted(set(configured).difference(SELECTION_COMPONENTS))
    missing = sorted(set(SELECTION_COMPONENTS).difference(configured))
    if unknown or missing:
        raise ValueError(
            "train.selection_metric_weights must contain exactly "
            f"{list(SELECTION_COMPONENTS)}: missing={missing} "
            f"unknown={unknown}"
        )
    weights = {
        name: float(configured[name])
        for name in SELECTION_COMPONENTS
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in weights.values()
    ):
        raise ValueError(
            "all train.selection_metric_weights must be finite and positive"
        )
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "train.selection_metric_weights must sum to 1.0: "
            f"observed={total}"
        )
    return weights


def _configured_selection_baseline(
    cfg: dict,
) -> dict[str, float] | None:
    configured = cfg.get("train", {}).get(
        "selection_metric_baseline"
    )
    if configured is None:
        return None
    if not isinstance(configured, dict) or set(configured) != set(
        SELECTION_COMPONENTS
    ):
        raise ValueError(
            "train.selection_metric_baseline must contain exactly "
            f"{list(SELECTION_COMPONENTS)}"
        )
    baseline = {
        name: float(configured[name])
        for name in SELECTION_COMPONENTS
    }
    _normalized_selection_metrics(
        baseline,
        baseline,
        DEFAULT_SELECTION_WEIGHTS,
    )
    return baseline


def _assert_selection_baseline_matches(
    observed: dict[str, float],
    expected: dict[str, float],
) -> None:
    for name in SELECTION_COMPONENTS:
        if not math.isclose(
            float(observed[name]),
            float(expected[name]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "shared initialization baseline mismatch: "
                f"component={name} observed={float(observed[name]):.9g} "
                f"expected={float(expected[name]):.9g}"
            )


def _selection_start_step(cfg: dict) -> int:
    loss_cfg = cfg.get("loss", {})
    endpoints = [
        int(loss_cfg.get("expert_supervision_start_step", 0))
        + int(loss_cfg.get("expert_supervision_ramp_steps", 0)),
    ]
    if float(loss_cfg.get("prototype_filter_weight", 0.0)) > 0.0:
        endpoints.append(
            int(loss_cfg.get("prototype_filter_start_step", 0))
            + int(loss_cfg.get("prototype_filter_ramp_steps", 0))
        )
    if float(loss_cfg.get("zhcc_response_weight", 0.0)) > 0.0:
        endpoints.append(
            int(loss_cfg.get("zhcc_response_start_step", 0))
            + int(loss_cfg.get("zhcc_response_ramp_steps", 0))
        )
    required = max(endpoints)
    configured = cfg.get("train", {}).get(
        "selection_early_stop_start_step"
    )
    if configured is None:
        return required
    return max(int(configured), required)


def _eligible_selection_epoch_count(
    *,
    current_global_step: int,
    steps_per_epoch: int,
    start_epoch: int,
    expected_epochs: int,
    selection_start_step: int,
    resume_batch_in_epoch: int = 0,
) -> int:
    """Count remaining epoch ends at which joint selection is valid."""

    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if resume_batch_in_epoch < 0 or resume_batch_in_epoch > steps_per_epoch:
        raise ValueError(
            "resume_batch_in_epoch must be within the current epoch"
        )
    if expected_epochs < start_epoch:
        return 0
    step = int(current_global_step)
    eligible = 0
    for epoch in range(start_epoch, expected_epochs + 1):
        completed_batches = (
            resume_batch_in_epoch if epoch == start_epoch else 0
        )
        step += steps_per_epoch - completed_batches
        if step >= selection_start_step:
            eligible += 1
    return eligible


def _selection_early_stop_requested(
    *,
    enabled: bool,
    eligible: bool,
    eligible_epochs: int,
    minimum_eligible_epochs: int,
    bad_epochs: int,
    patience: int,
) -> bool:
    """Gate joint-validation stopping by both evidence and patience."""

    return bool(
        enabled
        and eligible
        and eligible_epochs >= minimum_eligible_epochs
        and bad_epochs >= patience
    )


def _teacher_retention_metrics(
    embedding_metrics: dict[str, float],
    *,
    relation_weight: float,
) -> dict[str, float]:
    feature_by_teacher = {
        key[: -len("_feature_cosine")]: float(value)
        for key, value in embedding_metrics.items()
        if key.endswith("_feature_cosine")
    }
    relation_by_teacher = {
        key[: -len("_relation_mse")]: float(value)
        for key, value in embedding_metrics.items()
        if key.endswith("_relation_mse")
    }
    if not feature_by_teacher:
        raise ValueError(
            "teacher validation requires at least one feature cosine metric"
        )
    if (
        relation_by_teacher
        and relation_by_teacher.keys() != feature_by_teacher.keys()
    ) or (relation_weight > 0.0 and not relation_by_teacher):
        raise ValueError(
            "teacher validation requires paired feature cosine and relation "
            "metrics for every teacher"
        )
    feature_cosines = list(feature_by_teacher.values())
    relation_losses = (
        list(relation_by_teacher.values())
        if relation_by_teacher
        else [0.0] * len(feature_cosines)
    )
    if not all(
        math.isfinite(value)
        for value in (*feature_cosines, *relation_losses)
    ):
        raise FloatingPointError("non-finite fixed teacher validation metric")
    feature_distance = float(
        sum(1.0 - value for value in feature_cosines)
        / len(feature_cosines)
    )
    relation = float(sum(relation_losses) / len(relation_losses))
    total = feature_distance + float(relation_weight) * relation
    return {
        "teacher_alignment_score": float(
            sum(feature_cosines) / len(feature_cosines)
        ),
        "fixed_teacher_distance": feature_distance,
        "fixed_teacher_relation": relation,
        "teacher_validation_loss": total,
    }


def _normalized_selection_metrics(
    raw: dict[str, float],
    baseline: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name in SELECTION_COMPONENTS:
        value = float(raw[name])
        reference = float(baseline[name])
        if (
            not math.isfinite(value)
            or not math.isfinite(reference)
            or value < 0.0
            or reference <= 0.0
        ):
            raise FloatingPointError(
                "selection metrics and baselines must be finite with a "
                f"positive baseline: component={name} value={value} "
                f"baseline={reference}"
            )
        normalized[name] = value / reference
    selection_loss = sum(
        weights[name] * normalized[name]
        for name in SELECTION_COMPONENTS
    )
    return {
        **{
            f"selection_{name}_raw": float(raw[name])
            for name in SELECTION_COMPONENTS
        },
        **{
            f"selection_{name}_baseline": float(baseline[name])
            for name in SELECTION_COMPONENTS
        },
        **{
            f"selection_{name}_normalized": normalized[name]
            for name in SELECTION_COMPONENTS
        },
        **{
            f"selection_{name}_weight": weights[name]
            for name in SELECTION_COMPONENTS
        },
        "selection_loss": float(selection_loss),
    }


def _selection_raw_metrics(
    validation: dict[str, dict[str, float]],
    cfg: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    teacher = _teacher_retention_metrics(
        validation["embedding"],
        relation_weight=float(
            cfg.get("loss", {}).get("relation_weight", 0.0)
        ),
    )
    classification = validation["expert_classification"]
    spatial = validation["expert_spatial"]
    raw = {
        "teacher": teacher["teacher_validation_loss"],
        "classification": float(
            classification["classification_balanced_cross_entropy"]
        ),
        "spatial": float(spatial["spatial"]),
    }
    return raw, teacher


def _run_validation_streams(
    *,
    model,
    val_loader,
    expert_classification_val_loader,
    expert_spatial_val_loader,
    prototypes,
    optimizer,
    device,
    cfg: dict,
    expert_eval_cfg: dict,
    epoch: int,
    global_step: int,
    spatial_supervised_step: int,
    summary_writer=None,
) -> dict[str, dict[str, float]]:
    # Compilation is an optimizer-only concern. Validation contains small
    # fixed expert banks and unavoidable tail batches; sending those shapes
    # through the compiled wrapper creates one-off graphs/CUDA allocations
    # without changing the selected checkpoint.
    evaluation_model = getattr(model, "_orig_mod", model)
    # A capped population validation view must contain the same deterministic
    # tiles in every epoch; sample order must not become a hidden moving target.
    _set_loader_epoch(val_loader, 0)
    val_iterator = iter(val_loader)
    try:
        val_metrics, val_embeddings = run_epoch(
            evaluation_model,
            val_loader,
            prototypes,
            optimizer,
            device,
            cfg,
            train=False,
            max_batches=cfg["train"].get("max_val_batches"),
            epoch=epoch,
            global_step=global_step,
            spatial_supervised_step=spatial_supervised_step,
            summary_writer=summary_writer,
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
        {
            name: registry.to("cpu")
            for name, registry in prototypes.items()
        }
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
    population_classification_metrics = _classification_eval_metrics(
        supervised
    )
    del val_embeddings, student_by_teacher, teacher_by_name, supervised
    _release_host_memory()

    expert_classification_metrics: dict[str, float] = {}
    expert_spatial_metrics: dict[str, float] = {}
    if expert_classification_val_loader is not None:
        _set_loader_epoch(expert_classification_val_loader, 0)
        classification_eval_cfg = {
            **expert_eval_cfg,
            "loss": {
                **expert_eval_cfg["loss"],
                "classification_weight": 1.0,
                "spatial_weight": 0.0,
            },
        }
        _, expert_classification_embeddings = run_epoch(
            evaluation_model,
            expert_classification_val_loader,
            prototypes,
            optimizer,
            device,
            classification_eval_cfg,
            train=False,
            max_batches=None,
            epoch=epoch,
            global_step=global_step,
            spatial_supervised_step=spatial_supervised_step,
            collect_embeddings=True,
            max_eval_batches=None,
        )
        expert_supervised = expert_classification_embeddings[3]
        expert_classification_metrics = _classification_eval_metrics(
            expert_supervised
        )
        del expert_classification_embeddings, expert_supervised
        _release_host_memory()
    if expert_spatial_val_loader is not None:
        _set_loader_epoch(expert_spatial_val_loader, 0)
        spatial_eval_cfg = {
            **expert_eval_cfg,
            "loss": {
                **expert_eval_cfg["loss"],
                "classification_weight": 0.0,
                "spatial_weight": 1.0,
            },
        }
        expert_spatial_metrics = run_epoch(
            evaluation_model,
            expert_spatial_val_loader,
            prototypes,
            optimizer,
            device,
            spatial_eval_cfg,
            train=False,
            max_batches=None,
            epoch=epoch,
            global_step=global_step,
            spatial_supervised_step=spatial_supervised_step,
        )
        _release_host_memory()
    return {
        "population": val_metrics,
        "embedding": embedding_metrics,
        "population_classification": population_classification_metrics,
        "expert_classification": expert_classification_metrics,
        "expert_spatial": expert_spatial_metrics,
    }


def _truncate_csv_after_step(path: Path, maximum_step: int) -> None:
    """Atomically discard metric rows newer than an exact resume checkpoint."""

    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if not row.get("global_step")
            or int(float(row["global_step"])) <= maximum_step
        ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _development_early_stop_state_from_csv(
    path: Path,
    *,
    maximum_step: int,
    minimum_step: int,
    relative_delta: float,
) -> dict[str, float | int | bool | None]:
    state: dict[str, float | int | bool | None] = {
        "previous_loss": None,
        "consecutive_low_gain": 0,
        "last_probe_step": 0,
        "triggered": False,
    }
    if not path.exists():
        return state
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["global_step"]))
            if step > maximum_step:
                continue
            loss = float(row["loss"])
            previous = state["previous_loss"]
            if previous is not None and step >= minimum_step:
                gain = (float(previous) - loss) / max(
                    abs(float(previous)),
                    1e-12,
                )
                state["consecutive_low_gain"] = (
                    int(state["consecutive_low_gain"]) + 1
                    if gain < relative_delta
                    else 0
                )
            state["previous_loss"] = loss
            state["last_probe_step"] = step
    return state


def _update_development_early_stop_state(
    state: dict[str, float | int | bool | None],
    *,
    step: int,
    loss: float,
    enabled: bool,
    minimum_step: int,
    relative_delta: float,
    patience: int,
) -> tuple[float, bool]:
    relative_improvement = float("nan")
    previous_loss = state.get("previous_loss")
    if enabled and previous_loss is not None and step >= minimum_step:
        relative_improvement = (
            float(previous_loss) - loss
        ) / max(abs(float(previous_loss)), 1e-12)
        state["consecutive_low_gain"] = (
            int(state.get("consecutive_low_gain", 0)) + 1
            if relative_improvement < relative_delta
            else 0
        )
    state["previous_loss"] = float(loss)
    state["last_probe_step"] = int(step)
    triggered = bool(
        enabled
        and step >= minimum_step
        and int(state.get("consecutive_low_gain", 0)) >= patience
    )
    state["triggered"] = triggered
    return relative_improvement, triggered


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
    scheduler_contract: dict | None = None,
    resume_state: dict | None = None,
    prototype_refresh_loader=None,
    spatial_prototype_refresh_loader=None,
    expert_classification_val_loader=None,
    expert_spatial_val_loader=None,
) -> dict:
    output_dir = ensure_dir(cfg["runtime"]["output_dir"])
    checkpoints = ensure_dir(output_dir / "checkpoints")
    trajectory_snapshots_dir = (
        ensure_dir(output_dir / "trajectory_snapshots")
        if bool(
            cfg["train"].get(
                "retain_trajectory_snapshots",
                False,
            )
        )
        else None
    )
    resume_global_step = int((resume_state or {}).get("global_step", 0))
    if resume_state is not None:
        for metric_name in (
            "step_metrics.csv",
            "development_metrics.csv",
            "metrics.csv",
        ):
            _truncate_csv_after_step(
                output_dir / metric_name,
                resume_global_step,
            )
    write_json(output_dir / "resolved_config.json", cfg)
    best_loss = float((resume_state or {}).get("best_loss", float("inf")))
    best_teacher_alignment = float(
        (resume_state or {}).get("best_teacher_alignment", float("-inf"))
    )
    best_classification_accuracy = float(
        (resume_state or {}).get("best_classification_accuracy", float("-inf"))
    )
    best_selection_loss = float(
        (resume_state or {}).get("best_selection_loss", float("inf"))
    )
    best_significant_selection_loss = float(
        (resume_state or {}).get(
            "best_significant_selection_loss",
            float("inf"),
        )
    )
    best_selection_epoch = int(
        (resume_state or {}).get("best_selection_epoch", 0)
    )
    selection_bad_epochs = int(
        (resume_state or {}).get("selection_bad_epochs", 0)
    )
    selection_eligible_epochs = int(
        (resume_state or {}).get("selection_eligible_epochs", 0)
    )
    selection_baseline = {
        str(name): float(value)
        for name, value in (
            (resume_state or {}).get("selection_baseline", {})
        ).items()
    }
    trajectory_snapshots = list(
        (resume_state or {}).get("trajectory_snapshots", [])
    )
    best_metrics = dict((resume_state or {}).get("best_metrics", {}))
    last_metrics = dict(
        (resume_state or {}).get(
            "last_metrics",
            (resume_state or {}).get("best_metrics", {}),
        )
    )
    alignment_history = list((resume_state or {}).get("alignment_history", []))
    resume_batch_in_epoch = int(
        (resume_state or {}).get("batch_in_epoch", 0)
    )
    checkpoint_epoch = int((resume_state or {}).get("epoch", 0))
    start_epoch = (
        checkpoint_epoch
        if resume_batch_in_epoch > 0
        else checkpoint_epoch + 1
    )
    previous_expected_epochs = int(
        (resume_state or {}).get(
            "expected_epochs",
            cfg["train"]["epochs"],
        )
    )
    expected_epochs = int(cfg["train"]["epochs"])
    if expected_epochs < start_epoch - 1:
        raise ValueError(
            "configured train.epochs precedes the checkpoint epoch: "
            f"configured={expected_epochs} checkpoint={start_epoch - 1}"
        )
    global_step = int((resume_state or {}).get("global_step", 0))
    continuation_history = list(
        (resume_state or {}).get("continuation_history", [])
    )
    if resume_state is not None and expected_epochs > previous_expected_epochs:
        continuation_history.append(
            {
                "checkpoint_epoch": start_epoch - 1,
                "checkpoint_global_step": global_step,
                "previous_expected_epochs": previous_expected_epochs,
                "configured_epochs": expected_epochs,
                "lr_terminal_behavior": (
                    "clamp_at_min_lr"
                    if str(cfg["train"].get("scheduler", "none")).lower()
                    == "cosine"
                    else "unchanged"
                ),
            }
        )
    write_json(
        output_dir / "continuation_history.json",
        {"extensions": continuation_history},
    )
    spatial_supervised_step = int(
        (resume_state or {}).get("spatial_supervised_step", 0)
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
    checkpoint_interval_steps = int(
        cfg["train"].get("checkpoint_interval_steps", 1000)
    )
    development_early_stop_enabled = bool(
        cfg["train"].get("development_early_stop", False)
    )
    development_early_stop_min_step = int(
        cfg["train"].get("development_early_stop_min_step", 4000)
    )
    development_early_stop_relative_delta = float(
        cfg["train"].get("development_early_stop_relative_delta", 0.005)
    )
    development_early_stop_patience = int(
        cfg["train"].get("development_early_stop_patience", 2)
    )
    development_early_stop_contract = {
        "enabled": development_early_stop_enabled,
        "minimum_step": development_early_stop_min_step,
        "relative_delta": development_early_stop_relative_delta,
        "patience": development_early_stop_patience,
        "probe_interval_steps": int(
            cfg["train"].get("development_probe_interval_steps", 0) or 0
        ),
        "probe_batches": int(
            cfg["train"].get("development_probe_batches", 64)
        ),
    }
    expert_selection_enabled = bool(
        expert_classification_val_loader is not None
        and expert_spatial_val_loader is not None
    )
    selection_early_stop_enabled = bool(
        expert_selection_enabled
        and cfg["train"].get("selection_early_stop", False)
    )
    selection_metric_weights = _selection_weights(cfg)
    configured_selection_baseline = _configured_selection_baseline(cfg)
    selection_early_stop_start_step = _selection_start_step(cfg)
    selection_early_stop_patience = int(
        cfg["train"].get("selection_early_stop_patience", 4)
    )
    selection_early_stop_relative_delta = float(
        cfg["train"].get(
            "selection_early_stop_relative_delta",
            0.005,
        )
    )
    selection_minimum_eligible_epochs = int(
        cfg["train"].get("selection_minimum_eligible_epochs", 1)
    )
    if expert_selection_enabled:
        configured_max_batches = cfg["train"].get("max_train_batches")
        steps_per_epoch = len(train_loader)
        if configured_max_batches is not None:
            steps_per_epoch = min(
                steps_per_epoch,
                int(configured_max_batches),
            )
        eligible_epochs = _eligible_selection_epoch_count(
            current_global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            start_epoch=start_epoch,
            expected_epochs=expected_epochs,
            selection_start_step=selection_early_stop_start_step,
            resume_batch_in_epoch=resume_batch_in_epoch,
        )
        if eligible_epochs < selection_minimum_eligible_epochs:
            raise ValueError(
                "training budget cannot reach enough eligible joint-selection "
                "epochs after all active ramps: "
                f"steps_per_epoch={steps_per_epoch} "
                f"selection_start_step={selection_early_stop_start_step} "
                f"eligible_epochs={eligible_epochs} "
                f"required={selection_minimum_eligible_epochs}"
            )
    expert_eval_cfg = {
        **cfg,
        "train": {
            **cfg["train"],
            "progress": False,
            "log_interval": 0,
            "tensorboard_batch_interval": 0,
        },
        "loss": {
            **cfg["loss"],
            "expert_supervision_start_step": 0,
            "expert_supervision_ramp_steps": 0,
        },
    }
    saved_early_stop_state = (resume_state or {}).get(
        "development_early_stop_state"
    )
    saved_early_stop_contract = (resume_state or {}).get(
        "development_early_stop_contract"
    )
    early_stop_state = (
        dict(saved_early_stop_state)
        if (
            isinstance(saved_early_stop_state, dict)
            and saved_early_stop_contract
            == development_early_stop_contract
        )
        else _development_early_stop_state_from_csv(
            output_dir / "development_metrics.csv",
            maximum_step=resume_global_step,
            minimum_step=development_early_stop_min_step,
            relative_delta=development_early_stop_relative_delta,
        )
    )
    development_stop_requested = bool(
        development_early_stop_enabled
        and early_stop_state.get("triggered", False)
    )
    selection_stop_requested = False
    development_stop_summary: dict[str, float | int | bool] = {}

    def checkpoint_payload(
        *,
        epoch: int,
        batch_in_epoch: int,
        epoch_accumulator: dict | None,
        training_complete: bool,
    ) -> dict:
        raw_model = getattr(model, "_orig_mod", model)
        return {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_hyperparameters": _optimizer_hyperparameters(
                optimizer
            ),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "scheduler_contract": _scheduler_contract(
                cfg, len(train_loader)
            ) if scheduler_contract is None else dict(scheduler_contract),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "epoch_accumulator": epoch_accumulator,
            "global_step": global_step,
            "spatial_supervised_step": spatial_supervised_step,
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
            "best_classification_accuracy": (
                best_classification_accuracy
            ),
            "best_selection_loss": best_selection_loss,
            "best_significant_selection_loss": (
                best_significant_selection_loss
            ),
            "best_selection_epoch": best_selection_epoch,
            "selection_bad_epochs": selection_bad_epochs,
            "selection_eligible_epochs": selection_eligible_epochs,
            "selection_baseline": dict(selection_baseline),
            "trajectory_snapshots": list(trajectory_snapshots),
            "selection_metric_weights": dict(
                selection_metric_weights
            ),
            "selection_early_stop_start_step": (
                selection_early_stop_start_step
            ),
            "best_metrics": best_metrics,
            "last_metrics": last_metrics,
            "alignment_history": alignment_history,
            "rng_state": _rng_state(),
            "config": cfg,
            "training_complete": bool(training_complete),
            "expected_epochs": expected_epochs,
            "continuation_history": continuation_history,
            "development_early_stop_state": dict(early_stop_state),
            "development_early_stop_contract": dict(
                development_early_stop_contract
            ),
        }

    def archive_trajectory_snapshot(
        *,
        current_epoch: int,
        step: int,
        spatial_step: int,
        metrics: dict | None,
    ) -> None:
        nonlocal trajectory_snapshots
        if (
            trajectory_snapshots_dir is None
            or current_epoch <= 1
            or step < selection_early_stop_start_step
            or any(
                int(item["global_step"]) == int(step)
                for item in trajectory_snapshots
            )
        ):
            return
        reserve_gb = float(
            cfg["train"].get(
                "trajectory_snapshot_reserve_gb",
                10.0,
            )
        )
        free_bytes = shutil.disk_usage(output_dir).free
        if free_bytes < int(reserve_gb * (1024**3)) + 1024**3:
            _log(
                "trajectory_snapshot_skipped_disk_reserve "
                f"epoch={current_epoch} global_step={step} "
                f"free_gb={free_bytes / (1024**3):.2f} "
                f"reserve_gb={reserve_gb:.2f}"
            )
            return
        filename = f"step_{step:08d}.pt"
        raw_model = getattr(model, "_orig_mod", model)
        _atomic_torch_save(
            {
                "format": "hcc-sempath-trajectory-snapshot-v1",
                "model": raw_model.state_dict(),
                "epoch": int(current_epoch),
                "global_step": int(step),
                "spatial_supervised_step": int(spatial_step),
                "metrics": (
                    _scalar_epoch_metrics(metrics)
                    if metrics is not None
                    else {}
                ),
                "selection_baseline": dict(selection_baseline),
                "config": cfg,
            },
            trajectory_snapshots_dir / filename,
        )
        entry = {
            "epoch": int(current_epoch),
            "global_step": int(step),
            "spatial_supervised_step": int(spatial_step),
            "checkpoint": filename,
            "has_complete_validation": metrics is not None,
        }
        if metrics is not None and expert_selection_enabled:
            entry["raw"] = {
                name: float(metrics[f"selection_{name}_raw"])
                for name in SELECTION_COMPONENTS
            }
            entry["normalized"] = {
                name: float(
                    metrics[f"selection_{name}_normalized"]
                )
                for name in SELECTION_COMPONENTS
            }
        trajectory_snapshots.append(entry)
        trajectory_snapshots.sort(
            key=lambda item: int(item["global_step"])
        )
        write_json(
            trajectory_snapshots_dir / "index.json",
            {
                "reserve_gb": reserve_gb,
                "snapshots": trajectory_snapshots,
            },
        )

    def save_step_checkpoint(
        step: int,
        spatial_step: int,
        current_epoch: int,
        batch_in_epoch: int,
        epoch_accumulator: dict,
    ) -> None:
        nonlocal global_step, spatial_supervised_step
        if (
            checkpoint_interval_steps <= 0
            or step % checkpoint_interval_steps != 0
            or batch_in_epoch >= len(train_loader)
        ):
            return
        global_step = int(step)
        spatial_supervised_step = int(spatial_step)
        step_metrics_writer.flush()
        started = time.perf_counter()
        archive_trajectory_snapshot(
            current_epoch=current_epoch,
            step=step,
            spatial_step=spatial_step,
            metrics=None,
        )
        _atomic_torch_save(
            checkpoint_payload(
                epoch=current_epoch,
                batch_in_epoch=batch_in_epoch,
                epoch_accumulator=epoch_accumulator,
                training_complete=False,
            ),
            checkpoints / "last.pt",
        )
        _log(
            "step_checkpoint "
            f"epoch={current_epoch} batch={batch_in_epoch} "
            f"global_step={step} "
            f"seconds={time.perf_counter() - started:.2f}"
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
    ) -> bool:
        nonlocal development_stop_requested, development_stop_summary
        if (
            development_probe_interval <= 0
            or step % development_probe_interval != 0
        ):
            return False
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
                spatial_supervised_step=spatial_step,
            )
        finally:
            model.train(was_training)
        probe_loss_cfg = scheduled_loss_config(
            cfg,
            epoch=current_epoch,
            global_step=step,
        )
        probe_scalars = _scalar_epoch_metrics(probe_metrics)
        probe_loss = probe_scalars["loss"]
        relative_improvement, development_stop_requested = (
            _update_development_early_stop_state(
                early_stop_state,
                step=step,
                loss=probe_loss,
                enabled=development_early_stop_enabled,
                minimum_step=development_early_stop_min_step,
                relative_delta=development_early_stop_relative_delta,
                patience=development_early_stop_patience,
            )
        )
        development_stop_summary = {
            "epoch": int(current_epoch),
            "global_step": int(step),
            "spatial_supervised_step": int(spatial_step),
            "early_stopped": development_stop_requested,
            **probe_scalars,
        }
        append_csv(
            output_dir / "development_metrics.csv",
            {
                "epoch": current_epoch,
                "global_step": step,
                "spatial_supervised_step": spatial_step,
                "probe_batches": development_probe_batches,
                "scheduled_semantic_weight": float(
                    probe_loss_cfg["semantic_weight"]
                ),
                "scheduled_classification_weight": float(
                    probe_loss_cfg["classification_weight"]
                ),
                "scheduled_spatial_weight": float(
                    probe_loss_cfg["spatial_weight"]
                ),
                "relative_loss_improvement": relative_improvement,
                "early_stop_consecutive_low_gain": int(
                    early_stop_state.get("consecutive_low_gain", 0)
                ),
                "early_stop_triggered": development_stop_requested,
                **probe_scalars,
            },
        )
        if development_stop_requested:
            _log(
                "development_early_stop "
                f"global_step={step} "
                f"relative_delta={development_early_stop_relative_delta} "
                f"patience={development_early_stop_patience}"
            )
        return development_stop_requested

    if expert_selection_enabled:
        if (
            not selection_baseline
            and resume_state is not None
            and global_step > 0
        ):
            raise ValueError(
                "cannot reconstruct the shared-initialization selection "
                "baseline from a progressed checkpoint"
            )
        if selection_baseline:
            _normalized_selection_metrics(
                selection_baseline,
                selection_baseline,
                selection_metric_weights,
            )
            if configured_selection_baseline is not None:
                _assert_selection_baseline_matches(
                    selection_baseline,
                    configured_selection_baseline,
                )
        else:
            was_training = bool(model.training)
            try:
                if prototype_refresh_state is not None:
                    _maybe_refresh_prototypes(
                        model=model,
                        cfg=cfg,
                        device=device,
                        state=prototype_refresh_state,
                        global_step=global_step,
                    )
                initial_validation = _run_validation_streams(
                    model=model,
                    val_loader=val_loader,
                    expert_classification_val_loader=(
                        expert_classification_val_loader
                    ),
                    expert_spatial_val_loader=expert_spatial_val_loader,
                    prototypes=prototypes,
                    optimizer=optimizer,
                    device=device,
                    cfg=cfg,
                    expert_eval_cfg=expert_eval_cfg,
                    epoch=0,
                    global_step=global_step,
                    spatial_supervised_step=spatial_supervised_step,
                )
                observed_selection_baseline, initial_teacher_metrics = (
                    _selection_raw_metrics(initial_validation, cfg)
                )
                if configured_selection_baseline is not None:
                    _assert_selection_baseline_matches(
                        observed_selection_baseline,
                        configured_selection_baseline,
                    )
                    selection_baseline = dict(
                        configured_selection_baseline
                    )
                else:
                    selection_baseline = observed_selection_baseline
                _normalized_selection_metrics(
                    selection_baseline,
                    selection_baseline,
                    selection_metric_weights,
                )
            finally:
                model.train(was_training)
            write_json(
                output_dir / "selection_baseline.json",
                {
                    "global_step": global_step,
                    "metrics": selection_baseline,
                    "observed_metrics": observed_selection_baseline,
                    "weights": selection_metric_weights,
                    "teacher_diagnostics": initial_teacher_metrics,
                },
            )
            _log(
                "selection_baseline "
                + " ".join(
                    f"{name}={selection_baseline[name]:.6f}"
                    for name in SELECTION_COMPONENTS
                )
            )

    try:
        for epoch in range(start_epoch, expected_epochs + 1):
            _set_loader_epoch(train_loader, epoch - 1)
            epoch_accumulator = None
            if epoch == start_epoch and resume_batch_in_epoch > 0:
                _set_loader_batch_cursor(
                    train_loader,
                    resume_batch_in_epoch,
                )
                epoch_accumulator = (resume_state or {}).get(
                    "epoch_accumulator"
                )
                if epoch_accumulator is None:
                    raise ValueError(
                        "mid-epoch checkpoint has no epoch accumulator"
                    )
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
                spatial_supervised_step=spatial_supervised_step,
                summary_writer=writer,
                prototype_refresh_state=prototype_refresh_state,
                step_metrics_writer=step_metrics_writer,
                development_probe=run_development_probe,
                step_checkpoint=save_step_checkpoint,
                resume_epoch_accumulator=epoch_accumulator,
            )
            resume_batch_in_epoch = 0
            global_step = int(train_metrics["global_step_end"])
            spatial_supervised_step = int(
                train_metrics["spatial_supervised_step_end"]
            )
            if development_stop_requested:
                last_metrics = dict(development_stop_summary)
                _atomic_torch_save(
                    checkpoint_payload(
                        epoch=epoch,
                        batch_in_epoch=int(
                            train_metrics["batch_in_epoch_end"]
                        ),
                        epoch_accumulator=dict(
                            train_metrics["epoch_accumulator_end"]
                        ),
                        training_complete=True,
                    ),
                    checkpoints / "last.pt",
                )
                break
            validation = _run_validation_streams(
                model=model,
                val_loader=val_loader,
                expert_classification_val_loader=(
                    expert_classification_val_loader
                ),
                expert_spatial_val_loader=expert_spatial_val_loader,
                prototypes=prototypes,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                expert_eval_cfg=expert_eval_cfg,
                epoch=epoch,
                global_step=global_step,
                spatial_supervised_step=spatial_supervised_step,
                summary_writer=writer,
            )
            val_metrics = validation["population"]
            embedding_metrics = validation["embedding"]
            population_classification_metrics = validation[
                "population_classification"
            ]
            expert_classification_metrics = validation[
                "expert_classification"
            ]
            expert_spatial_metrics = validation["expert_spatial"]
            teacher_metrics = _teacher_retention_metrics(
                embedding_metrics,
                relation_weight=float(
                    cfg["loss"].get("relation_weight", 0.0)
                ),
            )
            teacher_alignment = teacher_metrics[
                "teacher_alignment_score"
            ]
            selection_loss = float("nan")
            selection_metrics: dict[str, float] = {}
            if expert_selection_enabled:
                raw_selection_metrics, teacher_metrics = (
                    _selection_raw_metrics(validation, cfg)
                )
                selection_metrics = _normalized_selection_metrics(
                    raw_selection_metrics,
                    selection_baseline,
                    selection_metric_weights,
                )
                selection_loss = selection_metrics["selection_loss"]
            loss_cfg = scheduled_loss_config(
                cfg,
                epoch=epoch,
                global_step=global_step,
            )
            classification_metrics = (
                expert_classification_metrics
                if expert_classification_metrics
                else population_classification_metrics
            )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "spatial_supervised_step": spatial_supervised_step,
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
                "scheduled_classification_weight": float(loss_cfg["classification_weight"]),
                "scheduled_spatial_weight": float(loss_cfg["spatial_weight"]),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "teacher_alignment_score": (
                    0.0 if not math.isfinite(teacher_alignment) else teacher_alignment
                ),
                **teacher_metrics,
                **selection_metrics,
                "selection_loss": selection_loss,
                "selection_start_step": (
                    selection_early_stop_start_step
                ),
                "selection_eligible": (
                    global_step >= selection_early_stop_start_step
                ),
                **embedding_metrics,
                **classification_metrics,
                **{
                    f"population_{key}": value
                    for key, value in (
                        population_classification_metrics.items()
                    )
                },
                **{
                    f"expert_val_{key}": value
                    for key, value in (
                        expert_classification_metrics.items()
                    )
                },
                **{
                    f"expert_val_{key}": value
                    for key, value in expert_spatial_metrics.items()
                    if key.startswith("spatial")
                },
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
            selection_eligible = bool(
                expert_selection_enabled
                and global_step >= selection_early_stop_start_step
            )
            improved_selection = bool(
                selection_eligible
                and selection_loss < best_selection_loss
            )
            significant_selection_improvement = bool(
                selection_eligible
                and (
                    not math.isfinite(
                        best_significant_selection_loss
                    )
                    or selection_loss
                    < best_significant_selection_loss
                    * (1.0 - selection_early_stop_relative_delta)
                )
            )
            improved_alignment = teacher_alignment > best_teacher_alignment
            classification_selection_metric = float(
                classification_metrics.get(
                    "classification_macro_f1",
                    classification_metrics["classification_accuracy"],
                )
            )
            improved_classification = (
                classification_metrics["classification_evaluated_tiles"] > 0
                and classification_selection_metric
                > best_classification_accuracy
            )
            if improved_loss:
                best_loss = val_metrics["loss"]
                if not expert_selection_enabled:
                    best_metrics = row
            if improved_selection:
                best_selection_loss = selection_loss
                best_selection_epoch = epoch
                best_metrics = row
            if selection_eligible:
                selection_eligible_epochs += 1
                if significant_selection_improvement:
                    best_significant_selection_loss = selection_loss
                    selection_bad_epochs = 0
                else:
                    selection_bad_epochs += 1
            if improved_alignment:
                best_teacher_alignment = teacher_alignment
            if improved_classification:
                best_classification_accuracy = (
                    classification_selection_metric
                )

            selection_stop_requested = _selection_early_stop_requested(
                enabled=selection_early_stop_enabled,
                eligible=selection_eligible,
                eligible_epochs=selection_eligible_epochs,
                minimum_eligible_epochs=(
                    selection_minimum_eligible_epochs
                ),
                bad_epochs=selection_bad_epochs,
                patience=selection_early_stop_patience,
            )
            archive_trajectory_snapshot(
                current_epoch=epoch,
                step=global_step,
                spatial_step=spatial_supervised_step,
                metrics=row,
            )
            checkpoint = checkpoint_payload(
                epoch=epoch,
                batch_in_epoch=0,
                epoch_accumulator=None,
                training_complete=(
                    epoch >= expected_epochs
                    or selection_stop_requested
                ),
            )
            _atomic_torch_save(checkpoint, checkpoints / "last.pt")
            if improved_selection or (
                improved_loss and not expert_selection_enabled
            ):
                _atomic_torch_save(checkpoint, checkpoints / "best.pt")
            if improved_loss:
                _atomic_torch_save(
                    checkpoint,
                    checkpoints / "best_population_loss.pt",
                )
            if improved_alignment:
                _atomic_torch_save(
                    checkpoint,
                    checkpoints / "best_teacher_alignment.pt",
                )
            if improved_classification:
                _atomic_torch_save(
                    checkpoint,
                    checkpoints / "best_classification_accuracy.pt",
                )
            if selection_stop_requested:
                _log(
                    "selection_early_stop "
                    f"epoch={epoch} "
                    f"global_step={global_step} "
                    f"best_epoch={best_selection_epoch} "
                    f"best_loss={best_selection_loss:.6f} "
                    f"patience={selection_early_stop_patience}"
                )
                break
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
    if expert_selection_enabled and not math.isfinite(best_selection_loss):
        raise RuntimeError(
            "training ended before the joint selection metric became "
            "eligible: "
            f"global_step={global_step} "
            f"selection_start_step={selection_early_stop_start_step}"
        )
    if expert_selection_enabled:
        best_checkpoint_path = checkpoints / "best.pt"
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            int(best_checkpoint.get("epoch", -1))
            != best_selection_epoch
            or not math.isclose(
                float(
                    best_checkpoint.get(
                        "best_selection_loss",
                        float("nan"),
                    )
                ),
                best_selection_loss,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                "best selection checkpoint disagrees with the completed run"
            )
        best_checkpoint.update(
            {
                "run_complete": True,
                "selection_finalized": True,
                "run_terminal_epoch": int(
                    last_metrics.get("epoch", best_selection_epoch)
                ),
                "run_terminal_global_step": int(global_step),
                "selection_stop_triggered": bool(
                    selection_stop_requested
                ),
            }
        )
        _atomic_torch_save(best_checkpoint, best_checkpoint_path)
    spatial_route = bool(cfg["data"].get("spatial_manifest_path"))
    summary_metrics = (
        best_metrics
        if expert_selection_enabled
        else (last_metrics if spatial_route else best_metrics)
    )
    if not summary_metrics:
        raise RuntimeError("training produced no summary metrics")
    write_json(output_dir / "summary.json", summary_metrics)
    return summary_metrics
