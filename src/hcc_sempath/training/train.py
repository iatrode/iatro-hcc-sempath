from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from . import _pipeline_probe as _probe

from iatro.iac import read_tables
from iatro.iac.adapters.tiles import read_package_metadata
from ..modeling.prototypes import PrototypeRegistry, load_prototype_registry
from ..modeling.models import (
    HCCSemPathModel,
    STUDENT_BACKBONE_NAME,
    STUDENT_IMAGE_SIZE,
    STUDENT_PATCH_SIZE,
    SPATIAL_OUTPUT_STRIDE,
    SPATIAL_PATCH_PADDING,
    STUDENT_PRETRAINED_PATH,
    STUDENT_PRETRAINED_SHA256,
)
from .config import (
    embedding_dim,
    image_tile_package_paths,
    load_config,
    manifest_data_paths,
    teacher_dims,
    teacher_feature_package_paths,
    teacher_names,
    validate_training_config,
)
from .datasets import (
    DistillationTileDataset,
    PackageSampledDistillationDataset,
    apply_split_overrides,
    collate_distillation,
    read_packaged_tile_records,
    validate_teacher_cache,
)
from .engine import _scheduler_contract, build_lr_scheduler, fit
from .manifest import load_training_manifest
from .prototype_labels import DEFAULT_CLASSIFICATION_CLASSES, load_prototype_labels
from .roi import (
    SpatialRoiTarget,
    build_spatial_roi_targets,
    empty_spatial_roi_target,
    load_spatial_tile_locations,
    spatial_component_names,
)
from .utils import seed_everything


@dataclass
class BatchSlot:
    """Destination of one decoded row: which ring buffer and which row position."""
    buffer_idx: int
    pos: int


@dataclass
class BatchBuffer:
    """One pre-allocated, pinned set of tensors for a single batch. Workers
    scatter decoded rows directly into these — there is no np.stack, no
    post-hoc pin_memory, and no per-batch collate. When the last slot is
    filled the buffer IS the finished batch."""
    images: torch.Tensor                       # (B, H, W, 3) uint8, pinned
    teacher_features: dict                       # name -> (B, dim) float32, pinned
    prototype_mask: torch.Tensor                 # (B,) bool
    prototype_classification: torch.Tensor               # (B,) long
    spatial_targets: list[SpatialRoiTarget | None]
    spatial_shape: tuple[int, int, int]
    tile_id: list = field(default_factory=list)  # (B,) str

    def view(self, count: int) -> dict:
        """Expose the filled prefix as the dict the engine expects."""
        selected = self.spatial_targets[:count]
        component_count, grid_h, grid_w = self.spatial_shape
        if any(item is not None for item in selected):
            empty = empty_spatial_roi_target(
                component_count,
                (grid_h, grid_w),
            )
            spatial = [item if item is not None else empty for item in selected]
            spatial_payload = {
                "spatial_point_centers": torch.stack(
                    [item.point_centers for item in spatial]
                ),
                "spatial_instance_exclusion_support": torch.stack(
                    [
                        item.instance_exclusion_support
                        if item.instance_exclusion_support is not None
                        else torch.zeros_like(
                            item.area_positive,
                            dtype=torch.bool,
                        )
                        for item in spatial
                    ]
                ),
                "spatial_brush_bag_ids": torch.stack(
                    [item.brush_bag_ids for item in spatial]
                ),
                "spatial_area_positive": torch.stack(
                    [item.area_positive for item in spatial]
                ),
                "spatial_explicit_negative": torch.stack(
                    [item.explicit_negative for item in spatial]
                ),
                "spatial_implicit_negative": torch.stack(
                    [item.implicit_negative for item in spatial]
                ),
                "spatial_supervised": torch.stack(
                    [item.supervised for item in spatial]
                ),
            }
        else:
            spatial_payload = {
                "spatial_point_centers": torch.zeros((count, 0, 0, 0)),
                "spatial_instance_exclusion_support": torch.zeros(
                    (count, 0, 0, 0),
                    dtype=torch.bool,
                ),
                "spatial_brush_bag_ids": torch.zeros(
                    (count, 0, 0, 0),
                    dtype=torch.long,
                ),
                "spatial_area_positive": torch.zeros(
                    (count, 0, 0, 0),
                    dtype=torch.bool,
                ),
                "spatial_explicit_negative": torch.zeros(
                    (count, 0, 0, 0),
                    dtype=torch.bool,
                ),
                "spatial_implicit_negative": torch.zeros(
                    (count, 0, 0, 0),
                    dtype=torch.bool,
                ),
                "spatial_supervised": torch.zeros(
                    (count, 0),
                    dtype=torch.bool,
                ),
            }
        return {
            "tile_id": list(self.tile_id[:count]),
            "images": self.images[:count],
            "images_uint8": True,
            "images_hwc": True,
            "teacher_features": {name: feat[:count] for name, feat in self.teacher_features.items()},
            "prototype_mask": self.prototype_mask[:count],
            "prototype_classification": self.prototype_classification[:count],
            **spatial_payload,
        }


def _alloc_batch_buffer(batch_size: int, spec: dict, pin: bool) -> BatchBuffer:
    h, w = spec["image_hw"]
    kw = {"pin_memory": True} if pin else {}
    spatial_k, spatial_h, spatial_w = spec.get("spatial_shape", (0, 0, 0))
    return BatchBuffer(
        images=torch.empty((batch_size, h, w, 3), dtype=torch.uint8, **kw),
        teacher_features={
            name: torch.empty((batch_size, dim), dtype=torch.float32, **kw)
            for name, dim in spec["teacher_dims"].items()
        },
        prototype_mask=torch.zeros((batch_size,), dtype=torch.bool),
        prototype_classification=torch.full((batch_size,), -1, dtype=torch.long),
        spatial_targets=[None] * batch_size,
        spatial_shape=(spatial_k, spatial_h, spatial_w),
        tile_id=[""] * batch_size,
    )



class _ChunkPlanBatchSampler:
    """Yields batches of GLOBAL dataset indices reproducing the seed-ordered
    package-chunk plan, for use as a multiprocessing DataLoader batch_sampler.

    The thread loader's tile sequence = chunks concatenated in plan order, then
    sliced into batch_size groups. We reproduce exactly that here so the
    multiprocessing path is numerically equivalent to the thread path. A fresh
    seed per epoch (seed + epoch) matches reshuffle_each_epoch semantics.
    """

    def __init__(self, dataset, batch_size: int, chunk_size: int, seed: int,
                 reshuffle_each_epoch: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._start_batch = 0

    def __iter__(self):
        epoch_seed = self.seed + self._epoch if self.reshuffle_each_epoch else self.seed
        if self.reshuffle_each_epoch:
            self._epoch += 1
        start_batch = self._start_batch
        self._start_batch = 0
        chunks = self.dataset.iter_global_index_chunks(self.chunk_size, epoch_seed)
        if not chunks:
            return
        flat = np.concatenate(chunks)
        if start_batch:
            flat = flat[start_batch * self.batch_size :]
        n = len(flat)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            yield flat[start:end].tolist()

    def __len__(self) -> int:
        n = int(self.dataset.sample_count)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._start_batch = 0

    def set_batch_cursor(self, batch: int) -> None:
        self._start_batch = max(0, int(batch))


class _PackageShuffleBatchLoader:
    """Deterministic parallel-decode batch loader.

    Design (see docs in code review):
      * A persistent pool of worker threads is created ONCE and reused across
        every epoch — this avoids the per-epoch ThreadPoolExecutor leak that
        previously accumulated hundreds of live threads.
      * The chunk plan (``dataset.iter_package_row_chunks``) is seed-ordered and
        materialized into an indexed task list. Workers PULL the next task index
        atomically, decode it, and write the decoded rows back into the slot at
        that index. "Which chunk a worker takes" does not matter for ordering
        because the decoded rows are placed back at the chunk's fixed position.
      * The consumer delivers rows in chunk-index order (``_consume_cursor``),
        so the per-batch tile_id sequence is identical to a single-threaded run
        regardless of thread scheduling — fully reproducible.
      * Workers never stop for the consumer: they keep pulling and decoding the
        next chunk until the number of decoded-but-unconsumed rows reaches
        ``max_outstanding_rows`` (back-pressure). A slow chunk only blocks the
        single ``get`` waiting on that exact slot (max stall = one chunk decode);
        other workers keep filling later slots until the back-pressure ceiling.

    Reproducibility is also preserved because shuffling is entirely a property
    of the seed-ordered chunk plan, not of decode/arrival order.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        prefetch_batches: int,
        collate_fn,
        seed: int = 13,
        chunk_size: int | None = None,
        buffer_batches: int = 1,
        reshuffle_each_epoch: bool = True,
        pin_memory: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = max(0, int(num_workers))
        self.prefetch_batches = max(0, int(prefetch_batches))
        self.collate_fn = collate_fn
        self.seed = int(seed)
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self._epoch = 0
        self._start_batch = 0
        self.chunk_size = max(1, int(chunk_size or self.batch_size))
        # Back-pressure ceiling, measured in decoded-but-unconsumed rows.
        self.max_outstanding_rows = max(self.batch_size, self.batch_size * max(1, int(buffer_batches)))
        self.pin_memory = bool(pin_memory)
        # Live-iteration teardown handle. Each __iter__ registers its stop_event
        # and threads here so a NEW __iter__ (e.g. prefetch of the next epoch
        # before the previous generator is closed) and a weakref finalizer can
        # both stop orphaned worker threads — preventing the thread/file-handle
        # leak that occurs when a consumer breaks out of iteration without
        # closing the generator.
        self._active: dict | None = None

    def _maybe_pin(self, batch: dict) -> dict:
        if self.pin_memory and torch.cuda.is_available():
            batch["images"] = batch["images"].pin_memory()
            batch["teacher_features"] = {
                name: feat.pin_memory()
                for name, feat in batch["teacher_features"].items()
            }
        return batch

    def set_epoch(self, epoch: int) -> None:
        self._stop_active()
        self._epoch = int(epoch)
        self._start_batch = 0

    def set_batch_cursor(self, batch: int) -> None:
        self._stop_active()
        self._start_batch = max(0, int(batch))

    def _draw_batch(self, buffer: list[dict], rng: np.random.Generator) -> list[dict]:
        take = min(self.batch_size, len(buffer))
        if bool(getattr(self.dataset, "sequential_iac_rows", False)):
            batch = buffer[:take]
            del buffer[:take]
            return batch
        chosen = rng.choice(len(buffer), size=take, replace=False)
        chosen_set = set(chosen)
        batch = [buffer[index] for index in chosen]
        buffer[:] = [item for index, item in enumerate(buffer) if index not in chosen_set]
        return batch

    @staticmethod
    def _teardown(active: dict) -> None:
        """Signal stop and join all threads of one iteration. Idempotent."""
        if active.get("done"):
            return
        active["done"] = True
        stop_event = active.get("stop_event")
        if stop_event is not None:
            stop_event.set()
        cond_lock = active.get("lock")
        conditions = active.get("conditions", [])
        if cond_lock is not None:
            with cond_lock:
                for cond in conditions:
                    cond.notify_all()
        finished = active.get("finished")
        if finished is not None:
            # Unblock a producer that is stuck on a full output queue.
            while True:
                try:
                    finished.get_nowait()
                except Empty:
                    break
        for thread in active.get("threads", []):
            if thread.is_alive():
                thread.join(timeout=5.0)

    def _stop_active(self) -> None:
        active = self._active
        if active is not None:
            self._active = None
            self._teardown(active)

    def _build_batch_plan(self, tasks: list, epoch_seed: int) -> tuple[list, list, list]:
        """Map the seed-ordered chunk plan onto (batch, position) destinations.

        Returns:
          batch_sizes: rows in each batch (last may be short)
          task_targets: task_targets[t] = list[BatchSlot] aligned to tasks[t].rows
          (positions are decided here, deterministically, BEFORE any decoding —
           so batch contents do not depend on decode/arrival order)
        """
        # Flatten chunk plan into a global row order, recording (task, k).
        global_seq: list[tuple[int, int]] = []
        for t, (_pkg, rows) in enumerate(tasks):
            for k in range(len(rows)):
                global_seq.append((t, k))
        total = len(global_seq)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        batch_sizes = [
            min(self.batch_size, total - b * self.batch_size) for b in range(num_batches)
        ]
        task_targets: list[list] = [[None] * len(rows) for _pkg, rows in tasks]
        for g, (t, k) in enumerate(global_seq):
            b = g // self.batch_size
            within = g % self.batch_size
            # Deterministic intra-batch shuffle keyed by (epoch_seed, batch).
            task_targets[t][k] = (b, within)
        # Apply per-batch permutation so positions are decorrelated but reproducible.
        for b in range(num_batches):
            perm = np.random.default_rng(epoch_seed + b).permutation(batch_sizes[b])
            self._batch_perms[b] = perm
        return batch_sizes, task_targets, []

    def __iter__(self):
        import threading

        epoch_seed = self.seed + self._epoch if self.reshuffle_each_epoch else self.seed
        if self.reshuffle_each_epoch:
            self._epoch += 1
        start_batch = self._start_batch
        self._start_batch = 0
        _probe.set_config(
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            prefetch_batches=self.prefetch_batches,
            n_buffers=max(2, self.prefetch_batches + 1),
            pin_memory=self.pin_memory,
            torch_threads=torch.get_num_threads(),
            torch_interop=torch.get_num_interop_threads(),
        )
        _probe.start_proc_sampler()

        tasks: list[tuple[int, np.ndarray]] = list(
            self.dataset.iter_package_row_chunks(self.chunk_size, epoch_seed)
        )
        skip_rows = start_batch * self.batch_size
        if skip_rows:
            remaining: list[tuple[int, np.ndarray]] = []
            for package_idx, rows in tasks:
                if skip_rows >= len(rows):
                    skip_rows -= len(rows)
                    continue
                if skip_rows:
                    rows = rows[skip_rows:]
                    skip_rows = 0
                remaining.append((package_idx, rows))
            tasks = remaining
        num_tasks = len(tasks)
        if num_tasks == 0:
            return

        if self.num_workers == 0:
            rng = np.random.default_rng(epoch_seed)
            sample_buffer: list[dict] = []
            for package_idx, rows in tasks:
                sample_buffer.extend(self.dataset.read_package_rows(package_idx, rows))
                while len(sample_buffer) >= self.batch_size:
                    yield self._maybe_pin(self.collate_fn(self._draw_batch(sample_buffer, rng)))
            if sample_buffer:
                yield self._maybe_pin(self.collate_fn(self._draw_batch(sample_buffer, rng)))
            return

        # If a previous iteration's threads are still alive (e.g. a consumer
        # broke out without closing the generator, or a prefetch created a new
        # iterator), stop them before starting fresh ones. Without this the
        # orphaned workers accumulate every epoch and exhaust threads/FDs.
        self._stop_active()

        spec = self.dataset.batch_buffer_spec()
        self._batch_perms: dict[int, np.ndarray] = {}
        batch_sizes, task_targets, _ = self._build_batch_plan(tasks, epoch_seed)
        num_batches = len(batch_sizes)

        # Apply the per-batch permutation to the (b, within) targets so the final
        # slot position is the shuffled one. Build BatchSlot lists per task.
        n_buffers = max(2, self.prefetch_batches + 1)
        positions_per_task: list[list] = []
        for t in range(num_tasks):
            slots_for_task = []
            for (b, within) in task_targets[t]:
                pos = int(self._batch_perms[b][within])
                slots_for_task.append(BatchSlot(buffer_idx=b % n_buffers, pos=pos))
            positions_per_task.append(slots_for_task)

        # Pre-allocate the ring of pinned buffers.
        buffers = [_alloc_batch_buffer(self.batch_size, spec, self.pin_memory) for _ in range(n_buffers)]

        # Per-batch fill counters and the buffer-generation guard. buffer_gen[i]
        # holds the batch index that currently OWNS ring slot i; a worker that
        # needs batch b must wait until buffer (b % n_buffers) is free for b
        # (i.e. the previous owner b - n_buffers has been consumed).
        fill_count = [0] * num_batches
        buffer_owner = [b for b in range(n_buffers)]  # initial owners: batches 0..n_buffers-1
        next_task_idx = [0]
        error_box: list[BaseException | None] = [None]
        stop_event = Event()
        lock = threading.Lock()
        batch_done = threading.Condition(lock)   # a batch reached full fill_count
        buffer_freed = threading.Condition(lock)  # a ring buffer was released by consumer

        finished: Queue = Queue(maxsize=max(1, self.prefetch_batches))
        sentinel = object()
        ready_batches: list[bool] = [False] * num_batches

        def worker() -> None:
            try:
                while not stop_event.is_set():
                    with _probe.section("w_get_task"):
                        with lock:
                            idx = next_task_idx[0]
                            if idx >= num_tasks:
                                _probe.flush_thread()
                                return
                            next_task_idx[0] = idx + 1
                    package_idx, rows = tasks[idx]
                    slots = positions_per_task[idx]
                    targets = task_targets[idx]
                    # Group this task's rows by destination batch, then process
                    # groups in batch order. Each group waits ONLY for its own
                    # ring buffer — so the front-batch portion is written even if
                    # a later batch's buffer is not yet free (avoids deadlock when
                    # a chunk straddles a batch boundary).
                    groups: dict[int, list[int]] = {}
                    for k, (b, _w) in enumerate(targets):
                        groups.setdefault(b, []).append(k)
                    for b in sorted(groups):
                        ring = b % n_buffers
                        with _probe.section("w_wait_ring"):
                            with lock:
                                while (not stop_event.is_set()) and buffer_owner[ring] != b:
                                    buffer_freed.wait()
                                if stop_event.is_set():
                                    return
                        ks = groups[b]
                        sub_rows = np.asarray([int(rows[k]) for k in ks], dtype=np.int64)
                        sub_slots = [slots[k] for k in ks]
                        with _probe.section("w_scatter"):
                            self.dataset.scatter_package_rows(package_idx, sub_rows, sub_slots, buffers)
                        with _probe.section("w_submit"):
                            with lock:
                                fill_count[b] += len(ks)
                                if fill_count[b] >= batch_sizes[b]:
                                    ready_batches[b] = True
                                    batch_done.notify_all()
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    if error_box[0] is None:
                        error_box[0] = exc
                    stop_event.set()
                    batch_done.notify_all()
                    buffer_freed.notify_all()

        workers = [Thread(target=worker, daemon=True) for _ in range(self.num_workers)]
        for thread in workers:
            thread.start()

        # Emitter: deliver batches in order as each becomes ready.
        def emitter() -> None:
            try:
                for b in range(num_batches):
                    with lock:
                        while (not ready_batches[b]) and error_box[0] is None and not stop_event.is_set():
                            batch_done.wait()
                        if error_box[0] is not None:
                            raise error_box[0]
                        if stop_event.is_set():
                            return
                    view = buffers[b % n_buffers].view(batch_sizes[b])
                    while not stop_event.is_set():
                        try:
                            finished.put((b, view), timeout=0.1)
                            break
                        except Full:
                            continue
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    if error_box[0] is None:
                        error_box[0] = exc
                try:
                    finished.put(exc, timeout=0.1)
                except Full:
                    pass
            finally:
                while not stop_event.is_set():
                    try:
                        finished.put(sentinel, timeout=0.1)
                        break
                    except Full:
                        continue

        emitter_thread = Thread(target=emitter, daemon=True)
        emitter_thread.start()

        # Register this iteration so a later __iter__ or a GC finalizer can stop
        # these threads even if the consumer never closes the generator.
        active = {
            "stop_event": stop_event,
            "lock": lock,
            "conditions": [batch_done, buffer_freed],
            "finished": finished,
            "threads": [*workers, emitter_thread],
            "done": False,
        }
        self._active = active
        finalizer = weakref.finalize(self, self._teardown, active)

        try:
            _emitted = 0
            while True:
                with _probe.section("c_wait_batch"):
                    item = finished.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                b, view = item
                with _probe.section("c_consume"):
                    yield view
                _emitted += 1
                if _probe.ON and (_emitted % 20 == 0):
                    _probe.flush_thread()
                    _probe.report(tag=f"epoch={self._epoch} batches={_emitted} workers={self.num_workers}")
                # Consumer is done with batch b: release its ring buffer to the
                # next owner (b + n_buffers) so a waiting worker can reuse it.
                with lock:
                    ring = b % n_buffers
                    buffer_owner[ring] = b + n_buffers
                    fill_count_next = b + n_buffers
                    if fill_count_next < num_batches:
                        fill_count[fill_count_next] = 0
                        ready_batches[fill_count_next] = False
                    buffer_freed.notify_all()
        finally:
            self._teardown(active)
            finalizer.detach()
            if self._active is active:
                self._active = None

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class _MaterializedExpertBank:
    """One in-memory copy of the small fixed classification/spatial expert union."""

    def __init__(self, batches: list[dict]) -> None:
        if not batches:
            raise ValueError("expert bank materialization produced no batches")
        self.tile_id = [
            tile_id
            for batch in batches
            for tile_id in batch["tile_id"]
        ]
        self.images = torch.cat(
            [batch["images"] for batch in batches],
            dim=0,
        )
        self.images_uint8 = bool(batches[0].get("images_uint8", False))
        self.images_hwc = bool(batches[0].get("images_hwc", False))
        tensor_keys = (
            "prototype_mask",
            "prototype_classification",
            "spatial_point_centers",
            "spatial_instance_exclusion_support",
            "spatial_brush_bag_ids",
            "spatial_area_positive",
            "spatial_explicit_negative",
            "spatial_implicit_negative",
            "spatial_supervised",
        )
        self.tensors = {
            key: torch.cat(
                [
                    (
                        batch[key]
                        if key in batch
                        else torch.zeros_like(
                            batch["spatial_area_positive"],
                            dtype=torch.bool,
                        )
                    )
                    for batch in batches
                ],
                dim=0,
            )
            for key in tensor_keys
        }
        teacher_names = list(batches[0]["teacher_features"])
        self.teacher_features = {
            name: torch.cat(
                [
                    batch["teacher_features"][name]
                    for batch in batches
                ],
                dim=0,
            )
            for name in teacher_names
        }
        if len(set(self.tile_id)) != len(self.tile_id):
            raise ValueError("materialized expert bank contains duplicate tiles")
        self.index_by_tile_id = {
            tile_id: index
            for index, tile_id in enumerate(self.tile_id)
        }

    def __len__(self) -> int:
        return len(self.tile_id)

    @property
    def memory_bytes(self) -> int:
        tensors = [
            self.images,
            *self.tensors.values(),
            *self.teacher_features.values(),
        ]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def batch(self, indices: torch.Tensor) -> dict:
        rows = [int(index) for index in indices.tolist()]
        payload = {
            "tile_id": [self.tile_id[index] for index in rows],
            "images": self.images.index_select(0, indices),
            "images_uint8": self.images_uint8,
            "teacher_features": {
                name: values.index_select(0, indices)
                for name, values in self.teacher_features.items()
            },
            **{
                key: values.index_select(0, indices)
                for key, values in self.tensors.items()
            },
        }
        if self.images_hwc:
            payload["images_hwc"] = True
        return payload


class _InMemoryExpertBatchLoader:
    """Deterministic shuffled views over one materialized expert bank."""

    def __init__(
        self,
        bank: _MaterializedExpertBank,
        *,
        indices: list[int] | None,
        batch_size: int,
        seed: int,
    ) -> None:
        self.bank = bank
        self.indices = np.asarray(
            list(range(len(bank))) if indices is None else indices,
            dtype=np.int64,
        )
        if self.indices.size == 0:
            raise ValueError("expert loader requires at least one bank row")
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self._epoch = 0
        self._cycle = 0
        self._start_batch = 0

    def __len__(self) -> int:
        return (
            len(self.indices) + self.batch_size - 1
        ) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self._epoch * 1_000_003 + self._cycle
        )
        self._cycle += 1
        start_batch = self._start_batch
        self._start_batch = 0
        order = rng.permutation(self.indices)
        for start in range(
            start_batch * self.batch_size,
            len(order),
            self.batch_size,
        ):
            selected = torch.from_numpy(
                order[start : start + self.batch_size].copy()
            ).long()
            yield self.bank.batch(selected)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._cycle = 0
        self._start_batch = 0

    def set_batch_cursor(self, batch: int) -> None:
        batch = max(0, int(batch))
        batches_per_cycle = len(self)
        self._cycle = batch // batches_per_cycle
        self._start_batch = batch % batches_per_cycle


def _global_indices_for_package_rows(
    dataset: PackageSampledDistillationDataset,
    rows_by_package: dict[int, np.ndarray],
) -> list[int]:
    offsets = {
        package_idx: int(dataset.block_offsets[position])
        for position, package_idx in enumerate(dataset.package_order)
    }
    indices: list[int] = []
    for package_idx, rows in sorted(rows_by_package.items()):
        sampled_rows = dataset.package_sample_rows[package_idx]
        if sampled_rows is not None:
            raise ValueError(
                "expert bank requires the complete package row space"
            )
        base = offsets[package_idx]
        indices.extend(base + int(row) for row in rows)
    return indices


def _materialize_expert_bank(
    dataset: PackageSampledDistillationDataset,
    rows_by_package: dict[int, np.ndarray],
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> _MaterializedExpertBank:
    indices = _global_indices_for_package_rows(
        dataset,
        rows_by_package,
    )
    loader_kwargs = {
        "batch_size": max(1, int(batch_size)),
        "num_workers": max(0, int(num_workers)),
        "shuffle": False,
        "collate_fn": dataset.collate,
        "pin_memory": False,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        loader_kwargs["persistent_workers"] = False
    bank = _MaterializedExpertBank(
        list(DataLoader(Subset(dataset, indices), **loader_kwargs))
    )
    print(
        "expert_bank_materialized "
        f"tiles={len(bank)} memory_mib={bank.memory_bytes / (1024**2):.1f}",
        flush=True,
    )
    return bank


class _InterleavedBatchLoader:
    """Insert expert replay before each fixed population-batch interval."""

    def __init__(self, population_loader, expert_loader, *, interval: int) -> None:
        self.population_loader = population_loader
        self.expert_loader = expert_loader
        self.interval = int(interval)
        self._start_batch = 0
        if self.interval <= 0:
            raise ValueError(
                f"expert replay interval must be positive, got {interval}"
            )

    def __len__(self) -> int:
        population_batches = len(self.population_loader)
        expert_batches = (
            population_batches + self.interval - 1
        ) // self.interval
        return population_batches + expert_batches

    @staticmethod
    def _batch_cursor_setter(loader):
        setter = getattr(loader, "set_batch_cursor", None)
        if callable(setter):
            return setter
        batch_sampler = getattr(loader, "batch_sampler", None)
        setter = getattr(batch_sampler, "set_batch_cursor", None)
        return setter if callable(setter) else None

    def __iter__(self):
        start_batch = self._start_batch
        self._start_batch = 0
        population_consumed = 0
        expert_consumed = 0
        for population_index in range(len(self.population_loader)):
            if population_index % self.interval == 0:
                if start_batch == 0:
                    break
                start_batch -= 1
                expert_consumed += 1
            if start_batch == 0:
                break
            start_batch -= 1
            population_consumed += 1
        first_expert_already_consumed = (
            population_consumed % self.interval == 0
            and expert_consumed
            > self._output_batches_for_population(population_consumed)
            - population_consumed
        )
        population_cursor = self._batch_cursor_setter(
            self.population_loader
        )
        expert_cursor = self._batch_cursor_setter(self.expert_loader)
        if population_consumed and not callable(population_cursor):
            raise RuntimeError(
                "population loader cannot resume from a batch cursor"
            )
        if expert_consumed and not callable(expert_cursor):
            raise RuntimeError(
                "expert loader cannot resume from a batch cursor"
            )
        if callable(population_cursor):
            population_cursor(population_consumed)
        if callable(expert_cursor):
            expert_cursor(expert_consumed)
        population_iterator = iter(self.population_loader)
        expert_iterator = iter(self.expert_loader)
        try:
            # Creating a multiprocessing DataLoader iterator starts its decode
            # workers. Yield the already materialized expert batch before
            # blocking on the first population result, so those workers can
            # fill the prefetch queue while the GPU processes expert
            # supervision.
            for batch_index in range(
                population_consumed + 1,
                len(self.population_loader) + 1,
            ):
                if (
                    (batch_index - 1) % self.interval == 0
                    and not (
                        first_expert_already_consumed
                        and batch_index == population_consumed + 1
                    )
                ):
                    try:
                        expert_batch = next(expert_iterator)
                    except StopIteration:
                        expert_iterator = iter(self.expert_loader)
                        expert_batch = next(expert_iterator)
                    yield expert_batch
                try:
                    batch = next(population_iterator)
                except StopIteration:
                    break
                yield batch
        finally:
            for iterator in (population_iterator, expert_iterator):
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

    def set_epoch(self, epoch: int) -> None:
        self._start_batch = 0
        for loader in (self.population_loader, self.expert_loader):
            setter = getattr(loader, "set_epoch", None)
            if callable(setter):
                setter(int(epoch))
                continue
            for name in ("batch_sampler", "sampler"):
                candidate = getattr(loader, name, None)
                setter = getattr(candidate, "set_epoch", None)
                if callable(setter):
                    setter(int(epoch))
                    break

    def set_batch_cursor(self, batch: int) -> None:
        self._start_batch = max(0, int(batch))

    def _output_batches_for_population(self, population_batches: int) -> int:
        expert_batches = (
            population_batches + self.interval - 1
        ) // self.interval
        return population_batches + expert_batches


def _target_rows_by_package(
    package_paths: list[str],
    target_locations: dict[str, tuple[str, int]],
    *,
    require_all: bool = True,
) -> dict[int, np.ndarray]:
    """Resolve fixed expert rows from annotation provenance alone."""

    if not target_locations:
        return {}
    normalized_packages = [
        str(Path(path).resolve()).replace("\\", "/")
        for path in package_paths
    ]
    targets_by_package: dict[int, dict[str, int]] = {}
    missing: list[str] = []
    for tile_id, (declared_path, row) in target_locations.items():
        suffix = str(declared_path).replace("\\", "/").lstrip("./")
        matches = [
            index
            for index, package_path in enumerate(normalized_packages)
            if package_path == suffix
            or package_path.endswith(f"/{suffix}")
        ]
        if not matches:
            basename = Path(suffix).name
            matches = [
                index
                for index, package_path in enumerate(normalized_packages)
                if Path(package_path).name == basename
            ]
        if len(matches) > 1:
            raise ValueError(
                "expert package provenance is ambiguous: "
                f"tile={tile_id} package={declared_path}"
            )
        if not matches:
            missing.append(tile_id)
            continue
        if int(row) < 0:
            raise ValueError(f"negative expert row: tile={tile_id} row={row}")
        targets_by_package.setdefault(matches[0], {})[tile_id] = int(row)
    if missing and require_all:
        raise ValueError(
            "expert supervision references tiles outside the training split: "
            f"count={len(missing)} sample={', '.join(missing[:3])}"
        )
    rows: dict[int, np.ndarray] = {}
    for package_index, targets in targets_by_package.items():
        _, _, record_table = read_tables(package_paths[package_index])
        tile_ids = record_table.column("tile_id")
        resolved_rows: list[int] = []
        for tile_id, declared_row in targets.items():
            observed = (
                str(tile_ids[declared_row].as_py())
                if declared_row < len(tile_ids)
                else "<row-out-of-range>"
            )
            if observed != tile_id:
                raise ValueError(
                    "stale expert package/row provenance: "
                    f"package={package_paths[package_index]} "
                    f"row={declared_row} expected={tile_id} "
                    f"observed={observed}"
                )
            resolved_rows.append(declared_row)
        rows[package_index] = np.asarray(
            sorted(resolved_rows),
            dtype=np.int64,
        )
    return rows


def _expert_package_subset(
    tile_packages: list[str],
    teacher_packages: dict[str, list[str]],
    target_locations: dict[str, tuple[str, int]],
) -> tuple[
    list[str],
    dict[str, list[str]],
    dict[int, np.ndarray],
    set[str],
]:
    """Select complete packages/rows needed by one fixed expert split."""

    resolved_rows = _target_rows_by_package(
        tile_packages,
        target_locations,
    )
    selected_indices = sorted(resolved_rows)
    selected_tiles = [
        tile_packages[index]
        for index in selected_indices
    ]
    selected_teachers = {
        name: [
            paths[index]
            for index in selected_indices
        ]
        for name, paths in teacher_packages.items()
    }
    remapped_rows = {
        new_index: resolved_rows[old_index]
        for new_index, old_index in enumerate(selected_indices)
    }
    return (
        selected_tiles,
        selected_teachers,
        remapped_rows,
        set(selected_tiles),
    )


def _validation_package_keep_indices(
    validation_packages: list[str],
    optimizer_visible_packages: list[str],
) -> list[int]:
    """Keep evaluation packages outside the frozen optimizer-visible set."""

    visible_paths = {
        str(Path(path).resolve())
        for path in optimizer_visible_packages
    }
    return [
        index
        for index, package_path in enumerate(validation_packages)
        if str(Path(package_path).resolve()) not in visible_paths
    ]


def _assert_disjoint_package_paths(
    train_packages: list[str],
    validation_packages: list[str],
) -> None:
    """Fail if an exact package path occurs in both prepared splits."""

    train_paths = {str(Path(path).resolve()) for path in train_packages}
    validation_paths = {
        str(Path(path).resolve()) for path in validation_packages
    }
    overlap = train_paths & validation_paths
    if overlap:
        raise ValueError(
            "train/validation package overlap: "
            f"count={len(overlap)} sample={next(iter(sorted(overlap)))}"
        )


def _package_cohort_ids(
    package_paths: list[str],
) -> tuple[set[str], set[str]]:
    """Read patient/slide identities for offline calibration provenance."""

    patient_ids: set[str] = set()
    slide_ids: set[str] = set()
    for package_path in package_paths:
        _, slide_table, _ = read_tables(package_path)
        for field, destination in (
            ("patient_id", patient_ids),
            ("slide_id", slide_ids),
        ):
            if field in slide_table.column_names:
                destination.update(
                    str(value)
                    for value in slide_table.column(field).to_pylist()
                    if value not in (None, "")
                )
    return patient_ids, slide_ids


def _assert_disjoint_package_cohorts(
    train_packages: list[str],
    validation_packages: list[str],
) -> None:
    """Verify patient/slide separation for offline evaluation artifacts."""

    train_patients, train_slides = _package_cohort_ids(train_packages)
    validation_patients, validation_slides = _package_cohort_ids(
        validation_packages
    )
    patient_overlap = train_patients & validation_patients
    slide_overlap = train_slides & validation_slides
    if patient_overlap or slide_overlap:
        raise ValueError(
            "train/validation cohort leakage: "
            f"patients={len(patient_overlap)} slides={len(slide_overlap)}"
        )


def _paths_from_data(cfg: dict, key: str) -> list[str]:
    value = cfg["data"].get(key)
    if value is None:
        raise ValueError(f"data.{key} is required")
    if isinstance(value, dict):
        return [str(path) for path in value.values()]
    return [str(path) for path in value]


def _teacher_paths_from_data(cfg: dict, key: str) -> dict[str, list[str]]:
    value = cfg["data"].get(key)
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a teacher->paths mapping")
    result = {}
    for name, paths in value.items():
        if isinstance(paths, (list, tuple)):
            result[str(name)] = [str(path) for path in paths]
        else:
            result[str(name)] = [str(paths)]
    return result


def _active_teacher_paths(
    cfg: dict,
    paths: dict[str, list[str]],
    *,
    key: str,
) -> dict[str, list[str]]:
    active = teacher_names(cfg)
    missing = sorted(set(active) - set(paths))
    if missing:
        raise ValueError(f"data.{key} is missing active teachers: {missing}")
    return {
        teacher: list(paths[teacher])
        for teacher in active
    }


def _resume_contract(cfg: dict) -> dict:
    """Freeze every numerical/data semantic while allowing host-only changes."""

    contract = copy.deepcopy(cfg)
    for key in ("device", "output_dir"):
        contract.get("runtime", {}).pop(key, None)
    for key in (
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "package_pin_memory",
        "optimizer_visible_tile_packages",
        "optimizer_visible_tile_package_sizes",
        "optimizer_visible_contract_sha256",
    ):
        contract.get("data", {}).pop(key, None)
    for key in (
        "log_interval",
        "progress",
        "progress_interval_sec",
        "tensorboard",
        "tensorboard_batch_interval",
        "tensorboard_log_dir",
        "epochs",
        "step_metrics_flush_steps",
        "checkpoint_interval_steps",
        "development_probe_interval_steps",
        "development_probe_batches",
        "development_early_stop",
        "development_early_stop_min_step",
        "development_early_stop_relative_delta",
        "development_early_stop_patience",
    ):
        contract.get("train", {}).pop(key, None)
    return contract


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _verify_formal_ablation_contract(cfg: dict) -> None:
    expected = cfg.get("data", {}).get(
        "formal_ablation_contract_sha256"
    )
    if expected is None:
        return
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
    ):
        raise ValueError("formal ablation contract digest is invalid")
    payload = copy.deepcopy(cfg)
    payload["data"].pop("formal_ablation_contract_sha256", None)
    observed = _canonical_sha256(payload)
    if observed != expected:
        raise ValueError(
            "resolved formal ablation config changed after contract freeze: "
            f"expected={expected} observed={observed}"
        )


def _source_tree_sha256(repo: Path) -> str:
    roots = (
        repo / "pyproject.toml",
        repo / "README.md",
        repo / "CHANGELOG.md",
        repo / "configs",
        repo / "docs",
        repo / "experiments" / "ablation",
        repo / "scripts",
        repo / "src",
        repo / "tests",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _verify_formal_source_contract(
    cfg: dict,
    *,
    repo: Path | None = None,
) -> None:
    expected = cfg.get("data", {}).get("formal_source")
    if expected is None:
        return
    if not isinstance(expected, dict):
        raise ValueError("data.formal_source must be a mapping")
    commit = str(expected.get("commit", "")).lower()
    mode = str(expected.get("source_mode", ""))
    tree_digest = str(expected.get("source_tree_sha256", "")).lower()
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or re.fullmatch(r"[0-9a-f]{64}", tree_digest) is None
        or mode not in {"clean_git_commit", "declared_archive"}
    ):
        raise ValueError("formal source contract is incomplete")
    repo = (
        Path(__file__).resolve().parents[3]
        if repo is None
        else Path(repo).resolve()
    )
    observed_tree = _source_tree_sha256(repo)
    if observed_tree != tree_digest:
        raise ValueError(
            "formal executable source tree content changed: "
            f"expected={tree_digest} observed={observed_tree}"
        )
    if mode == "clean_git_commit":
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            head.returncode != 0
            or head.stdout.strip().lower() != commit
            or status.returncode != 0
            or status.stdout.strip()
        ):
            raise ValueError(
                "formal training requires the exact clean committed source"
            )
    else:
        declared_commit = os.environ.get(
            "HCC_SEMPATH_SOURCE_COMMIT",
            "",
        ).strip().lower()
        if declared_commit != commit:
            raise ValueError(
                "formal source archive commit declaration changed"
            )
    cfg["data"]["formal_source_contract_sha256"] = (
        _canonical_sha256(expected)
    )


def _formal_static_asset_paths(cfg: dict) -> dict[str, Path]:
    data = cfg["data"]
    result: dict[str, Path] = {}
    prototype_paths = data.get("prototype_paths")
    if isinstance(prototype_paths, dict):
        for teacher, path in prototype_paths.items():
            result[f"prototype_{teacher}"] = Path(str(path)).resolve()
    for key in (
        "prototype_supervision_manifest_path",
        "spatial_manifest_path",
        "train_manifest_path",
    ):
        value = data.get(key)
        if value:
            result[key] = Path(str(value)).resolve()
    return result


def _verify_formal_asset_contract(
    cfg: dict,
    *,
    complete_tile_packages: list[str],
    complete_teacher_packages: dict[str, list[str]],
) -> None:
    """Re-resolve and re-hash the complete formal input contract per trial."""

    formal = cfg.get("data", {}).get("formal_asset_sha256")
    if formal is None:
        return
    if not isinstance(formal, dict):
        raise ValueError("data.formal_asset_sha256 must be a mapping")
    static_expected = formal.get("static_files")
    iac_expected = formal.get("iac_packages")
    student_expected = formal.get("student_pretrained")
    if (
        not isinstance(static_expected, dict)
        or not isinstance(iac_expected, dict)
        or not isinstance(student_expected, dict)
    ):
        raise ValueError("formal asset SHA-256 contract is incomplete")
    static_paths = _formal_static_asset_paths(cfg)
    if set(static_paths) != set(static_expected):
        raise ValueError(
            "formal static asset keys differ from the resolved config: "
            f"expected={sorted(static_expected)} "
            f"resolved={sorted(static_paths)}"
        )
    for name, path in static_paths.items():
        if not path.is_file() or _file_sha256(path) != str(
            static_expected[name]
        ):
            raise ValueError(
                f"formal static asset content changed: {name}={path}"
            )
    resolved_iac_paths = {
        str(Path(path).resolve())
        for path in complete_tile_packages
    }
    resolved_iac_paths.update(
        str(Path(path).resolve())
        for teacher_paths in complete_teacher_packages.values()
        for path in teacher_paths
    )
    expected_iac_paths = {
        str(Path(str(path)).resolve())
        for path in iac_expected
    }
    if resolved_iac_paths != expected_iac_paths:
        added = sorted(resolved_iac_paths - expected_iac_paths)
        removed = sorted(expected_iac_paths - resolved_iac_paths)
        raise ValueError(
            "formal tile/teacher IAC package set changed: "
            f"added={added[:10]} removed={removed[:10]}"
        )
    for raw_path in sorted(expected_iac_paths):
        path = Path(raw_path)
        expected_digest = iac_expected.get(raw_path)
        if expected_digest is None:
            raise ValueError(
                "formal tile/teacher IAC path is not canonical: "
                f"{raw_path}"
            )
        if not path.is_file() or _file_sha256(path) != str(
            expected_digest
        ):
            raise ValueError(
                f"formal tile/teacher IAC content changed: {path}"
            )
    student_path = Path(
        str(student_expected.get("path", ""))
    ).resolve()
    if (
        student_path != STUDENT_PRETRAINED_PATH.resolve()
        or not student_path.is_file()
        or _file_sha256(student_path)
        != str(student_expected.get("sha256", ""))
    ):
        raise ValueError(
            "formal DINOv2 initialization content changed"
        )
    cfg["data"]["formal_asset_contract_sha256"] = _canonical_sha256(
        formal
    )


def _freeze_supervision_asset_contract(cfg: dict) -> None:
    """Bind every mutable human/prototype supervision file by content."""

    data = cfg["data"]
    paths: dict[str, Path] = {}
    for key in (
        "prototype_supervision_manifest_path",
        "expert_replay_prototype_manifest_path",
        "spatial_manifest_path",
    ):
        value = data.get(key)
        if value:
            paths[key] = Path(str(value)).resolve()
    prototype_paths = data.get("prototype_paths")
    if isinstance(prototype_paths, dict):
        for name, value in prototype_paths.items():
            paths[f"prototype_paths.{name}"] = Path(str(value)).resolve()
    missing = [
        f"{name}={path}"
        for name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "supervision assets are missing: " + ", ".join(missing)
        )
    frozen = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for name, path in sorted(paths.items())
    }
    formal_static = (
        data.get("formal_asset_sha256", {}).get(
            "static_files",
            {},
        )
    )
    if formal_static:
        expected_digests = {
            str(value) for value in formal_static.values()
        }
        unmatched = [
            name
            for name, value in frozen.items()
            if value["sha256"] not in expected_digests
        ]
        if unmatched:
            raise ValueError(
                "supervision asset content differs from the formal A0 "
                f"contract: {unmatched}"
            )
    data["supervision_asset_sha256"] = frozen


def _excluded_rows_contract(
    package_paths: list[str],
    rows_by_package: dict[int, np.ndarray],
) -> dict[str, list[int]]:
    return {
        str(Path(package_paths[index]).resolve()): [
            int(value)
            for value in np.asarray(rows, dtype=np.int64).tolist()
        ]
        for index, rows in sorted(rows_by_package.items())
        if len(rows) > 0
    }


def _freeze_expert_split_exclusion_contract(
    cfg: dict,
    *,
    train_packages: list[str],
    train_excluded_rows: dict[int, np.ndarray],
    val_packages: list[str],
    val_excluded_rows: dict[int, np.ndarray],
) -> None:
    contract = {
        "population_train_excludes_expert_val": (
            _excluded_rows_contract(
                train_packages,
                train_excluded_rows,
            )
        ),
        "population_val_excludes_expert_train": (
            _excluded_rows_contract(
                val_packages,
                val_excluded_rows,
            )
        ),
    }
    cfg["data"]["expert_split_exclusion_contract"] = contract
    cfg["data"]["expert_split_exclusion_sha256"] = _canonical_sha256(
        contract
    )


def _freeze_optimizer_visible_contract(
    cfg: dict,
    *,
    population_packages: list[str],
    expert_packages: list[str],
    expert_replay_enabled: bool,
) -> None:
    """Freeze optimizer-visible package identity and content."""

    visible = list(population_packages)
    if expert_replay_enabled:
        visible.extend(expert_packages)
    frozen_packages = sorted(
        set(str(Path(path).resolve()) for path in visible)
    )
    cfg["data"]["optimizer_visible_tile_packages"] = frozen_packages
    cfg["data"]["optimizer_visible_tile_package_sizes"] = [
        Path(path).stat().st_size for path in frozen_packages
    ]
    formal_iac = (
        cfg["data"].get("formal_asset_sha256", {}).get(
            "iac_packages",
            {},
        )
    )
    if formal_iac and not isinstance(formal_iac, dict):
        raise ValueError("data.formal_asset_sha256.iac_packages is invalid")
    digests: list[str] = []
    for path in frozen_packages:
        digest = formal_iac.get(path) if formal_iac else None
        if formal_iac and digest is None:
            raise ValueError(
                "optimizer-visible package is absent from the formal "
                f"A0 asset contract: {path}"
            )
        if not formal_iac:
            digest = _file_sha256(path)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"invalid optimizer-visible package SHA-256: {path}"
            )
        digests.append(digest)
    cfg["data"]["optimizer_visible_tile_package_sha256"] = digests


def _verify_optimizer_visible_packages(cfg: dict) -> None:
    packages = cfg["data"].get("optimizer_visible_tile_packages")
    sizes = cfg["data"].get("optimizer_visible_tile_package_sizes")
    digests = cfg["data"].get(
        "optimizer_visible_tile_package_sha256"
    )
    if (
        not isinstance(packages, list)
        or not isinstance(sizes, list)
        or not isinstance(digests, list)
    ):
        raise ValueError("checkpoint has no optimizer-visible package list")
    if len(packages) != len(sizes) or len(packages) != len(digests):
        raise ValueError("optimizer-visible package count changed")
    current = [Path(path).stat().st_size for path in packages]
    if current != [int(size) for size in sizes]:
        raise ValueError("optimizer-visible package size changed")
    current_digests = [_file_sha256(path) for path in packages]
    if current_digests != [str(value) for value in digests]:
        raise ValueError("optimizer-visible package content changed")


def _validate_resume_contract(cfg: dict, resume_state: dict) -> None:
    saved_cfg = resume_state.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("resume checkpoint has no resolved training config")
    _verify_optimizer_visible_packages(saved_cfg)
    if _resume_contract(saved_cfg) != _resume_contract(cfg):
        raise ValueError(
            "resume config changes the model, loss, data, or optimization "
            "contract; start a separate run instead"
        )


def _resolve_configured_epochs(
    cfg: dict,
    resume_state: dict | None,
) -> int:
    configured = int(cfg["train"]["epochs"])
    checkpoint_epoch = int((resume_state or {}).get("epoch", 0))
    if configured < checkpoint_epoch:
        raise ValueError(
            "train.epochs "
            f"({configured}) precedes checkpoint epoch ({checkpoint_epoch})"
        )
    return configured


def _limit_records(records: list, limit: int, seed: int) -> list:
    if limit <= 0 or len(records) <= limit:
        return records
    groups: dict[str, list] = {}
    for item in records:
        package_path = getattr(item, "tile_package_path", None)
        key = str(package_path) if package_path is not None else item.record.slide_id
        groups.setdefault(key, []).append(item)
    if limit < len(groups):
        raise ValueError(
            f"max_records must be 0 or at least the selected package/group count so every group participates: "
            f"max_records={limit} groups={len(groups)}"
        )
    rng = np.random.default_rng(seed)
    group_keys = list(groups)
    group_sizes = np.asarray([len(groups[key]) for key in group_keys], dtype=np.int64)
    expected = group_sizes.astype(np.float64) * (limit / int(group_sizes.sum()))
    quotas = np.floor(expected).astype(np.int64)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, group_sizes)
    overflow = int(quotas.sum() - limit)
    while overflow > 0:
        candidates = np.flatnonzero(quotas > 1)
        chosen = rng.choice(candidates, size=min(overflow, len(candidates)), replace=False)
        quotas[chosen] -= 1
        overflow = int(quotas.sum() - limit)
    remainder = int(limit - quotas.sum())
    if remainder > 0:
        capacity = group_sizes - quotas
        candidates = np.flatnonzero(capacity > 0)
        weights = expected[candidates] - np.floor(expected[candidates])
        if float(weights.sum()) <= 0:
            chosen = rng.choice(candidates, size=remainder, replace=False)
        else:
            chosen = rng.choice(candidates, size=remainder, replace=False, p=weights / weights.sum())
        quotas[chosen] += 1

    selected = []
    for key, quota in zip(group_keys, quotas):
        items = groups[key]
        if int(quota) >= len(items):
            selected.extend(items)
            continue
        stride = len(items) / int(quota)
        offset = float(rng.random()) * stride
        rows = np.floor(offset + np.arange(int(quota), dtype=np.float64) * stride).astype(np.int64)
        selected.extend(items[int(row)] for row in np.minimum(rows, len(items) - 1))
    rng.shuffle(selected)
    return selected


def _load_prototype_map(
    cfg: dict,
    dims: dict[str, int],
    device: torch.device,
    expected_names: list[str] | tuple[str, ...] = DEFAULT_CLASSIFICATION_CLASSES,
) -> dict[str, PrototypeRegistry] | None:
    loss_cfg = cfg["loss"]
    prototype_responses_enabled = (
        float(loss_cfg.get("semantic_weight", 0.0)) > 0
        or float(loss_cfg.get("prototype_filter_weight", 0.0)) > 0
        or float(loss_cfg.get("zhcc_response_weight", 0.0)) > 0
    )
    if not prototype_responses_enabled:
        return None
    prototype_paths = cfg["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        registries = {
            name: load_prototype_registry(
                prototype_paths[name],
                expected_dim=dim,
            ).to(device)
            for name, dim in dims.items()
        }
    else:
        prototype_path = cfg["data"].get("prototype_path")
        if prototype_path is None:
            raise ValueError(
                "data.prototype_path or data.prototype_paths is required "
                "when prototype-response supervision is enabled"
            )
        registries = {
            name: load_prototype_registry(
                prototype_path,
                expected_dim=dim,
            ).to(device)
            for name, dim in dims.items()
        }
    expected = list(expected_names)
    for teacher, registry in registries.items():
        if registry.names != expected:
            raise ValueError(
                "teacher semantic prototype contract mismatch: "
                f"teacher={teacher} expected={expected} got={registry.names}"
            )
    return registries


def _prototype_source_splits(cfg: dict, key: str, default: list[str]) -> set[str] | None:
    value = cfg["data"].get(key, default)
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _cgroup_cpu_quota() -> int | None:
    """Usable CPU cores from the cgroup v2 quota, or None if unlimited/unknown.

    On shared hosts torch/OpenMP size their thread pools to the PHYSICAL core
    count (e.g. 208) while the container is limited to a far smaller quota
    (e.g. 25). The resulting oversubscription burns the quota on context
    switching instead of useful work, starving the decode workers.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as handle:
            quota_s, period_s = handle.read().split()
        if quota_s == "max":
            return None
        cores = int(quota_s) // int(period_s)
        return max(1, cores)
    except Exception:
        return None


def _cap_compute_threads(num_workers: int) -> None:
    quota = _cgroup_cpu_quota()
    if quota is None:
        return
    # torch.set_num_threads is PROCESS-GLOBAL, not main-thread-only: the decode
    # workers' scatter_package_rows() does ATen copy_() ops, so each of the
    # num_workers decode threads fans its copy out across this same intra-op
    # pool. A pool size of N therefore multiplies CPU use by ~N per worker
    # (num_workers * N threads), oversubscribing the cgroup quota — this is why
    # the loader hit the 25-core ceiling at only ~6 workers. The training hot
    # path (image normalize, model forward, loss) all runs on the GPU; the only
    # CPU-side torch work left in the main thread is scalar/small-tensor glue
    # and .cpu()/cat moves, which a single thread handles fine. Pin the pool to
    # 1 so the quota goes almost 1:1 to decode workers.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _build_optimizer(
    model,
    cfg: dict,
    device: torch.device,
) -> torch.optim.AdamW:
    """Build the numerically equivalent low-traffic CUDA optimizer when enabled."""

    kwargs = {
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
    }
    if bool(
        cfg["train"].get("fused_optimizer", False)
        and device.type == "cuda"
    ):
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _configure_compiled_training_for_gradient_diagnostics() -> None:
    """Allow diagnostic autograd passes before the optimizing backward pass."""

    from torch._functorch import config as functorch_config

    # AOTAutograd buffer donation requires one non-retained backward. The
    # scientific diagnostic intentionally takes two retained gradients from
    # the same forward graph before the optimizing backward.
    functorch_config.donated_buffer = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCC-SemPath distillation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    _probe.timeline_event("startup.begin", force=True)
    cfg = load_config(args.config)
    _probe.timeline_event("startup.config_loaded", force=True)
    _verify_formal_ablation_contract(cfg)
    cfg["research_contract"] = {
        "student_backbone": STUDENT_BACKBONE_NAME,
        "student_pretrained": True,
        "student_image_size": STUDENT_IMAGE_SIZE,
        "student_patch_size": STUDENT_PATCH_SIZE,
        "student_pretrained_file": STUDENT_PRETRAINED_PATH.name,
        "student_pretrained_sha256": STUDENT_PRETRAINED_SHA256,
    }
    _verify_formal_source_contract(cfg)
    _cap_compute_threads(int(cfg["data"].get("num_workers", 0)))
    seed_everything(int(cfg["runtime"]["seed"]))
    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    manifest_path = cfg["data"].get("train_manifest_path")
    explicit_split_packages = "train_image_tile_package_paths" in cfg["data"]
    if explicit_split_packages:
        train_tile_packages = _paths_from_data(cfg, "train_image_tile_package_paths")
        val_tile_packages = _paths_from_data(cfg, "val_image_tile_package_paths")
        train_teacher_packages = _active_teacher_paths(
            cfg,
            _teacher_paths_from_data(
                cfg,
                "train_teacher_feature_package_paths",
            ),
            key="train_teacher_feature_package_paths",
        )
        val_teacher_packages = _active_teacher_paths(
            cfg,
            _teacher_paths_from_data(
                cfg,
                "val_teacher_feature_package_paths",
            ),
            key="val_teacher_feature_package_paths",
        )
        names = teacher_names(cfg)
        complete_expert_tile_packages = (
            list(train_tile_packages) + list(val_tile_packages)
        )
        complete_expert_teacher_packages = {
            name: (
                list(train_teacher_packages[name])
                + list(val_teacher_packages[name])
            )
            for name in names
        }
        complete_train_tile_packages = list(
            complete_expert_tile_packages
        )
        complete_val_tile_packages = list(
            complete_expert_tile_packages
        )
        complete_train_teacher_packages = {
            name: list(paths)
            for name, paths in (
                complete_expert_teacher_packages.items()
            )
        }
        complete_val_teacher_packages = {
            name: list(paths)
            for name, paths in (
                complete_expert_teacher_packages.items()
            )
        }
    elif manifest_path:
        manifest = load_training_manifest(manifest_path)
        train_tile_packages, train_teacher_packages = manifest_data_paths(cfg, manifest, "train")
        val_tile_packages, val_teacher_packages = manifest_data_paths(cfg, manifest, "val")
        names = teacher_names(cfg)
        full_cfg = copy.deepcopy(cfg)
        full_cfg["data"]["train_tile_fraction"] = 1.0
        full_cfg["data"]["val_tile_fraction"] = 1.0
        expert_train_tiles, expert_train_teachers = manifest_data_paths(
            full_cfg,
            manifest,
            "train",
        )
        expert_val_tiles, expert_val_teachers = manifest_data_paths(
            full_cfg,
            manifest,
            "val",
        )
        complete_expert_tile_packages = (
            expert_train_tiles + expert_val_tiles
        )
        complete_expert_teacher_packages = {
            name: (
                expert_train_teachers[name]
                + expert_val_teachers[name]
            )
            for name in names
        }
        complete_train_tile_packages = list(
            complete_expert_tile_packages
        )
        complete_val_tile_packages = list(
            complete_expert_tile_packages
        )
        complete_train_teacher_packages = {
            name: list(paths)
            for name, paths in (
                complete_expert_teacher_packages.items()
            )
        }
        complete_val_teacher_packages = {
            name: list(paths)
            for name, paths in (
                complete_expert_teacher_packages.items()
            )
        }
    else:
        tile_packages = image_tile_package_paths(cfg)
        teacher_packages = _active_teacher_paths(
            cfg,
            teacher_feature_package_paths(cfg),
            key="teacher_feature_package_paths",
        )
        train_tile_packages = tile_packages
        val_tile_packages = tile_packages
        train_teacher_packages = teacher_packages
        val_teacher_packages = teacher_packages
        names = teacher_names(cfg)
        complete_train_tile_packages = list(train_tile_packages)
        complete_val_tile_packages = list(val_tile_packages)
        complete_train_teacher_packages = {
            name: list(paths)
            for name, paths in train_teacher_packages.items()
        }
        complete_val_teacher_packages = {
            name: list(paths)
            for name, paths in val_teacher_packages.items()
        }
    _verify_formal_asset_contract(
        cfg,
        complete_tile_packages=complete_train_tile_packages,
        complete_teacher_packages=complete_train_teacher_packages,
    )
    _probe.timeline_event(
        "startup.package_paths_resolved",
        force=True,
        train_packages=len(train_tile_packages),
        validation_packages=len(val_tile_packages),
        expert_train_packages=len(complete_train_tile_packages),
        expert_validation_packages=len(complete_val_tile_packages),
    )
    if manifest_path or explicit_split_packages:
        _assert_disjoint_package_paths(
            train_tile_packages,
            val_tile_packages,
        )
    validate_training_config(cfg, names)
    dims = teacher_dims(cfg, names)
    classification_class_names = [
        str(name)
        for name in cfg["model"].get("classification_class_names", DEFAULT_CLASSIFICATION_CLASSES)
    ]
    if tuple(classification_class_names) != DEFAULT_CLASSIFICATION_CLASSES:
        raise ValueError(
            f"fixed classification class contract must be {list(DEFAULT_CLASSIFICATION_CLASSES)}, got {classification_class_names}"
        )
    prototypes = _load_prototype_map(
        cfg,
        dims,
        device,
        expected_names=classification_class_names,
    )
    prototype_manifest_path = cfg["data"].get("prototype_supervision_manifest_path")
    train_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        classification_class_names,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_train_splits", ["train"]),
    )
    val_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        classification_class_names,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_val_splits", ["val"]),
    )
    replay_prototype_manifest_path = cfg["data"].get(
        "expert_replay_prototype_manifest_path",
        prototype_manifest_path,
    )
    replay_prototype_labels = (
        train_prototype_labels
        if replay_prototype_manifest_path == prototype_manifest_path
        else load_prototype_labels(
            replay_prototype_manifest_path,
            classification_class_names,
            allowed_source_splits=_prototype_source_splits(
                cfg,
                "prototype_supervision_train_splits",
                ["train"],
            ),
        )
    )
    _probe.timeline_event(
        "startup.classification_assets_loaded",
        force=True,
        train_labels=len(train_prototype_labels),
        replay_labels=len(replay_prototype_labels),
    )
    all_tile_packages = sorted(set(train_tile_packages + val_tile_packages))
    tile_metadata = read_package_metadata(all_tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    expected_image_size = (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE)
    if image_size != expected_image_size:
        raise ValueError(f"fixed student expects native tiles {expected_image_size}, got {image_size}")
    spatial_manifest_path = cfg["data"].get("spatial_manifest_path")
    component_names = spatial_component_names(spatial_manifest_path) if spatial_manifest_path else []
    cfg["data"]["spatial_component_names"] = list(component_names)
    spatial_stride = int(cfg["model"].get("spatial_output_stride", SPATIAL_OUTPUT_STRIDE))
    spatial_grid_size = (
        (
            (image_size[0] + 2 * SPATIAL_PATCH_PADDING - STUDENT_PATCH_SIZE) // spatial_stride + 1,
            (image_size[1] + 2 * SPATIAL_PATCH_PADDING - STUDENT_PATCH_SIZE) // spatial_stride + 1,
        )
        if spatial_manifest_path
        else (0, 0)
    )
    spatial_train_splits = set(
        cfg["data"].get("spatial_train_splits", ["train"])
    )
    train_spatial_targets = build_spatial_roi_targets(
        spatial_manifest_path,
        component_names=component_names,
        image_size=image_size,
        grid_size=spatial_grid_size,
        allowed_splits=spatial_train_splits,
        point_tolerance_cells=int(
            cfg["loss"].get("spatial_point_tolerance_cells", 1)
        ),
    )
    spatial_val_splits = set(
        cfg["data"].get("spatial_val_splits", ["val"])
    )
    val_spatial_targets = build_spatial_roi_targets(
        spatial_manifest_path,
        component_names=component_names,
        image_size=image_size,
        grid_size=spatial_grid_size,
        allowed_splits=spatial_val_splits,
        point_tolerance_cells=int(
            cfg["loss"].get("spatial_point_tolerance_cells", 1)
        ),
    )
    _probe.timeline_event(
        "startup.spatial_assets_loaded",
        force=True,
        train_spatial_tiles=len(train_spatial_targets),
        validation_spatial_tiles=len(val_spatial_targets),
    )
    expert_tile_ids = set(replay_prototype_labels).union(
        train_spatial_targets
    )
    validation_expert_tile_ids = set(val_prototype_labels).union(
        val_spatial_targets
    )
    supervision_overlap = expert_tile_ids & validation_expert_tile_ids
    if supervision_overlap:
        raise ValueError(
            "train/validation expert tile overlap: "
            f"count={len(supervision_overlap)} "
            f"sample={next(iter(sorted(supervision_overlap)))}"
        )
    if bool(cfg["data"].get("require_complete_expert_validation", False)):
        missing_classes = [
            classification_class_names[index]
            for index in range(len(classification_class_names))
            if not any(
                label.classification == index
                for label in val_prototype_labels.values()
            )
        ]
        if missing_classes:
            raise ValueError(
                "expert validation has no classification labels for: "
                + ", ".join(missing_classes)
            )
        missing_components = [
            component
            for index, component in enumerate(component_names)
            if not any(
                bool(target.supervised[index])
                for target in val_spatial_targets.values()
            )
        ]
        if missing_components:
            raise ValueError(
                "expert validation has no spatial supervision for: "
                + ", ".join(missing_components)
            )

    expert_locations: dict[str, tuple[str, int]] = {}
    for tile_id, label in replay_prototype_labels.items():
        if label.package_path is not None and label.row is not None:
            expert_locations[tile_id] = (label.package_path, label.row)
    for tile_id, location in load_spatial_tile_locations(
        spatial_manifest_path,
        allowed_splits=spatial_train_splits,
    ).items():
        previous = expert_locations.get(tile_id)
        if previous is not None and previous != location:
            raise ValueError(
                "classification/spatial provenance mismatch: "
                f"tile={tile_id} classification={previous} spatial={location}"
            )
        expert_locations[tile_id] = location
    missing_locations = sorted(expert_tile_ids.difference(expert_locations))
    if missing_locations:
        raise ValueError(
            "expert annotation is missing fixed IAC package/row provenance: "
            f"count={len(missing_locations)} "
            f"sample={', '.join(missing_locations[:3])}"
        )
    validation_expert_locations: dict[str, tuple[str, int]] = {}
    for tile_id, label in val_prototype_labels.items():
        if label.package_path is not None and label.row is not None:
            validation_expert_locations[tile_id] = (
                label.package_path,
                label.row,
            )
    for tile_id, location in load_spatial_tile_locations(
        spatial_manifest_path,
        allowed_splits=spatial_val_splits,
    ).items():
        previous = validation_expert_locations.get(tile_id)
        if previous is not None and previous != location:
            raise ValueError(
                "validation classification/spatial provenance mismatch: "
                f"tile={tile_id} classification={previous} spatial={location}"
            )
        validation_expert_locations[tile_id] = location
    missing_validation_locations = sorted(
        validation_expert_tile_ids.difference(
            validation_expert_locations
        )
    )
    if missing_validation_locations:
        raise ValueError(
            "validation expert annotation is missing fixed IAC "
            "package/row provenance: "
            f"count={len(missing_validation_locations)} "
            f"sample={', '.join(missing_validation_locations[:3])}"
        )

    expert_rows_by_package: dict[int, np.ndarray] = {}
    expert_tile_packages: list[str] = []
    expert_teacher_packages: dict[str, list[str]] = {
        name: []
        for name in names
    }
    if expert_tile_ids:
        (
            expert_tile_packages,
            expert_teacher_packages,
            expert_rows_by_package,
            _,
        ) = _expert_package_subset(
            complete_train_tile_packages,
            complete_train_teacher_packages,
            expert_locations,
        )
    validation_expert_rows_by_package: dict[int, np.ndarray] = {}
    validation_expert_tile_packages: list[str] = []
    validation_expert_teacher_packages: dict[str, list[str]] = {
        name: []
        for name in names
    }
    if validation_expert_tile_ids:
        (
            validation_expert_tile_packages,
            validation_expert_teacher_packages,
            validation_expert_rows_by_package,
            _,
        ) = _expert_package_subset(
            complete_val_tile_packages,
            complete_val_teacher_packages,
            validation_expert_locations,
        )
    # The finalized expert split is authoritative even when its source IACs
    # came from an older train/val package partition. Exact expert-validation
    # rows must never enter the optimizer through population distillation, and
    # exact expert-training rows must not leak into population validation.
    population_train_excluded_rows = _target_rows_by_package(
        train_tile_packages,
        validation_expert_locations,
        require_all=False,
    )
    population_val_excluded_rows = _target_rows_by_package(
        val_tile_packages,
        expert_locations,
        require_all=False,
    )
    cfg["data"]["population_train_excluded_expert_val_tiles"] = int(
        sum(len(rows) for rows in population_train_excluded_rows.values())
    )
    cfg["data"]["population_val_excluded_expert_train_tiles"] = int(
        sum(len(rows) for rows in population_val_excluded_rows.values())
    )
    _freeze_expert_split_exclusion_contract(
        cfg,
        train_packages=train_tile_packages,
        train_excluded_rows=population_train_excluded_rows,
        val_packages=val_tile_packages,
        val_excluded_rows=population_val_excluded_rows,
    )
    spatial_dataset_kwargs = {
        "spatial_component_count": len(component_names) if spatial_manifest_path else 0,
        "spatial_grid_size": spatial_grid_size,
    }
    dynamic_package_sampling = bool(cfg["data"].get("dynamic_package_sampling", False))
    if dynamic_package_sampling:
        tensor_collate = bool(cfg["data"].get("tensor_collate", device.type == "cuda"))
        common_dataset_kwargs = {
            "image_size": image_size,
            "mean": cfg["data"].get("mean"),
            "std": cfg["data"].get("std"),
            "tensor_collate": tensor_collate,
        }
        train_ds = PackageSampledDistillationDataset(
            train_tile_packages,
            train_teacher_packages,
            **common_dataset_kwargs,
            max_records=int(cfg["data"].get("max_train_records", 0)),
            seed=int(cfg["runtime"]["seed"]),
            expected_dims=dims,
            prototype_labels=train_prototype_labels,
            spatial_targets=train_spatial_targets,
            excluded_rows_by_package=population_train_excluded_rows,
            **spatial_dataset_kwargs,
        )
        val_ds = PackageSampledDistillationDataset(
            val_tile_packages,
            val_teacher_packages,
            **common_dataset_kwargs,
            max_records=int(cfg["data"].get("max_val_records", 0)),
            seed=int(cfg["runtime"]["seed"]) + 1,
            expected_dims=dims,
            prototype_labels={},
            spatial_targets={},
            excluded_rows_by_package=population_val_excluded_rows,
            **spatial_dataset_kwargs,
        )
    elif manifest_path or explicit_split_packages:
        train_records = read_packaged_tile_records(train_tile_packages)
        val_records = read_packaged_tile_records(val_tile_packages)
        train_records = [
            item
            for item in train_records
            if item.record.tile_id not in validation_expert_tile_ids
        ]
        val_records = [
            item
            for item in val_records
            if item.record.tile_id not in expert_tile_ids
        ]
        if not train_records or not val_records:
            raise ValueError("manifest must contain non-empty train and val splits")
        train_records = _limit_records(train_records, int(cfg["data"].get("max_train_records", 0)), int(cfg["runtime"]["seed"]))
        val_records = _limit_records(val_records, int(cfg["data"].get("max_val_records", 0)), int(cfg["runtime"]["seed"]) + 1)
        validate_teacher_cache(
            train_records,
            None,
            dims,
            teacher_cache_package_paths=train_teacher_packages,
        )
        validate_teacher_cache(
            val_records,
            None,
            dims,
            teacher_cache_package_paths=val_teacher_packages,
        )
        common_dataset_kwargs = {
            "teacher_cache_dir": None,
            "image_size": image_size,
            "mean": cfg["data"].get("mean"),
            "std": cfg["data"].get("std"),
        }
        train_ds = DistillationTileDataset(
            train_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=train_teacher_packages,
            prototype_labels=train_prototype_labels,
            spatial_targets=train_spatial_targets,
            **spatial_dataset_kwargs,
        )
        val_ds = DistillationTileDataset(
            val_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=val_teacher_packages,
            prototype_labels={},
            spatial_targets={},
            **spatial_dataset_kwargs,
        )
    else:
        records = read_packaged_tile_records(train_tile_packages)
        records = apply_split_overrides(
            records,
            cfg["data"].get("split_manifest_path"),
            cfg["data"].get("split_key", "slide_id"),
        )
        train_records = [item for item in records if item.record.split == "train"]
        val_records = [item for item in records if item.record.split == "val"]
        train_records = [
            item
            for item in train_records
            if item.record.tile_id not in validation_expert_tile_ids
        ]
        val_records = [
            item
            for item in val_records
            if item.record.tile_id not in expert_tile_ids
        ]
        if not train_records or not val_records:
            raise ValueError("manifest must contain non-empty train and val splits")
        train_records = _limit_records(train_records, int(cfg["data"].get("max_train_records", 0)), int(cfg["runtime"]["seed"]))
        val_records = _limit_records(val_records, int(cfg["data"].get("max_val_records", 0)), int(cfg["runtime"]["seed"]) + 1)
        validate_teacher_cache(
            train_records,
            None,
            dims,
            teacher_cache_package_paths=train_teacher_packages,
        )
        validate_teacher_cache(
            val_records,
            None,
            dims,
            teacher_cache_package_paths=val_teacher_packages,
        )
        common_dataset_kwargs = {
            "teacher_cache_dir": None,
            "image_size": image_size,
            "mean": cfg["data"].get("mean"),
            "std": cfg["data"].get("std"),
        }
        train_ds = DistillationTileDataset(
            train_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=train_teacher_packages,
            prototype_labels=train_prototype_labels,
            spatial_targets=train_spatial_targets,
            **spatial_dataset_kwargs,
        )
        val_ds = DistillationTileDataset(
            val_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=val_teacher_packages,
            prototype_labels={},
            spatial_targets={},
            **spatial_dataset_kwargs,
        )
    _probe.timeline_event(
        "startup.population_datasets_built",
        force=True,
        train_tiles=len(train_ds),
        validation_tiles=len(val_ds),
    )
    num_workers = int(cfg["data"]["num_workers"])
    loader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
    if dynamic_package_sampling:
        prefetch_batches = int(cfg["data"].get("prefetch_factor", 2))
        default_chunk_size = max(1, int(cfg["train"]["batch_size"]) // max(1, num_workers))
        package_chunk_size = int(cfg["data"].get("package_chunk_size", default_chunk_size))
        package_buffer_batches = int(cfg["data"].get("package_buffer_batches", 4))
        use_mp = bool(cfg["data"].get("package_multiprocessing", False)) and num_workers > 0
        if use_mp:
            # Process-per-worker path: each decode worker is a separate PROCESS
            # with its own GIL, so JXL decode no longer contends with the
            # main-thread GPU inference loop for one process-wide GIL (the cause
            # of the ~800% CPU ceiling and 7.3ms/tile vs 2ms in-bench decode).
            # Reproduces the thread loader's exact tile order via the chunk-plan
            # batch sampler, and reuses the dataset's __getitems__ + collate.
            bs = int(cfg["train"]["batch_size"])
            train_sampler = _ChunkPlanBatchSampler(
                train_ds, batch_size=bs, chunk_size=package_chunk_size,
                seed=int(cfg["runtime"]["seed"]), reshuffle_each_epoch=True,
            )
            val_sampler = _ChunkPlanBatchSampler(
                val_ds, batch_size=bs, chunk_size=package_chunk_size,
                seed=int(cfg["runtime"]["seed"]) + 1, reshuffle_each_epoch=False,
            )
            mp_kwargs = dict(
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
                persistent_workers=bool(cfg["data"].get("persistent_workers", True)),
            )
            train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                      collate_fn=train_ds.collate, **mp_kwargs)
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                                    collate_fn=val_ds.collate, **mp_kwargs)
        else:
            train_loader = _PackageShuffleBatchLoader(
                train_ds,
                batch_size=int(cfg["train"]["batch_size"]),
                num_workers=num_workers,
                prefetch_batches=prefetch_batches,
                collate_fn=train_ds.collate,
                seed=int(cfg["runtime"]["seed"]),
                chunk_size=package_chunk_size,
                buffer_batches=package_buffer_batches,
                reshuffle_each_epoch=True,
                pin_memory=bool(cfg["data"].get("package_pin_memory", False)),
            )
            val_loader = _PackageShuffleBatchLoader(
                val_ds,
                batch_size=int(cfg["train"]["batch_size"]),
                num_workers=num_workers,
                prefetch_batches=prefetch_batches,
                collate_fn=val_ds.collate,
                seed=int(cfg["runtime"]["seed"]) + 1,
                chunk_size=package_chunk_size,
                buffer_batches=package_buffer_batches,
                reshuffle_each_epoch=False,
                pin_memory=bool(cfg["data"].get("package_pin_memory", False)),
            )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            collate_fn=collate_distillation,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            collate_fn=collate_distillation,
            **loader_kwargs,
        )
    _probe.timeline_event("startup.population_loaders_built", force=True)
    replay_interval = int(
        cfg["data"].get("expert_replay_interval_batches", 16)
    )
    prototype_refresh_loader = None
    spatial_prototype_refresh_loader = None
    expert_classification_val_loader = None
    expert_spatial_val_loader = None
    expert_batch_size = int(
        cfg["data"].get(
            "expert_batch_size",
            min(64, int(cfg["train"]["batch_size"])),
        )
    )
    cfg["data"]["expert_batch_size"] = expert_batch_size
    prototype_batch_size = int(
        cfg["train"].get(
            "dynamic_prototype_batch_size",
            cfg["train"]["batch_size"],
        )
    )
    if expert_tile_ids:
        expert_ds = PackageSampledDistillationDataset(
            expert_tile_packages,
            expert_teacher_packages,
            image_size=image_size,
            max_records=0,
            seed=int(cfg["runtime"]["seed"]),
            mean=cfg["data"].get("mean"),
            std=cfg["data"].get("std"),
            expected_dims=dims,
            prototype_labels=train_prototype_labels,
            tensor_collate=bool(
                cfg["data"].get(
                    "tensor_collate",
                    device.type == "cuda",
                )
            ),
            spatial_targets=train_spatial_targets,
            **spatial_dataset_kwargs,
        )
        expert_bank = _materialize_expert_bank(
            expert_ds,
            expert_rows_by_package,
            batch_size=prototype_batch_size,
            num_workers=num_workers,
            prefetch_factor=int(
                cfg["data"].get("prefetch_factor", 2)
            ),
        )
        _probe.timeline_event(
            "startup.expert_bank_materialized",
            force=True,
            expert_tiles=len(expert_bank),
        )
        if train_prototype_labels:
            prototype_refresh_loader = _InMemoryExpertBatchLoader(
                expert_bank,
                indices=None,
                batch_size=prototype_batch_size,
                seed=int(cfg["runtime"]["seed"]) + 200_003,
            )
        if train_spatial_targets:
            spatial_indices = [
                expert_bank.index_by_tile_id[tile_id]
                for tile_id in sorted(train_spatial_targets)
            ]
            spatial_prototype_refresh_loader = (
                _InMemoryExpertBatchLoader(
                    expert_bank,
                    indices=spatial_indices,
                    batch_size=expert_batch_size,
                    seed=int(cfg["runtime"]["seed"]) + 300_007,
                )
            )
        if replay_interval > 0:
            expert_loader = _InMemoryExpertBatchLoader(
                expert_bank,
                indices=None,
                batch_size=expert_batch_size,
                seed=int(cfg["runtime"]["seed"]) + 100_003,
            )
            train_loader = _InterleavedBatchLoader(
                train_loader,
                expert_loader,
                interval=replay_interval,
            )
            cfg["data"]["expert_replay_interval_batches"] = (
                replay_interval
            )
            cfg["data"]["expert_replay_tiles"] = len(expert_tile_ids)
    if validation_expert_tile_ids:
        validation_expert_ds = PackageSampledDistillationDataset(
            validation_expert_tile_packages,
            validation_expert_teacher_packages,
            image_size=image_size,
            max_records=0,
            seed=int(cfg["runtime"]["seed"]) + 1,
            mean=cfg["data"].get("mean"),
            std=cfg["data"].get("std"),
            expected_dims=dims,
            prototype_labels=val_prototype_labels,
            tensor_collate=bool(
                cfg["data"].get(
                    "tensor_collate",
                    device.type == "cuda",
                )
            ),
            spatial_targets=val_spatial_targets,
            **spatial_dataset_kwargs,
        )
        validation_bank = _materialize_expert_bank(
            validation_expert_ds,
            validation_expert_rows_by_package,
            batch_size=prototype_batch_size,
            num_workers=num_workers,
            prefetch_factor=int(
                cfg["data"].get("prefetch_factor", 2)
            ),
        )
        if val_prototype_labels:
            classification_indices = [
                validation_bank.index_by_tile_id[tile_id]
                for tile_id in sorted(val_prototype_labels)
            ]
            expert_classification_val_loader = (
                _InMemoryExpertBatchLoader(
                    validation_bank,
                    indices=classification_indices,
                    batch_size=expert_batch_size,
                    seed=int(cfg["runtime"]["seed"]) + 400_009,
                )
            )
        if val_spatial_targets:
            spatial_validation_indices = [
                validation_bank.index_by_tile_id[tile_id]
                for tile_id in sorted(val_spatial_targets)
            ]
            expert_spatial_val_loader = _InMemoryExpertBatchLoader(
                validation_bank,
                indices=spatial_validation_indices,
                batch_size=expert_batch_size,
                seed=int(cfg["runtime"]["seed"]) + 500_009,
            )
        cfg["data"]["expert_validation_tiles"] = len(
            validation_expert_tile_ids
        )
    _freeze_supervision_asset_contract(cfg)
    _freeze_optimizer_visible_contract(
        cfg,
        population_packages=train_tile_packages,
        expert_packages=expert_tile_packages,
        expert_replay_enabled=bool(
            expert_tile_ids
            and (
                replay_interval > 0
                or prototype_refresh_loader is not None
                or spatial_prototype_refresh_loader is not None
            )
        ),
    )
    model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=not bool(args.resume),
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=bool(cfg["model"].get("grad_checkpointing", False)),
        classification_num_classes=len(classification_class_names),
        spatial_num_components=len(component_names) if spatial_manifest_path else 0,
        spatial_dim=int(cfg["model"].get("spatial_dim", 256)),
        spatial_output_stride=spatial_stride,
    ).to(device)
    _probe.timeline_event("startup.model_constructed", force=True)
    if model.spatial_head is not None:
        model.spatial_head.use_local_branch = bool(
            cfg["model"].get("spatial_use_local_branch", True)
        )
        model.spatial_head.use_semantic_branch = bool(
            cfg["model"].get("spatial_use_semantic_branch", True)
        )
        model.spatial_head.use_context = bool(
            cfg["model"].get("spatial_use_context", True)
        )
    if spatial_manifest_path and model.spatial_head is not None:
        if spatial_grid_size != (32, 32):
            raise ValueError(
                f"fixed 224px/14px-window spatial contract expects a 32x32 grid, got {spatial_grid_size}"
            )
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        _probe.timeline_event("startup.checkpoint_loaded", force=True)
        _validate_resume_contract(cfg, resume_state)
        state = {
            key.removeprefix("_orig_mod."): value
            for key, value in resume_state["model"].items()
        }
        model.load_state_dict(state)
    if bool(cfg["train"].get("compile", False)):
        _configure_compiled_training_for_gradient_diagnostics()
        model = torch.compile(model)
    _probe.timeline_event("startup.model_compile_wrapped", force=True)
    optimizer = _build_optimizer(model, cfg, device)
    _probe.timeline_event("startup.optimizer_built", force=True)
    if resume_state and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
    scheduler_cfg = (
        resume_state.get("config", cfg)
        if resume_state is not None
        else cfg
    )
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg, len(train_loader))
    if resume_state and scheduler is not None and resume_state.get("scheduler") is not None:
        scheduler.load_state_dict(resume_state["scheduler"])
    _resolve_configured_epochs(cfg, resume_state)
    metrics = fit(
        model,
        train_loader,
        val_loader,
        prototypes,
        optimizer,
        device,
        cfg,
        scheduler=scheduler,
        scheduler_contract=_scheduler_contract(
            scheduler_cfg,
            len(train_loader),
        ),
        resume_state=resume_state,
        prototype_refresh_loader=prototype_refresh_loader,
        spatial_prototype_refresh_loader=(
            spatial_prototype_refresh_loader
        ),
        expert_classification_val_loader=(
            expert_classification_val_loader
        ),
        expert_spatial_val_loader=expert_spatial_val_loader,
    )
    _probe.timeline_event("startup.fit_returned", force=True)
    print("train_ok " + " ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
