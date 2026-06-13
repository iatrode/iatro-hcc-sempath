from __future__ import annotations

import argparse
import weakref
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Thread
import torch
from torch.utils.data import DataLoader
import numpy as np

from . import _pipeline_probe as _probe

from ..io.tile_package import read_package_metadata
from ..modeling.prototypes import PrototypeRegistry, load_prototype_registry
from ..modeling.models import HCCSemPathModel
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
from .engine import build_lr_scheduler, fit
from .manifest import load_training_manifest
from .prototype_images import PrototypeImageBank, load_prototype_image_bank
from .prototype_labels import load_prototype_labels
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
    prototype_level1: torch.Tensor               # (B,) long
    prototype_level2: torch.Tensor               # (B, L2) float32 (L2 may be 0)
    tile_id: list = field(default_factory=list)  # (B,) str

    def view(self, count: int) -> dict:
        """Expose the filled prefix as the dict the engine expects."""
        return {
            "tile_id": list(self.tile_id[:count]),
            "images": self.images[:count],
            "images_uint8": True,
            "images_hwc": True,
            "teacher_features": {name: feat[:count] for name, feat in self.teacher_features.items()},
            "prototype_mask": self.prototype_mask[:count],
            "prototype_level1": self.prototype_level1[:count],
            "prototype_level2": self.prototype_level2[:count],
        }


def _alloc_batch_buffer(batch_size: int, spec: dict, pin: bool) -> BatchBuffer:
    h, w = spec["image_hw"]
    kw = {"pin_memory": True} if pin else {}
    return BatchBuffer(
        images=torch.empty((batch_size, h, w, 3), dtype=torch.uint8, **kw),
        teacher_features={
            name: torch.empty((batch_size, dim), dtype=torch.float32, **kw)
            for name, dim in spec["teacher_dims"].items()
        },
        prototype_mask=torch.zeros((batch_size,), dtype=torch.bool),
        prototype_level1=torch.full((batch_size,), -1, dtype=torch.long),
        prototype_level2=torch.zeros((batch_size, int(spec["level2_dim"])), dtype=torch.float32, **kw),
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

    def __iter__(self):
        epoch_seed = self.seed + self._epoch if self.reshuffle_each_epoch else self.seed
        if self.reshuffle_each_epoch:
            self._epoch += 1
        chunks = self.dataset.iter_global_index_chunks(self.chunk_size, epoch_seed)
        if not chunks:
            return
        flat = np.concatenate(chunks)
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
        num_tasks = len(tasks)
        if num_tasks == 0:
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


def _load_prototype_map(cfg: dict, dims: dict[str, int], device: torch.device) -> dict[str, PrototypeRegistry] | None:
    semantic_weight = float(cfg["loss"].get("semantic_weight", 0.0))
    prototype_filter_weight = float(cfg["loss"].get("prototype_filter_weight", 0.0))
    zhcc_proto_weight = float(cfg["loss"].get("zhcc_proto_weight", 0.0))
    if semantic_weight == 0 and prototype_filter_weight == 0 and zhcc_proto_weight == 0:
        return None
    prototype_paths = cfg["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        return {name: load_prototype_registry(prototype_paths[name], expected_dim=dim).to(device) for name, dim in dims.items()}
    prototype_path = cfg["data"].get("prototype_path")
    if prototype_path is None:
        raise ValueError(
            "data.prototype_path or data.prototype_paths is required when semantic_weight, "
            "prototype_filter_weight, or zhcc_proto_weight > 0"
        )
    return {name: load_prototype_registry(prototype_path, expected_dim=dim).to(device) for name, dim in dims.items()}


def _load_zhcc_prototypes(cfg: dict, device: torch.device) -> PrototypeRegistry | None:
    prototype_path = cfg["data"].get("zhcc_prototype_path")
    if prototype_path is None:
        if float(cfg["loss"].get("zhcc_response_weight", 0.0)) > 0 and not cfg["data"].get("zhcc_prototype_image_path"):
            raise ValueError(
                "data.zhcc_prototype_path or data.zhcc_prototype_image_path is required when loss.zhcc_response_weight > 0"
            )
        return None
    return load_prototype_registry(prototype_path, expected_dim=embedding_dim(cfg)).to(device)


def _load_zhcc_image_bank(cfg: dict) -> PrototypeImageBank | None:
    image_path = cfg["data"].get("zhcc_prototype_image_path")
    if image_path is None:
        if float(cfg["loss"].get("zhcc_proto_weight", 0.0)) > 0:
            raise ValueError(
                "data.zhcc_prototype_image_path is required when loss.zhcc_proto_weight > 0"
            )
        return None
    return load_prototype_image_bank(image_path)


def _label_contract_registry(
    *,
    cfg: dict,
    prototypes: dict[str, PrototypeRegistry] | None,
    zhcc_prototypes: PrototypeRegistry | None,
    zhcc_image_bank: PrototypeImageBank | None,
) -> PrototypeRegistry | None:
    if zhcc_prototypes is not None:
        return zhcc_prototypes.to("cpu")
    if zhcc_image_bank is not None:
        return zhcc_image_bank.label_contract(embedding_dim(cfg))
    if prototypes:
        return next(iter(prototypes.values())).to("cpu")
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCC-SemPath distillation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
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
        train_teacher_packages = _teacher_paths_from_data(cfg, "train_teacher_feature_package_paths")
        val_teacher_packages = _teacher_paths_from_data(cfg, "val_teacher_feature_package_paths")
        names = list(train_teacher_packages)
    elif manifest_path:
        manifest = load_training_manifest(manifest_path)
        train_tile_packages, train_teacher_packages = manifest_data_paths(cfg, manifest, "train")
        val_tile_packages, val_teacher_packages = manifest_data_paths(cfg, manifest, "val")
        names = teacher_names(cfg)
    else:
        tile_packages = image_tile_package_paths(cfg)
        teacher_packages = teacher_feature_package_paths(cfg)
        train_tile_packages = tile_packages
        val_tile_packages = tile_packages
        train_teacher_packages = teacher_packages
        val_teacher_packages = teacher_packages
        names = list(teacher_packages)
    validate_training_config(cfg, names)
    dims = teacher_dims(cfg, names)
    prototypes = _load_prototype_map(cfg, dims, device)
    zhcc_prototypes = _load_zhcc_prototypes(cfg, device)
    zhcc_image_bank = _load_zhcc_image_bank(cfg)
    label_contract = _label_contract_registry(
        cfg=cfg,
        prototypes=prototypes,
        zhcc_prototypes=zhcc_prototypes,
        zhcc_image_bank=zhcc_image_bank,
    )
    prototype_manifest_path = cfg["data"].get("prototype_supervision_manifest_path")
    prototype_label_required = (
        float(cfg["loss"].get("prototype_filter_weight", 0.0)) > 0
        and float(cfg["loss"].get("prototype_label_weight", 0.4)) > 0
    )
    if prototype_label_required and prototype_manifest_path is None:
        raise ValueError(
            "data.prototype_supervision_manifest_path is required when prototype-label adjudication is enabled"
        )
    train_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        label_contract,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_train_splits", ["train"]),
    )
    val_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        label_contract,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_val_splits", ["val"]),
    )
    all_tile_packages = sorted(set(train_tile_packages + val_tile_packages))
    tile_metadata = read_package_metadata(all_tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    for package_path in all_tile_packages[1:]:
        metadata = read_package_metadata(package_path)
        candidate_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))
        if candidate_size != image_size:
            raise ValueError(f"tile package size mismatch: {package_path} has {candidate_size}, expected {image_size}")
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
        )
        val_ds = PackageSampledDistillationDataset(
            val_tile_packages,
            val_teacher_packages,
            **common_dataset_kwargs,
            max_records=int(cfg["data"].get("max_val_records", 0)),
            seed=int(cfg["runtime"]["seed"]) + 1,
            expected_dims=dims,
            prototype_labels=val_prototype_labels,
        )
    elif manifest_path or explicit_split_packages:
        train_records = read_packaged_tile_records(train_tile_packages)
        val_records = read_packaged_tile_records(val_tile_packages)
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
        )
        val_ds = DistillationTileDataset(
            val_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=val_teacher_packages,
            prototype_labels=val_prototype_labels,
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
        )
        val_ds = DistillationTileDataset(
            val_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=val_teacher_packages,
            prototype_labels=val_prototype_labels,
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
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=cfg["model"]["pretrained"],
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=bool(cfg["model"].get("grad_checkpointing", False)),
    ).to(device)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
    if bool(cfg["train"].get("compile", False)):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if resume_state and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
    scheduler = build_lr_scheduler(optimizer, cfg, len(train_loader))
    if resume_state and scheduler is not None and resume_state.get("scheduler") is not None:
        scheduler.load_state_dict(resume_state["scheduler"])
    metrics = fit(
        model,
        train_loader,
        val_loader,
        prototypes,
        optimizer,
        device,
        cfg,
        scheduler=scheduler,
        zhcc_prototypes=zhcc_prototypes,
        zhcc_image_bank=zhcc_image_bank,
        resume_state=resume_state,
    )
    print("train_ok " + " ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
