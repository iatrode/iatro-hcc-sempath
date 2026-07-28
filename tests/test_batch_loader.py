from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from hcc_sempath.training.train import (
    _InMemoryExpertBatchLoader,
    _InterleavedBatchLoader,
    _MaterializedExpertBank,
    _PackageShuffleBatchLoader,
)


class _FakeDataset:
    """Mimics PackageSampledDistillationDataset's scatter-loader interface.

    Each "package" has a fixed number of rows; a row's image encodes its global
    integer id (filled into every pixel) so we can recover exact ordering from
    the emitted tensor. Decode time is jittered to provoke out-of-order finishes
    across worker threads.
    """

    H = W = 4
    TEACHER_DIMS = {"t0": 3, "t1": 5}
    def __init__(self, package_sizes, jitter=True):
        self.package_sizes = list(package_sizes)
        self.total = sum(self.package_sizes)
        self._base = []
        running = 0
        for size in self.package_sizes:
            self._base.append(running)
            running += size
        self.jitter = jitter
        self._lock = threading.Lock()

    def __len__(self):
        return self.total

    def global_id(self, package_idx, row):
        return self._base[package_idx] + int(row)

    def iter_package_row_chunks(self, chunk_size, seed):
        chunk_size = max(1, int(chunk_size))
        plan = []
        for package_idx, size in enumerate(self.package_sizes):
            for start in range(0, size, chunk_size):
                rows = np.arange(start, min(start + chunk_size, size), dtype=np.int64)
                plan.append((package_idx, rows))
        rng = np.random.default_rng(seed)
        rng.shuffle(plan)
        return plan

    def batch_buffer_spec(self):
        return {
            "image_hw": (self.H, self.W),
            "teacher_dims": dict(self.TEACHER_DIMS),
            "spatial_shape": (0, 0, 0),
        }

    def scatter_package_rows(self, package_idx, rows, positions, buffers):
        if self.jitter:
            time.sleep(((package_idx * 7 + int(rows[0])) % 5) * 0.001)
        for k, row in enumerate(rows):
            gid = self.global_id(package_idx, int(row))
            slot = positions[k]
            buf = buffers[slot.buffer_idx]
            buf.images[slot.pos].fill_(gid % 256)
            buf.images[slot.pos, 0, 0, 0] = gid % 256  # low byte; full id tracked via tile_id
            for name, dim in self.TEACHER_DIMS.items():
                buf.teacher_features[name][slot.pos].fill_(float(gid))
            buf.prototype_mask[slot.pos] = bool(gid % 2)
            buf.prototype_classification[slot.pos] = gid
            buf.tile_id[slot.pos] = str(gid)


def _drain_gids(loader):
    """Return list of batches, each a list of global ids (from tile_id)."""
    out = []
    for batch in loader:
        out.append([int(t) for t in batch["tile_id"]])
    return out


def _expected(dataset, chunk_size, seed, batch_size, reshuffle_epoch=0):
    plan = dataset.iter_package_row_chunks(chunk_size, seed)
    flat = [dataset.global_id(p, r) for p, rows in plan for r in rows]
    batches = [flat[i : i + batch_size] for i in range(0, len(flat), batch_size)]
    shuffled = []
    for b, batch in enumerate(batches):
        perm = np.random.default_rng(seed + b).permutation(len(batch))
        # loader writes row `within` to position perm[within]; reading positions
        # in order yields inverse-permuted sequence.
        placed = [None] * len(batch)
        for within, gid in enumerate(batch):
            placed[int(perm[within])] = gid
        shuffled.append(placed)
    return shuffled


def test_expert_batches_are_interleaved_at_a_fixed_population_interval():
    population = [["p0"], ["p1"], ["p2"], ["p3"], ["p4"]]
    expert = [["e0"], ["e1"]]
    loader = _InterleavedBatchLoader(
        population,
        expert,
        interval=2,
    )

    assert list(loader) == [
        ["e0"],
        ["p0"],
        ["p1"],
        ["e1"],
        ["p2"],
        ["p3"],
        ["e0"],
        ["p4"],
    ]
    assert len(loader) == 8


def test_first_expert_batch_does_not_wait_for_population_decode():
    class _SlowPopulationIterator:
        def __init__(self) -> None:
            self._done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            self._done = True
            time.sleep(0.1)
            return ["p0"]

    class _SlowPopulation:
        def __len__(self):
            return 1

        def __iter__(self):
            return _SlowPopulationIterator()

    iterator = iter(
        _InterleavedBatchLoader(
            _SlowPopulation(),
            [["e0"]],
            interval=1,
        )
    )
    start = time.perf_counter()

    assert next(iterator) == ["e0"]
    assert time.perf_counter() - start < 0.05
    assert next(iterator) == ["p0"]


def test_materialized_expert_loader_reuses_complete_bank() -> None:
    size = 4
    zeros = torch.zeros((size, 2, 1, 1), dtype=torch.bool)
    bank = _MaterializedExpertBank(
        [
            {
                "tile_id": [f"tile-{index}" for index in range(size)],
                "images": torch.arange(size).view(size, 1, 1, 1),
                "images_uint8": False,
                "teacher_features": {
                    "teacher": torch.arange(size).view(size, 1).float()
                },
                "prototype_mask": torch.ones(size, dtype=torch.bool),
                "prototype_classification": torch.arange(size),
                "spatial_point_centers": zeros.float(),
                "spatial_brush_bag_ids": zeros.long(),
                "spatial_area_positive": zeros,
                "spatial_explicit_negative": zeros,
                "spatial_implicit_negative": zeros,
                "spatial_supervised": torch.zeros(
                    (size, 2),
                    dtype=torch.bool,
                ),
            }
        ]
    )
    loader = _InMemoryExpertBatchLoader(
        bank,
        indices=None,
        batch_size=2,
        seed=3,
    )
    first_cycle = [
        tile_id
        for batch in loader
        for tile_id in batch["tile_id"]
    ]
    second_cycle = [
        tile_id
        for batch in loader
        for tile_id in batch["tile_id"]
    ]

    assert sorted(first_cycle) == [f"tile-{index}" for index in range(size)]
    assert sorted(second_cycle) == sorted(first_cycle)
    assert first_cycle != second_cycle


@pytest.mark.parametrize("num_workers", [1, 4, 8])
def test_batch_sequence_matches_single_thread(num_workers):
    ds = _FakeDataset([23, 17, 31, 5], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=num_workers, prefetch_batches=2,
        collate_fn=None, seed=99, chunk_size=4, buffer_batches=3,
        reshuffle_each_epoch=False,
    )
    got = _drain_gids(loader)
    expected = _expected(ds, chunk_size=4, seed=99, batch_size=8)
    assert got == expected


def test_reproducible_across_runs_with_workers():
    def run():
        ds = _FakeDataset([23, 17, 31, 5], jitter=True)
        loader = _PackageShuffleBatchLoader(
            ds, batch_size=8, num_workers=8, prefetch_batches=2,
            collate_fn=None, seed=7, chunk_size=4, buffer_batches=2,
            reshuffle_each_epoch=False,
        )
        return _drain_gids(loader)

    assert run() == run()


def test_all_samples_present_exactly_once():
    ds = _FakeDataset([23, 17, 31, 5], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=6, prefetch_batches=2,
        collate_fn=None, seed=3, chunk_size=4, buffer_batches=2,
        reshuffle_each_epoch=False,
    )
    flat = [gid for batch in _drain_gids(loader) for gid in batch]
    assert sorted(flat) == list(range(ds.total))


def test_teacher_and_prototype_payload_correct():
    ds = _FakeDataset([16], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=4, prefetch_batches=2,
        collate_fn=None, seed=5, chunk_size=2, buffer_batches=2,
        reshuffle_each_epoch=False,
    )
    for batch in loader:
        gids = [int(t) for t in batch["tile_id"]]
        for i, gid in enumerate(gids):
            assert torch.allclose(batch["teacher_features"]["t0"][i], torch.full((3,), float(gid)))
            assert torch.allclose(batch["teacher_features"]["t1"][i], torch.full((5,), float(gid)))
            assert int(batch["prototype_classification"][i]) == gid
            assert bool(batch["prototype_mask"][i]) == bool(gid % 2)
        assert batch["images_uint8"] is True
        assert batch["images"].dtype == torch.uint8
        assert batch["images"].shape[1:] == (4, 4, 3)


def test_reshuffle_changes_order_across_epochs():
    ds = _FakeDataset([23, 17, 31, 5], jitter=False)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=4, prefetch_batches=2,
        collate_fn=None, seed=11, chunk_size=4, buffer_batches=2,
        reshuffle_each_epoch=True,
    )
    epoch1 = _drain_gids(loader)
    epoch2 = _drain_gids(loader)
    assert epoch1 != epoch2
    assert sorted(g for b in epoch1 for g in b) == sorted(g for b in epoch2 for g in b)


def test_set_epoch_restores_package_shuffle_order() -> None:
    ds = _FakeDataset([23, 17, 31, 5], jitter=False)
    loader = _PackageShuffleBatchLoader(
        ds,
        batch_size=8,
        num_workers=1,
        prefetch_batches=1,
        collate_fn=None,
        seed=11,
        chunk_size=4,
        buffer_batches=2,
        reshuffle_each_epoch=True,
    )
    first = _drain_gids(loader)
    _drain_gids(loader)
    loader.set_epoch(0)

    assert _drain_gids(loader) == first


def test_early_exit_does_not_hang_or_leak_threads():
    ds = _FakeDataset([50, 50, 50], jitter=True)
    before = threading.active_count()
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=8, prefetch_batches=2,
        collate_fn=None, seed=5, chunk_size=4, buffer_batches=2,
        reshuffle_each_epoch=False,
    )
    it = iter(loader)
    for _ in range(3):
        next(it)
    it.close()
    time.sleep(0.3)
    after = threading.active_count()
    assert after <= before, f"thread leak: before={before} after={after}"


def test_no_thread_growth_across_epochs():
    ds = _FakeDataset([30, 30], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=8, prefetch_batches=2,
        collate_fn=None, seed=1, chunk_size=4, buffer_batches=2,
        reshuffle_each_epoch=True,
    )
    baseline = threading.active_count()
    counts = []
    for _ in range(5):
        _drain_gids(loader)
        time.sleep(0.05)
        counts.append(threading.active_count())
    assert max(counts) <= baseline, f"threads grew: baseline={baseline} counts={counts}"


def test_buffer_reuse_no_corruption_small_ring():
    # prefetch_batches=1 -> n_buffers=2, forcing aggressive ring reuse. Verify
    # no batch gets overwritten before it is consumed (ids stay correct).
    ds = _FakeDataset([40], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=4, num_workers=8, prefetch_batches=1,
        collate_fn=None, seed=2, chunk_size=2, buffer_batches=1,
        reshuffle_each_epoch=False,
    )
    flat = [gid for batch in _drain_gids(loader) for gid in batch]
    assert sorted(flat) == list(range(ds.total))


def test_break_without_close_does_not_leak_threads():
    # Real leak scenario: consumer breaks out of the loop (e.g. max_batches)
    # WITHOUT calling .close(). A new __iter__ must stop the orphaned threads.
    ds = _FakeDataset([60, 60, 60], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=8, prefetch_batches=2,
        collate_fn=None, seed=4, chunk_size=4, buffer_batches=2,
        reshuffle_each_epoch=True,
    )
    baseline = threading.active_count()
    peak = baseline
    for _ in range(6):
        n = 0
        for _batch in loader:          # iterates via a fresh generator each time
            n += 1
            if n >= 2:
                break                  # leave generator un-closed on purpose
        time.sleep(0.2)
        peak = max(peak, threading.active_count())
    time.sleep(0.3)
    after = threading.active_count()
    # Threads from prior un-closed generators must be reclaimed, not accumulated.
    assert peak <= baseline + loader.num_workers + 4, f"thread accumulation: baseline={baseline} peak={peak}"
    assert after <= baseline + 2, f"threads not reclaimed: baseline={baseline} after={after}"


def test_explicit_stop_active_idempotent():
    ds = _FakeDataset([40], jitter=True)
    loader = _PackageShuffleBatchLoader(
        ds, batch_size=8, num_workers=6, prefetch_batches=2,
        collate_fn=None, seed=9, chunk_size=4, buffer_batches=2,
        reshuffle_each_epoch=False,
    )
    it = iter(loader)
    next(it)
    loader._stop_active()
    loader._stop_active()  # idempotent, must not raise
    time.sleep(0.2)
    # A fresh full iteration still works correctly after a forced stop.
    flat = [gid for batch in _drain_gids(loader) for gid in batch]
    assert sorted(flat) == list(range(ds.total))
