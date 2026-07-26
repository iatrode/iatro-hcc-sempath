"""Lightweight, env-gated data-pipeline profiler.

Zero cost when ``HCC_PIPELINE_PROBE`` is unset: ``section()`` returns a shared
no-op context manager and the sampler thread never starts. When set, it
accumulates per-section wall time across all worker threads and (optionally)
samples Linux thread run-states from /proc to reveal GIL serialization.

Enable with::

    HCC_PIPELINE_PROBE=1            # section timing only
    HCC_PIPELINE_PROBE=1 HCC_PIPELINE_PROBE_PROC=1   # + /proc thread-state sampling

Report is printed by the loader every ``report()`` call (per N batches).
"""
from __future__ import annotations

import os
import tempfile
import threading
from collections import defaultdict
from time import perf_counter

ON = bool(os.environ.get("HCC_PIPELINE_PROBE"))
PROC_ON = bool(os.environ.get("HCC_PIPELINE_PROBE_PROC"))
LOG_PATH = os.environ.get("HCC_PIPELINE_PROBE_LOG", os.path.join(tempfile.gettempdir(), "hcc_pipeline_probe.log"))

_CLK_TCK = 100.0
try:
    _CLK_TCK = float(os.sysconf("SC_CLK_TCK"))
except (ValueError, OSError, AttributeError):
    pass


def _avail_cores() -> tuple[int, float]:
    """(affinity core count, cgroup-quota cores). 0/-1 when unknown."""
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = os.cpu_count() or 0
    quota = -1.0
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as fh:
            q, p = fh.read().split()
        if q != "max":
            quota = int(q) / int(p)
    except (OSError, ValueError):
        pass
    return aff, quota


def _proc_cpu_jiffies() -> float:
    """Process-wide CPU jiffies (utime+stime over all threads) from /proc."""
    try:
        with open("/proc/self/stat", "rb") as fh:
            data = fh.read()
        rparen = data.rfind(b")")
        fields = data[rparen + 2:].split()
        # after (comm), index 11=utime, 12=stime (0-based in this slice)
        return (int(fields[11]) + int(fields[12])) / _CLK_TCK
    except (OSError, ValueError, IndexError):
        return -1.0


# Config snapshot captured once by the loader; printed in every report.
_config: dict = {}


def set_config(**kw) -> None:
    if not ON:
        return
    _config.update({k: v for k, v in kw.items() if v is not None})


# section name -> [total_seconds, call_count]; merged from thread-local under _lock.
_totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
_lock = threading.Lock()
_tls = threading.local()


def _local() -> dict:
    d = getattr(_tls, "acc", None)
    if d is None:
        d = defaultdict(lambda: [0.0, 0])
        _tls.acc = d
    return d


class _Timer:
    __slots__ = ("name", "t0")

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        dt = perf_counter() - self.t0
        rec = _local()[self.name]
        rec[0] += dt
        rec[1] += 1
        return False


class _Noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NOOP = _Noop()


def section(name: str):
    """Context manager timing a named hot-path section (no-op when OFF)."""
    if not ON:
        return _NOOP
    return _Timer(name)


def add(name: str, seconds: float) -> None:
    """Record a pre-measured duration (for spots where a CM is awkward)."""
    if not ON:
        return
    rec = _local()[name]
    rec[0] += seconds
    rec[1] += 1


def flush_thread() -> None:
    """Merge this thread's local accumulators into the global totals."""
    if not ON:
        return
    d = getattr(_tls, "acc", None)
    if not d:
        return
    with _lock:
        for name, (sec, cnt) in d.items():
            g = _totals[name]
            g[0] += sec
            g[1] += cnt
    d.clear()


# ---- /proc thread run-state sampling (Linux only) ----
_sampler: dict | None = None


def _proc_thread_states() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    try:
        task_dir = "/proc/self/task"
        for tid in os.listdir(task_dir):
            try:
                with open(f"{task_dir}/{tid}/stat", "rb") as fh:
                    data = fh.read()
                # state is the field after the (comm) paren group.
                rparen = data.rfind(b")")
                state = data[rparen + 2:rparen + 3].decode("ascii", "replace")
                counts[state] += 1
            except (FileNotFoundError, ProcessLookupError, OSError):
                continue
    except OSError:
        pass
    return dict(counts)


def start_proc_sampler(interval: float = 0.2) -> None:
    """Background thread tallying R/S/D thread-state histograms + process CPU%."""
    global _sampler
    if not (ON and PROC_ON) or _sampler is not None:
        return
    state = {
        "stop": threading.Event(),
        "samples": 0,
        "hist": defaultdict(int),   # state char -> summed count across samples
        "cpu_pct_sum": 0.0,         # summed instantaneous process CPU% across samples
        "cpu_pct_peak": 0.0,
        "last_jiffies": _proc_cpu_jiffies(),
        "last_wall": perf_counter(),
        "lock": threading.Lock(),
    }

    def loop():
        while not state["stop"].wait(interval):
            snap = _proc_thread_states()
            now = perf_counter()
            j = _proc_cpu_jiffies()
            with state["lock"]:
                state["samples"] += 1
                for k, v in snap.items():
                    state["hist"][k] += v
                dw = now - state["last_wall"]
                dj = j - state["last_jiffies"]
                if dw > 0 and dj >= 0 and state["last_jiffies"] >= 0:
                    pct = 100.0 * dj / dw
                    state["cpu_pct_sum"] += pct
                    if pct > state["cpu_pct_peak"]:
                        state["cpu_pct_peak"] = pct
                state["last_jiffies"] = j
                state["last_wall"] = now

    t = threading.Thread(target=loop, name="probe-proc-sampler", daemon=True)
    state["thread"] = t
    _sampler = state
    t.start()


def _emit(text: str) -> None:
    print(text, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except OSError:
        pass


def report(tag: str = "") -> None:
    """Print + log accumulated section timings, CPU%, cores, config; then reset."""
    if not ON:
        return
    aff, quota = _avail_cores()
    with _lock:
        items = sorted(_totals.items(), key=lambda kv: kv[1][0], reverse=True)
        lines = [f"[pipeline-probe] {tag}".rstrip()]
        cfgline = "  config: " + "  ".join(f"{k}={v}" for k, v in sorted(_config.items()))
        if _config:
            lines.append(cfgline)
        lines.append(f"  cores: affinity={aff}  cgroup_quota={quota:.1f}")
        for name, (sec, cnt) in items:
            per = (sec / cnt * 1e3) if cnt else 0.0
            lines.append(f"  {name:<20} total={sec:8.3f}s  calls={cnt:>8}  per={per:7.3f}ms")
        _totals.clear()
    if _sampler is not None:
        with _sampler["lock"]:
            n = _sampler["samples"] or 1
            avg = {k: v / n for k, v in sorted(_sampler["hist"].items())}
            cpu_avg = _sampler["cpu_pct_sum"] / n
            cpu_peak = _sampler["cpu_pct_peak"]
            _sampler["samples"] = 0
            _sampler["hist"].clear()
            _sampler["cpu_pct_sum"] = 0.0
            _sampler["cpu_pct_peak"] = 0.0
        states = "  ".join(f"{k}={v:.1f}" for k, v in avg.items())
        lines.append(f"  proc-cpu: avg={cpu_avg:.0f}%  peak={cpu_peak:.0f}%  (affinity={aff} cores -> {aff*100} %=full)")
        lines.append(f"  proc-thread-states(avg/sample): {states}")
    _emit("\n".join(lines))
