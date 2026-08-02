from __future__ import annotations

import argparse
import time

import torch

from hcc_sempath.inference.model import load_release_model
from hcc_sempath.modeling.models import STUDENT_IMAGE_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end released SemPath model throughput."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Release directory containing config.json and model.safetensors.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("--batch-size and --steps must be positive")

    device = torch.device(args.device)
    release = load_release_model(args.model, device=device)
    batch = torch.randn(
        args.batch_size,
        3,
        STUDENT_IMAGE_SIZE,
        STUDENT_IMAGE_SIZE,
        device=device,
    )
    with torch.inference_mode():
        for _ in range(3):
            release.model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            release.model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tiles_per_sec = args.steps * args.batch_size / elapsed
    print(
        "benchmark_ok "
        f"batch_size={args.batch_size} steps={args.steps} "
        f"tiles_per_sec={tiles_per_sec:.3f}"
    )


if __name__ == "__main__":
    main()
