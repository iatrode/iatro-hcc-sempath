from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .models import StudentEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark student encoder throughput.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg["runtime"]["device"])
    model = StudentEncoder(cfg["model"]["backbone_name"], cfg["model"]["teacher_dim"], cfg["model"]["pretrained"]).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    batch = torch.randn(cfg["train"]["batch_size"], 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=device)
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tiles_per_sec = args.steps * batch.shape[0] / elapsed
    print(f"benchmark_ok tiles_per_sec={tiles_per_sec:.3f}")


if __name__ == "__main__":
    main()

