from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic prototypes from per-concept feature arrays.")
    parser.add_argument("--concept-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names, prototypes, counts = [], [], []
    for subdir in sorted(path for path in Path(args.concept_dir).iterdir() if path.is_dir()):
        features = [np.load(path).astype(np.float32).reshape(-1) for path in sorted(subdir.glob("*.npy"))]
        if not features:
            continue
        matrix = np.stack(features)
        names.append(subdir.name)
        prototypes.append(matrix.mean(axis=0))
        counts.append(len(features))
    if not prototypes:
        raise ValueError("no concept features found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": torch.from_numpy(np.stack(prototypes)), "names": names, "counts": counts}, output)
    print(f"prototypes_ok concepts={len(names)} output={output}")


if __name__ == "__main__":
    main()
