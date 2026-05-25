from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _append_concepts(
    concept_dir: Path,
    level: int,
    group: str | None,
    names: list[str],
    groups: list[str | None],
    levels: list[int],
    exclusive: list[bool],
    prototypes: list[np.ndarray],
    counts: list[int],
) -> None:
    for subdir in sorted(path for path in concept_dir.iterdir() if path.is_dir()):
        features = [np.load(path).astype(np.float32).reshape(-1) for path in sorted(subdir.glob("*.npy"))]
        if not features:
            continue
        matrix = np.stack(features)
        names.append(subdir.name)
        groups.append(group)
        levels.append(level)
        exclusive.append(level == 1)
        prototypes.append(matrix.mean(axis=0))
        counts.append(len(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic prototypes from per-concept feature arrays.")
    parser.add_argument("--primary-dir", required=True, help="Directory of level-1 mutually exclusive concepts.")
    parser.add_argument("--attribute-dir", default="", help="Optional directory of level-2 non-exclusive concepts.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary-group", default="primary_state")
    parser.add_argument("--attribute-group", default="")
    args = parser.parse_args()
    names, groups, levels, exclusive, prototypes, counts = [], [], [], [], [], []
    _append_concepts(
        concept_dir=Path(args.primary_dir),
        level=1,
        group=args.primary_group or None,
        names=names,
        groups=groups,
        levels=levels,
        exclusive=exclusive,
        prototypes=prototypes,
        counts=counts,
    )
    if args.attribute_dir:
        _append_concepts(
            concept_dir=Path(args.attribute_dir),
            level=2,
            group=args.attribute_group or None,
            names=names,
            groups=groups,
            levels=levels,
            exclusive=exclusive,
            prototypes=prototypes,
            counts=counts,
        )
    if not prototypes:
        raise ValueError("no concept features found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "prototypes": torch.from_numpy(np.stack(prototypes)),
            "names": names,
            "groups": groups,
            "levels": levels,
            "exclusive": exclusive,
            "counts": counts,
            "source": {
                "primary_dir": str(Path(args.primary_dir)),
                "attribute_dir": str(Path(args.attribute_dir)) if args.attribute_dir else "",
                "builder": "hcc-sempath build-prototypes",
            },
        },
        output,
    )
    print(f"prototypes_ok concepts={len(names)} output={output}")


if __name__ == "__main__":
    main()
