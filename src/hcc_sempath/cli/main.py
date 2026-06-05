from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS: dict[str, tuple[str, str]] = {
    "build-tile-cache": ("hcc_sempath.cli.tile_cache", "Build image-tile IAC packages from a WSI file or directory."),
    "validate-package": ("hcc_sempath.io.validate_package", "Validate one IAC package or a directory of IAC packages."),
    "view-iac": ("hcc_sempath.cli.view_iac", "Open a local browser UI for inspecting an IAC package."),
    "annotate-prototypes": ("hcc_sempath.cli.annotate_prototypes", "Open browser UI for L1/L2 prototype tile annotation."),
    "build-teacher-cache": ("hcc_sempath.teacher.cache", "Run a teacher model and write <teacher-name>.features.iac directly."),
    "build-prototypes": ("hcc_sempath.modeling.build_prototypes", "Build semantic prototypes from concept feature arrays."),
    "build-train-manifest": ("hcc_sempath.training.manifest", "Build a training manifest from per-WSI IAC packages."),
    "train": ("hcc_sempath.training.train", "Train the HCC-SemPath student model."),
    "evaluate": ("hcc_sempath.training.evaluate", "Evaluate a trained checkpoint."),
    "benchmark": ("hcc_sempath.training.benchmark", "Benchmark student encoder throughput."),
}

ALIASES = {
    "teacher-cache": "build-teacher-cache",
    "eval": "evaluate",
}


def _print_help() -> None:
    print("usage: hcc-sempath <command> [options]\n")
    print("commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nUse `hcc-sempath <command> --help` for command-specific options.")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return
    command = ALIASES.get(args[0], args[0])
    if command not in COMMANDS:
        available = ", ".join(COMMANDS)
        raise SystemExit(f"unknown hcc-sempath command: {args[0]}\navailable commands: {available}")
    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    sys.argv = [f"hcc-sempath {command}", *args[1:]]
    module.main()


if __name__ == "__main__":
    main()
