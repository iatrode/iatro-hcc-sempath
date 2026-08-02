from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    module: str
    description: str


BUILD_COMMANDS: dict[str, Command] = {
    "tiles": Command(
        "hcc_sempath.build.tiles",
        "Build image-tile IAC packages from a WSI file or directory.",
    ),
    "teacher-features": Command(
        "hcc_sempath.teacher.cache",
        "Build one teacher-feature IAC stream.",
    ),
    "training-cache": Command(
        "hcc_sempath.build.training_cache",
        "Merge teacher features and align shuffled training rows.",
    ),
    "manifest": Command(
        "hcc_sempath.training.manifest",
        "Build a training dataset manifest from tile and teacher assets.",
    ),
    "supervision": Command(
        "hcc_sempath.build.supervision",
        "Build prototype registries and supervision manifests from annotations.",
    ),
}

COMMANDS: dict[str, Command] = {
    "annotate": Command(
        "hcc_sempath.annotation.server",
        "Open the classification and spatial annotation UI.",
    ),
    "train": Command("hcc_sempath.training.train", "Train the HCC-SemPath student model."),
    "evaluate": Command("hcc_sempath.training.evaluate", "Evaluate a trained checkpoint."),
    "export": Command(
        "hcc_sempath.deployment.export",
        "Export a training checkpoint as a standalone released model.",
    ),
    "infer": Command(
        "hcc_sempath.inference.run",
        "Run a released model and write reconstructable prediction IAC packages.",
    ),
    "benchmark": Command(
        "hcc_sempath.inference.benchmark",
        "Benchmark released-model inference throughput.",
    ),
}


def _print_help() -> None:
    print("usage: hcc-sempath <command> [options]\n")
    print("commands:")
    commands = {
        "build": "Build reusable data and teacher assets.",
        **{name: command.description for name, command in COMMANDS.items()},
    }
    width = max(len(name) for name in commands)
    for name, description in commands.items():
        print(f"  {name:<{width}}  {description}")
    print("\nUse `hcc-sempath <command> --help` for command-specific options.")


def _print_build_help() -> None:
    print("usage: hcc-sempath build <asset> [options]\n")
    print("assets:")
    width = max(len(name) for name in BUILD_COMMANDS)
    for name, command in BUILD_COMMANDS.items():
        print(f"  {name:<{width}}  {command.description}")
    print("\nUse `hcc-sempath build <asset> --help` for asset-specific options.")


def _run(command: Command, program: str, args: list[str]) -> None:
    module = importlib.import_module(command.module)
    sys.argv = [program, *args]
    module.main()


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return
    command_name = args.pop(0)
    if command_name == "build":
        if not args or args[0] in {"-h", "--help"}:
            _print_build_help()
            return
        asset_name = args.pop(0)
        command = BUILD_COMMANDS.get(asset_name)
        if command is None:
            available = ", ".join(BUILD_COMMANDS)
            raise SystemExit(f"unknown build asset: {asset_name}\navailable assets: {available}")
        _run(command, f"hcc-sempath build {asset_name}", args)
        return
    command = COMMANDS.get(command_name)
    if command is None:
        available = ", ".join(("build", *COMMANDS))
        raise SystemExit(f"unknown hcc-sempath command: {command_name}\navailable commands: {available}")
    _run(command, f"hcc-sempath {command_name}", args)


if __name__ == "__main__":
    main()
