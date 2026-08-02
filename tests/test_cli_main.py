from __future__ import annotations

from types import SimpleNamespace

import pytest

from hcc_sempath.cli import main as cli_main


def test_top_level_help_exposes_workflows_not_legacy_commands(capsys) -> None:
    cli_main.main([])

    output = capsys.readouterr().out
    for command in ("build", "annotate", "train", "evaluate", "export", "infer", "benchmark"):
        assert command in output
    for legacy in (
        "build-tile-cache",
        "build-teacher-cache",
        "build-train-manifest",
        "annotate-prototypes",
        "build-priority-list",
        "build-roi-queue",
    ):
        assert legacy not in output


def test_build_help_lists_public_assets(capsys) -> None:
    cli_main.main(["build", "--help"])

    output = capsys.readouterr().out
    assert "hcc-sempath build <asset>" in output
    assert "tiles" in output
    assert "teacher-features" in output
    assert "training-cache" in output
    assert "manifest" in output
    assert "supervision" in output


@pytest.mark.parametrize(
    ("argv", "module_name", "program"),
    [
        (["build", "tiles", "--help"], "hcc_sempath.build.tiles", "hcc-sempath build tiles"),
        (["build", "teacher-features", "--help"], "hcc_sempath.teacher.cache", "hcc-sempath build teacher-features"),
        (["build", "training-cache", "--help"], "hcc_sempath.build.training_cache", "hcc-sempath build training-cache"),
        (["build", "manifest", "--help"], "hcc_sempath.training.manifest", "hcc-sempath build manifest"),
        (["build", "supervision", "--help"], "hcc_sempath.build.supervision", "hcc-sempath build supervision"),
        (["annotate", "--help"], "hcc_sempath.annotation.server", "hcc-sempath annotate"),
        (["train", "--help"], "hcc_sempath.training.train", "hcc-sempath train"),
        (["evaluate", "--help"], "hcc_sempath.training.evaluate", "hcc-sempath evaluate"),
        (["export", "--help"], "hcc_sempath.deployment.export", "hcc-sempath export"),
        (["infer", "--help"], "hcc_sempath.inference.run", "hcc-sempath infer"),
        (["benchmark", "--help"], "hcc_sempath.inference.benchmark", "hcc-sempath benchmark"),
    ],
)
def test_routes_public_command(monkeypatch, argv, module_name, program) -> None:
    calls: list[tuple[str, list[str]]] = []
    module = SimpleNamespace(main=lambda: calls.append((module_name, list(cli_main.sys.argv))))
    monkeypatch.setattr(cli_main.importlib, "import_module", lambda name: module if name == module_name else None)

    cli_main.main(argv)

    assert calls == [(module_name, [program, "--help"])]


@pytest.mark.parametrize(
    "legacy",
    [
        "build-tile-cache",
        "build-teacher-cache",
        "build-train-manifest",
        "annotate-prototypes",
        "build-priority-list",
        "build-roi-queue",
        "teacher-cache",
        "eval",
    ],
)
def test_legacy_commands_are_removed(legacy) -> None:
    with pytest.raises(SystemExit, match="unknown hcc-sempath command"):
        cli_main.main([legacy])
