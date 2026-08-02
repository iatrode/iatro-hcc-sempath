import json
from pathlib import Path

import pytest

from hcc_sempath.release_hub import (
    _release_root,
    resolve_cached_release,
)


def _release(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"format": "test"}), encoding="utf-8")
    (path / "hcc_sempath_release.pt").write_bytes(b"weights")
    return path


def test_explicit_release_requires_both_contract_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="incomplete"):
        resolve_cached_release(tmp_path)
    assert resolve_cached_release(_release(tmp_path / "model")) == (tmp_path / "model").resolve()


def test_auto_resolves_existing_hf_cache_without_network(tmp_path: Path) -> None:
    release = _release(_release_root(hub="hf", cache_dir=tmp_path))
    assert resolve_cached_release(hub="auto", cache_dir=tmp_path) == release.resolve()


def test_missing_cache_points_to_download_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hcc-sempath download"):
        resolve_cached_release(hub="auto", cache_dir=tmp_path)
