"""Resolve gated SemPath releases from local Hugging Face/ModelScope caches."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from hcc_sempath.inference.model import RELEASE_CONFIG_NAME, RELEASE_WEIGHTS_NAME


DEFAULT_HF_REPO = os.getenv("HCC_SEMPATH_HF_REPO", "iatrode/iatro-hcc-sempath")
DEFAULT_MODELSCOPE_REPO = os.getenv(
    "HCC_SEMPATH_MODELSCOPE_REPO",
    "iatrode/iatro-hcc-sempath",
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _country_code() -> str | None:
    try:
        with urlopen("https://ipapi.co/json/", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return str(payload.get("country_code") or "").upper() or None
    except Exception:
        return None


def _choose_hub(hub: str) -> str:
    if hub in {"hf", "modelscope"}:
        return hub
    if hub != "auto":
        raise ValueError(f"unsupported release hub: {hub!r}")
    return "modelscope" if _country_code() == "CN" else "hf"


def _hub_for_token(token: str) -> str:
    if token.startswith("hf_"):
        return "hf"
    if token.startswith("ms-"):
        return "modelscope"
    raise ValueError(
        "unrecognized gated token format; Hugging Face tokens begin with `hf_` "
        "and ModelScope tokens begin with `ms-`"
    )


def _repo(hub: str) -> str:
    return DEFAULT_HF_REPO if hub == "hf" else DEFAULT_MODELSCOPE_REPO


def _release_root(*, hub: str, cache_dir: str | Path | None) -> Path:
    root = Path(cache_dir or Path.home() / ".cache" / "hcc-sempath" / "releases")
    return root / hub / _repo(hub).replace("/", "--")


def _is_release(root: Path) -> bool:
    return (root / RELEASE_CONFIG_NAME).is_file() and (root / RELEASE_WEIGHTS_NAME).is_file()


def resolve_cached_release(
    model_dir: str | Path | None = None,
    *,
    hub: str = "auto",
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve a complete local release without performing network activity."""

    if model_dir is not None:
        root = Path(model_dir).expanduser()
        if not _is_release(root):
            raise FileNotFoundError(
                f"SemPath release is incomplete at {root}; expected "
                f"{RELEASE_CONFIG_NAME} and {RELEASE_WEIGHTS_NAME}"
            )
        return root.resolve()
    if hub not in {"auto", "hf", "modelscope"}:
        raise ValueError(f"unsupported release hub: {hub!r}")
    candidates = (
        [_release_root(hub=hub, cache_dir=cache_dir)]
        if hub != "auto"
        else [_release_root(hub=name, cache_dir=cache_dir) for name in ("hf", "modelscope")]
    )
    available = [candidate for candidate in candidates if _is_release(candidate)]
    if available:
        return available[0].resolve()
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "No downloaded SemPath release was found. Run `hcc-sempath download`, "
        f"or pass a local release with --model. Checked: {expected}"
    )


def _ensure_cli(hub: str) -> str:
    binary = "hf" if hub == "hf" else "modelscope"
    environment_binary = Path(sys.executable).with_name(binary)
    existing = str(environment_binary) if environment_binary.is_file() else shutil.which(binary)
    if existing:
        return existing
    package = "huggingface_hub" if hub == "hf" else "modelscope"
    print(f"SemPath · installing {binary} download client", flush=True)
    _run([sys.executable, "-m", "pip", "install", package])
    existing = str(environment_binary) if environment_binary.is_file() else shutil.which(binary)
    if not existing:
        raise RuntimeError(f"{binary} client installation completed but its executable is unavailable")
    return existing


def download_release(
    *,
    hub: str = "auto",
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    chosen = _hub_for_token(token) if token else _choose_hub(hub)
    if hub != "auto" and chosen != hub:
        raise ValueError(f"token belongs to {chosen}, not requested hub={hub}")
    root = _release_root(hub=chosen, cache_dir=cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cli = _ensure_cli(chosen)
    env_name = "HF_TOKEN" if chosen == "hf" else "MODELSCOPE_API_TOKEN"
    credential = token or os.getenv(env_name)
    if chosen == "hf":
        if credential:
            _run([cli, "auth", "login", "--token", credential])
        command = [cli, "download", _repo(chosen), "--repo-type", "model", "--local-dir", str(root)]
    else:
        if credential:
            _run([cli, "login", "--token", credential])
        command = [cli, "download", "--model", _repo(chosen), "--local_dir", str(root)]
    print(f"SemPath · downloading release through {chosen}", flush=True)
    try:
        _run(command)
    except subprocess.CalledProcessError as error:
        if credential:
            raise
        raise RuntimeError(
            f"Gated {chosen} access is unavailable. Log in with the official client, "
            f"set {env_name}, or rerun `hcc-sempath download --token <token>`."
        ) from error
    if not _is_release(root):
        raise RuntimeError(
            f"{chosen} download completed without {RELEASE_CONFIG_NAME} and "
            f"{RELEASE_WEIGHTS_NAME}: {root}"
        )
    return root.resolve()
