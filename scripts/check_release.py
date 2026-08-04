#!/usr/bin/env python3
"""Validate HCC-SemPath source metadata and release distributions."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import compat32
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "hcc-sempath"
RISKY_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".db",
    ".dcm",
    ".iac",
    ".key",
    ".mrxs",
    ".ndpi",
    ".onnx",
    ".p12",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".svs",
    ".zip",
}
PRIVATE_PATTERNS = {
    "macOS user path": re.compile(rb"/Users" rb"/"),
    "mounted-volume path": re.compile(rb"/Volumes" rb"/"),
    "private data marker": re.compile(
        rb"(?i)(?:private" rb"-data|MacData" rb"HD|temp-" rb"wsi)"
    ),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "credential token": re.compile(
        rb"\b(?:gh[pousr]_|github_pat_|hf_|sk-(?:proj-)?|AKIA)"
        rb"[A-Za-z0-9_-]{16,}"
    ),
}


def fail(message: str) -> None:
    raise SystemExit(message)


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def ensure_version_absent_from_pypi(version: str) -> None:
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
    request = Request(url, headers={"User-Agent": "hcc-sempath-release/1"})
    try:
        with urlopen(request, timeout=20) as response:
            json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return
        fail(f"PyPI preflight failed: HTTP {error.code}")
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        fail(f"PyPI preflight failed: {error}")
    fail(f"{PROJECT} {version} already exists on PyPI")


def validate_source(version: str, tag: str | None, check_pypi: bool) -> None:
    expected = Version(version)
    if str(expected) != version or expected.local is not None:
        fail(f"release version must be normalized for PyPI, got {version!r}")
    if project_version() != version:
        fail(f"pyproject version is {project_version()}, expected {version}")

    if tag is not None:
        expected_tag = f"v{version}"
        if tag != expected_tag:
            fail(f"release tag must be {expected_tag}, got {tag}")
        try:
            tag_commit = subprocess.check_output(
                ["git", "rev-parse", f"{tag}^{{commit}}"], cwd=ROOT, text=True
            ).strip()
            head_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except subprocess.CalledProcessError as error:
            fail(f"cannot resolve release tag {tag}: {error}")
        if tag_commit != head_commit:
            fail(f"HEAD {head_commit} is not release tag {tag} ({tag_commit})")

    if check_pypi:
        ensure_version_absent_from_pypi(version)
    print(f"source release metadata is synchronized at {version}")


def wheel_entries(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        return [(name, archive.read(name)) for name in archive.namelist()]


def sdist_entries(path: Path) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                fail(f"cannot read {member.name} from {path.name}")
            entries.append((member.name, stream.read()))
    return entries


def validate_archive_privacy(path: Path, entries: list[tuple[str, bytes]]) -> None:
    risky = [name for name, _ in entries if Path(name).suffix.lower() in RISKY_SUFFIXES]
    if risky:
        fail(f"private or binary asset included in {path.name}: {risky[0]}")
    for name, payload in entries:
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(payload):
                fail(f"{label} found in {path.name}:{name}")


def metadata_from_entries(
    path: Path, entries: list[tuple[str, bytes]]
) -> bytes:
    if path.suffix == ".whl":
        matches = [
            payload for name, payload in entries if name.endswith(".dist-info/METADATA")
        ]
    else:
        matches = [
            payload
            for name, payload in entries
            if len(Path(name).parts) == 2 and Path(name).name == "PKG-INFO"
        ]
    if len(matches) != 1:
        fail(f"expected one metadata file in {path.name}")
    return matches[0]


def validate_metadata(path: Path, raw: bytes, version: str) -> None:
    metadata = BytesParser(policy=compat32).parsebytes(raw)
    if canonicalize_name(metadata["Name"]) != canonicalize_name(PROJECT):
        fail(f"unexpected project name in {path.name}: {metadata['Name']}")
    if metadata["Version"] != version:
        fail(f"unexpected version in {path.name}: {metadata['Version']}")


def validate_dist(version: str, directory: Path) -> None:
    expected = Version(version)
    files = sorted(path for path in directory.iterdir() if path.is_file())
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    unexpected = [path.name for path in files if path not in wheels and path not in sdists]
    if unexpected or len(wheels) != 1 or len(sdists) != 1:
        fail(
            "expected exactly one wheel and one sdist; "
            f"found wheels={len(wheels)}, sdists={len(sdists)}, unexpected={unexpected}"
        )

    wheel = wheels[0]
    name, wheel_version, _build, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != canonicalize_name(PROJECT) or wheel_version != expected:
        fail(f"unexpected wheel identity: {wheel.name}")
    if {(tag.interpreter, tag.abi, tag.platform) for tag in tags} != {
        ("py3", "none", "any")
    }:
        fail(f"wheel is not the expected py3-none-any distribution: {wheel.name}")
    wheel_content = wheel_entries(wheel)
    validate_archive_privacy(wheel, wheel_content)
    validate_metadata(wheel, metadata_from_entries(wheel, wheel_content), version)
    entry_points = [
        payload
        for name, payload in wheel_content
        if name.endswith(".dist-info/entry_points.txt")
    ]
    if len(entry_points) != 1 or (
        b"hcc-sempath = hcc_sempath.cli.main:main" not in entry_points[0]
    ):
        fail("wheel does not expose the hcc-sempath console script")

    sdist = sdists[0]
    name, sdist_version = parse_sdist_filename(sdist.name)
    if canonicalize_name(name) != canonicalize_name(PROJECT) or sdist_version != expected:
        fail(f"unexpected sdist identity: {sdist.name}")
    sdist_content = sdist_entries(sdist)
    validate_archive_privacy(sdist, sdist_content)
    validate_metadata(sdist, metadata_from_entries(sdist, sdist_content), version)
    print(f"verified complete {version} release set with two distributions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version")
    source = subparsers.add_parser("source")
    source.add_argument("--version", required=True)
    source.add_argument("--tag")
    source.add_argument("--check-pypi", action="store_true")
    dist = subparsers.add_parser("dist")
    dist.add_argument("--version", required=True)
    dist.add_argument("--directory", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "version":
        print(project_version())
    elif args.command == "source":
        validate_source(args.version, args.tag, args.check_pypi)
    else:
        if not args.directory.is_dir():
            fail(f"distribution directory does not exist: {args.directory}")
        validate_dist(args.version, args.directory)


if __name__ == "__main__":
    main()
