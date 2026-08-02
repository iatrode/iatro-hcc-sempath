"""Download the complete gated SemPath release for local inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from hcc_sempath.release_hub import download_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", choices=("auto", "hf", "modelscope"), default="auto")
    parser.add_argument("--token", help="Optional gated Hugging Face or ModelScope token.")
    parser.add_argument("--cache-dir", type=Path, help="Optional SemPath release-cache root.")
    args = parser.parse_args()
    release_dir = download_release(hub=args.hub, token=args.token, cache_dir=args.cache_dir)
    print(f"SemPath · ready at {release_dir}", flush=True)


if __name__ == "__main__":
    main()
