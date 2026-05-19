from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download H-optimus-1 assets from Hugging Face.")
    parser.add_argument("--repo-id", default="bioptimus/H-optimus-1")
    parser.add_argument("--local-dir", default="weights/H-optimus-1")
    args = parser.parse_args()
    snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
    print(f"download_ok repo_id={args.repo_id} local_dir={args.local_dir}")


if __name__ == "__main__":
    main()

