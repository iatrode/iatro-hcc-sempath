from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an encoder-only HCC-SemPath release checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in payload["model"].items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"checkpoint has no encoder parameters: {checkpoint_path}")

    release_config = {
        "format": "hcc-sempath-encoder",
        "version": 1,
        "model": {
            "backbone_name": config["model"]["backbone_name"],
            "embedding_dim": config["model"]["embedding_dim"],
            "projector_type": config["model"].get("projector_type", "linear"),
            "projector_hidden_dim": config["model"].get("projector_hidden_dim", 2048),
        },
        "preprocessing": {
            "mean": config["data"].get("mean", [0.485, 0.456, 0.406]),
            "std": config["data"].get("std", [0.229, 0.224, 0.225]),
        },
        "output": {
            "name": "z_hcc",
            "normalized": True,
        },
    }
    torch.save(
        {
            "format": release_config["format"],
            "version": release_config["version"],
            "state_dict": encoder_state,
            "config": release_config,
        },
        output_dir / "model.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(release_config, indent=2) + "\n", encoding="utf-8")
    print(f"release_checkpoint_ok output={output_dir / 'model.pt'} parameters={len(encoder_state)}")


if __name__ == "__main__":
    main()
