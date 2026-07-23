from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hcc_sempath.modeling.models import (
    STUDENT_BACKBONE_NAME,
    STUDENT_PATCH_SIZE,
)
from hcc_sempath.training.config import load_config


def _encoder_release_contract(config: dict) -> dict:
    return {
        "model": {
            "embedding_dim": int(config["model"]["embedding_dim"]),
            "projector_type": config["model"].get(
                "projector_type",
                "linear",
            ),
            "projector_hidden_dim": int(
                config["model"].get("projector_hidden_dim", 2048)
            ),
        },
        "preprocessing": {
            "mean": config["data"].get(
                "mean",
                [0.485, 0.456, 0.406],
            ),
            "std": config["data"].get(
                "std",
                [0.229, 0.224, 0.225],
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an encoder-only HCC-SemPath release checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no resolved training config")
    if args.config:
        requested = load_config(Path(args.config))
        if _encoder_release_contract(requested) != _encoder_release_contract(
            config
        ):
            raise ValueError(
                "export config differs from the checkpoint encoder contract"
            )
    contract = payload.get("config", {}).get("research_contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no frozen research contract")
    if (
        contract.get("student_backbone") != STUDENT_BACKBONE_NAME
        or int(contract.get("student_patch_size", -1)) != STUDENT_PATCH_SIZE
    ):
        raise ValueError(
            "checkpoint backbone contract does not match the current "
            f"{STUDENT_BACKBONE_NAME} release"
        )
    model_state = {
        key.removeprefix("_orig_mod."): value
        for key, value in payload["model"].items()
    }
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"checkpoint has no encoder parameters: {checkpoint_path}")

    release_config = {
        "format": "hcc-sempath-encoder",
        "version": 1,
        "model": {
            "backbone_name": STUDENT_BACKBONE_NAME,
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
            "normalized": False,
            "recommended_readout": "L2-normalize the encoder output",
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
