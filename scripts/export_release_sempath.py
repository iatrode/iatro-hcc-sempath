import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add src/ to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.prototype_images import load_prototype_image_bank, build_student_prototype_registry

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        import yaml
        return yaml.safe_load(handle) or {}

def main() -> None:
    parser = argparse.ArgumentParser(description="Export a self-contained calibrated HCC-SemPath release checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt")
    parser.add_argument("--config", default="artifacts/models/hcc-sempath-full/resolved_config.json")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--l2-npz", default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_l2_probabilities.npz")
    parser.add_argument(
        "--l2-thresholds",
        default="artifacts/caches/local_cache/train_l2_thresholds/thresholds.json",
    )
    parser.add_argument("--output-dir", default="artifacts/release")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    l2_npz_path = Path(args.l2_npz)
    l2_threshold_path = Path(args.l2_thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading config and model weights...")
    cfg = _load_config(config_path)
    device = torch.device("cpu")

    # Load Model
    names = teacher_names(cfg)
    dims = teacher_dims(cfg, names)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=False,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    # Split state_dict
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in model.state_dict().items()
        if key.startswith("encoder.")
    }
    projector_state = {
        key.removeprefix("projector."): value
        for key, value in model.state_dict().items()
        if key.startswith("projector.")
    }

    # Build dynamic prototypes registry
    proto_dir = Path(args.prototype_dir)
    image_bank_path = proto_dir / "zhcc_hcc_prototype_images.pt"
    print("Loading prototype image bank...")
    image_bank = load_prototype_image_bank(image_bank_path)
    print("Building student prototype registry...")
    zhcc_registry = build_student_prototype_registry(
        model=model,
        image_bank=image_bank,
        cfg=cfg,
        device=device,
        batch_size=256
    )

    # Extract L1 and L2 prototype embeddings
    l1_prototypes = zhcc_registry.primary_prototypes.detach().cpu() # shape (4, embedding_dim)
    l2_prototypes = zhcc_registry.attribute_prototypes.detach().cpu() # shape (10, embedding_dim)
    l1_names = [zhcc_registry.names[idx] for idx in zhcc_registry.primary_indices]
    l2_names = [zhcc_registry.names[idx] for idx in zhcc_registry.attribute_indices]

    # Load the same robust calibration used by the review/retrieval export.
    print("Loading post-hoc L2 retrieval calibration...")
    if l2_npz_path.exists():
        npz_data = np.load(l2_npz_path)
        if "bias_pred_full" in npz_data and "temperature_pred_full" in npz_data:
            l2_biases = npz_data["bias_pred_full"]
            l2_temperatures = npz_data["temperature_pred_full"]
        else:
            probs_raw = npz_data["pred_full"]
            eps = 1e-6
            cos_sims = np.log(np.clip(probs_raw, eps, 1.0 - eps) / np.clip(1.0 - probs_raw, eps, 1.0)) * 0.1
            l2_biases = np.median(cos_sims, axis=0)
            q25, q75 = np.quantile(cos_sims, [0.25, 0.75], axis=0)
            l2_temperatures = np.maximum((q75 - q25) / (2.0 * np.log(3.0)), 1e-4)
        print(f"L2 biases: {l2_biases}")
        print(f"L2 temperatures: {l2_temperatures}")
    else:
        l2_biases = np.zeros(10)
        l2_temperatures = np.full(10, 0.1)
        print("Warning: L2 NPZ file not found. Setting calibration biases to zero.")

    l2_biases_tensor = torch.tensor(l2_biases, dtype=torch.float32)
    l2_temperatures_tensor = torch.tensor(l2_temperatures, dtype=torch.float32)

    # Instantiate the unified deployable classifier model
    print("Assembling the unified HCCSemPathModel...")
    deploy_model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims={},
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        l1_num_classes=len(l1_names),
        l2_num_attributes=len(l2_names),
    )

    # Load states
    deploy_model.encoder.load_state_dict(encoder_state)

    # Load pre-computed prototype matrices and calibration buffers
    deploy_model.l1_prototypes.copy_(l1_prototypes)
    deploy_model.l2_prototypes.copy_(l2_prototypes)
    deploy_model.l2_biases.copy_(l2_biases_tensor)
    
    l1_temp = float(cfg["loss"].get("zhcc_primary_temperature", 0.1))
    
    deploy_model.l1_temperature.copy_(torch.tensor(l1_temp))
    deploy_model.l2_temperature.copy_(l2_temperatures_tensor)

    threshold_payload = json.loads(l2_threshold_path.read_text(encoding="utf-8"))
    if threshold_payload["l2_names"] != l2_names:
        raise ValueError("L2 threshold names do not match release prototype names")

    release_config = {
        "format": "hcc-sempath-classifier-state-dict",
        "version": 1,
        "model": {
            "backbone_name": cfg["model"]["backbone_name"],
            "embedding_dim": cfg["model"]["embedding_dim"],
            "projector_type": cfg["model"].get("projector_type", "linear"),
            "projector_hidden_dim": cfg["model"].get("projector_hidden_dim", 2048),
            "l1_num_classes": len(l1_names),
            "l2_num_attributes": len(l2_names),
        },
        "preprocessing": {
            "mean": cfg["data"].get("mean", [0.485, 0.456, 0.406]),
            "std": cfg["data"].get("std", [0.229, 0.224, 0.225]),
        },
        "l1_names": l1_names,
        "l2_names": l2_names,
        "l1_temperature": l1_temp,
        "l2_temperature": l2_temperatures.tolist(),
        "l2_score_kind": "median_iqr_relative_retrieval_score_v1",
        "l2_decision_score": "raw_cosine",
        "l2_decision_thresholds": threshold_payload["thresholds"],
        "l2_threshold_target_prevalence": threshold_payload["target_prevalence"],
        "l2_threshold_method": threshold_payload["method"],
        "l2_threshold_training_tiles": threshold_payload["cached_tiles"],
        "l2_threshold_prototype_samples": 3000,
    }

    # Save the complete model state dict
    output_pt = output_dir / "hcc_sempath_release.pt"
    torch.save(deploy_model.state_dict(), output_pt)
    
    # Save config separately
    (output_dir / "config.json").write_text(json.dumps(release_config, indent=2) + "\n", encoding="utf-8")
    print(f"Successfully exported complete release checkpoint to {output_pt}")

if __name__ == "__main__":
    main()
