from __future__ import annotations

import torch

from hcc_sempath.teacher.cache import _local_weight_path, _pool_virchow2_features, _resolve_model_spec, _teacher_name


def test_supported_teacher_presets_resolve_expected_names() -> None:
    uni2_spec = _resolve_model_spec("uni2_h")
    assert uni2_spec["model_name"] == "hf-hub:MahmoodLab/UNI2-h"
    assert uni2_spec["feature_mode"] == "default"
    assert uni2_spec["model_kwargs"]["embed_dim"] == 1536
    assert _teacher_name("uni2_h", "") == "uni2_h"

    virchow2_spec = _resolve_model_spec("virchow2")
    assert virchow2_spec["model_name"] == "hf-hub:paige-ai/Virchow2"
    assert virchow2_spec["feature_mode"] == "virchow2"
    assert _teacher_name("virchow2", "") == "virchow2"


def test_pool_virchow2_features_concatenates_class_and_patch_mean() -> None:
    output = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    features = _pool_virchow2_features(output)

    expected = torch.cat([output[:, 0], output[:, 5:].mean(1)], dim=-1)
    assert features.shape == (2, 6)
    torch.testing.assert_close(features, expected)


def test_local_teacher_directory_uses_matching_preset(tmp_path) -> None:
    local_dir = tmp_path / "virchow2"
    local_dir.mkdir()

    spec = _resolve_model_spec(str(local_dir))
    assert spec["model_name"] == str(local_dir)
    assert spec["feature_mode"] == "virchow2"
    assert spec["model_kwargs"]["mlp_layer"] is not None


def test_local_weight_path_accepts_safetensors(tmp_path) -> None:
    model_dir = tmp_path / "teacher"
    model_dir.mkdir()
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"stub")

    assert _local_weight_path(model_dir, {}) == weight_path
