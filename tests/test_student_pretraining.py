from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors_file

from hcc_sempath.modeling.models import (
    STUDENT_PRETRAINED_ARTIFACTS,
    _read_fixed_student_pretraining,
    _resolve_fixed_student_pretraining,
)


def test_resolve_fixed_student_pretraining_prefers_first_existing(tmp_path: Path) -> None:
    safetensors_path = tmp_path / "student.safetensors"
    pth_path = tmp_path / "student.pth"
    pth_path.touch()

    assert _resolve_fixed_student_pretraining(
        ((safetensors_path, "safe"), (pth_path, "pth"))
    ) == (pth_path, "pth")

    safetensors_path.touch()
    assert _resolve_fixed_student_pretraining(
        ((safetensors_path, "safe"), (pth_path, "pth"))
    ) == (safetensors_path, "safe")


def test_fixed_student_pretraining_readers_preserve_tensor_values(tmp_path: Path) -> None:
    expected = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "bias": torch.tensor([1.0, -2.0, 3.0]),
    }
    pth_path = tmp_path / "student.pth"
    safetensors_path = tmp_path / "student.safetensors"
    torch.save(expected, pth_path)
    save_safetensors_file(expected, safetensors_path)

    pth_state = _read_fixed_student_pretraining(pth_path)
    safetensors_state = _read_fixed_student_pretraining(safetensors_path)

    assert pth_state.keys() == safetensors_state.keys() == expected.keys()
    for key, tensor in expected.items():
        assert torch.equal(pth_state[key], tensor)
        assert torch.equal(safetensors_state[key], tensor)


def test_modelscope_student_artifact_digest_is_fixed() -> None:
    artifacts = {path.name: digest for path, digest in STUDENT_PRETRAINED_ARTIFACTS}
    assert artifacts["dinov2_vits14_pretrain.safetensors"] == (
        "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
    )
    assert artifacts["dinov2_vits14_pretrain.pth"] == (
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    )
