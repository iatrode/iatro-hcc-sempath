from __future__ import annotations

import pytest

from hcc_sempath.training.config import _deep_merge, load_config


def test_deep_merge_unions_plain_dicts():
    base = {"train": {"lr": 0.1, "epochs": 100}}
    override = {"train": {"epochs": 20}}
    merged = _deep_merge(base, override)
    assert merged["train"] == {"lr": 0.1, "epochs": 20}


def test_deep_merge_replaces_teacher_keyed_maps_wholesale():
    base = {
        "model": {"teacher_dims": {"gigapath": 1536, "virchow2": 2560}},
        "loss": {"teacher_weights": {"gigapath": 1.0, "virchow2": 1.0}},
        "data": {"prototype_paths": {"gigapath": "g.pt", "virchow2": "v.pt"}},
    }
    override = {
        "model": {"teacher_dims": {"virchow2": 2560}},
        "loss": {"teacher_weights": {"virchow2": 1.0}},
        "data": {"prototype_paths": {"virchow2": "v.pt"}},
    }
    merged = _deep_merge(base, override)
    # Single-teacher override must drop the parent's other teachers, not union.
    assert merged["model"]["teacher_dims"] == {"virchow2": 2560}
    assert merged["loss"]["teacher_weights"] == {"virchow2": 1.0}
    assert merged["data"]["prototype_paths"] == {"virchow2": "v.pt"}


def test_deep_merge_keeps_sibling_keys_when_replacing_teacher_map():
    base = {"loss": {"teacher_weights": {"a": 1.0}, "semantic_weight": 0.02}}
    override = {"loss": {"teacher_weights": {"b": 1.0}}}
    merged = _deep_merge(base, override)
    assert merged["loss"]["teacher_weights"] == {"b": 1.0}
    assert merged["loss"]["semantic_weight"] == 0.02


@pytest.mark.parametrize(
    "override_yaml,expected_teachers",
    [
        ("", 4),
        (
            """
model:
  teacher_dims:
    virchow2: 2560
loss:
  teacher_weights:
    virchow2: 1.0
data:
  prototype_paths:
    virchow2: virchow2.pt
""",
            1,
        ),
    ],
)
def test_inherited_configs_resolve_expected_teacher_set(tmp_path, override_yaml, expected_teachers):
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        """
model:
  teacher_dims:
    gigapath: 1536
    h_optimus_1: 1536
    uni2_h: 1536
    virchow2: 2560
loss:
  teacher_weights:
    gigapath: 1.0
    h_optimus_1: 1.0
    uni2_h: 1.0
    virchow2: 1.0
data:
  prototype_paths:
    gigapath: gigapath.pt
    h_optimus_1: h_optimus_1.pt
    uni2_h: uni2_h.pt
    virchow2: virchow2.pt
""",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(f"inherits: parent.yaml\n{override_yaml}\n", encoding="utf-8")
    cfg = load_config(child)
    assert len(cfg["model"]["teacher_dims"]) == expected_teachers
    assert len(cfg["loss"]["teacher_weights"]) == expected_teachers
