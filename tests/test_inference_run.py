from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from hcc_sempath.inference.predictions import PredictionPackageReader
from hcc_sempath.inference.run import _parser, run


class _FakeModel:
    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        return {
            "classification_probabilities": torch.full((batch, 7), 1 / 7, device=images.device),
            "spatial_instance_probabilities": torch.full((batch, 11, 32, 32), 0.25, device=images.device),
            "spatial_abundance_probabilities": torch.full((batch, 11, 32, 32), 0.50, device=images.device),
        }


def test_infer_raster_pipeline_writes_canonical_iac_and_progress(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"fake release")
    source = tmp_path / "case.png"
    Image.new("RGB", (224, 224), (100, 40, 150)).save(source)
    release = SimpleNamespace(
        weights_path=weights,
        model_digest="d" * 64,
        model=_FakeModel(),
        config={"model": {"spatial_output_stride": 7}},
        classification_names=tuple(f"c{i}" for i in range(7)),
        spatial_component_names=tuple(f"s{i}" for i in range(11)),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    monkeypatch.setattr("hcc_sempath.inference.run.load_release_model", lambda *_args, **_kwargs: release)
    output = tmp_path / "output"
    args = _parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--input",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--workers",
            "1",
            "--tile-workers",
            "1",
        ]
    )

    manifest = run(args)

    assert manifest["records"] == 1
    assert (output / "case.tile.path.iac").is_file()
    prediction_path = output / "case.pred.path.iac"
    assert prediction_path.is_file()
    with PredictionPackageReader(prediction_path) as reader:
        decoded = reader.read_at(0)
        assert decoded["classification_probabilities"].shape == (7,)
        assert decoded["spatial_instance_probabilities"].shape == (11, 32, 32)
    captured = capsys.readouterr()
    assert "SemPath inference" in captured.err
