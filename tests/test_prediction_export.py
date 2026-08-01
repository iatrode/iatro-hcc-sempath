from __future__ import annotations

import numpy as np
import pyarrow as pa

from hcc_sempath.inference.predictions import (
    PredictionPackageReader,
    encode_prediction_payload,
    grid_cell_center_level0,
    prediction_header,
    prediction_index_table,
    source_index_sha256,
    write_prediction_package,
)


def _source_tables():
    header = {
        "coordinate_mode": "tile_grid",
        "origin": "top_left",
        "tile_width": 224,
        "tile_height": 224,
        "stride_x": 444,
        "stride_y": 444,
        "num_records": 1,
        "source": {"width": 1000, "height": 1000, "native_mpp_x": 0.25, "native_mpp_y": 0.25},
        "tiling": {"openslide_level": 0, "level_downsample": 1.0, "level_read_width": 444, "level_read_height": 444, "target_mpp": 0.5},
    }
    slides = pa.table({"slide_idx": pa.array([0], type=pa.uint8()), "slide_id": ["slide-a"], "patient_id": ["patient-a"]})
    index = pa.table({
        "slide_idx": pa.array([0], type=pa.uint8()),
        "tile_x": pa.array([2], type=pa.uint16()),
        "tile_y": pa.array([3], type=pa.uint16()),
        "tile_id": ["tile-a"],
        "split": ["val"],
        "offset": pa.array([0], type=pa.uint64()),
        "length": pa.array([10], type=pa.uint32()),
        "crc32": pa.array([123], type=pa.uint32()),
    })
    return header, slides, index


def test_prediction_package_roundtrip_and_coordinates(tmp_path):
    source_header, slides, source_index = _source_tables()
    header = prediction_header(
        source_path="slide-a.tiles.iac",
        source_header=source_header,
        source_index_digest=source_index_sha256(source_header, slides, source_index),
        checkpoint_path="best.pt",
        checkpoint_file_digest="a" * 64,
        checkpoint_model_digest="b" * 64,
        classification_names=["c0", "c1"],
        component_names=["s0", "s1"],
        grid_shape=(32, 32),
        spatial_stride=7,
        patch_size=14,
        patch_padding=4,
        spatial_dtype="uint8",
    )
    classification = np.array([0.2, 0.8], dtype=np.float32)
    instance = np.linspace(0, 1, 2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32)
    abundance = instance[::-1].copy()
    output = tmp_path / "predictions.iac"
    write_prediction_package(
        output,
        header=header,
        slide_table=slides,
        index_table=prediction_index_table(source_index, [0]),
        payloads=[encode_prediction_payload(classification, instance, abundance, spatial_dtype="uint8")],
    )

    with PredictionPackageReader(output) as reader:
        assert reader.record_count == 1
        assert reader.index_table.column("source_row")[0].as_py() == 0
        decoded = reader.read_at(0)
        np.testing.assert_allclose(decoded["classification_probabilities"], classification, atol=5e-4)
        np.testing.assert_allclose(decoded["spatial_instance_probabilities"], instance, atol=1 / 510)
        np.testing.assert_allclose(decoded["spatial_abundance_probabilities"], abundance, atol=1 / 510)
        x, y = grid_cell_center_level0(reader.header, tile_x=2, tile_y=3, row=0, column=0)
        assert x == 2 * 444 + 3 * (444 / 224)
        assert y == 3 * 444 + 3 * (444 / 224)


def test_uint16_probability_roundtrip_is_high_precision():
    rng = np.random.default_rng(13)
    classification = rng.random(7, dtype=np.float32)
    classification /= classification.sum()
    instance = rng.random((11, 4, 4), dtype=np.float32)
    abundance = rng.random((11, 4, 4), dtype=np.float32)
    from hcc_sempath.inference.predictions import decode_prediction_payload

    header = {
        "classification_class_names": [str(i) for i in range(7)],
        "spatial_component_names": [str(i) for i in range(11)],
        "spatial_grid_shape": [4, 4],
        "spatial_probability_encoding": {"dtype": "uint16"},
    }
    decoded = decode_prediction_payload(
        encode_prediction_payload(classification, instance, abundance, spatial_dtype="uint16"),
        header,
    )
    np.testing.assert_allclose(decoded["spatial_instance_probabilities"], instance, atol=1 / 131070)


def test_float16_probability_roundtrip():
    from hcc_sempath.inference.predictions import decode_prediction_payload

    classification = np.array([0.25, 0.75], dtype=np.float32)
    instance = np.full((2, 2, 2), 0.1234, dtype=np.float32)
    abundance = np.full((2, 2, 2), 0.9876, dtype=np.float32)
    header = {
        "classification_class_names": ["a", "b"],
        "spatial_component_names": ["x", "y"],
        "spatial_grid_shape": [2, 2],
        "spatial_probability_encoding": {"dtype": "float16"},
    }
    decoded = decode_prediction_payload(
        encode_prediction_payload(classification, instance, abundance, spatial_dtype="float16"),
        header,
    )
    np.testing.assert_allclose(decoded["spatial_instance_probabilities"], instance, atol=5e-4)
