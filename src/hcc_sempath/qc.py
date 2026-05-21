from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from .tile_package import iter_package_tiles


def render_tile_package_qc(
    package_path: str | Path,
    output_path: str | Path,
    max_tiles: int = 36,
    thumb_size: int = 160,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for record, image in iter_package_tiles(package_path):
        samples.append((record, image.copy()))
        if len(samples) >= max_tiles:
            break
    if not samples:
        raise ValueError(f"package has no tiles: {package_path}")

    cols = min(6, len(samples))
    rows = math.ceil(len(samples) / cols)
    label_h = 34
    pad = 8
    canvas_w = cols * thumb_size + (cols + 1) * pad
    canvas_h = rows * (thumb_size + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (record, image) in enumerate(samples):
        row = i // cols
        col = i % cols
        x0 = pad + col * (thumb_size + pad)
        y0 = pad + row * (thumb_size + label_h + pad)
        thumb = image.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
        canvas.paste(thumb, (x0, y0))
        draw.rectangle((x0, y0, x0 + thumb_size - 1, y0 + thumb_size - 1), outline=(40, 40, 40))
        label = f"{record.tile_id}\n({record.x},{record.y}) {record.split}"
        draw.text((x0, y0 + thumb_size + 3), label, fill=(0, 0, 0))

    canvas.save(output_path)
