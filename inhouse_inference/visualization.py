from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = {
    0: (0, 0, 0, 0),
    1: (34, 197, 94, 135),
    2: (220, 38, 38, 180),
}


def mask_to_rgba(mask: np.ndarray) -> Image.Image:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype="uint8")
    for class_id, color in PALETTE.items():
        rgba[mask == class_id] = color
    return Image.fromarray(rgba, mode="RGBA")


def boundary_to_rgba(boundary: np.ndarray) -> Image.Image:
    rgba = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype="uint8")
    rgba[boundary > 0] = (220, 38, 38, 220)
    return Image.fromarray(rgba, mode="RGBA")


def overlay_image(base: Image.Image, mask: np.ndarray) -> Image.Image:
    image = base.convert("RGBA")
    overlay = mask_to_rgba(mask).resize(image.size, Image.Resampling.NEAREST)
    image.alpha_composite(overlay)
    return image


def save_outputs(
    output_dir: str | Path,
    original: Image.Image,
    resized: Image.Image,
    mask_512: np.ndarray,
    mask_original: np.ndarray,
    boundary_512: np.ndarray,
    boundary_original: np.ndarray,
) -> dict[str, str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    files = {
        "input": path / "input.png",
        "resized_input": path / "resized_input.png",
        "mask_512": path / "mask_512.png",
        "mask_original": path / "mask_original.png",
        "boundary_512": path / "boundary_512.png",
        "boundary_original": path / "boundary_original.png",
        "overlay_512": path / "overlay_512.png",
        "overlay_original": path / "overlay_original.png",
    }
    original.save(files["input"])
    resized.save(files["resized_input"])
    mask_to_rgba(mask_512).save(files["mask_512"])
    mask_to_rgba(mask_original).save(files["mask_original"])
    boundary_to_rgba(boundary_512).save(files["boundary_512"])
    boundary_to_rgba(boundary_original).save(files["boundary_original"])
    overlay_image(resized, mask_512).save(files["overlay_512"])
    overlay_image(original, mask_original).save(files["overlay_original"])
    return {key: str(value) for key, value in files.items()}
