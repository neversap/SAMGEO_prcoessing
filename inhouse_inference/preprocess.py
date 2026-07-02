from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreprocessResult:
    original: Image.Image
    resized: Image.Image
    tensor: torch.Tensor
    original_size: tuple[int, int]
    model_input_size: tuple[int, int]
    scale_x: float
    scale_y: float


def load_image(value: str | Path | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.open(value).convert("RGB")


def preprocess_image(
    image: str | Path | Image.Image,
    size: int = 512,
    normalize: bool = False,
    stats_mean: list[float] | None = None,
    stats_std: list[float] | None = None,
) -> PreprocessResult:
    original = load_image(image)
    original_size = original.size
    resized = original.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype="float32") / 255.0
    chw = array.transpose(2, 0, 1)
    if normalize and stats_mean is not None and stats_std is not None:
        mean = np.asarray(stats_mean, dtype="float32")[:, None, None]
        std = np.maximum(np.asarray(stats_std, dtype="float32")[:, None, None], 1e-6)
        chw = (chw - mean) / std
    tensor = torch.from_numpy(np.ascontiguousarray(chw[None])).float()
    width, height = original_size
    return PreprocessResult(
        original=original,
        resized=resized,
        tensor=tensor,
        original_size=original_size,
        model_input_size=(size, size),
        scale_x=width / size,
        scale_y=height / size,
    )
