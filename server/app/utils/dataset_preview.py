from __future__ import annotations

import base64
import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
    import numpy as np
    import rasterio
except ModuleNotFoundError as exc:
    Image = None
    np = None
    rasterio = None
    _GEO_DEPENDENCY_ERROR = exc
else:
    _GEO_DEPENDENCY_ERROR = None


@dataclass(frozen=True)
class PreviewSample:
    metadata: dict[str, Any]
    image_png_base64: str
    mask_png_base64: str
    overlay_png_base64: str


def load_preview_samples(
    dataset_dir: str,
    split: str = "all",
    mode: str = "random",
    limit: int = 12,
    seed: int = 42,
    allowed_roots: list[Path] | None = None,
) -> list[PreviewSample]:
    _ensure_geo_dependencies()
    dataset_path = _validate_dataset_dir(dataset_dir, allowed_roots or [])
    rows = _read_patch_index(dataset_path)
    rows = _filter_rows(rows, split)
    rows = _select_rows(rows, mode, limit, seed)
    return [_row_to_sample(dataset_path, row) for row in rows]


def _ensure_geo_dependencies() -> None:
    if _GEO_DEPENDENCY_ERROR is None:
        return
    raise ModuleNotFoundError(
        "dataset preview requires geospatial dependencies. "
        "Install them with: pip install -r requirements-data-process.txt"
    ) from _GEO_DEPENDENCY_ERROR


def _validate_dataset_dir(dataset_dir: str, allowed_roots: list[Path]) -> Path:
    path = Path(dataset_dir).expanduser()
    if not path.is_absolute():
        raise ValueError("dataset_dir must be an absolute path")
    resolved = path.resolve()
    if allowed_roots and not any(_is_relative_to(resolved, root.resolve()) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"dataset_dir must be under one of: {roots}")
    patch_index = resolved / "metadata" / "patch_index.csv"
    if not patch_index.exists():
        raise FileNotFoundError(f"patch_index.csv does not exist: {patch_index}")
    return resolved


def _read_patch_index(dataset_dir: Path) -> list[dict[str, str]]:
    path = dataset_dir / "metadata" / "patch_index.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _filter_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    normalized = split.strip().lower()
    if normalized in {"", "all"}:
        return rows
    if normalized not in {"train", "val", "test"}:
        raise ValueError("split must be all, train, val, or test")
    return [row for row in rows if row.get("split", "").lower() == normalized]


def _select_rows(rows: list[dict[str, str]], mode: str, limit: int, seed: int) -> list[dict[str, str]]:
    limit = max(1, min(int(limit), 50))
    normalized = mode.strip().lower()
    if normalized == "random":
        selected = rows[:]
        random.Random(seed).shuffle(selected)
        return selected[:limit]
    if normalized == "high_cropland":
        return sorted(rows, key=lambda row: _float_value(row, "cropland_ratio"), reverse=True)[:limit]
    if normalized == "low_cropland":
        candidates = [row for row in rows if 0.0 < _float_value(row, "cropland_ratio") <= 0.15]
        return sorted(candidates, key=lambda row: _float_value(row, "cropland_ratio"))[:limit]
    if normalized == "empty":
        return [row for row in rows if _float_value(row, "cropland_ratio") == 0.0][:limit]
    if normalized == "high_ignore":
        return sorted(rows, key=lambda row: _float_value(row, "ignore_ratio"), reverse=True)[:limit]
    if normalized == "boundary":
        return sorted(rows, key=lambda row: _float_value(row, "boundary_ratio"), reverse=True)[:limit]
    raise ValueError("mode must be random, high_cropland, low_cropland, empty, high_ignore, or boundary")


def _row_to_sample(dataset_dir: Path, row: dict[str, str]) -> PreviewSample:
    image_path = _safe_child_path(dataset_dir, row.get("image_path", ""))
    mask_path = _safe_child_path(dataset_dir, row.get("mask_path", ""))
    image_rgb = _read_tif_rgb(image_path)
    mask = _read_mask(mask_path)
    overlay = _make_overlay(image_rgb, mask)
    mask_rgb = _mask_to_rgb(mask)
    metadata = {
        "patch_name": row.get("patch_name", ""),
        "source_tif": row.get("source_tif", ""),
        "x": _int_value(row, "x"),
        "y": _int_value(row, "y"),
        "width": _int_value(row, "width"),
        "height": _int_value(row, "height"),
        "cropland_ratio": _float_value(row, "cropland_ratio"),
        "ignore_ratio": _float_value(row, "ignore_ratio"),
        "interior_ratio": _float_value(row, "interior_ratio"),
        "boundary_ratio": _float_value(row, "boundary_ratio"),
        "background_ratio": _float_value(row, "background_ratio"),
        "patch_type": row.get("patch_type", ""),
        "split": row.get("split", ""),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
    }
    return PreviewSample(
        metadata=metadata,
        image_png_base64=_png_base64(image_rgb),
        mask_png_base64=_png_base64(mask_rgb),
        overlay_png_base64=_png_base64(overlay),
    )


def _safe_child_path(dataset_dir: Path, value: str) -> Path:
    if not value:
        raise ValueError("patch_index.csv contains an empty image_path or mask_path")
    path = Path(value)
    if not path.is_absolute():
        path = dataset_dir / path
    resolved = path.resolve()
    if not _is_relative_to(resolved, dataset_dir):
        raise ValueError(f"patch path is outside dataset_dir: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"patch file does not exist: {resolved}")
    return resolved


def _read_tif_rgb(path: Path):
    with rasterio.open(path) as src:
        band_count = min(src.count, 3)
        data = src.read(list(range(1, band_count + 1))).astype("float32")
        if band_count == 1:
            data = np.repeat(data, 3, axis=0)
        elif band_count == 2:
            data = np.concatenate([data, data[:1]], axis=0)
    bands = [_stretch_band(data[index]) for index in range(3)]
    return np.stack(bands, axis=-1)


def _read_mask(path: Path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    return mask


def _stretch_band(band):
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype="uint8")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1.0
    stretched = (band - low) / (high - low)
    return np.clip(stretched * 255, 0, 255).astype("uint8")


def _make_overlay(image_rgb, mask):
    overlay = image_rgb.copy().astype("float32")
    colors = {
        1: np.array([40, 175, 90], dtype="float32"),
        2: np.array([235, 180, 30], dtype="float32"),
        255: np.array([130, 105, 185], dtype="float32"),
    }
    for value, color in colors.items():
        selected = mask == value
        overlay[selected] = overlay[selected] * 0.45 + color * 0.55
    return np.clip(overlay, 0, 255).astype("uint8")


def _mask_to_rgb(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype="uint8")
    rgb[mask == 1] = [40, 175, 90]
    rgb[mask == 2] = [235, 180, 30]
    rgb[mask == 255] = [130, 105, 185]
    rgb[(mask > 0) & (mask != 1) & (mask != 2) & (mask != 255)] = [220, 45, 45]
    return rgb


def _png_base64(array) -> str:
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def _int_value(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except ValueError:
        return 0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
