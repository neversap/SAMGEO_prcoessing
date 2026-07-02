from __future__ import annotations

import base64
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


IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class FTWPreviewSample:
    metadata: dict[str, Any]
    image_png_base64: str
    mask_png_base64: str
    overlay_png_base64: str


def load_ftw_preview_samples(
    ftw_root: str,
    country: str = "all",
    window: str = "window_a",
    mask_type: str = "semantic_3class",
    mode: str = "random",
    limit: int = 12,
    seed: int = 42,
    allowed_roots: list[Path] | None = None,
) -> list[FTWPreviewSample]:
    _ensure_geo_dependencies()
    root = _validate_root(ftw_root, allowed_roots or [])
    rows = _discover_rows(root, country, window, mask_type)
    rows = _select_rows(rows, mode, limit, seed)
    return [_row_to_sample(row) for row in rows]


def _ensure_geo_dependencies() -> None:
    if _GEO_DEPENDENCY_ERROR is None:
        return
    raise ModuleNotFoundError(
        "FTW preview requires geospatial dependencies. "
        "Install them with: pip install -r requirements-data-process.txt"
    ) from _GEO_DEPENDENCY_ERROR


def _validate_root(ftw_root: str, allowed_roots: list[Path]) -> Path:
    path = Path(ftw_root).expanduser()
    if not path.is_absolute():
        raise ValueError("ftw_root must be an absolute path")
    resolved = path.resolve()
    if allowed_roots and not any(_is_relative_to(resolved, root.resolve()) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"ftw_root must be under one of: {roots}")
    if not resolved.exists():
        raise FileNotFoundError(f"ftw_root does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"ftw_root must be a directory: {resolved}")
    return resolved


def _discover_rows(root: Path, country: str, window: str, mask_type: str) -> list[dict[str, Any]]:
    normalized_window = window.strip().lower()
    if normalized_window not in {"window_a", "window_b"}:
        raise ValueError("window must be window_a or window_b")
    normalized_country = country.strip().lower()
    countries = [
        item
        for item in sorted(root.iterdir())
        if item.is_dir() and (normalized_country in {"", "all"} or item.name.lower() == normalized_country)
    ]
    if not countries:
        raise FileNotFoundError(f"No FTW country directory matched {country!r} under {root}")

    rows = []
    for country_dir in countries:
        image_dir = country_dir / "s2_images" / normalized_window
        mask_dir = country_dir / "label_masks" / mask_type
        if not image_dir.exists() or not mask_dir.exists():
            continue
        mask_by_stem = {
            _normalize_stem(path): path
            for path in _list_files(mask_dir)
        }
        for image_path in _list_files(image_dir):
            mask_path = mask_by_stem.get(_normalize_stem(image_path))
            if mask_path is None:
                continue
            ratios, width, height = _inspect_mask(mask_path)
            rows.append(
                {
                    "patch_name": f"{country_dir.name}_{image_path.stem}_{normalized_window}",
                    "source_tif": f"{country_dir.name}/{normalized_window}/{image_path.name}",
                    "country": country_dir.name,
                    "window": normalized_window,
                    "mask_type": mask_type,
                    "split": "ftw",
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    **ratios,
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No FTW image/mask pairs found under {root} for country={country!r}, "
            f"window={window!r}, mask_type={mask_type!r}"
        )
    return rows


def _list_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def _normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "_label", "_labels", "_semantic", "_semantic_3class", "_semantic_2class"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _select_rows(rows: list[dict[str, Any]], mode: str, limit: int, seed: int) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 50))
    normalized = mode.strip().lower()
    if normalized == "random":
        selected = rows[:]
        random.Random(seed).shuffle(selected)
        return selected[:limit]
    if normalized == "high_cropland":
        return sorted(rows, key=lambda row: row["cropland_ratio"], reverse=True)[:limit]
    if normalized == "low_cropland":
        candidates = [row for row in rows if 0.0 < row["cropland_ratio"] <= 0.15]
        return sorted(candidates, key=lambda row: row["cropland_ratio"])[:limit]
    if normalized == "empty":
        return [row for row in rows if row["cropland_ratio"] == 0.0][:limit]
    if normalized == "high_ignore":
        return sorted(rows, key=lambda row: row["ignore_ratio"], reverse=True)[:limit]
    if normalized == "boundary":
        return sorted(rows, key=lambda row: row["boundary_ratio"], reverse=True)[:limit]
    raise ValueError("mode must be random, high_cropland, low_cropland, empty, high_ignore, or boundary")


def _row_to_sample(row: dict[str, Any]) -> FTWPreviewSample:
    image_rgb = _read_image_rgb(row["image_path"])
    mask = _read_mask(row["mask_path"])
    mask = _align_mask_to_image(mask, image_rgb)
    overlay = _make_overlay(image_rgb, mask)
    mask_rgb = _mask_to_rgb(mask)
    metadata = {
        "patch_name": row["patch_name"],
        "source_tif": row["source_tif"],
        "x": row["x"],
        "y": row["y"],
        "width": row["width"],
        "height": row["height"],
        "cropland_ratio": row["cropland_ratio"],
        "ignore_ratio": row["ignore_ratio"],
        "interior_ratio": row["interior_ratio"],
        "boundary_ratio": row["boundary_ratio"],
        "background_ratio": row["background_ratio"],
        "patch_type": row["mask_type"],
        "split": row["split"],
        "image_path": str(row["image_path"]),
        "mask_path": str(row["mask_path"]),
    }
    return FTWPreviewSample(
        metadata=metadata,
        image_png_base64=_png_base64(image_rgb),
        mask_png_base64=_png_base64(mask_rgb),
        overlay_png_base64=_png_base64(overlay),
    )


def _align_mask_to_image(mask, image_rgb):
    image_height, image_width = image_rgb.shape[:2]
    if mask.shape == (image_height, image_width):
        return mask
    pil_mask = Image.fromarray(mask.astype("uint8"))
    resized = pil_mask.resize((image_width, image_height), resample=Image.Resampling.NEAREST)
    return np.array(resized, dtype=mask.dtype)


def _inspect_mask(mask_path: Path) -> tuple[dict[str, float], int, int]:
    mask = _read_mask(mask_path)
    total = float(mask.size)
    valid_total = max(float(np.count_nonzero(mask != 255)), 1.0)
    interior_ratio = float(np.count_nonzero(mask == 1) / valid_total)
    boundary_ratio = float(np.count_nonzero(mask == 2) / valid_total)
    background_ratio = float(np.count_nonzero(mask == 0) / valid_total)
    ignore_ratio = float(np.count_nonzero(mask == 255) / total)
    return (
        {
            "cropland_ratio": interior_ratio + boundary_ratio,
            "interior_ratio": interior_ratio,
            "boundary_ratio": boundary_ratio,
            "background_ratio": background_ratio,
            "ignore_ratio": ignore_ratio,
        },
        int(mask.shape[1]),
        int(mask.shape[0]),
    )


def _read_image_rgb(path: Path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            band_count = min(src.count, 3)
            data = src.read(list(range(1, band_count + 1))).astype("float32")
            if band_count == 1:
                data = np.repeat(data, 3, axis=0)
            elif band_count == 2:
                data = np.concatenate([data, data[:1]], axis=0)
        bands = [_stretch_band(data[index]) for index in range(3)]
        return np.stack(bands, axis=-1)
    image = Image.open(path).convert("RGB")
    return np.array(image, dtype="uint8")


def _read_mask(path: Path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            return src.read(1)
    return np.array(Image.open(path), dtype="uint8")


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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
