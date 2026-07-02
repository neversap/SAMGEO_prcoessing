from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_process_pipeline.dataloader import FTWIndexConfig
from data_process_pipeline.dataloader import build_ftw_official_index
from data_process_pipeline.dataloader import discover_ftw_official_rows
from data_process_pipeline.dataloader import infer_image_scale
from data_process_pipeline.dataloader import load_inhouse_index
from data_process_pipeline.dataloader import read_image_chw
from data_process_pipeline.dataloader import read_mask
from data_process_pipeline.dataloader import select_rows
from data_process_pipeline.dataloader import summarize_rows

try:
    import numpy as np
    from PIL import Image, ImageEnhance
except ModuleNotFoundError as exc:
    np = None
    Image = None
    ImageEnhance = None
    _DEPENDENCY_ERROR = exc
else:
    _DEPENDENCY_ERROR = None


@dataclass(frozen=True)
class TrainingPreviewSample:
    metadata: dict[str, Any]
    image_png_base64: str
    mask_png_base64: str
    overlay_png_base64: str
    augmented_image_png_base64: str
    augmented_mask_png_base64: str
    augmented_overlay_png_base64: str


def build_ftw_training_index(
    ftw_root: str,
    metadata_dir: str,
    country: str,
    window: str,
    mask_type: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_samples: int,
    allowed_roots: list[Path],
):
    root = _validate_root(ftw_root, allowed_roots, "ftw_root")
    metadata_path = _validate_output_dir(metadata_dir, allowed_roots, "metadata_dir")
    return build_ftw_official_index(
        FTWIndexConfig(
            ftw_root=root,
            metadata_dir=metadata_path,
            country=country,
            window=window,
            mask_type=mask_type,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            max_samples=max_samples,
        )
    )


def load_training_augmentation_preview(
    source: str,
    root_path: str,
    country: str = "all",
    window: str = "window_a",
    mask_type: str = "semantic_3class",
    split: str = "all",
    mode: str = "random",
    limit: int = 6,
    seed: int = 42,
    hflip: bool = True,
    vflip: bool = True,
    rotate90: bool = True,
    scale_jitter: float = 0.15,
    brightness: float = 0.12,
    contrast: float = 0.12,
    noise: float = 0.02,
    allowed_roots: list[Path] | None = None,
) -> tuple[list[TrainingPreviewSample], dict[str, Any]]:
    _ensure_dependencies()
    allowed = allowed_roots or []
    normalized_source = source.strip().lower()
    root = _validate_root(root_path, allowed, "root_path")
    if normalized_source == "ftw":
        rows = discover_ftw_official_rows(
            ftw_root=root,
            country=country,
            window=window,
            mask_type=mask_type,
            max_samples=0,
            seed=seed,
        )
    elif normalized_source == "inhouse":
        rows = load_inhouse_index(root)
    else:
        raise ValueError("source must be ftw or inhouse")

    selected = select_rows(rows, split=split, mode=mode, limit=limit, seed=seed)
    samples = [
        _row_to_augmented_sample(
            row=row,
            seed=seed + index * 101,
            hflip=hflip,
            vflip=vflip,
            rotate90=rotate90,
            scale_jitter=scale_jitter,
            brightness=brightness,
            contrast=contrast,
            noise=noise,
        )
        for index, row in enumerate(selected)
    ]
    return samples, summarize_rows(rows)


def _row_to_augmented_sample(
    row: dict[str, Any],
    seed: int,
    hflip: bool,
    vflip: bool,
    rotate90: bool,
    scale_jitter: float,
    brightness: float,
    contrast: float,
    noise: float,
) -> TrainingPreviewSample:
    image_rgb = _read_preview_rgb(Path(row["image_path"]))
    mask = read_mask(Path(row["mask_path"]))
    mask = _align_mask_to_image(mask, image_rgb)
    aug_image, aug_mask, applied = _augment_pair(
        image_rgb,
        mask,
        seed=seed,
        hflip=hflip,
        vflip=vflip,
        rotate90=rotate90,
        scale_jitter=scale_jitter,
        brightness=brightness,
        contrast=contrast,
        noise=noise,
    )
    metadata = {
        "sample_id": row.get("sample_id", ""),
        "patch_name": row.get("patch_name", ""),
        "source_dataset": row.get("source_dataset", ""),
        "source_tif": row.get("source_tif", ""),
        "country": row.get("country", ""),
        "window": row.get("window", ""),
        "split": row.get("split", ""),
        "bucket": row.get("bucket", ""),
        "fg_ratio": float(row.get("fg_ratio", row.get("cropland_ratio", 0.0))),
        "cropland_ratio": float(row.get("cropland_ratio", 0.0)),
        "interior_ratio": float(row.get("interior_ratio", 0.0)),
        "boundary_ratio": float(row.get("boundary_ratio", 0.0)),
        "ignore_ratio": float(row.get("ignore_ratio", 0.0)),
        "image_path": str(row.get("image_path", "")),
        "mask_path": str(row.get("mask_path", "")),
        "augmentation": applied,
    }
    return TrainingPreviewSample(
        metadata=metadata,
        image_png_base64=_png_base64(image_rgb),
        mask_png_base64=_png_base64(_mask_to_rgb(mask)),
        overlay_png_base64=_png_base64(_make_overlay(image_rgb, mask)),
        augmented_image_png_base64=_png_base64(aug_image),
        augmented_mask_png_base64=_png_base64(_mask_to_rgb(aug_mask)),
        augmented_overlay_png_base64=_png_base64(_make_overlay(aug_image, aug_mask)),
    )


def _read_preview_rgb(path: Path):
    image_chw = read_image_chw(path)
    scale = infer_image_scale(image_chw)
    image = image_chw.astype("float32")
    if scale > 1.0:
        image = image / scale
    image = np.clip(image, 0.0, 1.0)
    return (image.transpose(1, 2, 0) * 255.0).round().astype("uint8")


def _augment_pair(
    image_rgb,
    mask,
    seed: int,
    hflip: bool,
    vflip: bool,
    rotate90: bool,
    scale_jitter: float,
    brightness: float,
    contrast: float,
    noise: float,
):
    rng = random.Random(seed)
    applied: list[str] = []
    image = image_rgb
    label = mask

    if hflip and rng.random() < 0.5:
        image = np.flip(image, axis=1)
        label = np.flip(label, axis=1)
        applied.append("hflip")
    if vflip and rng.random() < 0.5:
        image = np.flip(image, axis=0)
        label = np.flip(label, axis=0)
        applied.append("vflip")
    if rotate90:
        turns = rng.randint(0, 3)
        if turns:
            image = np.rot90(image, k=turns)
            label = np.rot90(label, k=turns)
            applied.append(f"rotate90x{turns}")

    if scale_jitter > 0:
        factor = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
        image, label = _scale_to_original_size(image, label, factor)
        applied.append(f"scale={factor:.2f}")

    pil_image = Image.fromarray(np.ascontiguousarray(image.astype("uint8")), mode="RGB")
    if brightness > 0:
        factor = 1.0 + rng.uniform(-brightness, brightness)
        pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)
        applied.append(f"brightness={factor:.2f}")
    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)
        applied.append(f"contrast={factor:.2f}")

    image = np.asarray(pil_image, dtype="float32")
    if noise > 0:
        noise_rng = np.random.default_rng(seed)
        image = image + noise_rng.normal(0.0, noise * 255.0, size=image.shape)
        applied.append(f"noise={noise:.3f}")

    image = np.clip(image, 0, 255).astype("uint8")
    return image, np.ascontiguousarray(label), ", ".join(applied) or "none"


def _scale_to_original_size(image, mask, factor: float):
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * factor)))
    new_height = max(1, int(round(height * factor)))
    pil_image = Image.fromarray(np.ascontiguousarray(image.astype("uint8")), mode="RGB")
    pil_mask = Image.fromarray(np.ascontiguousarray(mask.astype("uint8")))
    resized_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    resized_mask = pil_mask.resize((new_width, new_height), Image.Resampling.NEAREST)
    if factor >= 1.0:
        left = max(0, (new_width - width) // 2)
        top = max(0, (new_height - height) // 2)
        resized_image = resized_image.crop((left, top, left + width, top + height))
        resized_mask = resized_mask.crop((left, top, left + width, top + height))
    else:
        canvas_image = Image.new("RGB", (width, height))
        canvas_mask = Image.new("L", (width, height), color=255)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        canvas_image.paste(resized_image, (left, top))
        canvas_mask.paste(resized_mask, (left, top))
        resized_image = canvas_image
        resized_mask = canvas_mask
    return np.asarray(resized_image, dtype="uint8"), np.asarray(resized_mask)


def _align_mask_to_image(mask, image_rgb):
    image_height, image_width = image_rgb.shape[:2]
    if mask.shape == (image_height, image_width):
        return mask
    pil_mask = Image.fromarray(mask.astype("uint8"))
    resized = pil_mask.resize((image_width, image_height), resample=Image.Resampling.NEAREST)
    return np.array(resized, dtype=mask.dtype)


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


def _validate_root(value: str, allowed_roots: list[Path], label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = path.resolve()
    if allowed_roots and not any(_is_relative_to(resolved, root.resolve()) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{label} must be under one of: {roots}")
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _validate_output_dir(value: str, allowed_roots: list[Path], label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = path.resolve()
    if allowed_roots and not any(_is_relative_to(resolved, root.resolve()) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{label} must be under one of: {roots}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_dependencies() -> None:
    if _DEPENDENCY_ERROR is None:
        return
    raise ModuleNotFoundError("training preview requires Pillow and numpy") from _DEPENDENCY_ERROR
