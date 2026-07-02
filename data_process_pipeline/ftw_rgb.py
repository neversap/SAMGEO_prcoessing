"""Build RGB-only FTW pretraining data from official FTW-style samples.

The converter keeps the official FTW data read-only and writes a normalized
RGB-only view that matches the in-house training format:

image: [3, H, W]
mask:  [H, W] with 0 background, 1 interior, 2 boundary, 255 ignore

Use a manifest when possible. It avoids guessing the official dataset layout:

image_path,mask_path,split,country,sample_id
/data/ftw/Rwanda/sample_001.tif,/data/ftw/Rwanda/sample_001_mask.tif,train,Rwanda,sample_001
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
    import rasterio
except ModuleNotFoundError as exc:
    np = None
    rasterio = None
    _FTW_DEPENDENCY_ERROR = exc
else:
    _FTW_DEPENDENCY_ERROR = None

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".tif", ".tiff", ".TIF", ".TIFF", ".npz")
MASK_KEYS = ("mask", "y", "label", "labels", "target")
IMAGE_KEYS = ("image", "x", "arr_0")
IGNORE_VALUE = 255


@dataclass(frozen=True)
class FTWRGBConfig:
    ftw_root: Path
    output_dir: Path
    metadata_dir: Path
    manifest_path: Path | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    use_both_windows: bool = True
    max_samples: int = 0
    copy_masks: bool = True


@dataclass(frozen=True)
class FTWSample:
    image_path: Path
    mask_path: Path | None
    split: str
    country: str
    sample_id: str


def run_ftw_rgb_pipeline(config: FTWRGBConfig) -> None:
    """Convert official FTW samples into RGB-only pretraining patches."""
    _ensure_dependencies()
    _validate_config(config)
    _create_output_dirs(config.output_dir, config.metadata_dir)

    samples = _load_samples(config)
    if config.max_samples > 0:
        samples = samples[: config.max_samples]
    if not samples:
        raise FileNotFoundError(f"No FTW samples found under {config.ftw_root}")

    inspection_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    stats = _BandStatsAccumulator()

    for index, sample in enumerate(samples, start=1):
        LOGGER.info("Converting FTW sample %s/%s: %s", index, len(samples), sample.image_path)
        image, mask = _read_sample(sample)
        _validate_image_mask(image, mask, sample)
        inspection_rows.append(_inspect_sample(sample, image, mask))

        windows = [(0, image[[0, 1, 2], :, :])]
        if config.use_both_windows:
            windows.append((1, image[[4, 5, 6], :, :]))

        for window_id, rgb in windows:
            patch_id = _patch_id(sample, window_id)
            image_out = config.output_dir / sample.split / "images" / f"{patch_id}.tif"
            mask_out = config.output_dir / sample.split / "masks" / f"{patch_id}_mask.tif"
            _write_rgb_image(image_out, rgb, sample.image_path)
            if config.copy_masks:
                _write_mask(mask_out, mask, sample.mask_path or sample.image_path)

            ratios = _mask_ratios(mask)
            stats.update(rgb, mask != IGNORE_VALUE)
            patch_rows.append(
                {
                    "patch_id": patch_id,
                    "source_dataset": "ftw",
                    "country": sample.country,
                    "source_sample": sample.sample_id,
                    "window_id": f"t{window_id + 1}",
                    "split": sample.split,
                    "image_path": str(image_out),
                    "mask_path": str(mask_out),
                    "interior_ratio": round(ratios["interior_ratio"], 8),
                    "boundary_ratio": round(ratios["boundary_ratio"], 8),
                    "background_ratio": round(ratios["background_ratio"], 8),
                    "ignore_ratio": round(ratios["ignore_ratio"], 8),
                }
            )

    _write_csv(
        config.metadata_dir / "ftw_inspection.csv",
        [
            "sample_id",
            "country",
            "split",
            "image_path",
            "mask_path",
            "channels",
            "height",
            "width",
            "dtype",
            "mask_values",
        ],
        inspection_rows,
    )
    _write_csv(
        config.metadata_dir / "ftw_patch_index.csv",
        [
            "patch_id",
            "source_dataset",
            "country",
            "source_sample",
            "window_id",
            "split",
            "interior_ratio",
            "boundary_ratio",
            "background_ratio",
            "ignore_ratio",
            "image_path",
            "mask_path",
        ],
        patch_rows,
    )
    _write_json(config.metadata_dir / "ftw_band_stats.json", stats.finalize())
    _write_json(config.metadata_dir / "ftw_rgb_config.json", _serialize_config(config))


def _ensure_dependencies() -> None:
    if _FTW_DEPENDENCY_ERROR is None:
        return
    raise ModuleNotFoundError(
        "FTW RGB preprocessing requires numpy and rasterio. "
        "Install them with: pip install -r requirements-data-process.txt"
    ) from _FTW_DEPENDENCY_ERROR


def _validate_config(config: FTWRGBConfig) -> None:
    if not config.ftw_root.exists():
        raise FileNotFoundError(f"ftw_root does not exist: {config.ftw_root}")
    if config.manifest_path is not None and not config.manifest_path.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {config.manifest_path}")
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def _create_output_dirs(output_dir: Path, metadata_dir: Path) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def _load_samples(config: FTWRGBConfig) -> list[FTWSample]:
    if config.manifest_path is not None:
        return _read_manifest(config.manifest_path, config.ftw_root, config)
    discovered = _discover_samples(config.ftw_root)
    splits = _assign_splits([sample.sample_id for sample in discovered], config)
    return [
        FTWSample(
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            split=splits[sample.sample_id],
            country=sample.country,
            sample_id=sample.sample_id,
        )
        for sample in discovered
    ]


def _read_manifest(path: Path, ftw_root: Path, config: FTWRGBConfig) -> list[FTWSample]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_path = _resolve_path(row["image_path"], ftw_root)
            mask_value = row.get("mask_path", "")
            mask_path = _resolve_path(mask_value, ftw_root) if mask_value else None
            sample_id = row.get("sample_id") or image_path.stem
            rows.append(
                FTWSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    split=(row.get("split") or "").strip().lower(),
                    country=row.get("country") or _infer_country(image_path, ftw_root),
                    sample_id=sample_id,
                )
            )

    needs_split = [sample for sample in rows if sample.split not in {"train", "val", "test"}]
    if not needs_split:
        return rows
    splits = _assign_splits([sample.sample_id for sample in rows], config)
    return [
        FTWSample(
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            split=sample.split if sample.split in {"train", "val", "test"} else splits[sample.sample_id],
            country=sample.country,
            sample_id=sample.sample_id,
        )
        for sample in rows
    ]


def _discover_samples(ftw_root: Path) -> list[FTWSample]:
    samples = []
    for path in sorted(ftw_root.rglob("*")):
        if not path.is_file() or path.suffix not in IMAGE_EXTENSIONS:
            continue
        if _looks_like_mask(path):
            continue
        mask_path = _find_mask_path(path)
        if path.suffix.lower() != ".npz" and mask_path is None:
            continue
        samples.append(
            FTWSample(
                image_path=path,
                mask_path=mask_path,
                split="",
                country=_infer_country(path, ftw_root),
                sample_id=path.stem,
            )
        )
    return samples


def _read_sample(sample: FTWSample):
    if sample.image_path.suffix.lower() == ".npz":
        return _read_npz_sample(sample.image_path, sample.mask_path)
    image = _read_raster_image(sample.image_path)
    if sample.mask_path is None:
        raise ValueError(f"mask_path is required for raster image sample: {sample.image_path}")
    mask = _read_mask(sample.mask_path)
    return image, mask


def _read_npz_sample(image_path: Path, mask_path: Path | None):
    with np.load(image_path) as data:
        image = _get_npz_array(data, IMAGE_KEYS, image_path)
        if mask_path is not None:
            mask = _read_mask(mask_path)
        else:
            mask = _get_npz_array(data, MASK_KEYS, image_path)
    image = _to_chw(image)
    return image, _to_hw_mask(mask).astype("uint8")


def _get_npz_array(data, keys: Sequence[str], path: Path):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"{path} does not contain any of keys: {', '.join(keys)}")


def _read_raster_image(path: Path):
    with rasterio.open(path) as src:
        return src.read()


def _read_mask(path: Path):
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return _to_hw_mask(_get_npz_array(data, MASK_KEYS, path)).astype("uint8")
    with rasterio.open(path) as src:
        return src.read(1).astype("uint8")


def _to_chw(array):
    if array.ndim != 3:
        raise ValueError(f"image array must be 3D, got shape {array.shape}")
    if array.shape[0] in {3, 4, 8}:
        return array
    if array.shape[-1] in {3, 4, 8}:
        return array.transpose(2, 0, 1)
    raise ValueError(f"cannot infer channel dimension from shape {array.shape}")


def _to_hw_mask(array):
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[:, :, 0]
    raise ValueError(f"mask array must be 2D or single-channel 3D, got shape {array.shape}")


def _validate_image_mask(image, mask, sample: FTWSample) -> None:
    if image.ndim != 3 or image.shape[0] < 8:
        raise ValueError(f"FTW image must have at least 8 channels: {sample.image_path} shape={image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"FTW mask must be 2D: {sample.mask_path or sample.image_path} shape={mask.shape}")
    if image.shape[1:] != mask.shape:
        raise ValueError(
            f"image/mask shape mismatch for {sample.sample_id}: image={image.shape}, mask={mask.shape}"
        )


def _inspect_sample(sample: FTWSample, image, mask) -> dict[str, Any]:
    values = sorted(int(value) for value in np.unique(mask))
    return {
        "sample_id": sample.sample_id,
        "country": sample.country,
        "split": sample.split,
        "image_path": str(sample.image_path),
        "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
        "channels": image.shape[0],
        "height": image.shape[1],
        "width": image.shape[2],
        "dtype": str(image.dtype),
        "mask_values": json.dumps(values),
    }


def _write_rgb_image(path: Path, rgb, reference_path: Path) -> None:
    profile = None
    if reference_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
    if profile is None:
        profile = {
            "driver": "GTiff",
            "height": rgb.shape[1],
            "width": rgb.shape[2],
            "count": 3,
            "dtype": str(rgb.dtype),
            "compress": "deflate",
        }
    else:
        profile.update(count=3, height=rgb.shape[1], width=rgb.shape[2], dtype=str(rgb.dtype), compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb)


def _write_mask(path: Path, mask, reference_path: Path) -> None:
    profile = None
    if reference_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
    if profile is None:
        profile = {
            "driver": "GTiff",
            "height": mask.shape[0],
            "width": mask.shape[1],
            "count": 1,
            "dtype": "uint8",
            "compress": "deflate",
        }
    else:
        profile.update(count=1, height=mask.shape[0], width=mask.shape[1], dtype="uint8", nodata=IGNORE_VALUE, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def _mask_ratios(mask) -> dict[str, float]:
    total = float(mask.size)
    valid_total = max(float(np.count_nonzero(mask != IGNORE_VALUE)), 1.0)
    return {
        "interior_ratio": float(np.count_nonzero(mask == 1) / valid_total),
        "boundary_ratio": float(np.count_nonzero(mask == 2) / valid_total),
        "background_ratio": float(np.count_nonzero(mask == 0) / valid_total),
        "ignore_ratio": float(np.count_nonzero(mask == IGNORE_VALUE) / total),
    }


def _patch_id(sample: FTWSample, window_id: int) -> str:
    safe_country = _safe_name(sample.country)
    safe_sample = _safe_name(sample.sample_id)
    return f"{safe_country}_{safe_sample}_t{window_id + 1}"


def _assign_splits(sample_ids: Sequence[str], config: FTWRGBConfig) -> dict[str, str]:
    names = list(sample_ids)
    random.Random(config.seed).shuffle(names)
    total = len(names)
    train_count = int(round(total * config.train_ratio))
    val_count = int(round(total * config.val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    splits = {}
    for index, name in enumerate(names):
        if index < train_count:
            splits[name] = "train"
        elif index < train_count + val_count:
            splits[name] = "val"
        else:
            splits[name] = "test"
    return splits


def _find_mask_path(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}"),
        image_path.with_name(f"{image_path.stem}_label{image_path.suffix}"),
        image_path.with_name(f"{image_path.stem}_labels{image_path.suffix}"),
        image_path.parent / "masks" / f"{image_path.stem}_mask{image_path.suffix}",
        image_path.parent / "labels" / f"{image_path.stem}_mask{image_path.suffix}",
        image_path.parent.parent / "masks" / f"{image_path.stem}_mask{image_path.suffix}",
        image_path.parent.parent / "labels" / f"{image_path.stem}_mask{image_path.suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _looks_like_mask(path: Path) -> bool:
    lowered = path.stem.lower()
    return lowered.endswith("_mask") or lowered.endswith("_label") or lowered.endswith("_labels")


def _infer_country(path: Path, ftw_root: Path) -> str:
    try:
        relative = path.relative_to(ftw_root)
    except ValueError:
        return "unknown"
    return relative.parts[0] if len(relative.parts) > 1 else "unknown"


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _serialize_config(config: FTWRGBConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("ftw_root", "output_dir", "metadata_dir", "manifest_path"):
        if value[key] is not None:
            value[key] = str(value[key])
    return value


class _BandStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)

    def update(self, rgb, valid_mask) -> None:
        data = rgb.astype(np.float64)
        valid = valid_mask.reshape(-1)
        flat = data.reshape(3, -1)
        if not np.any(valid):
            return
        flat = flat[:, valid]
        self.count += flat.shape[1]
        self.sum += np.sum(flat, axis=1)
        self.sum_sq += np.sum(flat * flat, axis=1)
        self.minimum = np.minimum(self.minimum, np.min(flat, axis=1))
        self.maximum = np.maximum(self.maximum, np.max(flat, axis=1))

    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            return {"pixel_count": 0, "bands": []}
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        return {
            "pixel_count": self.count,
            "bands": [
                {
                    "band": index + 1,
                    "name": ["R", "G", "B"][index],
                    "min": float(self.minimum[index]),
                    "max": float(self.maximum[index]),
                    "mean": float(mean[index]),
                    "std": float(std[index]),
                }
                for index in range(3)
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FTW official samples to RGB-only pretraining data.")
    parser.add_argument("--ftw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-window", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    config = FTWRGBConfig(
        ftw_root=Path(args.ftw_root),
        output_dir=Path(args.output_dir),
        metadata_dir=Path(args.metadata_dir),
        manifest_path=Path(args.manifest) if args.manifest else None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_both_windows=not args.single_window,
        max_samples=args.max_samples,
    )
    run_ftw_rgb_pipeline(config)


if __name__ == "__main__":
    main()
