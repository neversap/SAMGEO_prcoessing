from __future__ import annotations

import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
    import rasterio
    from PIL import Image
except ModuleNotFoundError as exc:
    np = None
    rasterio = None
    Image = None
    _DEPENDENCY_ERROR = exc
else:
    _DEPENDENCY_ERROR = None


IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
IGNORE_VALUE = 255
DEFAULT_BUCKET_WEIGHTS = {
    "boundary_rich": 3.0,
    "low_fg": 1.5,
    "mid_fg": 1.0,
    "high_fg": 0.7,
    "very_high_fg": 0.4,
    "near_background": 0.3,
}
DEFAULT_GDAL_CACHEMAX_MB = 64
_GDAL_CACHEMAX_MB = DEFAULT_GDAL_CACHEMAX_MB


def configure_gdal_cache(cachemax_mb: int | float | str | None = DEFAULT_GDAL_CACHEMAX_MB) -> None:
    global _GDAL_CACHEMAX_MB
    try:
        value = int(float(cachemax_mb or DEFAULT_GDAL_CACHEMAX_MB))
    except (TypeError, ValueError):
        value = DEFAULT_GDAL_CACHEMAX_MB
    _GDAL_CACHEMAX_MB = max(1, value)
    os.environ["GDAL_CACHEMAX"] = str(_GDAL_CACHEMAX_MB)


@dataclass(frozen=True)
class FTWIndexConfig:
    ftw_root: Path
    metadata_dir: Path
    country: str = "all"
    window: str = "both"
    mask_type: str = "semantic_3class"
    output_name: str = "ftw_dataloader_index.csv"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    max_samples: int = 0


@dataclass(frozen=True)
class IndexResult:
    index_path: Path
    stats_path: Path
    count: int
    buckets: dict[str, int]
    splits: dict[str, int]


def build_ftw_official_index(config: FTWIndexConfig) -> IndexResult:
    _ensure_dependencies()
    root = config.ftw_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ftw_root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"ftw_root must be a directory: {root}")
    _validate_ratios(config.train_ratio, config.val_ratio, config.test_ratio)

    rows = discover_ftw_official_rows(
        ftw_root=root,
        country=config.country,
        window=config.window,
        mask_type=config.mask_type,
        max_samples=config.max_samples,
        seed=config.seed,
    )
    split_by_id = _assign_splits(
        [row["sample_id"] for row in rows],
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    for row in rows:
        row["split"] = split_by_id[row["sample_id"]]

    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    index_path = config.metadata_dir / config.output_name
    stats_path = config.metadata_dir / "ftw_dataloader_stats.json"
    _write_csv(index_path, _index_fields(), rows)
    stats = summarize_rows(rows)
    _write_json(stats_path, stats)
    return IndexResult(
        index_path=index_path,
        stats_path=stats_path,
        count=len(rows),
        buckets=stats["buckets"],
        splits=stats["splits"],
    )


def discover_ftw_official_rows(
    ftw_root: Path,
    country: str = "all",
    window: str = "both",
    mask_type: str = "semantic_3class",
    max_samples: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    _ensure_dependencies()
    normalized_window = window.strip().lower()
    if normalized_window not in {"window_a", "window_b", "both", "all"}:
        raise ValueError("window must be window_a, window_b, or both")
    windows = ["window_a", "window_b"] if normalized_window in {"both", "all"} else [normalized_window]
    normalized_country = country.strip().lower()
    countries = [
        item
        for item in sorted(ftw_root.iterdir())
        if item.is_dir() and (normalized_country in {"", "all"} or item.name.lower() == normalized_country)
    ]
    if not countries:
        raise FileNotFoundError(f"No FTW country directory matched {country!r} under {ftw_root}")

    rows: list[dict[str, Any]] = []
    for country_dir in countries:
        mask_dir = country_dir / "label_masks" / mask_type
        if not mask_dir.exists():
            continue
        mask_by_stem = {_normalize_stem(path): path for path in _list_files(mask_dir)}
        for item_window in windows:
            image_dir = country_dir / "s2_images" / item_window
            if not image_dir.exists():
                continue
            for image_path in _list_files(image_dir):
                mask_path = mask_by_stem.get(_normalize_stem(image_path))
                if mask_path is None:
                    continue
                ratios, width, height = inspect_mask(mask_path)
                bucket = assign_bucket(
                    ratios["cropland_ratio"],
                    ratios["boundary_ratio"],
                    ratios["ignore_ratio"],
                )
                sample_id = f"{country_dir.name}_{image_path.stem}_{item_window}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "patch_name": sample_id,
                        "source_dataset": "ftw",
                        "source_tif": f"{country_dir.name}/{item_window}/{image_path.name}",
                        "country": country_dir.name,
                        "window": item_window,
                        "mask_type": mask_type,
                        "split": "",
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "fg_ratio": ratios["cropland_ratio"],
                        **ratios,
                        "bucket": bucket,
                        "use_for_train": 0 if bucket == "drop" else 1,
                        "note": "",
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"No FTW image/mask pairs found under {ftw_root} for "
            f"country={country!r}, window={window!r}, mask_type={mask_type!r}"
        )
    if max_samples > 0 and len(rows) > max_samples:
        selected = rows[:]
        random.Random(seed).shuffle(selected)
        rows = selected[:max_samples]
    return rows


def load_inhouse_index(dataset_dir: Path) -> list[dict[str, Any]]:
    patch_index = dataset_dir / "metadata" / "patch_index.csv"
    if not patch_index.exists():
        raise FileNotFoundError(f"patch_index.csv does not exist: {patch_index}")
    rows = _read_csv(patch_index)
    normalized = []
    for row in rows:
        cropland_ratio = _float_value(row, "cropland_ratio")
        boundary_ratio = _float_value(row, "boundary_ratio")
        ignore_ratio = _float_value(row, "ignore_ratio")
        sample_id = row.get("patch_name") or row.get("patch_id") or Path(row.get("image_path", "")).stem
        normalized.append(
            {
                "sample_id": sample_id,
                "patch_name": sample_id,
                "source_dataset": "inhouse",
                "source_tif": row.get("source_tif", ""),
                "country": "",
                "window": "",
                "mask_type": "inhouse",
                "split": row.get("split", ""),
                "x": _int_value(row, "x"),
                "y": _int_value(row, "y"),
                "width": _int_value(row, "width"),
                "height": _int_value(row, "height"),
                "image_path": str(_resolve_dataset_path(dataset_dir, row.get("image_path", ""))),
                "mask_path": str(_resolve_dataset_path(dataset_dir, row.get("mask_path", ""))),
                "fg_ratio": cropland_ratio,
                "cropland_ratio": cropland_ratio,
                "interior_ratio": _float_value(row, "interior_ratio"),
                "boundary_ratio": boundary_ratio,
                "background_ratio": _float_value(row, "background_ratio"),
                "ignore_ratio": ignore_ratio,
                "bucket": assign_bucket(cropland_ratio, boundary_ratio, ignore_ratio),
                "use_for_train": 1,
                "note": row.get("patch_type", ""),
            }
        )
    return normalized


def assign_bucket(
    fg_ratio: float,
    boundary_ratio: float,
    ignore_ratio: float,
    ignore_ratio_drop: float = 0.5,
    boundary_rich_threshold: float = 0.005,
    fg_near_background: float = 0.01,
    fg_low: float = 0.10,
    fg_high: float = 0.70,
    fg_very_high: float = 0.95,
) -> str:
    if ignore_ratio > ignore_ratio_drop:
        return "drop"
    if boundary_ratio > boundary_rich_threshold:
        return "boundary_rich"
    if fg_ratio < fg_near_background:
        return "near_background"
    if fg_ratio < fg_low:
        return "low_fg"
    if fg_ratio < fg_high:
        return "mid_fg"
    if fg_ratio < fg_very_high:
        return "high_fg"
    return "very_high_fg"


def make_weighted_sampler(metadata_csv: str | Path, bucket_weights: dict[str, float] | None = None):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Weighted sampler requires torch") from exc

    rows = [row for row in _read_csv(Path(metadata_csv)) if _int_text(row.get("use_for_train", "1")) == 1]
    weights_by_bucket = bucket_weights or DEFAULT_BUCKET_WEIGHTS
    weights = [float(weights_by_bucket.get(row.get("bucket", ""), 1.0)) for row in rows]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


class MetadataPatchDataset:
    def __init__(
        self,
        metadata_csv: str | Path,
        transform=None,
        normalize: bool = True,
        stats: dict[str, Sequence[float]] | None = None,
        image_scale: float | None = None,
    ) -> None:
        _ensure_dependencies()
        self.rows = [row for row in _read_csv(Path(metadata_csv)) if _int_text(row.get("use_for_train", "1")) == 1]
        self.transform = transform
        self.normalize = normalize
        self.stats = stats
        self.image_scale = image_scale

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("MetadataPatchDataset requires torch") from exc

        row = self.rows[index]
        image = read_image_chw(Path(row["image_path"]))
        mask = read_mask(Path(row["mask_path"]))
        image = image.astype("float32")
        scale = self.image_scale or infer_image_scale(image)
        if scale > 1.0:
            image = image / scale
        image = np.clip(image, 0.0, 1.0)
        image_hwc = image.transpose(1, 2, 0)
        if self.transform is not None:
            augmented = self.transform(image=image_hwc, mask=mask)
            image_hwc = augmented["image"]
            mask = augmented["mask"]
        image = image_hwc.transpose(2, 0, 1).astype("float32")
        if self.normalize and self.stats:
            mean = np.asarray(self.stats["mean"], dtype="float32")[:, None, None]
            std = np.asarray(self.stats["std"], dtype="float32")[:, None, None]
            image = (image - mean) / np.maximum(std, 1e-6)
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask.astype("int64")).long(),
            "meta": {
                "sample_id": row.get("sample_id", ""),
                "country": row.get("country", ""),
                "window": row.get("window", ""),
                "fg_ratio": _float_text(row.get("fg_ratio", "0")),
                "boundary_ratio": _float_text(row.get("boundary_ratio", "0")),
                "bucket": row.get("bucket", ""),
            },
        }


def read_image_chw(path: Path):
    _ensure_dependencies()
    if path.suffix.lower() in {".tif", ".tiff"}:
        with _rasterio_env(), rasterio.open(path) as src:
            band_count = min(src.count, 3)
            data = src.read(list(range(1, band_count + 1))).astype("float32")
            if band_count == 1:
                data = np.repeat(data, 3, axis=0)
            elif band_count == 2:
                data = np.concatenate([data, data[:1]], axis=0)
        return data
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype="float32").transpose(2, 0, 1)


def read_mask(path: Path):
    _ensure_dependencies()
    if path.suffix.lower() in {".tif", ".tiff"}:
        with _rasterio_env(), rasterio.open(path) as src:
            return src.read(1)
    return np.asarray(Image.open(path), dtype="uint8")


def infer_image_scale(image) -> float:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 1.0
    high = float(np.percentile(finite, 99))
    if high > 255.0:
        return 10000.0
    if high > 1.5:
        return 255.0
    return 1.0


def inspect_mask(mask_path: Path) -> tuple[dict[str, float], int, int]:
    mask = read_mask(mask_path)
    total = float(mask.size)
    valid_total = max(float(np.count_nonzero(mask != IGNORE_VALUE)), 1.0)
    interior_ratio = float(np.count_nonzero(mask == 1) / valid_total)
    boundary_ratio = float(np.count_nonzero(mask == 2) / valid_total)
    background_ratio = float(np.count_nonzero(mask == 0) / valid_total)
    ignore_ratio = float(np.count_nonzero(mask == IGNORE_VALUE) / total)
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


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, int] = {}
    splits: dict[str, int] = {}
    countries: dict[str, int] = {}
    for row in rows:
        buckets[row.get("bucket", "")] = buckets.get(row.get("bucket", ""), 0) + 1
        splits[row.get("split", "")] = splits.get(row.get("split", ""), 0) + 1
        countries[row.get("country", "")] = countries.get(row.get("country", ""), 0) + 1
    return {
        "count": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "splits": dict(sorted(splits.items())),
        "countries": dict(sorted(countries.items())),
    }


def select_rows(
    rows: list[dict[str, Any]],
    split: str = "all",
    mode: str = "random",
    limit: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    normalized_split = split.strip().lower()
    if normalized_split not in {"", "all"}:
        rows = [row for row in rows if str(row.get("split", "")).lower() == normalized_split]
    normalized_mode = mode.strip().lower()
    limit = max(1, min(int(limit), 50))
    if normalized_mode == "random":
        selected = rows[:]
        random.Random(seed).shuffle(selected)
        return selected[:limit]
    if normalized_mode == "boundary":
        return sorted(rows, key=lambda row: float(row.get("boundary_ratio", 0.0)), reverse=True)[:limit]
    if normalized_mode == "low_fg":
        candidates = [row for row in rows if 0.0 < float(row.get("fg_ratio", 0.0)) <= 0.15]
        return sorted(candidates, key=lambda row: float(row.get("fg_ratio", 0.0)))[:limit]
    if normalized_mode == "high_fg":
        return sorted(rows, key=lambda row: float(row.get("fg_ratio", 0.0)), reverse=True)[:limit]
    return [row for row in rows if row.get("bucket") == normalized_mode][:limit]


def _assign_splits(
    sample_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    names = list(sample_ids)
    random.Random(seed).shuffle(names)
    train_count = int(len(names) * train_ratio)
    val_count = int(len(names) * val_ratio)
    split_by_name = {}
    for index, name in enumerate(names):
        if index < train_count:
            split_by_name[name] = "train"
        elif index < train_count + val_count:
            split_by_name[name] = "val"
        else:
            split_by_name[name] = "test"
    return split_by_name


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def _index_fields() -> list[str]:
    return [
        "sample_id",
        "patch_name",
        "source_dataset",
        "source_tif",
        "country",
        "window",
        "mask_type",
        "split",
        "x",
        "y",
        "width",
        "height",
        "image_path",
        "mask_path",
        "fg_ratio",
        "cropland_ratio",
        "interior_ratio",
        "boundary_ratio",
        "background_ratio",
        "ignore_ratio",
        "bucket",
        "use_for_train",
        "note",
    ]


def _list_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def _normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "_label", "_labels", "_semantic", "_semantic_3class", "_semantic_2class"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_dataset_path(dataset_dir: Path, value: str) -> Path:
    if not value:
        return dataset_dir
    path = Path(value)
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _float_value(row: dict[str, str], key: str) -> float:
    return _float_text(row.get(key, "0"))


def _int_value(row: dict[str, str], key: str) -> int:
    return _int_text(row.get(key, "0"))


def _float_text(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _int_text(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _ensure_dependencies() -> None:
    if _DEPENDENCY_ERROR is None:
        return
    raise ModuleNotFoundError(
        "dataloader pipeline requires geospatial dependencies. "
        "Install them with: pip install -r requirements-data-process.txt"
    ) from _DEPENDENCY_ERROR


def _rasterio_env():
    if rasterio is None:
        return nullcontext()
    return rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHEMAX_MB)
