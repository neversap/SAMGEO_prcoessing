"""Build crop-segmentation training data from remote-sensing rasters and labels.

Expected dataset layout:

dataset/
  raw/images/*.tif
  raw/labels/*.shp
  # Or legacy layout:
  tif/*.tif
  shp/*.shp
  processed/
  metadata/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import box
except ModuleNotFoundError as exc:
    gpd = None
    np = None
    pd = None
    rasterio = None
    rasterize = None
    Window = None
    box = None
    _GEO_DEPENDENCY_ERROR = exc
else:
    _GEO_DEPENDENCY_ERROR = None

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".tif", ".tiff", ".TIF", ".TIFF")
ProgressCallback = Callable[[dict[str, Any]], None]
MASK_MODE_BINARY = "binary"
MASK_MODE_FIELD_BOUNDARY_3CLASS = "field_boundary_3class"
MASK_MODES = {MASK_MODE_BINARY, MASK_MODE_FIELD_BOUNDARY_3CLASS}
IGNORE_VALUE = 255


@dataclass(frozen=True)
class DataProcessConfig:
    dataset_dir: Path = Path("dataset")
    tile_size: int = 512
    overlap: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    mask_value: int = 1
    all_touched: bool = False
    keep_empty: bool = True
    min_patch_size: int = 128
    split_strategy: str = "image"
    test_process: bool = False
    mask_mode: str = MASK_MODE_BINARY
    boundary_width_pixels: int = 2
    background_keep_ratio: float = 0.2
    max_ignore_ratio: float = 0.5
    black_pixel_threshold: float = 0.0

    @property
    def raw_images_dir(self) -> Path:
        return _first_existing_dir(
            self.dataset_dir / "raw" / "images",
            self.dataset_dir / "tif",
        )

    @property
    def raw_labels_dir(self) -> Path:
        return _first_existing_dir(
            self.dataset_dir / "raw" / "labels",
            self.dataset_dir / "shp",
        )

    @property
    def processed_dir(self) -> Path:
        return self.dataset_dir / "processed"

    @property
    def full_masks_dir(self) -> Path:
        return self.processed_dir / "masks"

    @property
    def patches_dir(self) -> Path:
        return self.processed_dir / "patches"

    @property
    def metadata_dir(self) -> Path:
        return self.dataset_dir / "metadata"


def run_pipeline(config: DataProcessConfig, progress_callback: ProgressCallback | None = None) -> None:
    """Run the full preprocessing pipeline."""
    _ensure_geo_dependencies()
    _validate_config(config)
    _create_output_dirs(config)
    _emit_progress(progress_callback, "initializing", 0, 1, "Preparing dataset directories")

    image_paths = _list_images(config.raw_images_dir)
    if config.test_process:
        image_paths = image_paths[:1]
    _emit_progress(progress_callback, "loading labels", 0, 1, f"Loading labels from {config.raw_labels_dir}")
    labels = _load_labels(config.raw_labels_dir)

    if not image_paths:
        raise FileNotFoundError(f"No TIF images found in {config.raw_images_dir}")

    image_infos = []
    label_infos = _collect_label_info(labels)
    overlap_rows = []
    patch_rows = []
    band_stats_accumulator = _BandStatsAccumulator()
    image_splits = _assign_image_splits(image_paths, config)
    total_patches = _estimate_total_patches(image_paths, config)
    patches_done = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        LOGGER.info("Processing image: %s", image_path)
        _emit_progress(
            progress_callback,
            "rasterizing masks",
            image_index - 1,
            len(image_paths),
            f"Opening {image_path.name}",
        )
        with rasterio.open(image_path) as src:
            image_info = _collect_image_info(src, image_path, image_splits[image_path.name])
            image_infos.append(image_info)
            band_stats_accumulator.update(src, black_pixel_threshold=config.black_pixel_threshold)

            image_labels = _labels_for_image(labels, src)
            overlap_rows.append(_collect_overlap_info(image_path, image_labels))
            full_mask_path = _write_full_mask(src, image_labels, image_path, config)
            LOGGER.info("Wrote full mask: %s", full_mask_path)

            patch_split = image_splits[image_path.name]
            written_rows = _write_patches(
                src,
                full_mask_path,
                image_path,
                patch_split,
                config,
                progress_callback=progress_callback,
                progress_offset=patches_done,
                progress_total=total_patches,
            )
            patches_done += _count_windows(src.width, src.height, config.tile_size, config.overlap)
            patch_rows.extend(written_rows)
        _emit_progress(
            progress_callback,
            "rasterizing masks",
            image_index,
            len(image_paths),
            f"Finished mask for {image_path.name}",
        )

    if config.split_strategy == "patch":
        _emit_progress(progress_callback, "splitting patches", 0, 1, "Moving patches into split directories")
        patch_rows = _assign_patch_splits(patch_rows, config)

    _emit_progress(progress_callback, "writing metadata", 0, 5, "Writing image_info.csv")
    _write_image_info(config.metadata_dir / "image_info.csv", image_infos)
    _emit_progress(progress_callback, "writing metadata", 1, 5, "Writing label_info.csv")
    _write_label_info(config.metadata_dir / "label_info.csv", label_infos)
    _emit_progress(progress_callback, "writing metadata", 2, 5, "Writing overlap_report.csv")
    _write_overlap_report(config.metadata_dir / "overlap_report.csv", overlap_rows)
    _emit_progress(progress_callback, "writing metadata", 3, 5, "Writing patch_index.csv")
    _write_patch_index(config.metadata_dir / "patch_index.csv", patch_rows)
    _emit_progress(progress_callback, "writing metadata", 4, 5, "Writing band_stats.json")
    _write_band_stats(config.metadata_dir / "band_stats.json", band_stats_accumulator.finalize())
    _write_run_config(config.metadata_dir / "pipeline_config.json", config)
    _emit_progress(progress_callback, "done", 1, 1, f"Generated {len(patch_rows)} patches")


def _validate_config(config: DataProcessConfig) -> None:
    if config.tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if config.overlap < 0:
        raise ValueError("overlap must be non-negative")
    if config.overlap * 2 >= config.tile_size:
        raise ValueError("overlap must be less than half of tile_size")
    if config.min_patch_size <= 0:
        raise ValueError("min_patch_size must be positive")
    if config.split_strategy not in {"image", "patch"}:
        raise ValueError("split_strategy must be 'image' or 'patch'")
    if config.mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of: {', '.join(sorted(MASK_MODES))}")
    if config.boundary_width_pixels < 0:
        raise ValueError("boundary_width_pixels must be non-negative")
    if not 0.0 <= config.background_keep_ratio <= 1.0:
        raise ValueError("background_keep_ratio must be between 0 and 1")
    if not 0.0 <= config.max_ignore_ratio <= 1.0:
        raise ValueError("max_ignore_ratio must be between 0 and 1")
    if config.black_pixel_threshold < 0:
        raise ValueError("black_pixel_threshold must be non-negative")

    total = config.train_ratio + config.val_ratio + config.test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def _ensure_geo_dependencies() -> None:
    if _GEO_DEPENDENCY_ERROR is None:
        return
    raise ModuleNotFoundError(
        "data_process_pipeline requires geospatial dependencies. "
        "Install them with: pip install -r requirements-data-process.txt"
    ) from _GEO_DEPENDENCY_ERROR


def _create_output_dirs(config: DataProcessConfig) -> None:
    config.full_masks_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (config.patches_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (config.patches_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def _first_existing_dir(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(
        {
            "stage": stage,
            "current": int(current),
            "total": max(int(total), 1),
            "message": message,
        }
    )


def _list_images(raw_images_dir: Path) -> list[Path]:
    return sorted(
        path for path in raw_images_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    )


def _load_labels(raw_labels_dir: Path) -> gpd.GeoDataFrame:
    shp_files = sorted(raw_labels_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No SHP labels found in {raw_labels_dir}")

    frames = []
    for shp_path in shp_files:
        LOGGER.info("Loading label file: %s", shp_path)
        gdf = gpd.read_file(shp_path)
        if not gdf.empty:
            frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=None)

    base_crs = frames[0].crs
    aligned = []
    for frame in frames:
        if base_crs and frame.crs and frame.crs != base_crs:
            aligned.append(frame.to_crs(base_crs))
        else:
            aligned.append(frame)

    return gpd.GeoDataFrame(
        pd.concat(aligned, ignore_index=True),
        geometry="geometry",
        crs=base_crs,
    )


def _collect_image_info(src: rasterio.DatasetReader, image_path: Path, split: str) -> dict:
    bounds = src.bounds
    return {
        "image_name": image_path.name,
        "path": str(image_path),
        "width": src.width,
        "height": src.height,
        "count": src.count,
        "dtype": ",".join(src.dtypes),
        "crs": str(src.crs) if src.crs else "",
        "transform": json.dumps([src.transform.a, src.transform.b, src.transform.c, src.transform.d, src.transform.e, src.transform.f]),
        "bounds_left": bounds.left,
        "bounds_bottom": bounds.bottom,
        "bounds_right": bounds.right,
        "bounds_top": bounds.top,
        "nodata": "" if src.nodata is None else src.nodata,
        "split": split,
    }


def _labels_for_image(labels: gpd.GeoDataFrame, src: rasterio.DatasetReader) -> gpd.GeoDataFrame:
    if labels.empty:
        return labels
    if src.crs and labels.crs and labels.crs != src.crs:
        labels = labels.to_crs(src.crs)

    image_bounds = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    return labels[labels.geometry.notna() & labels.intersects(image_bounds)]


def _collect_label_info(labels: gpd.GeoDataFrame) -> list[dict]:
    if labels.empty:
        return []
    rows = []
    geometry_types = {
        str(key): int(value)
        for key, value in labels.geometry.type.value_counts().to_dict().items()
    }
    rows.append(
        {
            "feature_count": len(labels),
            "crs": str(labels.crs) if labels.crs else "",
            "columns": ",".join(str(column) for column in labels.columns),
            "geometry_types": json.dumps(geometry_types, ensure_ascii=False),
        }
    )
    return rows


def _collect_overlap_info(image_path: Path, labels: gpd.GeoDataFrame) -> dict:
    return {
        "image_name": image_path.name,
        "overlap_polygon_count": len(labels),
        "has_overlap": len(labels) > 0,
    }


def _write_full_mask(
    src: rasterio.DatasetReader,
    labels: gpd.GeoDataFrame,
    image_path: Path,
    config: DataProcessConfig,
) -> Path:
    shapes = [(geom, config.mask_value) for geom in labels.geometry if geom is not None and not geom.is_empty]
    if shapes and config.mask_mode == MASK_MODE_BINARY:
        mask = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=config.all_touched,
        )
    elif shapes:
        mask = _rasterize_field_boundary_mask(src, labels, config)
    else:
        mask = np.zeros((src.height, src.width), dtype=np.uint8)

    mask_path = config.full_masks_dir / f"{image_path.stem}_mask.tif"
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=IGNORE_VALUE if config.mask_mode == MASK_MODE_FIELD_BOUNDARY_3CLASS else None,
        compress="deflate",
        predictor=2,
    )
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
    return mask_path


def _rasterize_field_boundary_mask(
    src: rasterio.DatasetReader,
    labels: gpd.GeoDataFrame,
    config: DataProcessConfig,
) -> np.ndarray:
    valid_geometries = [
        geom
        for geom in labels.geometry
        if geom is not None and not geom.is_empty
    ]
    interior_shapes = [(geom, 1) for geom in valid_geometries]
    mask = rasterize(
        interior_shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=config.all_touched,
    )
    if config.boundary_width_pixels <= 0:
        return mask

    pixel_size = max(abs(float(src.res[0])), abs(float(src.res[1])))
    boundary_width = config.boundary_width_pixels * pixel_size
    boundary_shapes = []
    for geom in valid_geometries:
        boundary = geom.boundary.buffer(boundary_width)
        if boundary is not None and not boundary.is_empty:
            boundary_shapes.append((boundary, 2))

    if not boundary_shapes:
        return mask
    boundary = rasterize(
        boundary_shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    mask[boundary == 2] = 2
    return mask


def _write_patches(
    src: rasterio.DatasetReader,
    full_mask_path: Path,
    image_path: Path,
    split: str,
    config: DataProcessConfig,
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int = 1,
) -> list[dict]:
    rows = []
    with rasterio.open(full_mask_path) as mask_src:
        for patch_index, (x, y, width, height) in enumerate(
            _iter_windows(src.width, src.height, config.tile_size, config.overlap),
            start=1,
        ):
            _emit_progress(
                progress_callback,
                "writing patches",
                progress_offset + patch_index,
                progress_total,
                f"{image_path.name} x={x} y={y}",
            )
            if width < config.min_patch_size or height < config.min_patch_size:
                continue

            window = Window(x, y, width, height)
            image_patch = src.read(window=window)
            mask_patch = mask_src.read(1, window=window)

            invalid_pixels = _compute_invalid_pixels(src, image_patch, config.black_pixel_threshold)
            if config.mask_mode == MASK_MODE_FIELD_BOUNDARY_3CLASS:
                mask_patch = mask_patch.copy()
                mask_patch[invalid_pixels] = IGNORE_VALUE

            ratios = _compute_patch_ratios(mask_patch, config)
            ignore_ratio = ratios["ignore_ratio"]
            if ignore_ratio > config.max_ignore_ratio:
                continue
            patch_name = f"{image_path.stem}_x{x:06d}_y{y:06d}"
            if not _should_keep_patch(patch_name, ratios, config):
                continue

            cropland_ratio = ratios["cropland_ratio"]

            image_out = config.patches_dir / split / "images" / f"{patch_name}.tif"
            mask_out = config.patches_dir / split / "masks" / f"{patch_name}_mask.tif"

            patch_transform = src.window_transform(window)
            image_profile = src.profile.copy()
            image_profile.update(
                driver="GTiff",
                height=height,
                width=width,
                transform=patch_transform,
                compress="deflate",
            )
            mask_profile = mask_src.profile.copy()
            mask_profile.update(
                driver="GTiff",
                height=height,
                width=width,
                transform=patch_transform,
                compress="deflate",
                predictor=2,
            )

            with rasterio.open(image_out, "w", **image_profile) as dst:
                dst.write(image_patch)
            with rasterio.open(mask_out, "w", **mask_profile) as dst:
                dst.write(mask_patch, 1)

            rows.append(
                {
                    "patch_name": patch_name,
                    "source_tif": image_path.name,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "image_path": str(image_out),
                    "mask_path": str(mask_out),
                    "cropland_ratio": round(cropland_ratio, 8),
                    "ignore_ratio": round(ignore_ratio, 8),
                    "interior_ratio": round(ratios["interior_ratio"], 8),
                    "boundary_ratio": round(ratios["boundary_ratio"], 8),
                    "background_ratio": round(ratios["background_ratio"], 8),
                    "patch_type": ratios["patch_type"],
                    "split": split,
                }
            )
    return rows


def _iter_windows(width: int, height: int, tile_size: int, overlap: int) -> Iterable[tuple[int, int, int, int]]:
    stride = tile_size - 2 * overlap
    x_starts = _axis_starts(width, tile_size, stride)
    y_starts = _axis_starts(height, tile_size, stride)
    for y in y_starts:
        for x in x_starts:
            yield x, y, min(tile_size, width - x), min(tile_size, height - y)


def _count_windows(width: int, height: int, tile_size: int, overlap: int) -> int:
    stride = tile_size - 2 * overlap
    return len(_axis_starts(width, tile_size, stride)) * len(_axis_starts(height, tile_size, stride))


def _estimate_total_patches(image_paths: Sequence[Path], config: DataProcessConfig) -> int:
    total = 0
    for image_path in image_paths:
        with rasterio.open(image_path) as src:
            total += _count_windows(src.width, src.height, config.tile_size, config.overlap)
    return max(total, 1)


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


def _compute_invalid_pixels(src: rasterio.DatasetReader, image_patch: np.ndarray, black_pixel_threshold: float):
    invalid = np.zeros(image_patch.shape[1:], dtype=bool)
    if src.nodata is not None:
        invalid |= np.any(image_patch == src.nodata, axis=0)
    if black_pixel_threshold >= 0:
        invalid |= np.all(image_patch <= black_pixel_threshold, axis=0)
    return invalid


def _compute_patch_ratios(mask_patch: np.ndarray, config: DataProcessConfig) -> dict[str, float | str]:
    total = float(mask_patch.size)
    ignore_ratio = float(np.count_nonzero(mask_patch == IGNORE_VALUE) / total)
    valid_total = max(float(np.count_nonzero(mask_patch != IGNORE_VALUE)), 1.0)
    if config.mask_mode == MASK_MODE_FIELD_BOUNDARY_3CLASS:
        interior_ratio = float(np.count_nonzero(mask_patch == 1) / valid_total)
        boundary_ratio = float(np.count_nonzero(mask_patch == 2) / valid_total)
        background_ratio = float(np.count_nonzero(mask_patch == 0) / valid_total)
        cropland_ratio = interior_ratio + boundary_ratio
    else:
        interior_ratio = float(np.count_nonzero(mask_patch == config.mask_value) / valid_total)
        boundary_ratio = 0.0
        background_ratio = float(np.count_nonzero(mask_patch == 0) / valid_total)
        cropland_ratio = interior_ratio

    if ignore_ratio >= 1.0:
        patch_type = "ignored"
    elif boundary_ratio > 0.0:
        patch_type = "boundary"
    elif interior_ratio > 0.0:
        patch_type = "interior"
    else:
        patch_type = "background"

    return {
        "cropland_ratio": cropland_ratio,
        "interior_ratio": interior_ratio,
        "boundary_ratio": boundary_ratio,
        "background_ratio": background_ratio,
        "ignore_ratio": ignore_ratio,
        "patch_type": patch_type,
    }


def _should_keep_patch(patch_name: str, ratios: dict[str, float | str], config: DataProcessConfig) -> bool:
    patch_type = str(ratios["patch_type"])
    if patch_type in {"boundary", "interior"}:
        return True
    if patch_type == "ignored":
        return False
    if config.keep_empty:
        return True
    if config.mask_mode == MASK_MODE_BINARY:
        return False
    return random.Random(f"{config.seed}:{patch_name}").random() < config.background_keep_ratio


def _assign_image_splits(image_paths: Sequence[Path], config: DataProcessConfig) -> dict[str, str]:
    names = [path.name for path in image_paths]
    rng = random.Random(config.seed)
    rng.shuffle(names)
    return _split_names(names, config)


def _assign_patch_splits(rows: list[dict], config: DataProcessConfig) -> list[dict]:
    patch_names = [row["patch_name"] for row in rows]
    rng = random.Random(config.seed)
    rng.shuffle(patch_names)
    split_by_name = _split_names(patch_names, config)
    for row in rows:
        split = split_by_name[row["patch_name"]]
        if row["split"] == split:
            continue

        old_image = Path(row["image_path"])
        old_mask = Path(row["mask_path"])
        new_image = config.patches_dir / split / "images" / old_image.name
        new_mask = config.patches_dir / split / "masks" / old_mask.name
        new_image.parent.mkdir(parents=True, exist_ok=True)
        new_mask.parent.mkdir(parents=True, exist_ok=True)
        old_image.replace(new_image)
        old_mask.replace(new_mask)
        row["image_path"] = str(new_image)
        row["mask_path"] = str(new_mask)
        row["split"] = split
    return rows


def _split_names(names: Sequence[str], config: DataProcessConfig) -> dict[str, str]:
    total = len(names)
    train_count = int(round(total * config.train_ratio))
    val_count = int(round(total * config.val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)

    split_by_name = {}
    for idx, name in enumerate(names):
        if idx < train_count:
            split_by_name[name] = "train"
        elif idx < train_count + val_count:
            split_by_name[name] = "val"
        else:
            split_by_name[name] = "test"
    return split_by_name


def _write_image_info(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image_name",
        "path",
        "width",
        "height",
        "count",
        "dtype",
        "crs",
        "transform",
        "bounds_left",
        "bounds_bottom",
        "bounds_right",
        "bounds_top",
        "nodata",
        "split",
    ]
    _write_csv(path, fieldnames, rows)


def _write_label_info(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "feature_count",
        "crs",
        "columns",
        "geometry_types",
    ]
    _write_csv(path, fieldnames, rows)


def _write_overlap_report(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image_name",
        "overlap_polygon_count",
        "has_overlap",
    ]
    _write_csv(path, fieldnames, rows)


def _write_patch_index(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "patch_name",
        "source_tif",
        "x",
        "y",
        "width",
        "height",
        "cropland_ratio",
        "ignore_ratio",
        "interior_ratio",
        "boundary_ratio",
        "background_ratio",
        "patch_type",
        "split",
        "image_path",
        "mask_path",
    ]
    _write_csv(path, fieldnames, rows)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_band_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _write_run_config(path: Path, config: DataProcessConfig) -> None:
    serializable = asdict(config)
    serializable["dataset_dir"] = str(config.dataset_dir)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


class _BandStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum: np.ndarray | None = None
        self.sum_sq: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def update(self, src: rasterio.DatasetReader, block_size: int = 1024, black_pixel_threshold: float = 0.0) -> None:
        for y in range(0, src.height, block_size):
            for x in range(0, src.width, block_size):
                window = Window(x, y, min(block_size, src.width - x), min(block_size, src.height - y))
                data = src.read(window=window).astype(np.float64)
                valid = np.ones(data.shape[1:], dtype=bool)
                if src.nodata is not None:
                    valid &= ~np.any(data == src.nodata, axis=0)
                if black_pixel_threshold >= 0:
                    valid &= ~np.all(data <= black_pixel_threshold, axis=0)
                if not np.any(valid):
                    continue
                data = data[:, valid]

                if self.sum is None:
                    band_count = data.shape[0]
                    self.sum = np.zeros(band_count, dtype=np.float64)
                    self.sum_sq = np.zeros(band_count, dtype=np.float64)
                    self.minimum = np.full(band_count, np.inf, dtype=np.float64)
                    self.maximum = np.full(band_count, -np.inf, dtype=np.float64)

                self.count += data.shape[1]
                self.sum += np.sum(data, axis=1)
                self.sum_sq += np.sum(data * data, axis=1)
                self.minimum = np.minimum(self.minimum, np.min(data, axis=1))
                self.maximum = np.maximum(self.maximum, np.max(data, axis=1))

    def finalize(self) -> dict:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            return {"pixel_count": 0, "bands": []}

        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        bands = []
        for idx in range(len(mean)):
            bands.append(
                {
                    "band": idx + 1,
                    "min": float(self.minimum[idx]),
                    "max": float(self.maximum[idx]),
                    "mean": float(mean[idx]),
                    "std": float(std[idx]),
                }
            )
        return {"pixel_count": self.count, "bands": bands}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote-sensing segmentation training data.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-touched", action="store_true")
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--min-patch-size", type=int, default=128)
    parser.add_argument("--split-strategy", choices=("image", "patch"), default="image")
    parser.add_argument("--test-process", action="store_true")
    parser.add_argument("--mask-mode", choices=sorted(MASK_MODES), default=MASK_MODE_BINARY)
    parser.add_argument("--boundary-width-pixels", type=int, default=2)
    parser.add_argument("--background-keep-ratio", type=float, default=0.2)
    parser.add_argument("--max-ignore-ratio", type=float, default=0.5)
    parser.add_argument("--black-pixel-threshold", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    config = DataProcessConfig(
        dataset_dir=Path(args.dataset_dir),
        tile_size=args.tile_size,
        overlap=args.overlap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        all_touched=args.all_touched,
        keep_empty=not args.drop_empty,
        min_patch_size=args.min_patch_size,
        split_strategy=args.split_strategy,
        test_process=args.test_process,
        mask_mode=args.mask_mode,
        boundary_width_pixels=args.boundary_width_pixels,
        background_keep_ratio=args.background_keep_ratio,
        max_ignore_ratio=args.max_ignore_ratio,
        black_pixel_threshold=args.black_pixel_threshold,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
