from __future__ import annotations

import csv
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_process_pipeline.dataloader import configure_gdal_cache, infer_image_scale, load_inhouse_index
from data_process_pipeline.dataloader import read_image_chw, read_mask


class FTWRGBPretrainDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        source: str = "ftw",
        dataset_dir: str | Path | None = None,
        transform=None,
        image_scale: float | None = None,
        normalize: bool = False,
        stats_mean: list[float] | None = None,
        stats_std: list[float] | None = None,
        num_classes: int = 3,
        ignore_index: int = 255,
        max_samples: int = 0,
        seed: int = 42,
    ) -> None:
        self.rows = _load_rows(
            metadata_csv=metadata_csv,
            split=split,
            source=source,
            dataset_dir=dataset_dir,
        )
        if max_samples > 0 and len(self.rows) > max_samples:
            rows = self.rows[:]
            random.Random(seed).shuffle(rows)
            self.rows = rows[:max_samples]
        self.transform = transform
        self.image_scale = image_scale
        self.normalize = normalize
        self.stats_mean = np.asarray(stats_mean, dtype="float32") if stats_mean else None
        self.stats_std = np.asarray(stats_std, dtype="float32") if stats_std else None
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = read_image_chw(Path(row["image_path"])).astype("float32")
        mask = sanitize_mask(
            read_mask(Path(row["mask_path"])),
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        scale = self.image_scale or infer_image_scale(image)
        if scale > 1.0:
            image = image / scale
        image = np.clip(image, 0.0, 1.0)
        image = image.transpose(1, 2, 0)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = image.transpose(2, 0, 1).astype("float32")
        if self.normalize and self.stats_mean is not None and self.stats_std is not None:
            mean = self.stats_mean[:, None, None]
            std = np.maximum(self.stats_std[:, None, None], 1e-6)
            image = (image - mean) / std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask.astype("int64"))).long(),
            "meta": {
                "sample_id": row.get("sample_id", ""),
                "country": row.get("country", ""),
                "window": row.get("window", ""),
                "bucket": row.get("bucket", ""),
                "fg_ratio": float(row.get("fg_ratio") or row.get("cropland_ratio") or 0.0),
                "boundary_ratio": float(row.get("boundary_ratio") or 0.0),
            },
        }


class RemoteSensingTrainTransform:
    def __init__(
        self,
        hflip: float = 0.5,
        vflip: float = 0.5,
        rotate90: float = 0.5,
        scale_jitter: float = 0.15,
        brightness: float = 0.12,
        contrast: float = 0.12,
        noise: float = 0.02,
    ) -> None:
        self.hflip = hflip
        self.vflip = vflip
        self.rotate90 = rotate90
        self.scale_jitter = scale_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.noise = noise

    def __call__(self, image, mask):
        if random.random() < self.hflip:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < self.vflip:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        if random.random() < self.rotate90:
            turns = random.randint(1, 3)
            image = np.rot90(image, k=turns)
            mask = np.rot90(mask, k=turns)
        if self.scale_jitter > 0:
            image, mask = _scale_to_original_size(
                image,
                mask,
                1.0 + random.uniform(-self.scale_jitter, self.scale_jitter),
            )
        if self.brightness > 0 or self.contrast > 0:
            pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype("uint8"), mode="RGB")
            if self.brightness > 0:
                pil_image = ImageEnhance.Brightness(pil_image).enhance(
                    1.0 + random.uniform(-self.brightness, self.brightness)
                )
            if self.contrast > 0:
                pil_image = ImageEnhance.Contrast(pil_image).enhance(
                    1.0 + random.uniform(-self.contrast, self.contrast)
                )
            image = np.asarray(pil_image, dtype="float32") / 255.0
        if self.noise > 0:
            image = image + np.random.normal(0.0, self.noise, size=image.shape).astype("float32")
        return np.clip(image, 0.0, 1.0), np.ascontiguousarray(mask)


def create_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    input_config = config["input"]
    augmentation_config = config["augmentation"]
    class_config = config["classes"]
    metadata_csv = config["paths"]["metadata_csv"]
    source = str(data_config.get("source", "ftw")).strip().lower()
    dataset_dir = config.get("paths", {}).get("dataset_dir")
    seed = int(config["experiment"]["seed"])
    gdal_cachemax_mb = int(data_config.get("gdal_cachemax_mb", 64))
    configure_gdal_cache(gdal_cachemax_mb)
    transform = None
    if augmentation_config.get("enabled", True):
        transform = RemoteSensingTrainTransform(
            hflip=float(augmentation_config.get("hflip", 0.5)),
            vflip=float(augmentation_config.get("vflip", 0.5)),
            rotate90=float(augmentation_config.get("rotate90", 0.5)),
            scale_jitter=float(augmentation_config.get("scale_jitter", 0.15)),
            brightness=float(augmentation_config.get("brightness", 0.12)),
            contrast=float(augmentation_config.get("contrast", 0.12)),
            noise=float(augmentation_config.get("noise", 0.02)),
        )
    train_dataset = FTWRGBPretrainDataset(
        metadata_csv,
        split="train",
        source=source,
        dataset_dir=dataset_dir,
        transform=transform,
        image_scale=input_config.get("image_scale"),
        normalize=bool(input_config.get("normalize", False)),
        stats_mean=input_config.get("stats_mean"),
        stats_std=input_config.get("stats_std"),
        num_classes=int(class_config.get("num_classes", 3)),
        ignore_index=int(class_config.get("ignore_index", 255)),
        max_samples=int(data_config.get("max_train_samples", 0)),
        seed=seed,
    )
    val_dataset = FTWRGBPretrainDataset(
        metadata_csv,
        split="val",
        source=source,
        dataset_dir=dataset_dir,
        transform=None,
        image_scale=input_config.get("image_scale"),
        normalize=bool(input_config.get("normalize", False)),
        stats_mean=input_config.get("stats_mean"),
        stats_std=input_config.get("stats_std"),
        num_classes=int(class_config.get("num_classes", 3)),
        ignore_index=int(class_config.get("ignore_index", 255)),
        max_samples=int(data_config.get("max_val_samples", 0)),
        seed=seed,
    )
    sampler = None
    shuffle = True
    if config.get("sampler", {}).get("enabled", True):
        sampler = _weighted_sampler(train_dataset.rows, config["sampler"].get("bucket_weights", {}))
        shuffle = False

    loader_kwargs = {
        "batch_size": int(data_config.get("batch_size", 16)),
        "num_workers": int(data_config.get("num_workers", 8)),
        "pin_memory": bool(data_config.get("pin_memory", False)),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(data_config.get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 1))
        loader_kwargs["worker_init_fn"] = partial(_init_worker, gdal_cachemax_mb=gdal_cachemax_mb)
        multiprocessing_context = str(data_config.get("multiprocessing_context", "") or "").strip()
        if multiprocessing_context:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=bool(data_config.get("drop_last_train", True)),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def _load_rows(
    metadata_csv: str | Path,
    split: str,
    source: str = "ftw",
    dataset_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    if source == "inhouse":
        if dataset_dir:
            rows = load_inhouse_index(Path(dataset_dir))
        else:
            rows = _load_inhouse_rows_from_csv(Path(metadata_csv))
    elif source == "ftw":
        with Path(metadata_csv).open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("data.source must be ftw or inhouse")
    return [
        row
        for row in rows
        if row.get("split", "").lower() == split and int(float(row.get("use_for_train") or 1)) == 1
    ]


def _load_inhouse_rows_from_csv(metadata_csv: Path) -> list[dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    dataset_dir = metadata_csv.parent.parent
    normalized = []
    for row in rows:
        patch_name = row.get("patch_name") or row.get("patch_id") or row.get("sample_id") or Path(row.get("image_path", "")).stem
        image_path = _resolve_dataset_path(dataset_dir, row.get("image_path", ""))
        mask_path = _resolve_dataset_path(dataset_dir, row.get("mask_path", ""))
        normalized.append(
            {
                **row,
                "sample_id": patch_name,
                "patch_name": patch_name,
                "source_dataset": "inhouse",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "fg_ratio": row.get("fg_ratio") or row.get("cropland_ratio") or "0",
                "cropland_ratio": row.get("cropland_ratio") or row.get("fg_ratio") or "0",
                "boundary_ratio": row.get("boundary_ratio") or "0",
                "bucket": row.get("bucket") or row.get("patch_type") or "",
                "use_for_train": row.get("use_for_train") or "1",
            }
        )
    return normalized


def _resolve_dataset_path(dataset_dir: Path, value: str) -> Path:
    if not value:
        return dataset_dir
    path = Path(value)
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()


def _weighted_sampler(rows: list[dict[str, str]], bucket_weights: dict[str, float]) -> WeightedRandomSampler:
    weights = [float(bucket_weights.get(row.get("bucket", ""), 1.0)) for row in rows]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def _init_worker(worker_id: int, gdal_cachemax_mb: int) -> None:
    configure_gdal_cache(gdal_cachemax_mb)
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def sanitize_mask(mask, num_classes: int, ignore_index: int):
    sanitized = np.asarray(mask).astype("int64", copy=False)
    valid = (sanitized == ignore_index) | ((sanitized >= 0) & (sanitized < num_classes))
    if not np.all(valid):
        sanitized = sanitized.copy()
        sanitized[~valid] = ignore_index
    return sanitized.astype("uint8", copy=False)


def _scale_to_original_size(image, mask, factor: float):
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * factor)))
    new_height = max(1, int(round(height * factor)))
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype("uint8"), mode="RGB")
    pil_mask = Image.fromarray(mask.astype("uint8"))
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
    return np.asarray(resized_image, dtype="float32") / 255.0, np.asarray(resized_mask)
