from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    yaml = None
    _YAML_ERROR = exc
else:
    _YAML_ERROR = None


DEFAULT_FTW_ROOT = Path("/home/nvme1/datasets/ftw_datasets/data/ftw")
DEFAULT_METADATA_DIR = DEFAULT_FTW_ROOT / "metadata"
DEFAULT_INDEX_CSV = DEFAULT_METADATA_DIR / "ftw_dataloader_index.csv"
DEFAULT_STATS_JSON = DEFAULT_METADATA_DIR / "ftw_dataloader_stats.json"
DEFAULT_OUTPUT_ROOT = Path("/home/nvme1/datasets/ftw_datasets/outputs/ftw_rgb_unet_effb3_pretrain_v1")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        return config
    _ensure_yaml()
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return deep_update(config, loaded)


def default_config() -> dict[str, Any]:
    return {
        "experiment": {
            "name": "ftw_rgb_unet_effb3_pretrain_v1",
            "seed": 42,
        },
        "paths": {
            "ftw_root": str(DEFAULT_FTW_ROOT),
            "metadata_csv": str(DEFAULT_INDEX_CSV),
            "stats_json": str(DEFAULT_STATS_JSON),
            "output_root": str(DEFAULT_OUTPUT_ROOT),
        },
        "classes": {
            "num_classes": 3,
            "ignore_index": 255,
            "names": {
                0: "background",
                1: "field_interior",
                2: "field_boundary",
            },
        },
        "input": {
            "image_scale": None,
            "normalize": False,
            "stats_mean": None,
            "stats_std": None,
        },
        "data": {
            "source": "ftw",
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 1,
            "multiprocessing_context": "spawn",
            "gdal_cachemax_mb": 64,
            "drop_last_train": True,
            "max_train_samples": 0,
            "max_val_samples": 0,
        },
        "sampler": {
            "enabled": True,
            "bucket_weights": {
                "boundary_rich": 3.0,
                "low_fg": 1.5,
                "mid_fg": 1.0,
                "high_fg": 0.7,
                "very_high_fg": 0.4,
                "near_background": 0.3,
            },
        },
        "augmentation": {
            "enabled": True,
            "hflip": 0.5,
            "vflip": 0.5,
            "rotate90": 0.5,
            "scale_jitter": 0.15,
            "brightness": 0.12,
            "contrast": 0.12,
            "noise": 0.02,
        },
        "model": {
            "architecture": "unet",
            "encoder": "efficientnet-b3",
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": 3,
        },
        "loss": {
            "version": "ce_dice_v1",
            "class_weights": [0.05, 0.25, 0.70],
            "ce_weight": 1.0,
            "dice_weight": 1.0,
            "log_cosh_dice_weight": 0.0,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1.0e-6,
        },
        "trainer": {
            "device": "cuda",
            "cuda_visible_devices": "6,7",
            "data_parallel": False,
            "epochs": 100,
            "precision": "amp",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 20,
            "memory_log_every_n_steps": 50,
            "save_top_k": 3,
            "monitor": "val_boundary_f1",
            "monitor_mode": "max",
        },
        "checkpoint": {
            "init_from": "",
            "strict": False,
            "load_optimizer": False,
        },
        "stage_overrides": {
            "sanity_check": {
                "trainer": {"epochs": 5, "memory_log_every_n_steps": 1, "data_parallel": False},
                "data": {
                    "batch_size": 4,
                    "num_workers": 4,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "max_train_samples": 500,
                    "max_val_samples": 200,
                },
            },
            "small_country_pretrain": {
                "trainer": {"epochs": 30},
            },
            "full_pretrain": {},
        },
    }


def apply_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    overrides = config.get("stage_overrides", {}).get(stage, {})
    if not overrides:
        return config
    return deep_update(config, overrides)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _ensure_yaml() -> None:
    if _YAML_ERROR is None:
        return
    raise ModuleNotFoundError("training config requires PyYAML") from _YAML_ERROR
