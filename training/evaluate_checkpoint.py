from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from data_process_pipeline.dataloader import infer_image_scale, read_image_chw, read_mask
from training.config import load_config
from training.data import sanitize_mask
from training.model import build_model


PALETTE = {
    0: (0, 0, 0, 0),
    1: (34, 197, 94, 135),
    2: (220, 38, 38, 180),
    255: (0, 0, 0, 0),
}


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)

    ftw_rows = select_ftw_rows(
        metadata_csv=Path(args.ftw_metadata_csv),
        count=args.ftw_count,
    )
    inhouse_rows = select_inhouse_rows(
        dataset_dir=Path(args.inhouse_dataset_dir),
        count=args.inhouse_count,
        seed=args.seed,
    )
    selected = [
        {"source": "ftw", "row": row}
        for row in ftw_rows
    ] + [
        {"source": "inhouse", "row": row}
        for row in inhouse_rows
    ]

    results = []
    for index, item in enumerate(selected, start=1):
        result = infer_sample(
            model=model,
            row=item["row"],
            source=item["source"],
            index=index,
            samples_dir=samples_dir,
            device=device,
            num_classes=int(config["classes"].get("num_classes", 3)),
            ignore_index=int(config["classes"].get("ignore_index", 255)),
            image_scale=config["input"].get("image_scale"),
            normalize=bool(config["input"].get("normalize", False)),
            stats_mean=config["input"].get("stats_mean"),
            stats_std=config["input"].get("stats_std"),
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "inference_sample",
                    "current": index,
                    "total": len(selected),
                    "sample_id": result["sample_id"],
                    "source": result["source"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "config": str(Path(args.config)),
        "output_dir": str(output_dir),
        "count": len(results),
        "samples": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "metrics.csv", results)
    print(
        json.dumps(
            {
                "event": "inference_completed",
                "summary": str(output_dir / "summary.json"),
                "metrics": str(output_dir / "metrics.csv"),
                "count": len(results),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device):
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    clean_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(
        json.dumps(
            {
                "event": "checkpoint_loaded",
                "checkpoint": checkpoint_path,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model.eval()
    return model


def select_ftw_rows(metadata_csv: Path, count: int) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(metadata_csv)
        if row.get("split", "").lower() == "test"
        and int(float(row.get("use_for_train") or 1)) == 1
        and row.get("image_path")
        and row.get("mask_path")
    ]
    rows = sorted(rows, key=lambda row: float(row.get("cropland_ratio") or row.get("fg_ratio") or 0.0))
    return quantile_pick(rows, count)


def select_inhouse_rows(dataset_dir: Path, count: int, seed: int) -> list[dict[str, str]]:
    index_path = dataset_dir / "metadata" / "patch_index.csv"
    if not index_path.exists():
        return []
    rows = [
        normalize_inhouse_row(dataset_dir, row)
        for row in read_csv(index_path)
    ]
    rows = [
        row
        for row in rows
        if row.get("image_path") and row.get("mask_path")
    ]
    test_rows = [row for row in rows if row.get("split", "").lower() == "test"]
    candidates = test_rows or rows
    selected = candidates[:]
    random.Random(seed).shuffle(selected)
    return selected[: max(0, count)]


def normalize_inhouse_row(dataset_dir: Path, row: dict[str, str]) -> dict[str, str]:
    patch_name = row.get("patch_name") or row.get("patch_id") or row.get("sample_id") or ""
    split = (row.get("split") or "all").lower()
    image_path = resolve_path(dataset_dir, row.get("image_path", ""))
    mask_path = resolve_path(dataset_dir, row.get("mask_path", ""))
    if not image_path and patch_name:
        image_path = find_patch_file(dataset_dir / "processed" / "patches" / split / "images", patch_name)
    if not mask_path and patch_name:
        mask_path = find_patch_file(dataset_dir / "processed" / "patches" / split / "masks", patch_name)
    normalized = dict(row)
    normalized["sample_id"] = patch_name
    normalized["patch_name"] = patch_name
    normalized["split"] = split
    normalized["image_path"] = image_path
    normalized["mask_path"] = mask_path
    return normalized


def infer_sample(
    model,
    row: dict[str, str],
    source: str,
    index: int,
    samples_dir: Path,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    image_scale: float | None,
    normalize: bool,
    stats_mean: list[float] | None,
    stats_std: list[float] | None,
) -> dict[str, Any]:
    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])
    image_chw = read_image_chw(image_path).astype("float32")
    mask = sanitize_mask(read_mask(mask_path), num_classes=num_classes, ignore_index=ignore_index)
    scale = image_scale or infer_image_scale(image_chw)
    if scale > 1.0:
        image_chw = image_chw / scale
    image_chw = np.clip(image_chw, 0.0, 1.0)
    model_input = image_chw.astype("float32")
    if normalize and stats_mean is not None and stats_std is not None:
        mean = np.asarray(stats_mean, dtype="float32")[:, None, None]
        std = np.maximum(np.asarray(stats_std, dtype="float32")[:, None, None], 1e-6)
        model_input = (model_input - mean) / std
    tensor = torch.from_numpy(np.ascontiguousarray(model_input[None])).float().to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype("uint8")

    sample_id = row.get("sample_id") or row.get("patch_name") or image_path.stem
    stem = f"{index:02d}_{source}_{safe_name(sample_id)}"
    image_png = f"{stem}_image.png"
    gt_png = f"{stem}_gt.png"
    pred_png = f"{stem}_pred.png"
    overlay_png = f"{stem}_overlay.png"
    save_rgb(image_chw, samples_dir / image_png)
    save_mask(mask, samples_dir / gt_png)
    save_mask(pred, samples_dir / pred_png)
    save_overlay(image_chw, pred, samples_dir / overlay_png)
    metrics = compute_metrics(pred, mask, num_classes=num_classes, ignore_index=ignore_index)
    return {
        "id": index,
        "source": source,
        "sample_id": sample_id,
        "patch_name": row.get("patch_name", sample_id),
        "split": row.get("split", ""),
        "country": row.get("country", ""),
        "window": row.get("window", ""),
        "cropland_ratio": float(row.get("cropland_ratio") or row.get("fg_ratio") or 0.0),
        "ignore_ratio": float(row.get("ignore_ratio") or 0.0),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_png": f"samples/{image_png}",
        "gt_png": f"samples/{gt_png}",
        "pred_png": f"samples/{pred_png}",
        "overlay_png": f"samples/{overlay_png}",
        "metrics": metrics,
    }


def compute_metrics(pred: np.ndarray, truth: np.ndarray, num_classes: int, ignore_index: int) -> dict[str, float]:
    valid = truth != ignore_index
    total = max(int(valid.sum()), 1)
    pixel_accuracy = float(((pred == truth) & valid).sum() / total)
    ious = []
    f1s = []
    result = {"pixel_accuracy": pixel_accuracy}
    for class_id in range(num_classes):
        pred_class = (pred == class_id) & valid
        truth_class = (truth == class_id) & valid
        intersection = int((pred_class & truth_class).sum())
        union = int((pred_class | truth_class).sum())
        pred_count = int(pred_class.sum())
        truth_count = int(truth_class.sum())
        iou = intersection / max(union, 1)
        precision = intersection / max(pred_count, 1)
        recall = intersection / max(truth_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        ious.append(iou)
        f1s.append(f1)
        result[f"class_{class_id}_iou"] = float(iou)
        result[f"class_{class_id}_f1"] = float(f1)
    result["miou"] = float(sum(ious) / max(len(ious), 1))
    result["macro_f1"] = float(sum(f1s) / max(len(f1s), 1))
    result["boundary_iou"] = float(result.get("class_2_iou", 0.0))
    result["boundary_f1"] = float(result.get("class_2_f1", 0.0))
    return result


def save_rgb(image_chw: np.ndarray, path: Path) -> None:
    image = np.clip(image_chw[:3].transpose(1, 2, 0), 0.0, 1.0)
    Image.fromarray((image * 255).astype("uint8"), mode="RGB").save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype="uint8")
    for class_id, color in PALETTE.items():
        rgba[mask == class_id] = color
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_overlay(image_chw: np.ndarray, mask: np.ndarray, path: Path) -> None:
    base = Image.fromarray(
        (np.clip(image_chw[:3].transpose(1, 2, 0), 0.0, 1.0) * 255).astype("uint8"),
        mode="RGB",
    ).convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype="uint8")
    for class_id, color in PALETTE.items():
        overlay[mask == class_id] = color
    base.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
    base.save(path)


def write_metrics_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "source",
        "sample_id",
        "cropland_ratio",
        "pixel_accuracy",
        "miou",
        "macro_f1",
        "boundary_iou",
        "boundary_f1",
        "class_0_iou",
        "class_0_f1",
        "class_1_iou",
        "class_1_f1",
        "class_2_iou",
        "class_2_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = {
                "id": item["id"],
                "source": item["source"],
                "sample_id": item["sample_id"],
                "cropland_ratio": item["cropland_ratio"],
            }
            row.update(item["metrics"])
            writer.writerow(row)


def quantile_pick(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    picks = []
    used = set()
    for index in range(count):
        position = round(index * (len(rows) - 1) / (count - 1))
        while position in used and position + 1 < len(rows):
            position += 1
        used.add(position)
        picks.append(rows[position])
    return picks


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def find_patch_file(directory: Path, patch_name: str) -> str:
    for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        candidate = directory / f"{patch_name}{suffix}"
        if candidate.exists():
            return str(candidate)
    return ""


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretraining checkpoint on FTW and inhouse samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ftw-metadata-csv", required=True)
    parser.add_argument("--inhouse-dataset-dir", required=True)
    parser.add_argument("--ftw-count", type=int, default=10)
    parser.add_argument("--inhouse-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices", default="6,7")
    return parser.parse_args()


if __name__ == "__main__":
    main()
