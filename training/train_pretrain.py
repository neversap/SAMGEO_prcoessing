from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from training.config import apply_stage, load_config
from training.data import create_dataloaders
from training.losses import build_loss
from training.metrics import SegmentationMetrics
from training.model import build_model


_LIBC = None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_stage(config, args.stage)
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        config["data"]["max_val_samples"] = args.max_val_samples
    if args.init_checkpoint:
        config.setdefault("checkpoint", {})["init_from"] = args.init_checkpoint
    apply_cuda_visible_devices(config)
    train(config=config, stage=args.stage)


def train(config: dict[str, Any], stage: str) -> None:
    set_seed(int(config["experiment"]["seed"]))
    output_root = build_run_output_root(config, Path(config["paths"]["output_root"]))
    config["paths"]["output_root"] = str(output_root)
    checkpoint_dir = output_root / "checkpoints"
    report_dir = output_root / "reports"
    config_dir = output_root / "configs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_dir / f"{stage}_resolved_config.json", config)

    device = resolve_device(config["trainer"].get("device", "cuda"))
    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    if (
        device.type == "cuda"
        and bool(config["trainer"].get("data_parallel", True))
        and torch.cuda.device_count() > 1
    ):
        model = torch.nn.DataParallel(model)
    print(
        json.dumps(
            {
                "event": "model_ready",
                "device": str(device),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "data_parallel": hasattr(model, "module"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = int(config["trainer"]["epochs"])
    milestone_epochs = build_milestone_epochs(epochs, ratio=0.25)
    scheduler = build_scheduler(optimizer, config, epochs)
    use_amp = config["trainer"].get("precision", "amp") == "amp" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_path = report_dir / f"{stage}_training_log.csv"
    best_loss = math.inf
    best_boundary_f1 = -math.inf
    best_miou = -math.inf
    print(
        json.dumps(
            {
                "event": "training_started",
                "stage": stage,
                "epochs": epochs,
                "output_root": str(output_root),
                "training_log": str(log_path),
                "checkpoints": str(checkpoint_dir),
                "reports": str(report_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_miou",
                "val_boundary_f1",
                "val_pixel_accuracy",
                "lr",
                "seconds",
                "epochs",
            ],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_loss = run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp,
                gradient_clip_val=float(config["trainer"].get("gradient_clip_val", 0.0)),
                memory_log_every_n_steps=int(config["trainer"].get("memory_log_every_n_steps", 50)),
                epoch=epoch,
            )
            log_memory("after_train_epoch", epoch=epoch)
            val_loss, val_metrics = run_validation(
                model,
                val_loader,
                criterion,
                device,
                memory_log_every_n_steps=int(config["trainer"].get("memory_log_every_n_steps", 50)),
                epoch=epoch,
            )
            log_memory("after_validation", epoch=epoch)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": val_metrics["miou"],
                "val_boundary_f1": val_metrics["boundary_f1"],
                "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                "lr": lr,
                "seconds": round(time.time() - started, 3),
                "epochs": epochs,
            }
            writer.writerow(row)
            f.flush()
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, row)
            log_memory("after_last_checkpoint", epoch=epoch)
            if epoch in milestone_epochs:
                milestone_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(milestone_path, model, optimizer, epoch, config, row)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_saved",
                            "kind": "milestone",
                            "epoch": epoch,
                            "path": str(milestone_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(checkpoint_dir / "best_val_loss.pt", model, optimizer, epoch, config, row)
            if val_metrics["boundary_f1"] > best_boundary_f1:
                best_boundary_f1 = val_metrics["boundary_f1"]
                save_checkpoint(checkpoint_dir / "best_val_boundary_f1.pt", model, optimizer, epoch, config, row)
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                save_checkpoint(checkpoint_dir / "best_val_miou.pt", model, optimizer, epoch, config, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)


def run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    gradient_clip_val: float,
    memory_log_every_n_steps: int,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    log_memory("train_epoch_start", epoch=epoch)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        if gradient_clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        progress.set_postfix(loss=total_loss / max(total_items, 1))
        del batch, images, masks, logits, loss
        if memory_log_every_n_steps > 0 and step % memory_log_every_n_steps == 0:
            log_memory("train_batch", epoch=epoch, step=step)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def run_validation(
    model,
    loader,
    criterion,
    device,
    memory_log_every_n_steps: int,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    metrics = SegmentationMetrics(num_classes=3, ignore_index=255)
    log_memory("validation_start", epoch=epoch)
    for step, batch in enumerate(tqdm(loader, desc="validation", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        metrics.update(logits, masks)
        del batch, images, masks, logits, loss
        if memory_log_every_n_steps > 0 and step % memory_log_every_n_steps == 0:
            log_memory("validation_batch", epoch=epoch, step=step)
    return total_loss / max(total_items, 1), metrics.compute()


def log_memory(event: str, epoch: int | None = None, step: int | None = None) -> None:
    gc.collect()
    trimmed = malloc_trim()
    payload: dict[str, Any] = {
        "event": "memory",
        "where": event,
        "pid": os.getpid(),
        "malloc_trim": trimmed,
        "process_rss_mb": round(_rss_mb(os.getpid()), 2),
        "process_pss_mb": round(_pss_mb(os.getpid()), 2),
    }
    children = _child_pids(os.getpid())
    child_rss = sum(_rss_mb(pid) for pid in children)
    child_pss = sum(_pss_mb(pid) for pid in children)
    payload["child_count"] = len(children)
    payload["child_rss_mb"] = round(child_rss, 2)
    payload["child_pss_mb"] = round(child_pss, 2)
    payload["total_rss_mb"] = round(payload["process_rss_mb"] + child_rss, 2)
    payload["total_pss_mb"] = round(payload["process_pss_mb"] + child_pss, 2)
    if epoch is not None:
        payload["epoch"] = epoch
    if step is not None:
        payload["step"] = step
    if torch.cuda.is_available():
        payload["cuda"] = []
        for index in range(torch.cuda.device_count()):
            payload["cuda"].append(
                {
                    "device": index,
                    "allocated_mb": round(torch.cuda.memory_allocated(index) / 1024 / 1024, 2),
                    "reserved_mb": round(torch.cuda.memory_reserved(index) / 1024 / 1024, 2),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated(index) / 1024 / 1024, 2),
                }
            )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def malloc_trim() -> bool:
    global _LIBC
    if os.name != "posix":
        return False
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        return bool(_LIBC.malloc_trim(0))
    except Exception:
        return False


def _rss_mb(pid: int) -> float:
    status_path = Path(f"/proc/{pid}/status")
    try:
        with status_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def _pss_mb(pid: int) -> float:
    rollup_path = Path(f"/proc/{pid}/smaps_rollup")
    try:
        with rollup_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Pss:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def _child_pids(parent_pid: int) -> list[int]:
    proc_root = Path("/proc")
    children: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return children
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat_path = entry / "stat"
        try:
            text = stat_path.read_text(encoding="utf-8")
        except OSError:
            continue
        right = text.rfind(")")
        if right < 0:
            continue
        fields = text[right + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            ppid = int(fields[1])
        except ValueError:
            continue
        if ppid == parent_pid:
            children.append(int(entry.name))
    return children


def build_scheduler(optimizer, config: dict[str, Any], epochs: int):
    scheduler_config = config["scheduler"]
    min_lr = float(scheduler_config.get("min_lr", 1e-6))
    base_lr = float(config["optimizer"]["lr"])
    warmup_epochs = int(scheduler_config.get("warmup_epochs", 5))

    def lr_lambda(epoch: int) -> float:
        current = epoch + 1
        if warmup_epochs > 0 and current <= warmup_epochs:
            return max(current / warmup_epochs, min_lr / base_lr)
        if epochs <= warmup_epochs:
            return 1.0
        progress = (current - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_milestone_epochs(epochs: int, ratio: float = 0.25) -> set[int]:
    if epochs <= 0:
        return set()
    steps = max(1, int(round(1.0 / ratio)))
    return {
        max(1, min(epochs, int(round(epochs * index / steps))))
        for index in range(1, steps + 1)
    }


def build_run_output_root(config: dict[str, Any], base_output_root: Path) -> Path:
    timestamp = time.strftime("%y%m%d:%H:%M")
    run_name = "__".join(
        [
            model_slug(config.get("model", {})),
            slugify(str(config.get("loss", {}).get("version", "loss"))),
            timestamp,
        ]
    )
    candidate = base_output_root / run_name
    if not candidate.exists():
        return candidate
    for suffix in range(2, 1000):
        numbered = base_output_root / f"{run_name}_{suffix:02d}"
        if not numbered.exists():
            return numbered
    raise RuntimeError(f"Could not allocate a unique output directory under {base_output_root}")


def model_slug(model_config: dict[str, Any]) -> str:
    architecture = slugify(str(model_config.get("architecture", "model")))
    encoder = slugify(str(model_config.get("encoder", "")))
    if encoder:
        return f"{architecture}_{encoder}"
    return architecture


def slugify(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in normalized).strip("_")


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_initial_checkpoint(model, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_config = config.get("checkpoint", {})
    checkpoint_path = str(checkpoint_config.get("init_from", "") or "").strip()
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint.init_from does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    clean_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    strict = bool(checkpoint_config.get("strict", False))
    result = model.load_state_dict(clean_state, strict=strict)
    print(
        json.dumps(
            {
                "event": "initial_checkpoint_loaded",
                "path": str(path),
                "strict": strict,
                "missing_keys": len(result.missing_keys),
                "unexpected_keys": len(result.unexpected_keys),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def apply_cuda_visible_devices(config: dict[str, Any]) -> None:
    value = str(config.get("trainer", {}).get("cuda_visible_devices", "") or "").strip()
    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FTW RGB-only pretraining model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="sanity_check", choices=["sanity_check", "small_country_pretrain", "full_pretrain"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--init-checkpoint", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
