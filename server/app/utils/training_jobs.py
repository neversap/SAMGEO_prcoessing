from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_CONFIG = "/app/configs/pretrain/ftw_rgb_unet_effb3_pretrain_v1.yaml"
TRAINING_CUDA_VISIBLE_DEVICES = "6,7"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingJobRequest:
    config_path: str = DEFAULT_CONFIG
    stage: str = "sanity_check"
    epochs: int = 0
    batch_size: int = 0
    max_train_samples: int = 0
    max_val_samples: int = 0
    init_checkpoint: str = ""


@dataclass
class TrainingJobState:
    job_id: str
    status: str
    config_path: str
    stage: str
    progress: float = 0.0
    current: int = 0
    total: int = 1
    message: str = "Queued"
    error: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class TrainingJobManager:
    def __init__(self, jobs_dir: Path, allowed_roots: list[Path] | None = None, max_workers: int = 1) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [root.resolve() for root in allowed_roots or []]
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="training")
        self._lock = threading.Lock()
        self._jobs: dict[str, TrainingJobState] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._load_existing_jobs()

    def create_job(self, request: TrainingJobRequest) -> TrainingJobState:
        config_path = self._validate_config_path(request.config_path)
        self._validate_request(request)
        job_id = f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        state = TrainingJobState(
            job_id=job_id,
            status="queued",
            config_path=str(config_path),
            stage=request.stage,
            output_paths={
                "config": str(config_path),
                "default_output_root": "/home/nvme1/datasets/ftw_datasets/outputs/ftw_rgb_unet_effb3_pretrain_v1",
            },
        )
        self._store_state(state)
        self._executor.submit(self._run_job, job_id, request, config_path)
        return state

    def get_job(self, job_id: str) -> TrainingJobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state:
            return state
        return self._read_state(job_id)

    def list_jobs(self) -> list[TrainingJobState]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> TrainingJobState | None:
        state = self.get_job(job_id)
        if state is None:
            return None
        if state.status in TERMINAL_STATUSES:
            return state
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            process.terminate()
        self._update_state(
            job_id,
            status="cancelled",
            finished_at=_now_iso(),
            message="Training cancellation requested",
        )
        return self.get_job(job_id)

    def read_metrics(self, job_id: str) -> list[dict[str, float]]:
        state = self.get_job(job_id)
        if state is None:
            raise FileNotFoundError("training job not found")
        log_path = state.output_paths.get("training_log")
        if not log_path:
            default_root = state.output_paths.get("default_output_root")
            if default_root:
                log_path = str(Path(default_root) / "reports" / f"{state.stage}_training_log.csv")
            else:
                return _metrics_from_logs(state.logs)
        path = Path(log_path)
        if not path.exists():
            return _metrics_from_logs(state.logs)
        with path.open("r", encoding="utf-8", newline="") as f:
            return [_metric_row(row) for row in csv.DictReader(f)]

    def _run_job(self, job_id: str, request: TrainingJobRequest, config_path: Path) -> None:
        command = [
            sys.executable,
            "-m",
            "training.train_pretrain",
            "--config",
            str(config_path),
            "--stage",
            request.stage,
        ]
        if request.epochs > 0:
            command.extend(["--epochs", str(request.epochs)])
        if request.batch_size > 0:
            command.extend(["--batch-size", str(request.batch_size)])
        if request.max_train_samples > 0:
            command.extend(["--max-train-samples", str(request.max_train_samples)])
        if request.max_val_samples > 0:
            command.extend(["--max-val-samples", str(request.max_val_samples)])
        if request.init_checkpoint.strip():
            command.extend(["--init-checkpoint", request.init_checkpoint.strip()])

        self._update_state(job_id, status="running", message="Starting FTW RGB pretraining")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": TRAINING_CUDA_VISIBLE_DEVICES},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._processes[job_id] = process
            assert process.stdout is not None
            for line in process.stdout:
                text = line.strip()
                if not text:
                    continue
                self._handle_log_line(job_id, text)
            return_code = process.wait()
        except Exception as exc:
            logger.exception("Training job failed: %s", job_id)
            self._update_state(job_id, status="failed", error=str(exc), finished_at=_now_iso(), message=str(exc))
            return
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

        state = self.get_job(job_id)
        if state and state.status == "cancelled":
            return
        if return_code == 0:
            self._update_state(
                job_id,
                status="completed",
                progress=1.0,
                current=1,
                total=1,
                finished_at=_now_iso(),
                message="Training completed",
            )
        else:
            self._update_state(
                job_id,
                status="failed",
                error=f"Training process exited with code {return_code}",
                finished_at=_now_iso(),
                message=f"Training process exited with code {return_code}",
            )

    def _handle_log_line(self, job_id: str, text: str) -> None:
        progress = None
        current = None
        total = None
        message = text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("event") == "training_started":
            output_paths = dict(self.get_job(job_id).output_paths)
            for key in ("output_root", "training_log", "checkpoints", "reports"):
                if payload.get(key):
                    output_paths[key] = str(payload[key])
            total = int(payload.get("epochs", 1) or 1)
            message = f"training started: {payload.get('stage', '')}"
            self._update_state(
                job_id,
                status="running",
                progress=0.0,
                current=0,
                total=total,
                message=message,
                output_paths=output_paths,
            )
            return
        if isinstance(payload, dict) and "epoch" in payload:
            current = int(payload["epoch"])
            total = max(int(payload.get("epochs", current)), 1)
            progress = max(0.0, min(current / total, 1.0))
            message = (
                f"epoch {current}: train_loss={float(payload.get('train_loss', 0)):.4f}, "
                f"val_boundary_f1={float(payload.get('val_boundary_f1', 0)):.4f}"
            )
        self._update_state(
            job_id,
            status="running",
            progress=progress if progress is not None else self.get_job(job_id).progress,
            current=current if current is not None else self.get_job(job_id).current,
            total=total if total is not None else self.get_job(job_id).total,
            message=message,
        )

    def _validate_config_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"training config does not exist: {resolved}")
        return resolved

    def _validate_request(self, request: TrainingJobRequest) -> None:
        if request.stage not in {"sanity_check", "small_country_pretrain", "full_pretrain"}:
            raise ValueError("stage must be sanity_check, small_country_pretrain, or full_pretrain")
        if request.epochs < 0 or request.batch_size < 0:
            raise ValueError("epochs and batch_size must be non-negative")
        if request.max_train_samples < 0 or request.max_val_samples < 0:
            raise ValueError("max sample overrides must be non-negative")

    def _load_existing_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Failed to load training job state: %s", path)
                continue
            if state.status not in TERMINAL_STATUSES:
                state.status = "failed"
                state.error = "Server restarted before this job finished"
                state.finished_at = _now_iso()
                state.updated_at = _now_iso()
                self._write_state(state)
            self._jobs[state.job_id] = state

    def _read_state(self, job_id: str) -> TrainingJobState | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Failed to read training job state: %s", path)
            return None
        with self._lock:
            self._jobs[job_id] = state
        return state

    def _store_state(self, state: TrainingJobState) -> None:
        with self._lock:
            self._jobs[state.job_id] = state
            self._write_state(state)

    def _update_state(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            state = self._jobs[job_id]
            for key, value in changes.items():
                setattr(state, key, value)
            state.updated_at = _now_iso()
            if "message" in changes and changes["message"]:
                _append_log(state, str(changes["message"]))
            self._write_state(state)

    def _write_state(self, state: TrainingJobState) -> None:
        path = self.jobs_dir / f"{state.job_id}.json"
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _append_log(state: TrainingJobState, message: str) -> None:
    line = f"{_now_iso()} {message}"
    if state.logs and state.logs[-1].endswith(message):
        return
    state.logs.append(line)
    state.logs = state.logs[-120:]


def _state_from_dict(data: dict[str, Any]) -> TrainingJobState:
    return TrainingJobState(
        job_id=data["job_id"],
        status=data["status"],
        config_path=data["config_path"],
        stage=data["stage"],
        progress=float(data.get("progress", 0.0)),
        current=int(data.get("current", 0)),
        total=int(data.get("total", 1)),
        message=data.get("message", ""),
        error=data.get("error"),
        created_at=data.get("created_at", _now_iso()),
        updated_at=data.get("updated_at", _now_iso()),
        finished_at=data.get("finished_at"),
        output_paths=data.get("output_paths", {}),
        logs=data.get("logs", []),
    )


def _metrics_from_logs(logs: list[str]) -> list[dict[str, float]]:
    rows = []
    for line in logs:
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "epoch" in payload:
            rows.append(_metric_row(payload))
    return rows


def _metric_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "epoch": _float_value(row.get("epoch")),
        "train_loss": _float_value(row.get("train_loss")),
        "val_loss": _float_value(row.get("val_loss")),
        "val_miou": _float_value(row.get("val_miou")),
        "val_boundary_f1": _float_value(row.get("val_boundary_f1")),
        "val_pixel_accuracy": _float_value(row.get("val_pixel_accuracy")),
        "lr": _float_value(row.get("lr")),
        "seconds": _float_value(row.get("seconds")),
    }


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
