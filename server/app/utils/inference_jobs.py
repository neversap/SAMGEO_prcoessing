from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_CONFIG = "/app/configs/pretrain/ftw_rgb_unet_effb3_pretrain_v1.yaml"
DEFAULT_FTW_METADATA = "/home/nvme1/datasets/ftw_datasets/data/ftw/metadata/ftw_dataloader_index.csv"
DEFAULT_INHOUSE_DATASET = "/home/nvme1/datasets"
DEFAULT_CHECKPOINT = "/home/nvme1/datasets/ftw_datasets/outputs/ftw_rgb_unet_effb3_pretrain_v1/checkpoints/best_val_boundary_f1.pt"
INFERENCE_CUDA_VISIBLE_DEVICES = "6,7"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InferenceJobRequest:
    checkpoint_path: str = DEFAULT_CHECKPOINT
    config_path: str = DEFAULT_CONFIG
    ftw_metadata_csv: str = DEFAULT_FTW_METADATA
    inhouse_dataset_dir: str = DEFAULT_INHOUSE_DATASET
    ftw_count: int = 10
    inhouse_count: int = 10
    seed: int = 42


@dataclass
class InferenceJobState:
    job_id: str
    status: str
    checkpoint_path: str
    config_path: str
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


class InferenceJobManager:
    def __init__(self, jobs_dir: Path, allowed_roots: list[Path] | None = None, max_workers: int = 1) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [root.resolve() for root in allowed_roots or []]
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="inference")
        self._lock = threading.Lock()
        self._jobs: dict[str, InferenceJobState] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._load_existing_jobs()

    def create_job(self, request: InferenceJobRequest) -> InferenceJobState:
        checkpoint = self._validate_existing_path(request.checkpoint_path, "checkpoint")
        config = self._validate_existing_path(request.config_path, "config")
        ftw_metadata = self._validate_existing_path(request.ftw_metadata_csv, "ftw metadata")
        inhouse_dataset = self._validate_existing_path(request.inhouse_dataset_dir, "inhouse dataset")
        self._validate_request(request)
        job_id = f"infer_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.jobs_dir / job_id / "outputs"
        state = InferenceJobState(
            job_id=job_id,
            status="queued",
            checkpoint_path=str(checkpoint),
            config_path=str(config),
            total=max(int(request.ftw_count) + int(request.inhouse_count), 1),
            output_paths={
                "checkpoint": str(checkpoint),
                "config": str(config),
                "output_dir": str(output_dir),
            },
        )
        self._store_state(state)
        self._executor.submit(self._run_job, job_id, request, checkpoint, config, ftw_metadata, inhouse_dataset, output_dir)
        return state

    def get_job(self, job_id: str) -> InferenceJobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state:
            return state
        return self._read_state(job_id)

    def cancel_job(self, job_id: str) -> InferenceJobState | None:
        state = self.get_job(job_id)
        if state is None:
            return None
        if state.status in TERMINAL_STATUSES:
            return state
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            process.terminate()
        self._update_state(job_id, status="cancelled", finished_at=_now_iso(), message="Inference cancellation requested")
        return self.get_job(job_id)

    def read_summary(self, job_id: str) -> dict[str, Any]:
        state = self.get_job(job_id)
        if state is None:
            raise FileNotFoundError("inference job not found")
        summary_value = state.output_paths.get("summary", "")
        if not summary_value:
            raise FileNotFoundError("inference summary is not ready")
        summary_path = Path(summary_value)
        if not summary_path.exists() or not summary_path.is_file():
            raise FileNotFoundError("inference summary is not ready")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def resolve_output_file(self, job_id: str, relative_path: str) -> Path:
        state = self.get_job(job_id)
        if state is None:
            raise FileNotFoundError("inference job not found")
        output_dir = Path(state.output_paths.get("output_dir", ""))
        path = (output_dir / relative_path).resolve()
        if output_dir.resolve() not in path.parents and path != output_dir.resolve():
            raise ValueError("invalid inference output path")
        if not path.exists():
            raise FileNotFoundError("inference output file not found")
        return path

    def _run_job(
        self,
        job_id: str,
        request: InferenceJobRequest,
        checkpoint: Path,
        config: Path,
        ftw_metadata: Path,
        inhouse_dataset: Path,
        output_dir: Path,
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "training.evaluate_checkpoint",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--ftw-metadata-csv",
            str(ftw_metadata),
            "--inhouse-dataset-dir",
            str(inhouse_dataset),
            "--ftw-count",
            str(request.ftw_count),
            "--inhouse-count",
            str(request.inhouse_count),
            "--seed",
            str(request.seed),
            "--cuda-visible-devices",
            INFERENCE_CUDA_VISIBLE_DEVICES,
        ]
        self._update_state(job_id, status="running", message="Starting checkpoint inference")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": INFERENCE_CUDA_VISIBLE_DEVICES},
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
                if text:
                    self._handle_log_line(job_id, text)
            return_code = process.wait()
        except Exception as exc:
            logger.exception("Inference job failed: %s", job_id)
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
                finished_at=_now_iso(),
                message="Inference completed",
            )
        else:
            self._update_state(
                job_id,
                status="failed",
                error=f"Inference process exited with code {return_code}",
                finished_at=_now_iso(),
                message=f"Inference process exited with code {return_code}",
            )

    def _handle_log_line(self, job_id: str, text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self._update_state(job_id, message=text)
            return
        if payload.get("event") == "inference_sample":
            current = int(payload.get("current", 0))
            total = max(int(payload.get("total", 1)), 1)
            self._update_state(
                job_id,
                current=current,
                total=total,
                progress=max(0.0, min(current / total, 1.0)),
                message=f"{payload.get('source', '')}: {payload.get('sample_id', '')}",
            )
            return
        if payload.get("event") == "inference_completed":
            output_paths = dict(self.get_job(job_id).output_paths)
            output_paths["summary"] = str(payload.get("summary", ""))
            output_paths["metrics"] = str(payload.get("metrics", ""))
            self._update_state(
                job_id,
                current=int(payload.get("count", self.get_job(job_id).current)),
                progress=1.0,
                message="Inference outputs written",
                output_paths=output_paths,
            )
            return
        self._update_state(job_id, message=text)

    def _validate_existing_path(self, value: str, label: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
        return resolved

    def _validate_request(self, request: InferenceJobRequest) -> None:
        if request.ftw_count < 0 or request.ftw_count > 100:
            raise ValueError("ftw_count must be between 0 and 100")
        if request.inhouse_count < 0 or request.inhouse_count > 100:
            raise ValueError("inhouse_count must be between 0 and 100")

    def _load_existing_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Failed to load inference job state: %s", path)
                continue
            if state.status not in TERMINAL_STATUSES:
                state.status = "failed"
                state.error = "Server restarted before this job finished"
                state.finished_at = _now_iso()
                state.updated_at = _now_iso()
                self._write_state(state)
            self._jobs[state.job_id] = state

    def _read_state(self, job_id: str) -> InferenceJobState | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Failed to read inference job state: %s", path)
            return None
        with self._lock:
            self._jobs[job_id] = state
        return state

    def _store_state(self, state: InferenceJobState) -> None:
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

    def _write_state(self, state: InferenceJobState) -> None:
        path = self.jobs_dir / f"{state.job_id}.json"
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _append_log(state: InferenceJobState, message: str) -> None:
    line = f"{_now_iso()} {message}"
    if state.logs and state.logs[-1].endswith(message):
        return
    state.logs.append(line)
    state.logs = state.logs[-120:]


def _state_from_dict(data: dict[str, Any]) -> InferenceJobState:
    return InferenceJobState(
        job_id=data["job_id"],
        status=data["status"],
        checkpoint_path=data["checkpoint_path"],
        config_path=data["config_path"],
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
