from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_process_pipeline import DataProcessConfig, run_pipeline

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PreprocessJobRequest:
    dataset_dir: str
    tile_size: int = 512
    overlap: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    all_touched: bool = False
    drop_empty: bool = False
    min_patch_size: int = 128
    split_strategy: str = "image"
    test_process: bool = False
    mask_mode: str = "binary"
    boundary_width_pixels: int = 2
    background_keep_ratio: float = 0.2
    max_ignore_ratio: float = 0.5
    black_pixel_threshold: float = 0.0


@dataclass
class PreprocessJobState:
    job_id: str
    status: str
    dataset_dir: str
    progress: float = 0.0
    stage: str = "queued"
    current: int = 0
    total: int = 1
    message: str = "Queued"
    error: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class PreprocessJobManager:
    def __init__(self, jobs_dir: Path, allowed_roots: list[Path] | None = None, max_workers: int = 1) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [root.resolve() for root in allowed_roots or []]
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="preprocess")
        self._lock = threading.Lock()
        self._jobs: dict[str, PreprocessJobState] = {}
        self._load_existing_jobs()

    def create_job(self, request: PreprocessJobRequest) -> PreprocessJobState:
        dataset_dir = self._validate_dataset_dir(request.dataset_dir)
        self._validate_request(request)
        job_id = f"pre_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        state = PreprocessJobState(
            job_id=job_id,
            status="queued",
            dataset_dir=str(dataset_dir),
            output_paths=_output_paths(dataset_dir),
        )
        self._store_state(state)
        self._executor.submit(self._run_job, job_id, request, dataset_dir)
        return state

    def get_job(self, job_id: str) -> PreprocessJobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state:
            return state
        return self._read_state(job_id)

    def list_jobs(self) -> list[PreprocessJobState]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> PreprocessJobState | None:
        state = self.get_job(job_id)
        if not state:
            return None
        if state.status in TERMINAL_STATUSES:
            return state
        self._update_state(
            job_id,
            status="cancelled",
            finished_at=_now_iso(),
            message="Cancellation requested. Running file write may finish before stopping.",
        )
        return self.get_job(job_id)

    def clear_processed_data(self, dataset_dir: str) -> dict[str, list[str]]:
        resolved = self._validate_dataset_dir(dataset_dir)
        with self._lock:
            active_job = next(
                (
                    job
                    for job in self._jobs.values()
                    if Path(job.dataset_dir).resolve() == resolved and job.status not in TERMINAL_STATUSES
                ),
                None,
            )
        if active_job is not None:
            raise ValueError(f"Cannot clear processed data while job is active: {active_job.job_id}")
        cleared = []
        recreated = []
        for target in (resolved / "processed", resolved / "metadata"):
            if target.exists():
                for child in target.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                    cleared.append(str(child))
            target.mkdir(parents=True, exist_ok=True)
            recreated.append(str(target))
        return {"cleared": cleared, "recreated": recreated}

    def _run_job(self, job_id: str, request: PreprocessJobRequest, dataset_dir: Path) -> None:
        self._update_state(job_id, status="running", stage="initializing", message="Starting preprocessing")
        config = DataProcessConfig(
            dataset_dir=dataset_dir,
            tile_size=request.tile_size,
            overlap=request.overlap,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            seed=request.seed,
            all_touched=request.all_touched,
            keep_empty=not request.drop_empty,
            min_patch_size=request.min_patch_size,
            split_strategy=request.split_strategy,
            test_process=request.test_process,
            mask_mode=request.mask_mode,
            boundary_width_pixels=request.boundary_width_pixels,
            background_keep_ratio=request.background_keep_ratio,
            max_ignore_ratio=request.max_ignore_ratio,
            black_pixel_threshold=request.black_pixel_threshold,
        )

        def progress_callback(payload: dict[str, Any]) -> None:
            state = self.get_job(job_id)
            if state and state.status == "cancelled":
                raise RuntimeError("Preprocessing job was cancelled")
            current = int(payload.get("current", 0))
            total = max(int(payload.get("total", 1)), 1)
            progress = max(0.0, min(current / total, 1.0))
            self._update_state(
                job_id,
                status="running",
                progress=progress,
                stage=str(payload.get("stage", "running")),
                current=current,
                total=total,
                message=str(payload.get("message", "")),
            )

        try:
            run_pipeline(config, progress_callback=progress_callback)
        except Exception as exc:
            logger.exception("Data preprocessing job failed: %s", job_id)
            status = "cancelled" if "cancelled" in str(exc).lower() else "failed"
            self._update_state(
                job_id,
                status=status,
                error=str(exc),
                finished_at=_now_iso(),
                message=str(exc),
            )
            return

        self._update_state(
            job_id,
            status="completed",
            progress=1.0,
            stage="done",
            current=1,
            total=1,
            message="Preprocessing completed",
            finished_at=_now_iso(),
            output_paths=_output_paths(dataset_dir),
        )

    def _validate_dataset_dir(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("dataset_dir must be an absolute path")
        resolved = path.resolve()
        if self.allowed_roots and not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise ValueError(f"dataset_dir must be under one of: {roots}")
        if not resolved.exists():
            raise FileNotFoundError(f"dataset_dir does not exist: {resolved}")
        if not _has_supported_data_layout(resolved):
            raise FileNotFoundError(
                f"Missing supported data layout under {resolved}. "
                "Expected raw/images + raw/labels or tif + shp."
            )
        return resolved

    def _validate_request(self, request: PreprocessJobRequest) -> None:
        if request.overlap * 2 >= request.tile_size:
            raise ValueError("overlap must be less than half of tile_size")
        ratio_total = request.train_ratio + request.val_ratio + request.test_ratio
        if abs(ratio_total - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if request.split_strategy not in {"image", "patch"}:
            raise ValueError("split_strategy must be image or patch")
        if request.mask_mode not in {"binary", "field_boundary_3class"}:
            raise ValueError("mask_mode must be binary or field_boundary_3class")
        if request.boundary_width_pixels < 0:
            raise ValueError("boundary_width_pixels must be non-negative")
        if not 0.0 <= request.background_keep_ratio <= 1.0:
            raise ValueError("background_keep_ratio must be between 0 and 1")
        if not 0.0 <= request.max_ignore_ratio <= 1.0:
            raise ValueError("max_ignore_ratio must be between 0 and 1")
        if request.black_pixel_threshold < 0:
            raise ValueError("black_pixel_threshold must be non-negative")

    def _load_existing_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Failed to load preprocess job state: %s", path)
                continue
            if state.status not in TERMINAL_STATUSES:
                state.status = "failed"
                state.error = "Server restarted before this job finished"
                state.finished_at = _now_iso()
                state.updated_at = _now_iso()
                self._write_state(state)
            self._jobs[state.job_id] = state

    def _read_state(self, job_id: str) -> PreprocessJobState | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Failed to read preprocess job state: %s", path)
            return None
        with self._lock:
            self._jobs[job_id] = state
        return state

    def _store_state(self, state: PreprocessJobState) -> None:
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

    def _write_state(self, state: PreprocessJobState) -> None:
        path = self.jobs_dir / f"{state.job_id}.json"
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _append_log(state: PreprocessJobState, message: str) -> None:
    line = f"{_now_iso()} {message}"
    if state.logs and state.logs[-1].endswith(message):
        return
    state.logs.append(line)
    state.logs = state.logs[-80:]


def _state_from_dict(data: dict[str, Any]) -> PreprocessJobState:
    return PreprocessJobState(
        job_id=data["job_id"],
        status=data["status"],
        dataset_dir=data["dataset_dir"],
        progress=float(data.get("progress", 0.0)),
        stage=data.get("stage", "queued"),
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


def _output_paths(dataset_dir: Path) -> dict[str, str]:
    return {
        "masks": str(dataset_dir / "processed" / "masks"),
        "patches": str(dataset_dir / "processed" / "patches"),
        "image_info": str(dataset_dir / "metadata" / "image_info.csv"),
        "label_info": str(dataset_dir / "metadata" / "label_info.csv"),
        "overlap_report": str(dataset_dir / "metadata" / "overlap_report.csv"),
        "patch_index": str(dataset_dir / "metadata" / "patch_index.csv"),
        "band_stats": str(dataset_dir / "metadata" / "band_stats.json"),
    }


def _has_supported_data_layout(dataset_dir: Path) -> bool:
    modern = (dataset_dir / "raw" / "images").exists() and (dataset_dir / "raw" / "labels").exists()
    legacy = (dataset_dir / "tif").exists() and (dataset_dir / "shp").exists()
    return modern or legacy


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
