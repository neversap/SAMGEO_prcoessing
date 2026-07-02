from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_process_pipeline import FTWRGBConfig, run_ftw_rgb_pipeline

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FTWDownloadRequest:
    ftw_root: str
    countries: list[str]
    extra_args: str = ""
    ftw_command: str = "ftw"


@dataclass
class FTWPreprocessRequest:
    ftw_root: str
    output_dir: str
    metadata_dir: str
    manifest_path: str | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    use_both_windows: bool = True
    max_samples: int = 0


@dataclass
class FTWJobState:
    job_id: str
    status: str
    job_type: str
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


class FTWJobManager:
    def __init__(self, jobs_dir: Path, allowed_roots: list[Path] | None = None, max_workers: int = 1) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [root.resolve() for root in allowed_roots or []]
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ftw")
        self._lock = threading.Lock()
        self._jobs: dict[str, FTWJobState] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._load_existing_jobs()

    def create_download_job(self, request: FTWDownloadRequest) -> FTWJobState:
        ftw_root = self._validate_output_dir(request.ftw_root)
        countries = _normalize_countries(request.countries)
        if not countries:
            raise ValueError("countries is required")
        job_id = self._new_job_id("ftw_down")
        state = FTWJobState(
            job_id=job_id,
            status="queued",
            job_type="download",
            dataset_dir=str(ftw_root),
            output_paths={"ftw_root": str(ftw_root)},
            message=f"Queued FTW download for {', '.join(countries)}",
        )
        self._store_state(state)
        self._executor.submit(self._run_download_job, job_id, request, ftw_root, countries)
        return state

    def create_preprocess_job(self, request: FTWPreprocessRequest) -> FTWJobState:
        ftw_root = self._validate_existing_dir(request.ftw_root, "ftw_root")
        output_dir = self._validate_output_dir(request.output_dir)
        metadata_dir = self._validate_output_dir(request.metadata_dir)
        manifest_path = self._validate_optional_file(request.manifest_path, "manifest_path")
        self._validate_ratios(request.train_ratio, request.val_ratio, request.test_ratio)
        if request.max_samples < 0:
            raise ValueError("max_samples must be non-negative")

        job_id = self._new_job_id("ftw_pre")
        state = FTWJobState(
            job_id=job_id,
            status="queued",
            job_type="preprocess",
            dataset_dir=str(ftw_root),
            output_paths=_ftw_preprocess_output_paths(output_dir, metadata_dir),
            message="Queued FTW RGB preprocessing",
        )
        self._store_state(state)
        self._executor.submit(
            self._run_preprocess_job,
            job_id,
            request,
            ftw_root,
            output_dir,
            metadata_dir,
            manifest_path,
        )
        return state

    def get_job(self, job_id: str) -> FTWJobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state:
            return state
        return self._read_state(job_id)

    def list_jobs(self) -> list[FTWJobState]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> FTWJobState | None:
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
            progress=1.0,
            finished_at=_now_iso(),
            message="Cancellation requested",
        )
        return self.get_job(job_id)

    def _run_download_job(self, job_id: str, request: FTWDownloadRequest, ftw_root: Path, countries: list[str]) -> None:
        command = [request.ftw_command, "data", "download", f"--countries={','.join(countries)}"]
        if request.extra_args.strip():
            command.extend(shlex.split(request.extra_args))
        env = os.environ.copy()
        env.setdefault("FTW_DATA_DIR", str(ftw_root))
        env.setdefault("FTW_DATA_ROOT", str(ftw_root))
        self._update_state(
            job_id,
            status="running",
            progress=0.05,
            stage="downloading",
            current=0,
            total=1,
            message=f"Running: {' '.join(command)}",
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ftw_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            self._fail_job(job_id, f"FTW CLI not found: {request.ftw_command}")
            return
        except Exception as exc:
            self._fail_job(job_id, str(exc))
            return

        with self._lock:
            self._processes[job_id] = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if self._is_cancelled(job_id):
                    process.terminate()
                    return
                message = line.strip()
                if message:
                    self._update_state(job_id, progress=0.35, stage="downloading", message=message)
            return_code = process.wait()
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

        if self._is_cancelled(job_id):
            return
        if return_code != 0:
            self._fail_job(job_id, f"FTW download failed with exit code {return_code}")
            return
        self._update_state(
            job_id,
            status="completed",
            progress=1.0,
            stage="done",
            current=1,
            total=1,
            message="FTW download completed",
            finished_at=_now_iso(),
        )

    def _run_preprocess_job(
        self,
        job_id: str,
        request: FTWPreprocessRequest,
        ftw_root: Path,
        output_dir: Path,
        metadata_dir: Path,
        manifest_path: Path | None,
    ) -> None:
        self._update_state(
            job_id,
            status="running",
            progress=0.1,
            stage="preprocessing",
            message="Starting FTW RGB preprocessing",
        )
        config = FTWRGBConfig(
            ftw_root=ftw_root,
            output_dir=output_dir,
            metadata_dir=metadata_dir,
            manifest_path=manifest_path,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            seed=request.seed,
            use_both_windows=request.use_both_windows,
            max_samples=request.max_samples,
        )
        try:
            run_ftw_rgb_pipeline(config)
        except Exception as exc:
            logger.exception("FTW preprocessing job failed: %s", job_id)
            status = "cancelled" if "cancelled" in str(exc).lower() else "failed"
            self._update_state(
                job_id,
                status=status,
                progress=1.0,
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
            message="FTW RGB preprocessing completed",
            finished_at=_now_iso(),
            output_paths=_ftw_preprocess_output_paths(output_dir, metadata_dir),
        )

    def _new_job_id(self, prefix: str) -> str:
        return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _validate_existing_dir(self, value: str, name: str) -> Path:
        path = self._validate_path(value, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"{name} must be a directory: {path}")
        return path

    def _validate_output_dir(self, value: str) -> Path:
        path = self._validate_path(value, "path")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _validate_optional_file(self, value: str | None, name: str) -> Path | None:
        if not value:
            return None
        path = self._validate_path(value, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{name} must be a file: {path}")
        return path

    def _validate_path(self, value: str, name: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
        resolved = path.resolve()
        if self.allowed_roots and not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise ValueError(f"{name} must be under one of: {roots}")
        return resolved

    def _validate_ratios(self, train: float, val: float, test: float) -> None:
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    def _is_cancelled(self, job_id: str) -> bool:
        state = self.get_job(job_id)
        return bool(state and state.status == "cancelled")

    def _fail_job(self, job_id: str, message: str) -> None:
        logger.error("FTW job failed: %s: %s", job_id, message)
        self._update_state(
            job_id,
            status="failed",
            progress=1.0,
            error=message,
            finished_at=_now_iso(),
            message=message,
        )

    def _load_existing_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Failed to load FTW job state: %s", path)
                continue
            if state.status not in TERMINAL_STATUSES:
                state.status = "failed"
                state.error = "Server restarted before this job finished"
                state.finished_at = _now_iso()
                state.updated_at = _now_iso()
                self._write_state(state)
            self._jobs[state.job_id] = state

    def _read_state(self, job_id: str) -> FTWJobState | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            state = _state_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Failed to read FTW job state: %s", path)
            return None
        with self._lock:
            self._jobs[job_id] = state
        return state

    def _store_state(self, state: FTWJobState) -> None:
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

    def _write_state(self, state: FTWJobState) -> None:
        path = self.jobs_dir / f"{state.job_id}.json"
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _append_log(state: FTWJobState, message: str) -> None:
    line = f"{_now_iso()} {message}"
    if state.logs and state.logs[-1].endswith(message):
        return
    state.logs.append(line)
    state.logs = state.logs[-120:]


def _state_from_dict(data: dict[str, Any]) -> FTWJobState:
    return FTWJobState(
        job_id=data["job_id"],
        status=data["status"],
        job_type=data.get("job_type", "ftw"),
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


def _normalize_countries(countries: list[str]) -> list[str]:
    normalized = []
    for item in countries:
        for country in item.split(","):
            value = country.strip()
            if value:
                normalized.append(value)
    return normalized


def _ftw_preprocess_output_paths(output_dir: Path, metadata_dir: Path) -> dict[str, str]:
    return {
        "ftw_rgb": str(output_dir),
        "ftw_inspection": str(metadata_dir / "ftw_inspection.csv"),
        "ftw_patch_index": str(metadata_dir / "ftw_patch_index.csv"),
        "ftw_band_stats": str(metadata_dir / "ftw_band_stats.json"),
        "ftw_config": str(metadata_dir / "ftw_rgb_config.json"),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
