import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.app.adapters import create_segmenter
from server.app.adapters.base import SegmentInput
from server.app.schemas import ClearProcessedDataRequest, ClearProcessedDataResponse
from server.app.schemas import DatasetPreviewResponse, DatasetPreviewSample
from server.app.schemas import FTWDownloadRequest as FTWDownloadCreateRequest
from server.app.schemas import FTWJobResponse, FTWPreprocessRequest as FTWPreprocessCreateRequest
from server.app.schemas import HealthResponse, MaskResult, ProposalGroupResult
from server.app.schemas import PreprocessJobCreateRequest, PreprocessJobListResponse
from server.app.schemas import PreprocessJobResponse
from server.app.schemas import ProposalResponse, ProposalResult, SegmentResponse
from server.app.schemas import TrainingAugmentPreviewResponse, TrainingAugmentPreviewSample
from server.app.schemas import InferenceJobCreateRequest, InferenceJobResponse
from server.app.schemas import InferenceSampleResponse, InferenceSummaryResponse
from server.app.schemas import TrainingJobCreateRequest, TrainingJobListResponse, TrainingJobResponse
from server.app.schemas import TrainingMetricPoint, TrainingMetricsResponse
from server.app.schemas import TrainingIndexRequest, TrainingIndexResponse
from server.app.settings import settings
from server.app.utils.images import bbox_from_mask, mask_to_png_base64
from server.app.utils.images import instance_masks_to_png_base64
from server.app.utils.images import postprocess_masks, read_image
from server.app.utils.images import semantic_mask_to_png_base64
from server.app.utils.dataset_preview import load_preview_samples
from server.app.utils.ftw_jobs import FTWDownloadRequest, FTWJobManager, FTWPreprocessRequest
from server.app.utils.ftw_preview import load_ftw_preview_samples
from server.app.utils.proposals import edges_to_png_base64
from server.app.utils.proposals import generate_opencv_proposals
from server.app.utils.proposals import preprocess_to_png_base64
from server.app.utils.proposals import Proposal
from server.app.utils.proposals import proposals_to_png_base64
from server.app.utils.preprocess_jobs import PreprocessJobManager
from server.app.utils.preprocess_jobs import PreprocessJobRequest
from server.app.utils.training_preview import build_ftw_training_index
from server.app.utils.training_preview import load_training_augmentation_preview
from server.app.utils.training_jobs import TrainingJobManager, TrainingJobRequest
from server.app.utils.inference_jobs import InferenceJobManager, InferenceJobRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sam_geo")
logger.setLevel(logging.INFO)
app = FastAPI(title="SAM GEO API", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
segmenter = create_segmenter(
    backend=settings.backend,
    model_dir=settings.model_dir,
    device=settings.device,
)
data_process_allowed_roots = [
    Path(item.strip())
    for item in settings.data_process_allowed_roots.split(",")
    if item.strip()
]
preprocess_jobs = PreprocessJobManager(
    jobs_dir=Path(settings.data_process_jobs_dir),
    allowed_roots=data_process_allowed_roots,
    max_workers=settings.data_process_max_workers,
)
ftw_jobs = FTWJobManager(
    jobs_dir=Path(settings.data_process_jobs_dir) / "ftw",
    allowed_roots=data_process_allowed_roots,
    max_workers=settings.data_process_max_workers,
)
training_jobs = TrainingJobManager(
    jobs_dir=Path(settings.data_process_jobs_dir) / "training",
    allowed_roots=data_process_allowed_roots,
    max_workers=1,
)
inference_jobs = InferenceJobManager(
    jobs_dir=Path(settings.data_process_jobs_dir) / "inference",
    allowed_roots=data_process_allowed_roots,
    max_workers=1,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def preload_model() -> None:
    segmenter.preload()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/preprocess")
def preprocess_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "preprocess.html")


@app.get("/augmentation")
def augmentation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "augmentation.html")


@app.get("/training")
def training_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "training.html")


@app.get("/inference")
def inference_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "inference.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend=segmenter.name, device=settings.device)


@app.post("/ftw/download", response_model=FTWJobResponse)
def create_ftw_download_job(payload: FTWDownloadCreateRequest) -> FTWJobResponse:
    try:
        state = ftw_jobs.create_download_job(
            FTWDownloadRequest(
                ftw_root=payload.ftw_root,
                countries=[payload.countries],
                extra_args=payload.extra_args,
                ftw_command=payload.ftw_command,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ftw_job_response(state)


@app.post("/ftw/preprocess", response_model=FTWJobResponse)
def create_ftw_preprocess_job(payload: FTWPreprocessCreateRequest) -> FTWJobResponse:
    try:
        state = ftw_jobs.create_preprocess_job(
            FTWPreprocessRequest(
                ftw_root=payload.ftw_root,
                output_dir=payload.output_dir,
                metadata_dir=payload.metadata_dir,
                manifest_path=payload.manifest_path,
                train_ratio=payload.train_ratio,
                val_ratio=payload.val_ratio,
                test_ratio=payload.test_ratio,
                seed=payload.seed,
                use_both_windows=payload.use_both_windows,
                max_samples=payload.max_samples,
            )
        )
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ftw_job_response(state)


@app.get("/ftw/jobs/{job_id}", response_model=FTWJobResponse)
def get_ftw_job(job_id: str) -> FTWJobResponse:
    state = ftw_jobs.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="FTW job not found")
    return ftw_job_response(state)


@app.post("/ftw/jobs/{job_id}/cancel", response_model=FTWJobResponse)
def cancel_ftw_job(job_id: str) -> FTWJobResponse:
    state = ftw_jobs.cancel_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="FTW job not found")
    return ftw_job_response(state)


@app.get("/ftw/preview", response_model=DatasetPreviewResponse)
def preview_ftw_dataset(
    ftw_root: str = Query(...),
    country: str = Query("all"),
    window: str = Query("window_a"),
    mask_type: str = Query("semantic_3class"),
    mode: str = Query("random"),
    limit: int = Query(12, ge=1, le=50),
    seed: int = Query(42),
) -> DatasetPreviewResponse:
    try:
        samples = load_ftw_preview_samples(
            ftw_root=ftw_root,
            country=country,
            window=window,
            mask_type=mask_type,
            mode=mode,
            limit=limit,
            seed=seed,
            allowed_roots=data_process_allowed_roots,
        )
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DatasetPreviewResponse(
        dataset_dir=ftw_root,
        split=window,
        mode=mode,
        count=len(samples),
        samples=[
            DatasetPreviewSample(
                **item.metadata,
                image_png_base64=item.image_png_base64,
                mask_png_base64=item.mask_png_base64,
                overlay_png_base64=item.overlay_png_base64,
            )
            for item in samples
        ],
    )


@app.post("/preprocess/jobs", response_model=PreprocessJobResponse)
def create_preprocess_job(payload: PreprocessJobCreateRequest) -> PreprocessJobResponse:
    try:
        state = preprocess_jobs.create_job(
            PreprocessJobRequest(
                dataset_dir=payload.dataset_dir,
                tile_size=payload.tile_size,
                overlap=payload.overlap,
                train_ratio=payload.train_ratio,
                val_ratio=payload.val_ratio,
                test_ratio=payload.test_ratio,
                seed=payload.seed,
                all_touched=payload.all_touched,
                drop_empty=payload.drop_empty,
                min_patch_size=payload.min_patch_size,
                split_strategy=payload.split_strategy,
                test_process=payload.test_process,
                mask_mode=payload.mask_mode,
                boundary_width_pixels=payload.boundary_width_pixels,
                background_keep_ratio=payload.background_keep_ratio,
                max_ignore_ratio=payload.max_ignore_ratio,
                black_pixel_threshold=payload.black_pixel_threshold,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return preprocess_job_response(state)


@app.get("/preprocess/jobs", response_model=PreprocessJobListResponse)
def list_preprocess_jobs() -> PreprocessJobListResponse:
    return PreprocessJobListResponse(
        jobs=[preprocess_job_response(state) for state in preprocess_jobs.list_jobs()]
    )


@app.get("/preprocess/jobs/{job_id}", response_model=PreprocessJobResponse)
def get_preprocess_job(job_id: str) -> PreprocessJobResponse:
    state = preprocess_jobs.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="preprocess job not found")
    return preprocess_job_response(state)


@app.post("/preprocess/jobs/{job_id}/cancel", response_model=PreprocessJobResponse)
def cancel_preprocess_job(job_id: str) -> PreprocessJobResponse:
    state = preprocess_jobs.cancel_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="preprocess job not found")
    return preprocess_job_response(state)


@app.post("/preprocess/clear", response_model=ClearProcessedDataResponse)
def clear_processed_data(payload: ClearProcessedDataRequest) -> ClearProcessedDataResponse:
    try:
        result = preprocess_jobs.clear_processed_data(payload.dataset_dir)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ClearProcessedDataResponse(
        dataset_dir=payload.dataset_dir,
        cleared=result["cleared"],
        recreated=result["recreated"],
    )


@app.get("/preprocess/preview", response_model=DatasetPreviewResponse)
def preview_dataset(
    dataset_dir: str = Query(...),
    split: str = Query("all"),
    mode: str = Query("random"),
    limit: int = Query(12, ge=1, le=50),
    seed: int = Query(42),
) -> DatasetPreviewResponse:
    try:
        samples = load_preview_samples(
            dataset_dir=dataset_dir,
            split=split,
            mode=mode,
            limit=limit,
            seed=seed,
            allowed_roots=data_process_allowed_roots,
        )
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DatasetPreviewResponse(
        dataset_dir=dataset_dir,
        split=split,
        mode=mode,
        count=len(samples),
        samples=[
            DatasetPreviewSample(
                **item.metadata,
                image_png_base64=item.image_png_base64,
                mask_png_base64=item.mask_png_base64,
                overlay_png_base64=item.overlay_png_base64,
            )
            for item in samples
        ],
    )


@app.post("/training/index/ftw", response_model=TrainingIndexResponse)
def create_training_ftw_index(payload: TrainingIndexRequest) -> TrainingIndexResponse:
    try:
        result = build_ftw_training_index(
            ftw_root=payload.ftw_root,
            metadata_dir=payload.metadata_dir,
            country=payload.country,
            window=payload.window,
            mask_type=payload.mask_type,
            train_ratio=payload.train_ratio,
            val_ratio=payload.val_ratio,
            test_ratio=payload.test_ratio,
            seed=payload.seed,
            max_samples=payload.max_samples,
            allowed_roots=data_process_allowed_roots,
        )
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TrainingIndexResponse(
        index_path=str(result.index_path),
        stats_path=str(result.stats_path),
        count=result.count,
        buckets=result.buckets,
        splits=result.splits,
    )


@app.get("/training/augment-preview", response_model=TrainingAugmentPreviewResponse)
def preview_training_augmentation(
    source: str = Query("ftw"),
    root_path: str = Query(...),
    country: str = Query("all"),
    window: str = Query("window_a"),
    mask_type: str = Query("semantic_3class"),
    split: str = Query("all"),
    mode: str = Query("random"),
    limit: int = Query(6, ge=1, le=24),
    seed: int = Query(42),
    hflip: bool = Query(True),
    vflip: bool = Query(True),
    rotate90: bool = Query(True),
    scale_jitter: float = Query(0.15, ge=0.0, le=0.5),
    brightness: float = Query(0.12, ge=0.0, le=0.5),
    contrast: float = Query(0.12, ge=0.0, le=0.5),
    noise: float = Query(0.02, ge=0.0, le=0.2),
) -> TrainingAugmentPreviewResponse:
    try:
        samples, stats = load_training_augmentation_preview(
            source=source,
            root_path=root_path,
            country=country,
            window=window,
            mask_type=mask_type,
            split=split,
            mode=mode,
            limit=limit,
            seed=seed,
            hflip=hflip,
            vflip=vflip,
            rotate90=rotate90,
            scale_jitter=scale_jitter,
            brightness=brightness,
            contrast=contrast,
            noise=noise,
            allowed_roots=data_process_allowed_roots,
        )
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TrainingAugmentPreviewResponse(
        source=source,
        root_path=root_path,
        split=split,
        mode=mode,
        count=len(samples),
        stats=stats,
        samples=[
            TrainingAugmentPreviewSample(
                **item.metadata,
                image_png_base64=item.image_png_base64,
                mask_png_base64=item.mask_png_base64,
                overlay_png_base64=item.overlay_png_base64,
                augmented_image_png_base64=item.augmented_image_png_base64,
                augmented_mask_png_base64=item.augmented_mask_png_base64,
                augmented_overlay_png_base64=item.augmented_overlay_png_base64,
            )
            for item in samples
        ],
    )


@app.post("/training/jobs", response_model=TrainingJobResponse)
def create_training_job(payload: TrainingJobCreateRequest) -> TrainingJobResponse:
    try:
        state = training_jobs.create_job(
            TrainingJobRequest(
                config_path=payload.config_path,
                stage=payload.stage,
                epochs=payload.epochs,
                batch_size=payload.batch_size,
                max_train_samples=payload.max_train_samples,
                max_val_samples=payload.max_val_samples,
                init_checkpoint=payload.init_checkpoint,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return training_job_response(state)


@app.get("/training/jobs", response_model=TrainingJobListResponse)
def list_training_jobs() -> TrainingJobListResponse:
    return TrainingJobListResponse(
        jobs=[training_job_response(state) for state in training_jobs.list_jobs()]
    )


@app.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str) -> TrainingJobResponse:
    state = training_jobs.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="training job not found")
    return training_job_response(state)


@app.get("/training/jobs/{job_id}/metrics", response_model=TrainingMetricsResponse)
def get_training_metrics(job_id: str) -> TrainingMetricsResponse:
    state = training_jobs.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="training job not found")
    try:
        metrics = training_jobs.read_metrics(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TrainingMetricsResponse(
        job_id=job_id,
        status=state.status,
        metrics=[TrainingMetricPoint(**row) for row in metrics],
    )


@app.post("/training/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
def cancel_training_job(job_id: str) -> TrainingJobResponse:
    state = training_jobs.cancel_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="training job not found")
    return training_job_response(state)


@app.post("/inference/jobs", response_model=InferenceJobResponse)
def create_inference_job(payload: InferenceJobCreateRequest) -> InferenceJobResponse:
    try:
        state = inference_jobs.create_job(
            InferenceJobRequest(
                checkpoint_path=payload.checkpoint_path,
                config_path=payload.config_path,
                ftw_metadata_csv=payload.ftw_metadata_csv,
                inhouse_dataset_dir=payload.inhouse_dataset_dir,
                ftw_count=payload.ftw_count,
                inhouse_count=payload.inhouse_count,
                seed=payload.seed,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return inference_job_response(state)


@app.get("/inference/jobs/{job_id}", response_model=InferenceJobResponse)
def get_inference_job(job_id: str) -> InferenceJobResponse:
    state = inference_jobs.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="inference job not found")
    return inference_job_response(state)


@app.post("/inference/jobs/{job_id}/cancel", response_model=InferenceJobResponse)
def cancel_inference_job(job_id: str) -> InferenceJobResponse:
    state = inference_jobs.cancel_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="inference job not found")
    return inference_job_response(state)


@app.get("/inference/jobs/{job_id}/summary", response_model=InferenceSummaryResponse)
def get_inference_summary(job_id: str) -> InferenceSummaryResponse:
    state = inference_jobs.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="inference job not found")
    try:
        summary = inference_jobs.read_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return InferenceSummaryResponse(
        job_id=job_id,
        status=state.status,
        checkpoint=summary.get("checkpoint", ""),
        config=summary.get("config", ""),
        count=int(summary.get("count", 0)),
        samples=[
            InferenceSampleResponse(
                id=int(item.get("id", 0)),
                source=item.get("source", ""),
                sample_id=item.get("sample_id", ""),
                patch_name=item.get("patch_name", ""),
                split=item.get("split", ""),
                country=item.get("country", ""),
                window=item.get("window", ""),
                cropland_ratio=float(item.get("cropland_ratio", 0.0)),
                ignore_ratio=float(item.get("ignore_ratio", 0.0)),
                image_path=item.get("image_path", ""),
                mask_path=item.get("mask_path", ""),
                image_url=f"/inference/jobs/{job_id}/files/{item.get('image_png', '')}",
                gt_url=f"/inference/jobs/{job_id}/files/{item.get('gt_png', '')}",
                pred_url=f"/inference/jobs/{job_id}/files/{item.get('pred_png', '')}",
                overlay_url=f"/inference/jobs/{job_id}/files/{item.get('overlay_png', '')}",
                metrics=item.get("metrics", {}),
            )
            for item in summary.get("samples", [])
        ],
    )


@app.get("/inference/jobs/{job_id}/files/{file_path:path}")
def get_inference_file(job_id: str, file_path: str) -> FileResponse:
    try:
        path = inference_jobs.resolve_output_file(job_id, file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


@app.post("/segment", response_model=SegmentResponse)
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form("object"),
    threshold: float = Form(0.5),
    postprocess: str | None = Form(None),
    inference_mode: str | None = Form(None),
    box: str | None = Form(None),
    points: str | None = Form(None),
    use_opencv_proposals: bool = Form(False),
    max_proposals: int = Form(30),
) -> SegmentResponse:
    image_bytes = await image.read()
    pil_image = read_image(image_bytes)
    proposals = []
    proposals_png_base64 = None
    preprocess_png_base64 = None
    edges_png_base64 = None

    try:
        inference_mode, postprocess_mode, quad_mode = parse_segment_mode(
            postprocess,
            inference_mode,
        )
        should_generate_proposals = use_opencv_proposals
        if should_generate_proposals:
            proposals = generate_opencv_proposals(
                pil_image,
                max_proposals=parse_max_proposals(max_proposals),
            )
            edges_png_base64 = edges_to_png_base64(pil_image)
            preprocess_png_base64 = preprocess_to_png_base64(pil_image)
            proposals_png_base64 = proposals_to_png_base64(
                proposals,
                width=pil_image.width,
                height=pil_image.height,
            )
        payload = SegmentInput(
            image=pil_image,
            prompt=prompt,
            threshold=parse_threshold(threshold),
            box=parse_box(box),
            points=None if inference_mode == "sam_cascade" else parse_points(points),
            inference_mode=inference_mode,
            proposals=[
                {
                    "bbox": item.bbox,
                    "score": item.score,
                }
                for item in proposals
            ],
            max_proposals=parse_max_proposals(max_proposals),
        )
        masks = segmenter.segment(payload)
        if inference_mode == "sam_cascade":
            proposals = [
                Proposal(
                    bbox=item["bbox"],
                    point=item["point"],
                    score=item["score"],
                    area=item["area"],
                    angle=item["angle"],
                    polygon=item["polygon"],
                )
                for item in payload.proposals or []
            ]
            proposals_png_base64 = proposals_to_png_base64(
                proposals,
                width=pil_image.width,
                height=pil_image.height,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("SAM3 segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        processed_masks = postprocess_masks(
            [item.mask for item in masks],
            mode=postprocess_mode,
            epsilon_ratio=settings.polygon_epsilon_ratio,
            min_area=settings.min_mask_area,
            max_area=settings.max_mask_area if settings.max_mask_area > 0 else None,
            fill_holes=settings.fill_mask_holes,
            max_hole_area=settings.max_hole_area,
            connectivity=settings.component_connectivity,
            orthogonal_min_edge=getattr(settings, "orthogonal_min_edge", 4),
            orthogonal_max_expand_ratio=getattr(
                settings,
                "orthogonal_max_expand_ratio",
                0.35,
            ),
            quad_mode=quad_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    processed_results = [
        (item, mask)
        for item, mask in zip(masks, processed_masks, strict=True)
        if mask.any()
    ]
    bool_masks = [mask for _, mask in processed_results]

    return SegmentResponse(
        backend=segmenter.name,
        width=pil_image.width,
        height=pil_image.height,
        object_count=len(processed_results),
        semantic_png_base64=semantic_mask_to_png_base64(
            bool_masks,
            width=pil_image.width,
            height=pil_image.height,
        ),
        instances_png_base64=instance_masks_to_png_base64(
            bool_masks,
            width=pil_image.width,
            height=pil_image.height,
        ),
        masks=[
            MaskResult(
                id=index + 1,
                score=item.score,
                bbox=bbox_from_mask(mask),
                area=int(mask.sum()),
                area_ratio=float(mask.sum() / (pil_image.width * pil_image.height)),
                png_base64=mask_to_png_base64(mask),
            )
            for index, (item, mask) in enumerate(processed_results)
        ],
        proposals_png_base64=proposals_png_base64,
        preprocess_png_base64=preprocess_png_base64,
        edges_png_base64=edges_png_base64,
        proposals=serialize_proposals(proposals),
    )


@app.post("/proposals", response_model=ProposalResponse)
async def proposals(
    image: UploadFile = File(...),
    max_proposals: int = Form(30),
) -> ProposalResponse:
    image_bytes = await image.read()
    pil_image = read_image(image_bytes)
    try:
        generated = generate_opencv_proposals(
            pil_image,
            max_proposals=parse_max_proposals(max_proposals),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProposalResponse(
        width=pil_image.width,
        height=pil_image.height,
        proposal_count=len(generated),
        proposals_png_base64=proposals_to_png_base64(
            generated,
            width=pil_image.width,
            height=pil_image.height,
        ),
        edges_png_base64=edges_to_png_base64(pil_image),
        preprocess_png_base64=preprocess_to_png_base64(pil_image),
        proposals=serialize_proposals(generated),
    )


def parse_box(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("box must be formatted as x1,y1,x2,y2")
    return tuple(int(float(item)) for item in parts)


def parse_threshold(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return float(value)


def parse_max_proposals(value: int) -> int:
    if value < 1 or value > 1000:
        raise ValueError("max_proposals must be between 1 and 1000")
    return int(value)


def parse_segment_mode(
    postprocess: str | None,
    inference_mode: str | None = None,
) -> tuple[str, str, str]:
    normalized_inference = (inference_mode or "text").strip().lower()
    if normalized_inference in {"", "default"}:
        normalized_inference = "text"
    if normalized_inference in {
        "proposal_boxes",
        "proposal-boxes",
        "proposal_box_points",
        "proposal-box-points",
    }:
        normalized_inference = "sam_cascade"
    if normalized_inference not in {"text", "sam_cascade"}:
        raise ValueError("inference_mode must be text or sam_cascade")
    if normalized_inference == "sam_cascade":
        return "sam_cascade", "clean", "rotated"

    if not postprocess:
        return "text", settings.mask_postprocess, getattr(settings, "quad_mode", "axis")

    normalized = postprocess.strip().lower()
    if normalized == "polygon":
        return normalized_inference, "polygon", getattr(settings, "quad_mode", "axis")
    if normalized in {"quad_rotated", "quad+rotated", "quad-rotated"}:
        return normalized_inference, "quad", "rotated"
    if normalized in {
        "sam_cascade",
        "sam-cascade",
        "proposal_boxes",
        "proposal-boxes",
        "proposal_box_points",
        "proposal-box-points",
        "proposal",
    }:
        return "sam_cascade", "clean", "rotated"
    raise ValueError("postprocess must be polygon, quad_rotated, or sam_cascade")


def parse_points(value: str | None) -> list[tuple[int, int, int]] | None:
    if not value:
        return None
    raw_points = json.loads(value)
    parsed = []
    for point in raw_points:
        if len(point) != 3:
            raise ValueError("each point must be [x, y, label]")
        parsed.append((int(point[0]), int(point[1]), int(point[2])))
    return parsed


def serialize_proposals(proposals):
    return [
        ProposalResult(
            id=index + 1,
            score=item.score,
            bbox=item.bbox,
            point=item.point,
            area=item.area,
            angle=item.angle,
            polygon=item.polygon,
        )
        for index, item in enumerate(proposals)
    ]


def serialize_proposal_groups(groups):
    return [
        ProposalGroupResult(
            id=index + 1,
            bbox=item.bbox,
            points=[[int(x), int(y), 1] for x, y in item.points],
            proposal_ids=item.proposal_ids,
            proposal_count=item.proposal_count,
        )
        for index, item in enumerate(groups)
    ]


def preprocess_job_response(state) -> PreprocessJobResponse:
    return PreprocessJobResponse(
        job_id=state.job_id,
        status=state.status,
        dataset_dir=state.dataset_dir,
        progress=state.progress,
        stage=state.stage,
        current=state.current,
        total=state.total,
        message=state.message,
        error=state.error,
        created_at=state.created_at,
        updated_at=state.updated_at,
        finished_at=state.finished_at,
        output_paths=state.output_paths,
        logs=state.logs,
    )


def ftw_job_response(state) -> FTWJobResponse:
    return FTWJobResponse(
        job_id=state.job_id,
        status=state.status,
        job_type=state.job_type,
        dataset_dir=state.dataset_dir,
        progress=state.progress,
        stage=state.stage,
        current=state.current,
        total=state.total,
        message=state.message,
        error=state.error,
        created_at=state.created_at,
        updated_at=state.updated_at,
        finished_at=state.finished_at,
        output_paths=state.output_paths,
        logs=state.logs,
    )


def training_job_response(state) -> TrainingJobResponse:
    return TrainingJobResponse(
        job_id=state.job_id,
        status=state.status,
        config_path=state.config_path,
        stage=state.stage,
        progress=state.progress,
        current=state.current,
        total=state.total,
        message=state.message,
        error=state.error,
        created_at=state.created_at,
        updated_at=state.updated_at,
        finished_at=state.finished_at,
        output_paths=state.output_paths,
        logs=state.logs,
    )


def inference_job_response(state) -> InferenceJobResponse:
    return InferenceJobResponse(
        job_id=state.job_id,
        status=state.status,
        checkpoint_path=state.checkpoint_path,
        config_path=state.config_path,
        progress=state.progress,
        current=state.current,
        total=state.total,
        message=state.message,
        error=state.error,
        created_at=state.created_at,
        updated_at=state.updated_at,
        finished_at=state.finished_at,
        output_paths=state.output_paths,
        logs=state.logs,
    )


if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
