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
from server.app.schemas import HealthResponse, MaskResult, ProposalGroupResult
from server.app.schemas import PreprocessJobCreateRequest, PreprocessJobListResponse
from server.app.schemas import PreprocessJobResponse
from server.app.schemas import ProposalResponse, ProposalResult, SegmentResponse
from server.app.settings import settings
from server.app.utils.images import bbox_from_mask, mask_to_png_base64
from server.app.utils.images import instance_masks_to_png_base64
from server.app.utils.images import postprocess_masks, read_image
from server.app.utils.images import semantic_mask_to_png_base64
from server.app.utils.dataset_preview import load_preview_samples
from server.app.utils.proposals import edges_to_png_base64
from server.app.utils.proposals import generate_opencv_proposals
from server.app.utils.proposals import preprocess_to_png_base64
from server.app.utils.proposals import Proposal
from server.app.utils.proposals import proposals_to_png_base64
from server.app.utils.preprocess_jobs import PreprocessJobManager
from server.app.utils.preprocess_jobs import PreprocessJobRequest

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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend=segmenter.name, device=settings.device)


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


if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
