import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.app.adapters import create_segmenter
from server.app.adapters.base import SegmentInput
from server.app.schemas import HealthResponse, MaskResult, SegmentResponse
from server.app.settings import settings
from server.app.utils.images import bbox_from_mask, mask_to_png_base64
from server.app.utils.images import instance_masks_to_png_base64
from server.app.utils.images import postprocess_masks, read_image
from server.app.utils.images import semantic_mask_to_png_base64

logger = logging.getLogger("sam_geo")
app = FastAPI(title="SAM GEO API", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
segmenter = create_segmenter(
    backend=settings.backend,
    model_dir=settings.model_dir,
    device=settings.device,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def preload_model() -> None:
    segmenter.preload()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend=segmenter.name, device=settings.device)


@app.post("/segment", response_model=SegmentResponse)
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form("object"),
    threshold: float = Form(0.5),
    postprocess: str | None = Form(None),
    box: str | None = Form(None),
    points: str | None = Form(None),
) -> SegmentResponse:
    image_bytes = await image.read()
    pil_image = read_image(image_bytes)

    try:
        payload = SegmentInput(
            image=pil_image,
            prompt=prompt,
            threshold=parse_threshold(threshold),
            box=parse_box(box),
            points=parse_points(points),
        )
        masks = segmenter.segment(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("SAM3 segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        postprocess_mode, quad_mode = parse_postprocess(postprocess)
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
            quad_min_protrusion_area=getattr(
                settings,
                "quad_min_protrusion_area",
                32,
            ),
            quad_max_connect_gap=getattr(settings, "quad_max_connect_gap", 8),
            quad_max_expand_ratio=getattr(settings, "quad_max_expand_ratio", 0.45),
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


def parse_postprocess(value: str | None) -> tuple[str, str]:
    if not value:
        return settings.mask_postprocess, getattr(settings, "quad_mode", "axis")

    normalized = value.strip().lower()
    if normalized == "polygon":
        return "polygon", getattr(settings, "quad_mode", "axis")
    if normalized in {"quad_rotated", "quad+rotated", "quad-rotated"}:
        return "quad", "rotated"
    raise ValueError("postprocess must be polygon or quad_rotated")


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


if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
