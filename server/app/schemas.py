from pydantic import BaseModel, Field


class MaskResult(BaseModel):
    id: int
    score: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
    area: int
    area_ratio: float
    png_base64: str


class SegmentResponse(BaseModel):
    backend: str
    width: int
    height: int
    object_count: int
    semantic_png_base64: str
    instances_png_base64: str
    masks: list[MaskResult]


class HealthResponse(BaseModel):
    status: str
    backend: str
    device: str
