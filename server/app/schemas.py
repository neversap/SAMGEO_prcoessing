from pydantic import BaseModel, Field


class MaskResult(BaseModel):
    id: int
    score: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
    area: int
    area_ratio: float
    png_base64: str


class ProposalResult(BaseModel):
    id: int
    score: float
    bbox: list[int]
    point: list[int]
    area: int
    angle: float
    polygon: list[list[int]]


class ProposalGroupResult(BaseModel):
    id: int
    bbox: list[int]
    points: list[list[int]]
    proposal_ids: list[int]
    proposal_count: int


class SegmentResponse(BaseModel):
    backend: str
    width: int
    height: int
    object_count: int
    semantic_png_base64: str
    instances_png_base64: str
    masks: list[MaskResult]
    proposals_png_base64: str | None = None
    preprocess_png_base64: str | None = None
    edges_png_base64: str | None = None
    proposals: list[ProposalResult] = Field(default_factory=list)
    proposal_groups_png_base64: str | None = None
    proposal_groups: list[ProposalGroupResult] = Field(default_factory=list)


class ProposalResponse(BaseModel):
    width: int
    height: int
    proposal_count: int
    proposals_png_base64: str
    preprocess_png_base64: str
    edges_png_base64: str
    proposals: list[ProposalResult]
    proposal_groups_png_base64: str | None = None
    proposal_groups: list[ProposalGroupResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    backend: str
    device: str
