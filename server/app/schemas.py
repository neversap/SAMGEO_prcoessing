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


class PreprocessJobCreateRequest(BaseModel):
    dataset_dir: str
    tile_size: int = Field(default=512, ge=64, le=8192)
    overlap: int = Field(default=64, ge=0, le=4096)
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 42
    all_touched: bool = False
    drop_empty: bool = False
    min_patch_size: int = Field(default=128, ge=1, le=8192)
    split_strategy: str = "image"
    test_process: bool = False
    mask_mode: str = "binary"
    boundary_width_pixels: int = Field(default=2, ge=0, le=32)
    background_keep_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    max_ignore_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    black_pixel_threshold: float = Field(default=0.0, ge=0.0)


class PreprocessJobResponse(BaseModel):
    job_id: str
    status: str
    dataset_dir: str
    progress: float
    stage: str
    current: int
    total: int
    message: str
    error: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)


class FTWDownloadRequest(BaseModel):
    ftw_root: str
    countries: str
    extra_args: str = ""
    ftw_command: str = "ftw"


class FTWPreprocessRequest(BaseModel):
    ftw_root: str
    output_dir: str
    metadata_dir: str
    manifest_path: str | None = None
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 42
    use_both_windows: bool = True
    max_samples: int = Field(default=0, ge=0)


class FTWJobResponse(PreprocessJobResponse):
    job_type: str


class PreprocessJobListResponse(BaseModel):
    jobs: list[PreprocessJobResponse]


class ClearProcessedDataRequest(BaseModel):
    dataset_dir: str


class ClearProcessedDataResponse(BaseModel):
    dataset_dir: str
    cleared: list[str]
    recreated: list[str]


class DatasetPreviewSample(BaseModel):
    patch_name: str
    source_tif: str
    x: int
    y: int
    width: int
    height: int
    cropland_ratio: float
    ignore_ratio: float
    interior_ratio: float = 0.0
    boundary_ratio: float = 0.0
    background_ratio: float = 0.0
    patch_type: str = ""
    split: str
    image_path: str
    mask_path: str
    image_png_base64: str
    mask_png_base64: str
    overlay_png_base64: str


class DatasetPreviewResponse(BaseModel):
    dataset_dir: str
    split: str
    mode: str
    count: int
    samples: list[DatasetPreviewSample]


class TrainingIndexRequest(BaseModel):
    ftw_root: str
    metadata_dir: str
    country: str = "all"
    window: str = "both"
    mask_type: str = "semantic_3class"
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 42
    max_samples: int = Field(default=0, ge=0)


class TrainingIndexResponse(BaseModel):
    index_path: str
    stats_path: str
    count: int
    buckets: dict[str, int]
    splits: dict[str, int]


class TrainingAugmentPreviewSample(BaseModel):
    sample_id: str
    patch_name: str
    source_dataset: str
    source_tif: str
    country: str
    window: str
    split: str
    bucket: str
    fg_ratio: float
    cropland_ratio: float
    interior_ratio: float
    boundary_ratio: float
    ignore_ratio: float
    image_path: str
    mask_path: str
    augmentation: str
    image_png_base64: str
    mask_png_base64: str
    overlay_png_base64: str
    augmented_image_png_base64: str
    augmented_mask_png_base64: str
    augmented_overlay_png_base64: str


class TrainingAugmentPreviewResponse(BaseModel):
    source: str
    root_path: str
    split: str
    mode: str
    count: int
    stats: dict
    samples: list[TrainingAugmentPreviewSample]


class TrainingJobCreateRequest(BaseModel):
    config_path: str = "/app/configs/pretrain/ftw_rgb_unet_effb3_pretrain_v1.yaml"
    stage: str = "sanity_check"
    epochs: int = Field(default=0, ge=0)
    batch_size: int = Field(default=0, ge=0)
    max_train_samples: int = Field(default=0, ge=0)
    max_val_samples: int = Field(default=0, ge=0)
    init_checkpoint: str = ""


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    config_path: str
    stage: str
    progress: float
    current: int
    total: int
    message: str
    error: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)


class TrainingJobListResponse(BaseModel):
    jobs: list[TrainingJobResponse]


class TrainingMetricPoint(BaseModel):
    epoch: float
    train_loss: float
    val_loss: float
    val_miou: float
    val_boundary_f1: float
    val_pixel_accuracy: float
    lr: float
    seconds: float


class TrainingMetricsResponse(BaseModel):
    job_id: str
    status: str
    metrics: list[TrainingMetricPoint]


class InferenceJobCreateRequest(BaseModel):
    checkpoint_path: str = "/home/nvme1/datasets/ftw_datasets/outputs/ftw_rgb_unet_effb3_pretrain_v1/checkpoints/best_val_boundary_f1.pt"
    config_path: str = "/app/configs/pretrain/ftw_rgb_unet_effb3_pretrain_v1.yaml"
    ftw_metadata_csv: str = "/home/nvme1/datasets/ftw_datasets/data/ftw/metadata/ftw_dataloader_index.csv"
    inhouse_dataset_dir: str = "/home/nvme1/datasets"
    ftw_count: int = Field(default=10, ge=0, le=100)
    inhouse_count: int = Field(default=10, ge=0, le=100)
    seed: int = 42


class InferenceJobResponse(BaseModel):
    job_id: str
    status: str
    checkpoint_path: str
    config_path: str
    progress: float
    current: int
    total: int
    message: str
    error: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)


class InferenceSampleResponse(BaseModel):
    id: int
    source: str
    sample_id: str
    patch_name: str
    split: str
    country: str = ""
    window: str = ""
    cropland_ratio: float
    ignore_ratio: float
    image_path: str
    mask_path: str
    image_url: str
    gt_url: str
    pred_url: str
    overlay_url: str
    metrics: dict[str, float]


class InferenceSummaryResponse(BaseModel):
    job_id: str
    status: str
    checkpoint: str
    config: str
    count: int
    samples: list[InferenceSampleResponse]
