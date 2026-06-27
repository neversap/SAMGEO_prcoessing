from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    backend: str = "mock"
    model_dir: str = "/models/sam3"
    device: str = "6,7"
    mask_postprocess: str = "polygon"
    polygon_epsilon_ratio: float = 0.003
    min_mask_area: int = 64
    max_mask_area: int = 0
    fill_mask_holes: bool = True
    max_hole_area: int = 256
    component_connectivity: int = 8
    orthogonal_min_edge: int = 4
    orthogonal_max_expand_ratio: float = 0.35
    quad_mode: str = "axis"
    data_process_allowed_roots: str = ""
    data_process_jobs_dir: str = "runtime/preprocess_jobs"
    data_process_max_workers: int = 1
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAM_GEO_",
        extra="ignore",
    )


settings = Settings()
