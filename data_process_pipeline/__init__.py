"""Training data preprocessing pipeline for remote-sensing segmentation."""

__all__ = ["DataProcessConfig", "FTWRGBConfig", "run_ftw_rgb_pipeline", "run_pipeline"]


def __getattr__(name: str):
    if name in {"DataProcessConfig", "run_pipeline"}:
        from .pipeline import DataProcessConfig, run_pipeline

        return {"DataProcessConfig": DataProcessConfig, "run_pipeline": run_pipeline}[name]
    if name in {"FTWRGBConfig", "run_ftw_rgb_pipeline"}:
        from .ftw_rgb import FTWRGBConfig, run_ftw_rgb_pipeline

        return {
            "FTWRGBConfig": FTWRGBConfig,
            "run_ftw_rgb_pipeline": run_ftw_rgb_pipeline,
        }[name]
    raise AttributeError(name)
