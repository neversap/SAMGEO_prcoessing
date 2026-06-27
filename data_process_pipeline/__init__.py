"""Training data preprocessing pipeline for remote-sensing segmentation."""

__all__ = ["DataProcessConfig", "run_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import DataProcessConfig, run_pipeline

        return {"DataProcessConfig": DataProcessConfig, "run_pipeline": run_pipeline}[name]
    raise AttributeError(name)
