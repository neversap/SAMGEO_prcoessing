from server.app.adapters.base import Segmenter
from server.app.adapters.mock import MockSegmenter
from server.app.adapters.sam3 import Sam3Segmenter


def create_segmenter(backend: str, model_dir: str, device: str) -> Segmenter:
    normalized = backend.strip().lower()
    if normalized == "mock":
        return MockSegmenter()
    if normalized == "sam3":
        return Sam3Segmenter(model_dir=model_dir, device=device)
    raise ValueError(f"Unsupported backend: {backend}")

