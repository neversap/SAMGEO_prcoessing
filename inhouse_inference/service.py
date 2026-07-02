from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from inhouse_inference.predictor import InhousePredictor


_predictor: InhousePredictor | None = None


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(mask, dtype="uint8").reshape(-1)
    if flat.size == 0:
        return {"shape": list(mask.shape), "encoding": "value_counts", "counts": []}
    counts: list[list[int]] = []
    current = int(flat[0])
    length = 1
    for value in flat[1:]:
        value_int = int(value)
        if value_int == current:
            length += 1
            continue
        counts.append([current, length])
        current = value_int
        length = 1
    counts.append([current, length])
    return {
        "shape": list(mask.shape),
        "encoding": "value_counts",
        "counts": counts,
    }


def get_predictor() -> InhousePredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return _predictor


def create_app(
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    device: str | None = None,
    input_size: int | None = None,
    strict: bool | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _predictor
        resolved_config = config_path or os.getenv("INHOUSE_CONFIG_PATH")
        resolved_checkpoint = checkpoint_path or os.getenv("INHOUSE_CHECKPOINT_PATH")
        resolved_device = device or os.getenv("INHOUSE_DEVICE", "cuda")
        resolved_input_size = input_size or int(os.getenv("INHOUSE_INPUT_SIZE", "512"))
        resolved_strict = strict if strict is not None else os.getenv("INHOUSE_STRICT_LOAD", "0") == "1"
        if not resolved_config:
            raise RuntimeError("INHOUSE_CONFIG_PATH is required")
        if not resolved_checkpoint:
            raise RuntimeError("INHOUSE_CHECKPOINT_PATH is required")
        _predictor = InhousePredictor(
            config_path=resolved_config,
            checkpoint_path=resolved_checkpoint,
            device=resolved_device,
            input_size=resolved_input_size,
            strict=resolved_strict,
        )
        yield

    app = FastAPI(title="Inhouse Farmland Inference", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        predictor = get_predictor()
        return {
            "ok": True,
            "device": str(predictor.device),
            "input_size": predictor.input_size,
            "config": predictor.config_path,
            "checkpoint": predictor.checkpoint_path,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        min_area: int = 16,
        epsilon_ratio: float = 0.003,
    ) -> dict[str, Any]:
        predictor = get_predictor()
        payload = await file.read()
        try:
            image = Image.open(BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
        prediction = predictor.predict(
            image=image,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
        )
        result = prediction.to_json_dict()
        result["mask_512"] = encode_rle(prediction.mask_512)
        result["mask_original"] = encode_rle(prediction.mask_original)
        result["boundary_512"] = encode_rle(prediction.boundary_512)
        result["boundary_original"] = encode_rle(prediction.boundary_original)
        return result

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone inhouse inference service.")
    parser.add_argument("--host", default=os.getenv("INHOUSE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("INHOUSE_PORT", "8088")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("INHOUSE_WORKERS", "1")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "inhouse_inference.service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
