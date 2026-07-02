from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from inhouse_inference.config import get_nested, load_config
from inhouse_inference.model_loader import build_model, load_checkpoint
from inhouse_inference.postprocess import extract_boundary, extract_polygons, resize_mask_nearest
from inhouse_inference.preprocess import preprocess_image
from inhouse_inference.schemas import InhousePrediction


class InhousePredictor:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        input_size: int = 512,
        strict: bool = False,
    ) -> None:
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.config = load_config(config_path)
        self.input_size = input_size
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = build_model(self.config).to(self.device)
        self.checkpoint_info = load_checkpoint(self.model, str(checkpoint_path), self.device, strict=strict)
        self.model.eval()

    def predict(
        self,
        image,
        min_area: int = 16,
        epsilon_ratio: float = 0.003,
    ) -> InhousePrediction:
        preprocessed = preprocess_image(
            image=image,
            size=self.input_size,
            normalize=bool(get_nested(self.config, "input.normalize", False)),
            stats_mean=get_nested(self.config, "input.stats_mean"),
            stats_std=get_nested(self.config, "input.stats_std"),
        )
        tensor = preprocessed.tensor.to(self.device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.model(tensor)
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype("uint8")
        mask_original = resize_mask_nearest(pred, preprocessed.original_size)
        boundary_512 = extract_boundary(pred)
        boundary_original = resize_mask_nearest(boundary_512, preprocessed.original_size)
        polygons = extract_polygons(
            pred,
            scale_x=preprocessed.scale_x,
            scale_y=preprocessed.scale_y,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
        )
        field_pixels = int(((mask_original == 1) | (mask_original == 2)).sum())
        boundary_pixels = int((boundary_original > 0).sum())
        total = max(mask_original.size, 1)
        return InhousePrediction(
            original_size=preprocessed.original_size,
            model_input_size=preprocessed.model_input_size,
            scale_x=preprocessed.scale_x,
            scale_y=preprocessed.scale_y,
            mask_512=pred,
            mask_original=mask_original,
            boundary_512=boundary_512,
            boundary_original=boundary_original,
            polygons=polygons,
            stats={
                "field_area_ratio": float(field_pixels / total),
                "boundary_area_ratio": float(boundary_pixels / total),
                "checkpoint": self.checkpoint_path,
                "config": self.config_path,
                "checkpoint_missing_keys": self.checkpoint_info["missing_keys"],
                "checkpoint_unexpected_keys": self.checkpoint_info["unexpected_keys"],
            },
        )

    def predict_to_files(
        self,
        image,
        output_dir: str | Path,
        min_area: int = 16,
        epsilon_ratio: float = 0.003,
    ) -> dict[str, Any]:
        from inhouse_inference.preprocess import preprocess_image
        from inhouse_inference.visualization import save_outputs
        import json

        preprocessed = preprocess_image(
            image=image,
            size=self.input_size,
            normalize=bool(get_nested(self.config, "input.normalize", False)),
            stats_mean=get_nested(self.config, "input.stats_mean"),
            stats_std=get_nested(self.config, "input.stats_std"),
        )
        prediction = self.predict(image, min_area=min_area, epsilon_ratio=epsilon_ratio)
        files = save_outputs(
            output_dir=output_dir,
            original=preprocessed.original,
            resized=preprocessed.resized,
            mask_512=prediction.mask_512,
            mask_original=prediction.mask_original,
            boundary_512=prediction.boundary_512,
            boundary_original=prediction.boundary_original,
        )
        result = prediction.to_json_dict()
        result["files"] = files
        result_path = Path(output_dir) / "result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["files"]["result"] = str(result_path)
        return result
