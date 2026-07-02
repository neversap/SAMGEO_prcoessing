from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PolygonResult:
    class_id: int
    label: str
    points: list[list[int]]
    points_512: list[list[int]]
    bbox: list[int]
    bbox_512: list[int]
    area: int
    area_512: int


@dataclass
class InhousePrediction:
    original_size: tuple[int, int]
    model_input_size: tuple[int, int]
    scale_x: float
    scale_y: float
    mask_512: np.ndarray
    mask_original: np.ndarray
    boundary_512: np.ndarray
    boundary_original: np.ndarray
    polygons: list[PolygonResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "original_size": list(self.original_size),
            "model_input_size": list(self.model_input_size),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "field_area_ratio": self.stats.get("field_area_ratio", 0.0),
            "boundary_area_ratio": self.stats.get("boundary_area_ratio", 0.0),
            "polygon_count": len(self.polygons),
            "polygons": [
                {
                    "class_id": item.class_id,
                    "label": item.label,
                    "points": item.points,
                    "points_512": item.points_512,
                    "bbox": item.bbox,
                    "bbox_512": item.bbox_512,
                    "area": item.area,
                    "area_512": item.area_512,
                }
                for item in self.polygons
            ],
        }
