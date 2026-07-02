from __future__ import annotations

import numpy as np
from PIL import Image

from inhouse_inference.schemas import PolygonResult


CLASS_NAMES = {
    0: "background",
    1: "field_interior",
    2: "field_boundary",
}


def resize_mask_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype("uint8"), mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype="uint8")


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    field = ((mask == 1) | (mask == 2)).astype("uint8")
    try:
        import cv2
    except ModuleNotFoundError:
        return (mask == 2).astype("uint8")
    kernel = np.ones((3, 3), dtype="uint8")
    eroded = cv2.erode(field, kernel, iterations=1)
    return ((field - eroded) > 0).astype("uint8")


def extract_polygons(
    mask_512: np.ndarray,
    scale_x: float,
    scale_y: float,
    min_area: int = 16,
    epsilon_ratio: float = 0.003,
) -> list[PolygonResult]:
    field = ((mask_512 == 1) | (mask_512 == 2)).astype("uint8")
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("polygon extraction requires opencv-python-headless") from exc
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        area_512 = int(round(float(cv2.contourArea(contour))))
        if area_512 < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(1.0, perimeter * epsilon_ratio)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points_512 = [[int(point[0][0]), int(point[0][1])] for point in approx]
        if len(points_512) < 3:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        bbox_512 = [int(x), int(y), int(x + w), int(y + h)]
        points = scale_points(points_512, scale_x, scale_y)
        bbox = scale_bbox(bbox_512, scale_x, scale_y)
        results.append(
            PolygonResult(
                class_id=1,
                label="field",
                points=points,
                points_512=points_512,
                bbox=bbox,
                bbox_512=bbox_512,
                area=int(round(area_512 * scale_x * scale_y)),
                area_512=area_512,
            )
        )
    return sorted(results, key=lambda item: item.area, reverse=True)


def scale_points(points: list[list[int]], scale_x: float, scale_y: float) -> list[list[int]]:
    return [
        [int(round(x * scale_x)), int(round(y * scale_y))]
        for x, y in points
    ]


def scale_bbox(bbox: list[int], scale_x: float, scale_y: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]
