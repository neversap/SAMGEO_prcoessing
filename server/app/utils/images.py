import base64
import io

import cv2
import numpy as np
from PIL import Image


def read_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def mask_to_png_base64(mask: np.ndarray) -> str:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    image = Image.fromarray(mask_uint8, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def polygonize_mask(
    mask: np.ndarray,
    epsilon_ratio: float = 0.003,
    min_area: int = 64,
) -> np.ndarray:
    binary = mask.astype(bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return np.zeros_like(mask, dtype=bool)

    output = np.zeros_like(binary, dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(1.0, epsilon_ratio * perimeter)
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        if polygon.shape[0] >= 3:
            cv2.fillPoly(output, [polygon], 255)

    return output.astype(bool)


def orthogonalize_mask(
    mask: np.ndarray,
    epsilon_ratio: float = 0.003,
    min_area: int = 64,
    min_edge: int = 4,
    max_expand_ratio: float = 0.35,
) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    source_area = int(binary.sum())
    mask_uint8 = binary.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    output = np.zeros_like(mask_uint8, dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(1.0, epsilon_ratio * perimeter)
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        polygon = complete_polygon_corners(
            polygon,
            binary.shape,
            binary,
            min_edge=min_edge,
            max_expand_ratio=max_expand_ratio,
            source_area=source_area,
        )
        if polygon.shape[0] >= 3:
            cv2.fillPoly(output, [polygon.astype(np.int32)], 255)

    return (output > 0) | binary


def quadify_mask(
    mask: np.ndarray,
    mode: str = "axis",
    min_area: int = 64,
) -> np.ndarray:
    binary = mask.astype(bool).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(mask, dtype=bool)

    output = np.zeros_like(binary, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = (labels == label).astype(np.uint8)
        if mode.lower() == "rotated":
            component_quad = quadify_component_rotated(component)
        elif mode.lower() == "axis":
            component_quad = quadify_component(component)
        else:
            raise ValueError(f"unsupported quad mode: {mode}")
        output[component_quad] = 1

    return output.astype(bool)


def quadify_component(
    component: np.ndarray,
    min_protrusion_area: int = 32,
    max_connect_gap: int = 8,
    max_expand_ratio: float = 0.45,
) -> np.ndarray:
    source_area = int(component.sum())
    body = estimate_body_region(component)
    body_box = mask_bbox(body)
    if body_box is None:
        body_box = mask_bbox(component)
    if body_box is None:
        return np.zeros_like(component, dtype=bool)

    output = np.zeros_like(component, dtype=np.uint8)
    fill_box(output, body_box)

    residual = (component > 0) & (output == 0)
    residual_count, residual_labels, residual_stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8),
        connectivity=8,
    )
    for label in range(1, residual_count):
        area = int(residual_stats[label, cv2.CC_STAT_AREA])
        if area < min_protrusion_area:
            continue

        protrusion_box = stats_bbox(residual_stats[label])
        if bbox_gap(body_box, protrusion_box) > max_connect_gap:
            continue

        candidate = output.copy()
        fill_connector(candidate, body_box, protrusion_box)
        fill_box(candidate, protrusion_box)
        added = int((candidate.astype(bool) & (component == 0)).sum())
        max_added = int(max(source_area * max_expand_ratio, min_protrusion_area))
        if added <= max_added:
            output = candidate

    return output.astype(bool)


def quadify_component_rotated(
    component: np.ndarray,
) -> np.ndarray:
    decomposition = decompose_component_by_projection(component)
    if decomposition is None:
        return np.zeros_like(component, dtype=bool)

    body_polygon, body_angle, protrusion_seed = decomposition

    output = np.zeros_like(component, dtype=np.uint8)
    fill_polygon(output, body_polygon)

    body_box = polygon_bbox(body_polygon)
    residual_count, residual_labels, residual_stats, _ = cv2.connectedComponentsWithStats(
        protrusion_seed.astype(np.uint8),
        connectivity=8,
    )
    min_protrusion_area = max(8, int(component.sum() * 0.015))
    max_part_expand_ratio = 0.75
    max_total_expand_ratio = 0.55
    for label in range(1, residual_count):
        area = int(residual_stats[label, cv2.CC_STAT_AREA])
        if area < min_protrusion_area:
            continue

        protrusion = (residual_labels == label).astype(np.uint8)
        protrusion_polygon = fixed_angle_rect_polygon(protrusion, body_angle)
        if protrusion_polygon is None:
            continue
        protrusion_raster = rasterize_polygon(protrusion_polygon, component.shape)
        protrusion_added = int((protrusion_raster & (component == 0)).sum())
        max_part_added = int(max(area * max_part_expand_ratio, min_protrusion_area))
        if protrusion_added > max_part_added:
            continue

        protrusion_box = polygon_bbox(protrusion_polygon)
        candidate = output.copy()
        fill_aligned_connector(candidate, body_box, protrusion_box, body_angle)
        fill_polygon(candidate, protrusion_polygon)
        added = int((candidate.astype(bool) & (component == 0)).sum())
        max_added = int(max(component.sum() * max_total_expand_ratio, min_protrusion_area))
        if added <= max_added:
            output = candidate

    return output.astype(bool)


def decompose_component_by_projection(
    component: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    base_polygon = min_area_rect_polygon(component)
    if base_polygon is None:
        return None

    angle = polygon_long_edge_angle(base_polygon)
    ys, xs = np.where(component > 0)
    if xs.size < 3 or ys.size < 3:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    local_points, inverse_rotation = to_local_points(points, center, angle)
    core_bounds = trim_sparse_projection_bounds(local_points)
    core_polygon = local_bounds_to_polygon(core_bounds, center, inverse_rotation)
    core_mask = rasterize_polygon(core_polygon, component.shape)
    protrusion_seed = (component > 0) & ~core_mask
    if not protrusion_seed.any():
        return base_polygon, angle, protrusion_seed

    return core_polygon, angle, protrusion_seed


def to_local_points(
    points: np.ndarray,
    center: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(angle_degrees)
    cos_value = float(np.cos(radians))
    sin_value = float(np.sin(radians))
    rotation = np.array(
        [[cos_value, sin_value], [-sin_value, cos_value]],
        dtype=np.float32,
    )
    inverse_rotation = np.array(
        [[cos_value, -sin_value], [sin_value, cos_value]],
        dtype=np.float32,
    )
    return (points - center) @ rotation.T, inverse_rotation


def trim_sparse_projection_bounds(
    local_points: np.ndarray,
) -> tuple[float, float, float, float]:
    min_x, min_y = local_points.min(axis=0)
    max_x, max_y = local_points.max(axis=0)
    trim_x_min, trim_x_max = dense_projection_range(local_points[:, 0], local_points[:, 1])
    trim_y_min, trim_y_max = dense_projection_range(local_points[:, 1], local_points[:, 0])

    if trim_x_max > trim_x_min:
        min_x, max_x = trim_x_min, trim_x_max
    if trim_y_max > trim_y_min:
        min_y, max_y = trim_y_min, trim_y_max
    return float(min_x), float(min_y), float(max_x), float(max_y)


def dense_projection_range(primary: np.ndarray, secondary: np.ndarray) -> tuple[float, float]:
    span = float(primary.max() - primary.min())
    if span < 6:
        return float(primary.min()), float(primary.max())

    bin_count = int(max(8, min(64, round(span / 2))))
    counts, edges = np.histogram(primary, bins=bin_count)
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return float(primary.min()), float(primary.max())

    threshold = max(2, int(np.percentile(nonzero, 55) * 0.45))
    dense = counts >= threshold
    if not dense.any():
        return float(primary.min()), float(primary.max())

    dense = close_dense_bins(dense)
    runs = dense_runs(dense)
    if not runs:
        return float(primary.min()), float(primary.max())

    best_start, best_end = max(
        runs,
        key=lambda item: counts[item[0] : item[1] + 1].sum(),
    )
    return float(edges[best_start]), float(edges[best_end + 1])


def close_dense_bins(dense: np.ndarray) -> np.ndarray:
    result = dense.copy()
    for index in range(1, len(result) - 1):
        if not result[index] and result[index - 1] and result[index + 1]:
            result[index] = True
    return result


def dense_runs(dense: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(dense):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(dense) - 1))
    return runs


def local_bounds_to_polygon(
    bounds: tuple[float, float, float, float],
    center: np.ndarray,
    inverse_rotation: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    local_box = np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=np.float32,
    )
    polygon = local_box @ inverse_rotation.T + center
    return np.round(polygon).astype(np.int32)


def min_area_rect_polygon(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if xs.size < 3 or ys.size < 3:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    polygon = cv2.boxPoints(rect)
    return np.round(polygon).astype(np.int32)


def fixed_angle_rect_polygon(mask: np.ndarray, angle_degrees: float) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if xs.size < 3 or ys.size < 3:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    radians = np.deg2rad(angle_degrees)
    cos_value = float(np.cos(radians))
    sin_value = float(np.sin(radians))
    rotation = np.array(
        [[cos_value, sin_value], [-sin_value, cos_value]],
        dtype=np.float32,
    )
    inverse_rotation = np.array(
        [[cos_value, -sin_value], [sin_value, cos_value]],
        dtype=np.float32,
    )

    local_points = (points - center) @ rotation.T
    min_x, min_y = local_points.min(axis=0)
    max_x, max_y = local_points.max(axis=0)
    local_box = np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=np.float32,
    )
    polygon = local_box @ inverse_rotation.T + center
    return np.round(polygon).astype(np.int32)


def polygon_long_edge_angle(polygon: np.ndarray) -> float:
    points = polygon.astype(np.float32)
    best_angle = 0.0
    best_length = -1.0
    for index, start in enumerate(points):
        end = points[(index + 1) % points.shape[0]]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = dx * dx + dy * dy
        if length > best_length:
            best_length = length
            best_angle = float(np.rad2deg(np.arctan2(dy, dx)))
    return best_angle


def polygon_bbox(polygon: np.ndarray) -> tuple[int, int, int, int]:
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fill_polygon(mask: np.ndarray, polygon: np.ndarray) -> None:
    height, width = mask.shape
    polygon = polygon.astype(np.int32).copy()
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon], 1)


def estimate_body_region(
    component: np.ndarray,
    open_ratio: float = 0.08,
    erode_size: int = 0,
) -> np.ndarray:
    area = int(component.sum())
    kernel_size = int(max(3, min(51, round(area**0.5 * open_ratio))))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if opened.any() and erode_size > 0:
        erode_kernel_size = max(1, erode_size)
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), dtype=np.uint8)
        eroded = cv2.erode(opened, erode_kernel, iterations=1)
        if eroded.any():
            opened = eroded
    if opened.any():
        return largest_component(opened)
    return largest_component(component)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas) + 1)
    return (labels == label).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def stats_bbox(stats_row: np.ndarray) -> tuple[int, int, int, int]:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x, y, x + w - 1, y + h - 1


def fill_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1


def bbox_gap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    dx = max(bx1 - ax2 - 1, ax1 - bx2 - 1, 0)
    dy = max(by1 - ay2 - 1, ay1 - by2 - 1, 0)
    return max(dx, dy)


def fill_connector(
    mask: np.ndarray,
    body_box: tuple[int, int, int, int],
    protrusion_box: tuple[int, int, int, int],
) -> None:
    ax1, ay1, ax2, ay2 = body_box
    bx1, by1, bx2, by2 = protrusion_box
    x1 = max(min(ax1, bx1), min(max(ax1, bx1), min(ax2, bx2)))
    x2 = min(max(ax2, bx2), max(min(ax2, bx2), max(ax1, bx1)))
    y1 = max(min(ay1, by1), min(max(ay1, by1), min(ay2, by2)))
    y2 = min(max(ay2, by2), max(min(ay2, by2), max(ay1, by1)))

    if ax2 < bx1:
        x1, x2 = ax2, bx1
    elif bx2 < ax1:
        x1, x2 = bx2, ax1

    if ay2 < by1:
        y1, y2 = ay2, by1
    elif by2 < ay1:
        y1, y2 = by2, ay1

    fill_box(mask, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))


def fill_aligned_connector(
    mask: np.ndarray,
    body_box: tuple[int, int, int, int],
    protrusion_box: tuple[int, int, int, int],
    angle_degrees: float,
) -> None:
    connector = np.zeros_like(mask, dtype=np.uint8)
    fill_connector(connector, body_box, protrusion_box)
    connector_polygon = fixed_angle_rect_polygon(connector, angle_degrees)
    if connector_polygon is None:
        fill_connector(mask, body_box, protrusion_box)
        return
    fill_polygon(mask, connector_polygon)


def complete_polygon_corners(
    polygon: np.ndarray,
    shape: tuple[int, int],
    source: np.ndarray,
    min_edge: int = 4,
    max_expand_ratio: float = 0.35,
    source_area: int | None = None,
) -> np.ndarray:
    points = remove_short_edges(polygon.astype(np.int32), min_edge=min_edge)
    if points.shape[0] < 3:
        return polygon

    source_area = int(source.sum()) if source_area is None else source_area
    max_iterations = max(8, points.shape[0] * 3)
    for _ in range(max_iterations):
        edge_index = find_diagonal_edge(points, min_edge=min_edge)
        if edge_index is None:
            break

        start = points[edge_index]
        end = points[(edge_index + 1) % points.shape[0]]
        corner_candidates = [
            np.array([end[0], start[1]], dtype=np.int32),
            np.array([start[0], end[1]], dtype=np.int32),
        ]
        candidate_polygons = [
            insert_corner(points, edge_index, corner)
            for corner in corner_candidates
            if not np.array_equal(corner, start) and not np.array_equal(corner, end)
        ]
        if not candidate_polygons:
            break

        points = max(
            candidate_polygons,
            key=lambda item: score_fill_first_polygon(
                item,
                shape=shape,
                source=source,
                source_area=source_area,
                max_expand_ratio=max_expand_ratio,
            ),
        )
        points = remove_short_edges(points, min_edge=min_edge)
        if points.shape[0] < 3:
            return polygon

    return points


def find_diagonal_edge(points: np.ndarray, min_edge: int = 4) -> int | None:
    for index, start in enumerate(points):
        end = points[(index + 1) % points.shape[0]]
        dx = abs(int(end[0]) - int(start[0]))
        dy = abs(int(end[1]) - int(start[1]))
        if dx >= min_edge and dy >= min_edge:
            return index
    return None


def insert_corner(points: np.ndarray, edge_index: int, corner: np.ndarray) -> np.ndarray:
    return np.insert(points, edge_index + 1, corner, axis=0)


def score_fill_first_polygon(
    polygon: np.ndarray,
    shape: tuple[int, int],
    source: np.ndarray,
    source_area: int,
    max_expand_ratio: float,
) -> tuple[int, int, int]:
    rasterized = rasterize_polygon(polygon, shape)
    overlap = int((rasterized & source).sum())
    added = int((rasterized & ~source).sum())
    lost = source_area - overlap
    max_added = int(source_area * max_expand_ratio)
    expand_penalty = max(0, added - max_added)
    return (-lost, -expand_penalty, added)


def rasterize_polygon(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    if polygon.shape[0] >= 3:
        cv2.fillPoly(canvas, [polygon.astype(np.int32)], 1)
    return canvas.astype(bool)


def remove_short_edges(points: np.ndarray, min_edge: int = 4) -> np.ndarray:
    if points.shape[0] < 4:
        return points

    kept = [points[0]]
    for point in points[1:]:
        previous = kept[-1]
        length = max(abs(int(point[0]) - int(previous[0])), abs(int(point[1]) - int(previous[1])))
        if length >= min_edge:
            kept.append(point)

    if len(kept) > 2:
        first = kept[0]
        last = kept[-1]
        length = max(abs(int(first[0]) - int(last[0])), abs(int(first[1]) - int(last[1])))
        if length < min_edge:
            kept.pop()

    return np.array(kept, dtype=np.int32)


def clean_mask_components(
    mask: np.ndarray,
    min_area: int = 64,
    max_area: int | None = None,
    connectivity: int = 8,
    fill_holes: bool = True,
    max_hole_area: int = 256,
) -> np.ndarray:
    connectivity = 8 if connectivity == 8 else 4
    binary = mask.astype(bool).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=connectivity,
    )

    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        cleaned[labels == label] = 1

    if fill_holes and cleaned.any():
        cleaned = fill_small_holes(
            cleaned,
            connectivity=connectivity,
            max_hole_area=max_hole_area,
        )

    return cleaned.astype(bool)


def fill_small_holes(
    mask: np.ndarray,
    connectivity: int = 8,
    max_hole_area: int = 256,
) -> np.ndarray:
    binary = mask.astype(bool).astype(np.uint8)
    inverse = (binary == 0).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=connectivity,
    )

    height, width = binary.shape
    filled = binary.copy()
    for label in range(1, label_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and area <= max_hole_area:
            filled[labels == label] = 1

    return filled


def postprocess_masks(
    masks: list[np.ndarray],
    mode: str = "polygon",
    epsilon_ratio: float = 0.003,
    min_area: int = 64,
    max_area: int | None = None,
    fill_holes: bool = True,
    max_hole_area: int = 256,
    connectivity: int = 8,
    orthogonal_min_edge: int = 4,
    orthogonal_max_expand_ratio: float = 0.35,
    quad_mode: str = "axis",
) -> list[np.ndarray]:
    if mode.lower() in {"", "none", "off", "false"}:
        return [mask.astype(bool) for mask in masks]
    if mode.lower() not in {"clean", "polygon", "orthogonal", "quad"}:
        raise ValueError(f"unsupported mask postprocess mode: {mode}")

    cleaned_masks = [
        clean_mask_components(
            mask,
            min_area=min_area,
            max_area=max_area,
            connectivity=connectivity,
            fill_holes=fill_holes,
            max_hole_area=max_hole_area,
        )
        for mask in masks
    ]
    if mode.lower() == "clean":
        return cleaned_masks
    if mode.lower() == "orthogonal":
        return [
            orthogonalize_mask(
                mask,
                epsilon_ratio=epsilon_ratio,
                min_area=min_area,
                min_edge=orthogonal_min_edge,
                max_expand_ratio=orthogonal_max_expand_ratio,
            )
            for mask in cleaned_masks
        ]
    if mode.lower() == "quad":
        return [
            quadify_mask(
                mask,
                mode=quad_mode,
                min_area=min_area,
            )
            for mask in cleaned_masks
        ]
    return [
        polygonize_mask(mask, epsilon_ratio=epsilon_ratio, min_area=min_area)
        for mask in cleaned_masks
    ]


def semantic_mask_to_png_base64(
    masks: list[np.ndarray],
    width: int = 1,
    height: int = 1,
) -> str:
    if not masks:
        return mask_to_png_base64(np.zeros((height, width), dtype=bool))
    semantic = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        semantic |= mask.astype(bool)
    return mask_to_png_base64(semantic)


def instance_masks_to_png_base64(
    masks: list[np.ndarray],
    width: int = 1,
    height: int = 1,
) -> str:
    if not masks:
        return transparent_png_base64(width, height)

    height, width = masks[0].shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    fill_color = np.array((248, 113, 113, 90), dtype=np.uint8)
    border_color = np.array((127, 29, 29, 230), dtype=np.uint8)
    for mask in masks:
        mask_bool = mask.astype(bool)
        rgba[mask_bool] = fill_color

        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        border = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(border, contours, -1, 1, thickness=1)
        rgba[border.astype(bool)] = border_color

    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def transparent_png_base64(width: int, height: int) -> str:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def instance_color(index: int) -> tuple[int, int, int, int]:
    palette = [
        (20, 184, 166, 150),
        (245, 158, 11, 150),
        (37, 99, 235, 150),
        (225, 29, 72, 150),
        (132, 204, 22, 150),
        (14, 165, 233, 150),
        (168, 85, 247, 150),
        (249, 115, 22, 150),
    ]
    return palette[index % len(palette)]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
