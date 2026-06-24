from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Proposal:
    bbox: list[int]
    point: list[int]
    score: float
    area: int
    angle: float
    polygon: list[list[int]]


@dataclass
class ProposalGroup:
    bbox: list[int]
    points: list[list[int]]
    proposal_ids: list[int]
    proposal_count: int


def generate_opencv_proposals(
    image: Image.Image,
    max_proposals: int = 6,
    box_padding_ratio: float = 0.5,
    min_box_padding: int = 4,
    duplicate_iou_threshold: float = 0.35,
    containment_threshold: float = 0.60,
) -> list[Proposal]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    image_area = width * height
    if image_area <= 0:
        return []

    edge_mask, _ = _preprocess_masks(rgb)

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        rect_area = float(rw * rh)
        rectangularity = contour_area / rect_area if rect_area > 0 else 0.0

        boundary = np.zeros((height, width), dtype=np.uint8)
        if rect_area > 0:
            box_points = cv2.boxPoints(rect).astype(np.int32)
        else:
            box_points = _padded_axis_box_points(x, y, bw, bh, width, height)
            cx = x + bw / 2
            cy = y + bh / 2
            angle = 0.0
        cv2.polylines(boundary, [box_points], True, 255, 2)
        edge_hits = cv2.countNonZero(cv2.bitwise_and(edge_mask, boundary))
        boundary_pixels = max(1, cv2.countNonZero(boundary))
        edge_density = edge_hits / boundary_pixels

        area_score = min(1.0, contour_area / max(1.0, image_area * 0.01))
        score = (0.55 * rectangularity) + (0.35 * edge_density) + (0.10 * area_score)

        x1, y1, x2, y2 = x, y, x + bw, y + bh
        polygon = _axis_aligned_polygon_from_bbox(x1, y1, x2, y2)
        candidates.append(
            Proposal(
                bbox=[x1, y1, x2, y2],
                point=[int(np.clip(cx, 0, width)), int(np.clip(cy, 0, height))],
                score=float(score),
                area=int(contour_area),
                angle=0.0,
                polygon=polygon,
            )
        )

    padded_candidates = [
        _pad_proposal(
            proposal,
            image_width=width,
            image_height=height,
            padding_ratio=box_padding_ratio,
            min_padding=min_box_padding,
        )
        for proposal in candidates
    ]
    padded_candidates = [
        proposal
        for proposal in padded_candidates
        if _is_prompt_sized_proposal(
            proposal,
            image_width=width,
            image_height=height,
        )
    ]
    selected = _select_diverse_proposals(
        padded_candidates,
        max_proposals=max(1, max_proposals),
        image_width=width,
        image_height=height,
        duplicate_iou_threshold=duplicate_iou_threshold,
        containment_threshold=containment_threshold,
    )
    return selected


def group_proposals(
    proposals: list[Proposal],
    image_width: int,
    image_height: int,
    merge_gap_ratio: float = 0.8,
    max_group_area_ratio: float = 0.12,
    max_group_width_ratio: float = 0.35,
    max_group_height_ratio: float = 0.35,
    group_padding_ratio: float = 0.08,
    min_group_padding: int = 8,
) -> list[ProposalGroup]:
    if not proposals:
        return []

    groups = [
        {
            "bbox": proposal.bbox[:],
            "points": [proposal.point[:]],
            "proposal_ids": [index + 1],
        }
        for index, proposal in enumerate(proposals)
    ]

    changed = True
    while changed:
        changed = False
        best_pair = None
        best_gap = None
        for first_index in range(len(groups)):
            for second_index in range(first_index + 1, len(groups)):
                first = groups[first_index]
                second = groups[second_index]
                gap = _bbox_gap(first["bbox"], second["bbox"])
                if not _should_merge_group_boxes(
                    first["bbox"],
                    second["bbox"],
                    gap=gap,
                    image_width=image_width,
                    image_height=image_height,
                    merge_gap_ratio=merge_gap_ratio,
                    max_group_area_ratio=max_group_area_ratio,
                    max_group_width_ratio=max_group_width_ratio,
                    max_group_height_ratio=max_group_height_ratio,
                ):
                    continue
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_pair = (first_index, second_index)

        if best_pair is None:
            continue

        first_index, second_index = best_pair
        first = groups[first_index]
        second = groups[second_index]
        first["bbox"] = _union_bbox(first["bbox"], second["bbox"])
        first["points"].extend(second["points"])
        first["proposal_ids"].extend(second["proposal_ids"])
        del groups[second_index]
        changed = True

    result = []
    for group in groups:
        padded_bbox = _pad_existing_bbox(
            group["bbox"],
            image_width=image_width,
            image_height=image_height,
            padding_ratio=group_padding_ratio,
            min_padding=min_group_padding,
        )
        proposal_ids = sorted(group["proposal_ids"])
        result.append(
            ProposalGroup(
                bbox=padded_bbox,
                points=group["points"],
                proposal_ids=proposal_ids,
                proposal_count=len(proposal_ids),
            )
        )

    result.sort(key=lambda item: (_bbox_area(item.bbox), item.proposal_count), reverse=True)
    return result


def preprocess_to_png_base64(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"))
    _, closed_mask = _preprocess_masks(rgb)
    return _mask_to_png_base64(closed_mask)


def edges_to_png_base64(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"))
    edge_mask, _ = _preprocess_masks(rgb)
    return _mask_to_png_base64(edge_mask)


def _mask_to_png_base64(mask: np.ndarray) -> str:
    preview = Image.fromarray(mask, mode="L").convert("RGBA")
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def proposals_to_png_base64(
    proposals: list[Proposal],
    width: int,
    height: int,
) -> str:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for index, proposal in enumerate(proposals, start=1):
        polygon = [tuple(point) for point in proposal.polygon]
        draw.polygon(polygon, fill=(37, 99, 235, 34))
        draw.line(polygon + [polygon[0]], fill=(37, 99, 235, 220), width=2)

        x, y = proposal.point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(250, 204, 21, 240))
        draw.text((x + 5, y + 5), str(index), fill=(15, 23, 42, 230))

    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def proposal_groups_to_png_base64(
    groups: list[ProposalGroup],
    width: int,
    height: int,
) -> str:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for index, group in enumerate(groups, start=1):
        x1, y1, x2, y2 = group.bbox
        draw.rectangle((x1, y1, x2, y2), fill=(20, 184, 166, 28), outline=(15, 118, 110, 235), width=3)
        for x, y in group.points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(250, 204, 21, 230))
        draw.text((x1 + 5, y1 + 5), f"{index}:{group.proposal_count}", fill=(15, 118, 110, 255))

    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _nms(
    candidates: list[Proposal],
    max_proposals: int,
    iou_threshold: float,
) -> list[Proposal]:
    selected = []
    for candidate in candidates:
        if all(_bbox_iou(candidate.bbox, item.bbox) < iou_threshold for item in selected):
            selected.append(candidate)
        if len(selected) >= max_proposals:
            break
    return selected


def _select_diverse_proposals(
    candidates: list[Proposal],
    max_proposals: int,
    image_width: int,
    image_height: int,
    duplicate_iou_threshold: float = 0.35,
    containment_threshold: float = 0.60,
    overlap_penalty_weight: float = 0.55,
) -> list[Proposal]:
    remaining = [
        (
            proposal,
            _proposal_rank_score(
                proposal,
                image_width=image_width,
                image_height=image_height,
            ),
        )
        for proposal in candidates
    ]
    selected = []

    while remaining and len(selected) < max_proposals:
        best_index = None
        best_key = None
        for index, (candidate, base_score) in enumerate(remaining):
            max_iou = 0.0
            duplicate = False
            for chosen in selected:
                intersection = _bbox_intersection_area(
                    candidate.bbox,
                    chosen.bbox,
                )
                if intersection <= 0:
                    continue

                candidate_area = max(1, _bbox_area(candidate.bbox))
                chosen_area = max(1, _bbox_area(chosen.bbox))
                candidate_coverage = intersection / candidate_area
                chosen_coverage = intersection / chosen_area
                iou = _bbox_iou(candidate.bbox, chosen.bbox)
                max_iou = max(max_iou, iou)
                if (
                    iou >= duplicate_iou_threshold
                    or candidate_coverage >= containment_threshold
                    or chosen_coverage >= containment_threshold
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            adjusted_score = base_score - overlap_penalty_weight * max_iou
            key = (adjusted_score, base_score, candidate.score, candidate.area)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index

        if best_index is None:
            break

        proposal, _ = remaining.pop(best_index)
        selected.append(proposal)

    return selected


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1, area_a + area_b - intersection)


def _bbox_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _bbox_gap(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(max(dx, dy))


def _union_bbox(a: list[int], b: list[int]) -> list[int]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def _should_merge_group_boxes(
    first: list[int],
    second: list[int],
    gap: float,
    image_width: int,
    image_height: int,
    merge_gap_ratio: float,
    max_group_area_ratio: float,
    max_group_width_ratio: float,
    max_group_height_ratio: float,
) -> bool:
    merged = _union_bbox(first, second)
    merged_width = max(0, merged[2] - merged[0])
    merged_height = max(0, merged[3] - merged[1])
    image_area = max(1, image_width * image_height)
    if merged_width / max(1, image_width) > max_group_width_ratio:
        return False
    if merged_height / max(1, image_height) > max_group_height_ratio:
        return False
    if (merged_width * merged_height) / image_area > max_group_area_ratio:
        return False

    if _bbox_intersection_area(first, second) > 0:
        return True

    first_size = max(1.0, _bbox_diagonal(first))
    second_size = max(1.0, _bbox_diagonal(second))
    return gap <= merge_gap_ratio * ((first_size + second_size) / 2)


def _bbox_diagonal(box: list[int]) -> float:
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return float((width * width + height * height) ** 0.5)


def _padded_axis_box_points(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    padding: int = 2,
) -> np.ndarray:
    x1 = int(np.clip(x - padding, 0, image_width - 1))
    y1 = int(np.clip(y - padding, 0, image_height - 1))
    x2 = int(np.clip(x + width + padding, 0, image_width - 1))
    y2 = int(np.clip(y + height + padding, 0, image_height - 1))
    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.int32,
    )


def _padded_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_padding: int,
) -> tuple[int, int, int, int]:
    padding = max(min_padding, int(round(max(width, height) * padding_ratio)))
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(image_width, int(x + width + padding))
    y2 = min(image_height, int(y + height + padding))
    return x1, y1, x2, y2


def _pad_existing_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_padding: int,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    padding = max(min_padding, int(round(max(width, height) * padding_ratio)))
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding),
    ]


def _pad_proposal(
    proposal: Proposal,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_padding: int,
) -> Proposal:
    padded_bbox = _pad_existing_bbox(
        proposal.bbox,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
        min_padding=min_padding,
    )
    return Proposal(
        bbox=padded_bbox,
        point=proposal.point[:],
        score=proposal.score,
        area=proposal.area,
        angle=proposal.angle,
        polygon=_axis_aligned_polygon_from_bbox(*padded_bbox),
    )


def _axis_aligned_polygon_from_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> list[list[int]]:
    return [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]


def _expand_polygon_from_center(
    polygon: np.ndarray,
    center: tuple[float, float],
    padding_ratio: float,
    min_padding: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = polygon.astype(np.float32)
    center_point = np.array(center, dtype=np.float32)
    vectors = points - center_point
    lengths = np.linalg.norm(vectors, axis=1)
    max_length = float(lengths.max()) if lengths.size else 0.0
    if max_length <= 0:
        return polygon.astype(np.int32)

    padding = max(float(min_padding), max_length * padding_ratio)
    scale = (max_length + padding) / max_length
    expanded = center_point + vectors * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_height - 1)
    return np.round(expanded).astype(np.int32)


def _proposal_sort_key(
    proposal: Proposal,
    image_width: int,
    image_height: int,
) -> tuple[float, float, int]:
    rank_score = _proposal_rank_score(
        proposal,
        image_width=image_width,
        image_height=image_height,
    )
    return rank_score, proposal.score, proposal.area


def _proposal_rank_score(
    proposal: Proposal,
    image_width: int,
    image_height: int,
    center_weight: float = 0.60,
    area_weight: float = 0.30,
    quality_weight: float = 0.10,
    ideal_area_min: float = 0.08,
    ideal_area_max: float = 0.20,
    maximum_area_ratio: float = 0.55,
) -> float:
    x1, y1, x2, y2 = proposal.bbox
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = max(1, image_width * image_height)
    area_ratio = min(1.0, bbox_area / image_area)
    if area_ratio < ideal_area_min:
        area_score = area_ratio / max(ideal_area_min, 1e-9)
    elif area_ratio <= ideal_area_max:
        area_score = 1.0
    else:
        area_score = max(
            0.0,
            1.0
            - (area_ratio - ideal_area_max)
            / max(maximum_area_ratio - ideal_area_max, 1e-9),
        )

    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0
    image_cx = image_width / 2.0
    image_cy = image_height / 2.0
    center_distance = float(
        ((box_cx - image_cx) ** 2 + (box_cy - image_cy) ** 2) ** 0.5
    )
    max_center_distance = max(
        1.0,
        float((image_cx**2 + image_cy**2) ** 0.5),
    )
    normalized_distance = min(1.0, center_distance / max_center_distance)
    center_score = 1.0 - normalized_distance
    size_gated_center_score = center_score * (0.35 + 0.65 * area_score)

    quality_score = float(np.clip(proposal.score, 0.0, 1.0))
    return (
        center_weight * size_gated_center_score
        + area_weight * area_score
        + quality_weight * quality_score
    )


def _is_prompt_sized_proposal(
    proposal: Proposal,
    image_width: int,
    image_height: int,
    minimum_area_ratio: float = 0.005,
    minimum_short_side_ratio: float = 0.05,
    maximum_area_ratio: float = 0.55,
) -> bool:
    x1, y1, x2, y2 = proposal.bbox
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    image_area = max(1, image_width * image_height)
    area_ratio = (box_width * box_height) / image_area
    short_side_ratio = min(box_width, box_height) / max(
        1,
        min(image_width, image_height),
    )
    return (
        minimum_area_ratio <= area_ratio <= maximum_area_ratio
        and short_side_ratio >= minimum_short_side_ratio
    )


def _preprocess_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median + 20))
    edges = cv2.Canny(blurred, lower, max(lower + 1, upper))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return edges, closed
