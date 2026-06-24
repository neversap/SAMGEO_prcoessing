import os
import sys
import logging
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from server.app.adapters.base import SegmentInput, SegmentMask, Segmenter
from server.app.utils.images import clean_mask_components, quadify_mask


DEFAULT_VISIBLE_DEVICES = "6,7"
DEFAULT_ENTRYPOINT = ""
DEFAULT_INFERENCE_DTYPE = "bfloat16"
logger = logging.getLogger("sam_geo.sam3")


class Sam3Segmenter(Segmenter):
    name = "sam3"

    def __init__(self, model_dir: str, device: str = DEFAULT_VISIBLE_DEVICES) -> None:
        self.model_dir = model_dir
        self.visible_devices = (device or DEFAULT_VISIBLE_DEVICES).strip()
        self.device = self._configure_device(self.visible_devices)
        self.code_path = os.getenv("SAM_GEO_CODE_PATH", "").strip()
        self.entrypoint = os.getenv("SAM_GEO_SAM3_ENTRYPOINT", DEFAULT_ENTRYPOINT).strip()
        self.checkpoint_path = os.getenv("SAM_GEO_CHECKPOINT_PATH", "").strip()
        self.inference_dtype = os.getenv(
            "SAM_GEO_INFERENCE_DTYPE",
            DEFAULT_INFERENCE_DTYPE,
        ).strip().lower()
        self.model = None
        self.torch = None
        self.dtype_hook_handles = []

    def _configure_device(self, device: str) -> str:
        if device.lower() == "cpu":
            return "cpu"
        if "," in device or device.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            return "cuda:0"
        if device == "cuda":
            return "cuda:0"
        return device

    def preload(self) -> None:
        if self.model is None:
            self.model = self._load_model()

    def _load_model(self) -> Any:
        if self.code_path and self.code_path not in sys.path:
            sys.path.insert(0, self.code_path)

        if not self.entrypoint:
            return self._load_official_image_processor()

        factory = self._import_entrypoint(self.entrypoint)
        return factory(model_dir=self.model_dir, device=self.device)

    def _load_official_image_processor(self) -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cannot import torch. The image must install "
                "torch==2.10.0 torchvision from the official cu128 index before "
                "loading SAM3."
            ) from exc

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as exc:
            missing_name = exc.name or "unknown"
            raise RuntimeError(
                "Cannot import official SAM3 image API. Missing module: "
                f"{missing_name}. Make sure sam3-main.zip was copied into the Docker "
                "build context, the image was rebuilt without cache, and SAM3 package "
                "dependencies from pyproject.toml were installed."
            ) from exc

        self.torch = torch
        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                f"SAM3 is configured for {self.visible_devices}, but CUDA is not "
                "available inside the container. Check Docker GPU access and the "
                "PyTorch CUDA build."
            )
        if self.device != "cpu":
            torch.cuda.set_device(self.device)

        checkpoint_path = self._resolve_checkpoint_path()
        with torch.inference_mode():
            model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                device=self.device,
                enable_inst_interactivity=True,
            )
            if self.inference_dtype == "float32" and hasattr(model, "float"):
                model = model.float()
            if self.device != "cpu" and hasattr(model, "to"):
                model = model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            self._install_dtype_alignment_hooks(model)
            return Sam3Processor(model, device=self.device)

    def _resolve_checkpoint_path(self) -> str:
        if self.checkpoint_path:
            path = Path(self.checkpoint_path)
            if not path.exists():
                raise RuntimeError(f"SAM_GEO_CHECKPOINT_PATH does not exist: {path}")
            return str(path)

        model_dir = Path(self.model_dir)
        candidates = [
            model_dir / "sam3.pt",
            model_dir / "sam3.1_multiplex.pt",
        ]
        candidates.extend(sorted(model_dir.glob("*.pt")))
        for path in candidates:
            if path.exists():
                return str(path)

        raise RuntimeError(
            "No local SAM3 checkpoint found. Set SAM_GEO_CHECKPOINT_PATH to the "
            "checkpoint file, or place sam3.pt under SAM_GEO_MODEL_DIR. Remote "
            "Hugging Face download is disabled in this service."
        )

    def segment(self, payload: SegmentInput) -> list[SegmentMask]:
        self.preload()
        if (
            payload.inference_mode == "sam_cascade"
            and self._is_official_image_processor(self.model)
        ):
            return self._run_official_sam_cascade_predictor(payload)
        raw_output = self._run_predictor(payload)
        return self._normalize_output(raw_output)

    def _import_entrypoint(self, value: str):
        if ":" not in value:
            raise RuntimeError(
                "SAM_GEO_SAM3_ENTRYPOINT must be formatted as module:function, "
                f"got {value!r}."
            )
        module_name, function_name = value.split(":", 1)
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cannot import SAM3 adapter module. Set SAM_GEO_CODE_PATH to the "
                "directory containing your server_sam3_adapter.py or install the "
                f"module in the image. Missing module: {module_name}"
            ) from exc

        try:
            factory = getattr(module, function_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"SAM3 adapter module {module_name!r} has no function {function_name!r}."
            ) from exc
        return factory

    def _run_predictor(self, payload: SegmentInput) -> Any:
        image = payload.image
        if self._is_official_image_processor(self.model):
            if payload.box or payload.points:
                raise RuntimeError(
                    "Official SAM3 image processor is currently wired for text "
                    "prompts only. Omit box and points, or configure a custom "
                    "SAM_GEO_SAM3_ENTRYPOINT bridge."
                )
            with self._inference_context():
                inference_state = self.model.set_image(image)
                if hasattr(self.model, "set_confidence_threshold"):
                    self.model.set_confidence_threshold(
                        payload.threshold,
                        state=inference_state,
                    )
                return self.model.set_text_prompt(
                    state=inference_state,
                    prompt=payload.prompt,
                )

        if hasattr(self.model, "predict"):
            return self.model.predict(
                image=image,
                prompt=payload.prompt,
                box=payload.box,
                points=payload.points,
            )
        if callable(self.model):
            return self.model(
                image=image,
                prompt=payload.prompt,
                box=payload.box,
                points=payload.points,
            )
        raise RuntimeError(
            "SAM3 factory must return a callable or an object with predict(image, "
            "prompt, box, points)."
        )

    def _run_official_sam_cascade_predictor(
        self,
        payload: SegmentInput,
    ) -> list[SegmentMask]:
        if not hasattr(self.model, "_forward_grounding"):
            raise RuntimeError(
                "The current SAM3 processor does not expose _forward_grounding; "
                "SAM cascade inference requires the official Meta SAM3 "
                "image processor."
            )

        image = payload.image
        with self._inference_context():
            state = self.model.set_image(image)
            if hasattr(self.model, "set_confidence_threshold"):
                self.model.set_confidence_threshold(0.40, state=state)
            first_output = self.model.set_text_prompt(
                state=state,
                prompt=payload.prompt,
            )
            first_masks = self._postprocess_cascade_masks(
                self._normalize_output(first_output)
            )
            proposals = self._build_cascade_proposals(
                first_masks,
                image_width=image.width,
                image_height=image.height,
                max_proposals=payload.max_proposals,
            )
            payload.proposals = proposals
            logger.info(
                "SAM cascade first pass produced %d masks and selected %d boxes",
                len(first_masks),
                len(proposals),
            )
            if not proposals:
                logger.info("SAM cascade has no valid boxes; returning first-pass masks")
                return first_masks

            boxes = [
                self._xyxy_to_normalized_cxcywh(
                    item["bbox"],
                    image.width,
                    image.height,
                )
                for item in proposals
            ]
            sam_model = getattr(self.model, "model", None)
            if sam_model is None or not hasattr(sam_model, "_get_dummy_prompt"):
                raise RuntimeError(
                    "SAM3 processor does not expose the model prompt reset API "
                    "required for cascade inference."
                )
            state["geometric_prompt"] = sam_model._get_dummy_prompt()
            if "geometric_prompt" not in state:
                raise RuntimeError(
                    "SAM3 did not initialize geometric_prompt for cascade inference."
                )
            self.model.confidence_threshold = payload.threshold

            box_tensor = self.torch.tensor(
                boxes,
                device=self.device,
                dtype=self.torch.float32,
            ).view(len(boxes), 1, 4)
            box_labels = self.torch.ones(
                (len(boxes), 1),
                device=self.device,
                dtype=self.torch.bool,
            )
            box_mask = self.torch.zeros(
                (1, len(boxes)),
                device=self.device,
                dtype=self.torch.bool,
            )
            state["geometric_prompt"].append_boxes(box_tensor, box_labels, box_mask)
            second_output = self.model._forward_grounding(state)
            second_masks = self._postprocess_cascade_masks(
                self._normalize_output(second_output)
            )
            second_masks = self._filter_masks_by_proposals(
                second_masks,
                proposals,
            )
            return self._merge_cascade_masks(first_masks, second_masks)

    def _build_cascade_proposals(
        self,
        masks: list[SegmentMask],
        image_width: int,
        image_height: int,
        max_proposals: int,
        padding_pixels: int = 10,
        mask_iou_threshold: float = 0.70,
        duplicate_iou_threshold: float = 0.75,
        containment_threshold: float = 0.85,
        confidence_weight: float = 0.10,
        area_weight: float = 0.15,
        distance_weight: float = 0.55,
        region_weight: float = 0.20,
        overlap_penalty_weight: float = 0.30,
        minimum_box_area_ratio: float = 0.001,
        minimum_short_side_ratio: float = 0.015,
    ) -> list[dict]:
        image_area = max(1, image_width * image_height)
        candidates = []
        for item in masks:
            cleaned = clean_mask_components(
                item.mask,
                min_area=64,
                connectivity=8,
                fill_holes=True,
                max_hole_area=256,
            )
            quad_mask = quadify_mask(cleaned, mode="rotated", min_area=64)
            if not quad_mask.any():
                continue
            bbox = self._quad_vertices_bbox(quad_mask)
            if bbox is None:
                continue
            area = int(quad_mask.sum())
            area_score = float((area / image_area) ** 0.5)
            candidates.append(
                {
                    "mask": quad_mask,
                    "bbox": bbox,
                    "score": float(item.score),
                    "area": area,
                    "area_score": area_score,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["area_score"],
            ),
            reverse=True,
        )
        mask_selected = []
        for candidate in candidates:
            if any(
                self._mask_iou(candidate["mask"], chosen["mask"])
                >= mask_iou_threshold
                for chosen in mask_selected
            ):
                continue
            mask_selected.append(candidate)

        prepared = []
        for candidate in mask_selected:
            padded_bbox = self._pad_bbox_pixels(
                candidate["bbox"],
                image_width=image_width,
                image_height=image_height,
                padding_pixels=padding_pixels,
            )
            if not self._is_prompt_box_size(
                padded_bbox,
                image_width=image_width,
                image_height=image_height,
                minimum_area_ratio=minimum_box_area_ratio,
                minimum_short_side_ratio=minimum_short_side_ratio,
            ):
                continue
            x1, y1, x2, y2 = padded_bbox
            prepared.append(
                {
                    "bbox": padded_bbox,
                    "point": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "score": candidate["score"],
                    "area": candidate["area"],
                    "angle": 0.0,
                    "polygon": [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                    "area_score": candidate["area_score"],
                    "center_distance": self._normalized_image_center_distance(
                        padded_bbox,
                        image_width=image_width,
                        image_height=image_height,
                    ),
                }
            )

        selected = self._select_spatially_diverse_boxes(
            prepared,
            max_proposals=max_proposals,
            image_width=image_width,
            image_height=image_height,
            duplicate_iou_threshold=duplicate_iou_threshold,
            containment_threshold=containment_threshold,
            confidence_weight=confidence_weight,
            area_weight=area_weight,
            distance_weight=distance_weight,
            region_weight=region_weight,
            overlap_penalty_weight=overlap_penalty_weight,
        )
        prepared_regions = self._proposal_region_counts(prepared)
        selected_regions = self._proposal_region_counts(selected)
        for proposal in selected:
            proposal.pop("area_score", None)
            proposal.pop("center_distance", None)
        logger.info(
            "SAM cascade proposal stages: %d candidates, %d after mask NMS, "
            "%d size-valid boxes (%s), %d selected (%s)",
            len(candidates),
            len(mask_selected),
            len(prepared),
            prepared_regions,
            len(selected),
            selected_regions,
        )
        return selected

    def _filter_masks_by_proposals(
        self,
        masks: list[SegmentMask],
        proposals: list[dict],
        minimum_overlap_ratio: float = 0.30,
    ) -> list[SegmentMask]:
        if not proposals or not masks:
            return masks
        union = np.zeros_like(masks[0].mask, dtype=bool)
        height, width = union.shape
        for proposal in proposals:
            x1, y1, x2, y2 = proposal["bbox"]
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2)))
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2)))
            union[y1:y2, x1:x2] = True

        filtered = []
        for item in masks:
            mask = item.mask.astype(bool)
            area = int(mask.sum())
            if area <= 0:
                continue
            overlap_ratio = int((mask & union).sum()) / area
            if overlap_ratio >= minimum_overlap_ratio:
                filtered.append(item)
        logger.info(
            "SAM cascade second pass kept %d of %d masks after spatial filtering",
            len(filtered),
            len(masks),
        )
        return filtered

    def _postprocess_cascade_masks(
        self,
        masks: list[SegmentMask],
        min_area: int = 64,
    ) -> list[SegmentMask]:
        processed = []
        for item in masks:
            cleaned = clean_mask_components(
                item.mask,
                min_area=min_area,
                connectivity=8,
                fill_holes=True,
                max_hole_area=256,
            )
            quad_mask = quadify_mask(
                cleaned,
                mode="rotated",
                min_area=min_area,
            )
            if not quad_mask.any():
                continue
            processed.append(
                SegmentMask(
                    mask=quad_mask,
                    score=item.score,
                    bbox=self._mask_bbox(quad_mask),
                )
            )
        return processed

    def _merge_cascade_masks(
        self,
        first_masks: list[SegmentMask],
        second_masks: list[SegmentMask],
        cross_iou_threshold: float = 0.45,
        cross_smaller_coverage_threshold: float = 0.75,
        second_coverage_threshold: float = 0.65,
        second_iou_threshold: float = 0.60,
        second_smaller_coverage_threshold: float = 0.80,
    ) -> list[SegmentMask]:
        deduplicated_second = []
        for candidate in sorted(
            second_masks,
            key=lambda item: item.score,
            reverse=True,
        ):
            if any(
                self._masks_are_duplicate(
                    candidate.mask,
                    chosen.mask,
                    iou_threshold=second_iou_threshold,
                    smaller_coverage_threshold=(
                        second_smaller_coverage_threshold
                    ),
                )
                for chosen in deduplicated_second
            ):
                continue
            deduplicated_second.append(candidate)

        supplements = []
        for candidate in deduplicated_second:
            duplicate_first = False
            for first in first_masks:
                metrics = self._mask_overlap_metrics(
                    first.mask,
                    candidate.mask,
                )
                if (
                    metrics["iou"] >= cross_iou_threshold
                    or metrics["smaller_coverage"]
                    >= cross_smaller_coverage_threshold
                    or metrics["second_coverage"] >= second_coverage_threshold
                ):
                    duplicate_first = True
                    break
            if not duplicate_first:
                supplements.append(candidate)

        logger.info(
            "SAM cascade fusion kept %d first-pass masks and %d of %d "
            "second-pass masks",
            len(first_masks),
            len(supplements),
            len(second_masks),
        )
        # The renderer draws later masks on top, so keep first-pass masks last.
        return supplements + first_masks

    def _masks_are_duplicate(
        self,
        first: np.ndarray,
        second: np.ndarray,
        iou_threshold: float,
        smaller_coverage_threshold: float,
    ) -> bool:
        metrics = self._mask_overlap_metrics(first, second)
        return (
            metrics["iou"] >= iou_threshold
            or metrics["smaller_coverage"] >= smaller_coverage_threshold
        )

    def _mask_overlap_metrics(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> dict[str, float]:
        first_bool = first.astype(bool)
        second_bool = second.astype(bool)
        intersection = int((first_bool & second_bool).sum())
        first_area = int(first_bool.sum())
        second_area = int(second_bool.sum())
        union = first_area + second_area - intersection
        return {
            "iou": intersection / max(1, union),
            "smaller_coverage": intersection
            / max(1, min(first_area, second_area)),
            "second_coverage": intersection / max(1, second_area),
        }

    def _mask_bbox(self, mask: np.ndarray) -> list[int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return [0, 0, 0, 0]
        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

    def _mask_iou(self, first: np.ndarray, second: np.ndarray) -> float:
        intersection = int((first.astype(bool) & second.astype(bool)).sum())
        if intersection <= 0:
            return 0.0
        union = int((first.astype(bool) | second.astype(bool)).sum())
        return intersection / max(1, union)

    def _bbox_iou(self, first: list[int], second: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        if intersection <= 0:
            return 0.0
        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        return intersection / max(1, first_area + second_area - intersection)

    def _bbox_containment(self, first: list[int], second: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        if intersection <= 0:
            return 0.0
        smaller_area = min(
            max(1, (ax2 - ax1) * (ay2 - ay1)),
            max(1, (bx2 - bx1) * (by2 - by1)),
        )
        return intersection / smaller_area

    def _quad_vertices_bbox(self, mask: np.ndarray) -> list[int] | None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        vertices = [
            contour.reshape(-1, 2)
            for contour in contours
            if contour.shape[0] >= 3
        ]
        if not vertices:
            return None
        points = np.concatenate(vertices, axis=0)
        return [
            int(points[:, 0].min()),
            int(points[:, 1].min()),
            int(points[:, 0].max()),
            int(points[:, 1].max()),
        ]

    def _pad_bbox_pixels(
        self,
        bbox: list[int],
        image_width: int,
        image_height: int,
        padding_pixels: int,
    ) -> list[int]:
        x1, y1, x2, y2 = bbox
        padding = max(0, int(padding_pixels))
        return [
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image_width, x2 + padding),
            min(image_height, y2 + padding),
        ]

    def _select_spatially_diverse_boxes(
        self,
        candidates: list[dict],
        max_proposals: int,
        image_width: int,
        image_height: int,
        duplicate_iou_threshold: float,
        containment_threshold: float,
        confidence_weight: float,
        area_weight: float,
        distance_weight: float,
        region_weight: float,
        overlap_penalty_weight: float,
    ) -> list[dict]:
        if not candidates:
            return []

        remaining = [dict(item) for item in candidates]
        selected = []
        selection_pattern = ("inner", "inner", "outer")
        image_diagonal = max(
            1.0,
            float((image_width**2 + image_height**2) ** 0.5),
        )

        while remaining and len(selected) < max(1, max_proposals):
            remaining = [
                candidate
                for candidate in remaining
                if not any(
                    self._bbox_iou(candidate["bbox"], chosen["bbox"])
                    >= duplicate_iou_threshold
                    or self._bbox_containment(
                        candidate["bbox"],
                        chosen["bbox"],
                    )
                    >= containment_threshold
                    for chosen in selected
                )
            ]
            if not remaining:
                break

            target_region = selection_pattern[len(selected) % len(selection_pattern)]
            eligible_indices = [
                index
                for index, candidate in enumerate(remaining)
                if self._matches_selection_region(candidate, target_region)
            ]
            if not eligible_indices:
                eligible_indices = list(range(len(remaining)))

            best_index = None
            best_key = None
            for index in eligible_indices:
                candidate = remaining[index]
                max_iou = 0.0
                minimum_distance = 1.0
                for chosen in selected:
                    iou = self._bbox_iou(candidate["bbox"], chosen["bbox"])
                    max_iou = max(max_iou, iou)
                    minimum_distance = min(
                        minimum_distance,
                        self._bbox_center_distance(
                            candidate["bbox"],
                            chosen["bbox"],
                        )
                        / image_diagonal,
                    )

                region_score = self._region_match_score(
                    candidate,
                    target_region,
                )
                selection_score = (
                    confidence_weight * float(candidate["score"])
                    + area_weight * float(candidate["area_score"])
                    + distance_weight * minimum_distance
                    + region_weight * region_score
                    - overlap_penalty_weight * max_iou
                )
                key = (
                    selection_score,
                    minimum_distance,
                    candidate["area"],
                    candidate["score"],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index

            if best_index is None:
                break
            selected.append(remaining.pop(best_index))

        return selected

    def _matches_selection_region(
        self,
        candidate: dict,
        target_region: str,
    ) -> bool:
        distance = float(candidate["center_distance"])
        if target_region == "outer":
            return distance > 0.60
        return distance <= 0.60

    def _region_match_score(
        self,
        candidate: dict,
        target_region: str,
    ) -> float:
        distance = float(candidate["center_distance"])
        if target_region == "outer":
            return distance
        return 1.0 - distance

    def _normalized_image_center_distance(
        self,
        bbox: list[int],
        image_width: int,
        image_height: int,
    ) -> float:
        box_x = (bbox[0] + bbox[2]) / 2.0
        box_y = (bbox[1] + bbox[3]) / 2.0
        image_x = image_width / 2.0
        image_y = image_height / 2.0
        distance = float(
            ((box_x - image_x) ** 2 + (box_y - image_y) ** 2) ** 0.5
        )
        maximum_distance = max(
            1.0,
            float((image_x**2 + image_y**2) ** 0.5),
        )
        return min(1.0, distance / maximum_distance)

    def _is_prompt_box_size(
        self,
        bbox: list[int],
        image_width: int,
        image_height: int,
        minimum_area_ratio: float,
        minimum_short_side_ratio: float,
    ) -> bool:
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        area_ratio = (width * height) / max(1, image_width * image_height)
        short_side_ratio = min(width, height) / max(
            1,
            min(image_width, image_height),
        )
        return (
            area_ratio >= minimum_area_ratio
            and short_side_ratio >= minimum_short_side_ratio
        )

    def _proposal_region_counts(self, proposals: list[dict]) -> str:
        counts = {"center": 0, "transition": 0, "outer": 0}
        for proposal in proposals:
            distance = float(proposal["center_distance"])
            if distance <= 0.30:
                counts["center"] += 1
            elif distance <= 0.60:
                counts["transition"] += 1
            else:
                counts["outer"] += 1
        return (
            f"center={counts['center']}, "
            f"transition={counts['transition']}, "
            f"outer={counts['outer']}"
        )

    def _bbox_center_distance(
        self,
        first: list[int],
        second: list[int],
    ) -> float:
        first_x = (first[0] + first[2]) / 2.0
        first_y = (first[1] + first[3]) / 2.0
        second_x = (second[0] + second[2]) / 2.0
        second_y = (second[1] + second[3]) / 2.0
        return float(
            ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5
        )

    def _is_official_image_processor(self, model: Any) -> bool:
        return hasattr(model, "set_image") and hasattr(model, "set_text_prompt")

    def _xyxy_to_normalized_cxcywh(
        self,
        bbox: list[int],
        width: int,
        height: int,
    ) -> list[float]:
        x1, y1, x2, y2 = bbox
        box_width = max(1.0, float(x2 - x1))
        box_height = max(1.0, float(y2 - y1))
        cx = float(x1) + box_width / 2.0
        cy = float(y1) + box_height / 2.0
        return [
            cx / max(1.0, float(width)),
            cy / max(1.0, float(height)),
            box_width / max(1.0, float(width)),
            box_height / max(1.0, float(height)),
        ]

    @contextmanager
    def _inference_context(self):
        if self.torch is None:
            with nullcontext():
                yield
            return
        if self.device == "cpu":
            with self.torch.inference_mode():
                yield
            return
        autocast_dtype = self._torch_dtype()
        if autocast_dtype is None:
            with (
                self.torch.inference_mode(),
                self.torch.cuda.device(self.device),
                self.torch.autocast(device_type="cuda", enabled=False),
            ):
                yield
            return
        with (
            self.torch.inference_mode(),
            self.torch.cuda.device(self.device),
            self.torch.autocast(device_type="cuda", dtype=autocast_dtype),
        ):
            yield

    def _torch_dtype(self):
        if self.torch is None:
            return None
        if self.inference_dtype in {"float32", "fp32"}:
            return None
        if self.inference_dtype in {"bfloat16", "bf16"}:
            return self.torch.bfloat16
        if self.inference_dtype in {"float16", "fp16"}:
            return self.torch.float16
        raise RuntimeError(
            "SAM_GEO_INFERENCE_DTYPE must be one of bfloat16, float16, or float32; "
            f"got {self.inference_dtype!r}."
        )

    def _install_dtype_alignment_hooks(self, model: Any) -> None:
        if self.torch is None:
            return
        for handle in self.dtype_hook_handles:
            handle.remove()
        self.dtype_hook_handles = []

        for module in model.modules():
            target = self._module_dtype_target(module)
            if target is None:
                continue
            self.dtype_hook_handles.append(
                module.register_forward_pre_hook(self._make_dtype_alignment_hook())
            )

    def _module_dtype_target(self, module: Any):
        for attr_name in ("weight", "in_proj_weight"):
            weight = getattr(module, attr_name, None)
            if (
                weight is not None
                and hasattr(weight, "dtype")
                and hasattr(weight, "device")
                and getattr(weight, "is_floating_point", lambda: False)()
            ):
                return weight
        return None

    def _make_dtype_alignment_hook(self):
        def hook(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            target = self._module_dtype_target(module)
            if target is None:
                return inputs
            return tuple(self._align_tensor_tree(item, target) for item in inputs)

        return hook

    def _align_tensor_tree(self, value: Any, target: Any) -> Any:
        if self.torch is not None and isinstance(value, self.torch.Tensor):
            if not value.is_floating_point():
                return value
            if value.dtype == target.dtype and value.device == target.device:
                return value
            return value.to(device=target.device, dtype=target.dtype)
        if isinstance(value, tuple):
            return tuple(self._align_tensor_tree(item, target) for item in value)
        if isinstance(value, list):
            return [self._align_tensor_tree(item, target) for item in value]
        if isinstance(value, dict):
            return {
                key: self._align_tensor_tree(item, target)
                for key, item in value.items()
            }
        return value

    def _normalize_output(self, raw_output: Any) -> list[SegmentMask]:
        if isinstance(raw_output, dict):
            masks = raw_output.get("masks")
            if masks is None:
                masks = raw_output.get("mask")
            scores = raw_output.get("scores")
            if scores is None:
                scores = raw_output.get("score")
            boxes = raw_output.get("boxes")
            if boxes is None:
                boxes = raw_output.get("box")
        else:
            masks = raw_output
            scores = None
            boxes = None

        if masks is None:
            raise RuntimeError("SAM3 predictor output must contain masks.")

        mask_list = self._as_mask_list(masks)
        score_list = self._as_score_list(scores, len(mask_list))
        box_list = self._as_box_list(boxes, len(mask_list))
        return [
            SegmentMask(
                mask=self._to_bool_mask(mask),
                score=float(score),
                bbox=box,
            )
            for mask, score, box in zip(mask_list, score_list, box_list, strict=True)
        ]

    def _as_mask_list(self, masks: Any) -> list[Any]:
        if isinstance(masks, Image.Image):
            return [masks]
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                return [masks]
            return [masks[index] for index in range(masks.shape[0])]
        if hasattr(masks, "detach"):
            array = self._tensor_to_numpy(masks)
            return self._as_mask_list(array)
        if isinstance(masks, list | tuple):
            return list(masks)
        raise RuntimeError(f"Unsupported SAM3 mask output type: {type(masks)!r}")

    def _as_score_list(self, scores: Any, count: int) -> list[float]:
        if scores is None:
            return [1.0] * count
        if isinstance(scores, int | float):
            return [float(scores)] * count
        if hasattr(scores, "detach"):
            scores = self._tensor_to_numpy(scores)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        score_list = [float(score) for score in scores]
        if len(score_list) != count:
            raise RuntimeError(
                f"SAM3 returned {count} masks but {len(score_list)} scores."
            )
        return score_list

    def _as_box_list(self, boxes: Any, count: int) -> list[list[int] | None]:
        if boxes is None:
            return [None] * count
        if hasattr(boxes, "detach"):
            boxes = self._tensor_to_numpy(boxes)
        if isinstance(boxes, np.ndarray):
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            boxes = boxes.tolist()
        if len(boxes) != count:
            raise RuntimeError(f"SAM3 returned {count} masks but {len(boxes)} boxes.")
        return [
            [int(round(float(value))) for value in box[:4]]
            for box in boxes
        ]

    def _to_bool_mask(self, mask: Any) -> np.ndarray:
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        elif hasattr(mask, "detach"):
            mask = self._tensor_to_numpy(mask)
        else:
            mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise RuntimeError(f"SAM3 mask must be 2D after squeeze, got {mask.shape}.")
        return mask > 0

    def _tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        if self.torch is not None and isinstance(tensor, self.torch.Tensor):
            tensor = tensor.detach()
            if tensor.is_floating_point():
                tensor = tensor.float()
            return tensor.cpu().numpy()
        return np.asarray(tensor)
