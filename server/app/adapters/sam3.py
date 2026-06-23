import os
import sys
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from server.app.adapters.base import SegmentInput, SegmentMask, Segmenter


DEFAULT_VISIBLE_DEVICES = "6,7"
DEFAULT_ENTRYPOINT = ""
DEFAULT_INFERENCE_DTYPE = "bfloat16"


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

    def _is_official_image_processor(self, model: Any) -> bool:
        return hasattr(model, "set_image") and hasattr(model, "set_text_prompt")

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
