from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SegmentInput:
    image: Image.Image
    prompt: str
    threshold: float = 0.5
    box: tuple[int, int, int, int] | None = None
    points: list[tuple[int, int, int]] | None = None


@dataclass
class SegmentMask:
    mask: np.ndarray
    score: float
    bbox: list[int] | None = None


class Segmenter(ABC):
    name: str

    def preload(self) -> None:
        return None

    @abstractmethod
    def segment(self, payload: SegmentInput) -> list[SegmentMask]:
        raise NotImplementedError
