import numpy as np

from server.app.adapters.base import SegmentInput, SegmentMask, Segmenter


class MockSegmenter(Segmenter):
    name = "mock"

    def segment(self, payload: SegmentInput) -> list[SegmentMask]:
        width, height = payload.image.size
        mask = np.zeros((height, width), dtype=bool)

        if payload.box:
            x1, y1, x2, y2 = payload.box
            x1, x2 = sorted((max(0, x1), min(width, x2)))
            y1, y2 = sorted((max(0, y1), min(height, y2)))
            mask[y1:y2, x1:x2] = True
        elif payload.points:
            yy, xx = np.ogrid[:height, :width]
            radius = max(8, min(width, height) // 12)
            for x, y, label in payload.points:
                disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
                if label > 0:
                    mask |= disk
                else:
                    mask &= ~disk
        else:
            x1, x2 = width // 4, width * 3 // 4
            y1, y2 = height // 4, height * 3 // 4
            mask[y1:y2, x1:x2] = True

        return [SegmentMask(mask=mask, score=1.0)]

