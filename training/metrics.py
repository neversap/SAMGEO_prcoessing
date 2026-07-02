from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, logits, target) -> None:
        pred = torch.argmax(logits.detach(), dim=1).cpu()
        truth = target.detach().cpu()
        valid = truth != self.ignore_index
        pred = pred[valid].view(-1)
        truth = truth[valid].view(-1)
        if truth.numel() == 0:
            return
        encoded = truth * self.num_classes + pred
        counts = torch.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).double()

    def compute(self) -> dict[str, float]:
        tp = torch.diag(self.confusion)
        gt = self.confusion.sum(dim=1)
        pred = self.confusion.sum(dim=0)
        union = gt + pred - tp
        iou = tp / torch.clamp(union, min=1.0)
        precision = tp / torch.clamp(pred, min=1.0)
        recall = tp / torch.clamp(gt, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        total = torch.clamp(self.confusion.sum(), min=1.0)
        result = {
            "miou": float(iou.mean().item()),
            "macro_f1": float(f1.mean().item()),
            "pixel_accuracy": float(tp.sum().item() / total.item()),
            "boundary_iou": float(iou[2].item()) if self.num_classes > 2 else 0.0,
            "boundary_f1": float(f1[2].item()) if self.num_classes > 2 else 0.0,
        }
        for class_id in range(self.num_classes):
            result[f"class_{class_id}_iou"] = float(iou[class_id].item())
            result[f"class_{class_id}_f1"] = float(f1[class_id].item())
        return result
