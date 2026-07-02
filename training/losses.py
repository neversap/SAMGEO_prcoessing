from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        class_weights: list[float],
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        log_cosh_dice_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.log_cosh_dice_weight = log_cosh_dice_weight
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)

    def forward(self, logits, target):
        validate_target_labels(target, self.num_classes, self.ignore_index)
        loss = logits.new_tensor(0.0)
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce(logits, target)
        dice = multiclass_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            class_weights=self.class_weights,
        )
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * dice
        if self.log_cosh_dice_weight > 0:
            loss = loss + self.log_cosh_dice_weight * torch.log(torch.cosh(dice))
        return loss


def build_loss(config: dict) -> CEDiceLoss:
    loss_config = config["loss"]
    if loss_config.get("version") == "log_cosh_dice_v2":
        ce_weight = 0.0
        dice_weight = 0.0
        log_cosh_dice_weight = 1.0
    else:
        ce_weight = float(loss_config.get("ce_weight", 1.0))
        dice_weight = float(loss_config.get("dice_weight", 1.0))
        log_cosh_dice_weight = float(loss_config.get("log_cosh_dice_weight", 0.0))
    return CEDiceLoss(
        num_classes=int(config["classes"]["num_classes"]),
        ignore_index=int(config["classes"]["ignore_index"]),
        class_weights=list(loss_config.get("class_weights", [0.05, 0.25, 0.70])),
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        log_cosh_dice_weight=log_cosh_dice_weight,
    )


def multiclass_dice_loss(logits, target, num_classes: int, ignore_index: int, class_weights):
    probs = F.softmax(logits, dim=1)
    valid = target != ignore_index
    total = logits.new_tensor(0.0)
    weight_sum = logits.new_tensor(0.0)
    for class_id in range(num_classes):
        class_weight = class_weights[class_id]
        pred = probs[:, class_id]
        truth = (target == class_id) & valid
        truth_count = truth.sum()
        valid_count = valid.sum()
        if valid_count == 0:
            continue
        intersection = pred[truth].sum()
        denominator = pred[valid].sum() + truth_count.to(dtype=pred.dtype)
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        total = total + class_weight * (1.0 - dice)
        weight_sum = weight_sum + class_weight
    return total / torch.clamp(weight_sum, min=1e-6)


def validate_target_labels(target, num_classes: int, ignore_index: int) -> None:
    invalid = (target != ignore_index) & ((target < 0) | (target >= num_classes))
    if torch.any(invalid):
        labels = torch.unique(target[invalid]).detach().cpu().tolist()
        raise ValueError(
            "mask contains labels outside the valid training range. "
            f"valid labels are 0..{num_classes - 1} and ignore_index={ignore_index}; "
            f"invalid labels={labels[:20]}"
        )
