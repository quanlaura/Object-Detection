
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.box_utils import box_iou, encode_boxes, decode_boxes


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.5,
                       smoothing=0.01, class_weights=None):
    if smoothing > 0:
        targets = targets * (1 - smoothing) + smoothing * 0.5
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    correct_probability = (
        probabilities * targets + (1 - probabilities) * (1 - targets)
    )
    balancing = alpha * targets + (1 - alpha) * (1 - targets)
    loss = balancing * (1 - correct_probability).pow(gamma) * cross_entropy
    if class_weights is not None:
        weights = class_weights.to(logits.device)
        sample_weights = 1 + (targets * weights).sum(dim=1)
        loss = loss * sample_weights[:, None]
    return loss.sum()


def _matched_geometry(pred_boxes, gt_boxes):
    """Các đại lượng của từng cặp box, tính bằng float32."""
    predicted = pred_boxes.float()
    expected = gt_boxes.float()
    predicted_size = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0)
    expected_size = (expected[:, 2:] - expected[:, :2]).clamp(min=0)
    intersection_start = torch.maximum(predicted[:, :2], expected[:, :2])
    intersection_end = torch.minimum(predicted[:, 2:], expected[:, 2:])
    intersection = (
        intersection_end - intersection_start
    ).clamp(min=0).prod(dim=1)
    union = predicted_size.prod(dim=1) + expected_size.prod(dim=1) - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing_start = torch.minimum(predicted[:, :2], expected[:, :2])
    enclosing_end = torch.maximum(predicted[:, 2:], expected[:, 2:])
    return predicted, expected, iou, union, enclosing_end - enclosing_start


def giou_loss(pred_boxes, gt_boxes):
    _, _, iou, union, enclosing_size = _matched_geometry(pred_boxes, gt_boxes)
    enclosing_area = enclosing_size.clamp(min=0).prod(dim=1)
    penalty = (enclosing_area - union) / enclosing_area.clamp(min=1e-7)
    return (1 - iou + penalty).sum()


def ciou_loss(pred_boxes, gt_boxes):
    predicted, expected, iou, _, enclosing_size = _matched_geometry(
        pred_boxes, gt_boxes
    )
    diagonal_squared = enclosing_size.square().sum(dim=1).clamp(min=1e-7)
    predicted_center = (predicted[:, :2] + predicted[:, 2:]) * 0.5
    expected_center = (expected[:, :2] + expected[:, 2:]) * 0.5
    distance_squared = (
        predicted_center - expected_center
    ).square().sum(dim=1)

    predicted_size = (predicted[:, 2:] - predicted[:, :2]).clamp(min=1e-6)
    expected_size = (expected[:, 2:] - expected[:, :2]).clamp(min=1e-6)
    angle_difference = (
        torch.atan(expected_size[:, 0] / expected_size[:, 1])
        - torch.atan(predicted_size[:, 0] / predicted_size[:, 1])
    )
    aspect_penalty = (4 / math.pi ** 2) * angle_difference.square()
    with torch.no_grad():
        aspect_weight = aspect_penalty / (
            1 - iou + aspect_penalty
        ).clamp(min=1e-7)
    return (
        1 - iou
        + distance_squared / diagonal_squared
        + aspect_weight * aspect_penalty
    ).sum()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, fg_iou_thresh=0.5, bg_iou_thresh=0.40,
                 alpha=0.25, gamma=2.5, reg_weight=2.0,
                 reg_loss_type="giou", smoothing=0.01, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.reg_loss_type = reg_loss_type
        self.smoothing = smoothing
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer(
                "class_weights", torch.as_tensor(class_weights, dtype=torch.float32)
            )

    def _match(self, anchors, gt_boxes):
        states = torch.zeros(
            len(anchors), dtype=torch.long, device=anchors.device
        )
        if gt_boxes.numel() == 0:
            return states, torch.zeros_like(states)

        overlaps = box_iou(anchors, gt_boxes)
        best_overlap, assignments = overlaps.max(dim=1)
        states.fill_(-1)
        states[best_overlap < self.bg_iou_thresh] = 0
        states[best_overlap >= self.fg_iou_thresh] = 1

        # Giữ forced matching của mẫu, kể cả khi nhiều GT chọn cùng anchor.
        preferred_anchors = overlaps.argmax(dim=0)
        states[preferred_anchors] = 1
        assignments[preferred_anchors] = torch.arange(
            len(gt_boxes), device=anchors.device
        )
        return states, assignments

    def forward(self, cls_logits, reg_preds, anchors, targets):
        classification_sum = torch.zeros((), device=cls_logits.device)
        regression_sum = torch.zeros((), device=cls_logits.device)
        positive_count = 0

        for index, target in enumerate(targets):
            gt_boxes = target["boxes"].to(cls_logits.device)
            gt_labels = target["labels"].to(cls_logits.device)
            states, assignments = self._match(anchors, gt_boxes)
            active = states >= 0
            positive = states == 1
            count = int(positive.sum().item())
            positive_count += count

            desired_classes = torch.zeros_like(cls_logits[index])
            if count:
                desired_classes[
                    positive, gt_labels[assignments[positive]]
                ] = 1

            classification_sum = classification_sum + sigmoid_focal_loss(
                cls_logits[index][active],
                desired_classes[active],
                alpha=self.alpha,
                gamma=self.gamma,
                smoothing=self.smoothing,
                class_weights=self.class_weights,
            )

            if not count:
                continue

            expected_boxes = gt_boxes[assignments[positive]]
            deltas = reg_preds[index][positive]

            if self.reg_loss_type in ("giou", "ciou"):
                decoded = decode_boxes(anchors[positive], deltas)
                regression_function = (
                    ciou_loss if self.reg_loss_type == "ciou" else giou_loss
                )
                regression_term = regression_function(decoded, expected_boxes)
            else:
                desired_deltas = encode_boxes(
                    anchors[positive], expected_boxes
                ).clamp(-10, 10)
                regression_term = F.smooth_l1_loss(
                    deltas, desired_deltas, beta=1 / 9, reduction="sum"
                )

            # Giữ cách bỏ qua regression loss không hữu hạn của mẫu.
            if torch.isfinite(regression_term).item():
                regression_sum = regression_sum + regression_term

        denominator = max(positive_count, 1)
        return (
            classification_sum / denominator,
            self.reg_weight * regression_sum / denominator,
        )
