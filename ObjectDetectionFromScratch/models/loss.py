"""Gán anchor với hộp thật và tính loss cho hai nhánh RetinaNet."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.box_utils import box_area, box_iou, decode_boxes, encode_boxes


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.5, smoothing=0.01,
                       class_weights=None):
    logits = logits.float()
    targets = targets.float() * (1 - smoothing) + smoothing / 2
    probability = logits.sigmoid()
    correct_probability = targets * probability + (1 - targets) * (1 - probability)
    balance = targets * alpha + (1 - targets) * (1 - alpha)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = balance * (1 - correct_probability).pow(gamma) * bce
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = loss * (1 + targets @ weights)[:, None]
    return loss.sum()


def _matched_geometry(pred, target):
    """Hình học theo từng cặp cùng hàng, kích thước O(số cặp)."""
    overlap_start = torch.maximum(pred[:, :2], target[:, :2])
    overlap_end = torch.minimum(pred[:, 2:], target[:, 2:])
    intersection = (overlap_end - overlap_start).clamp_min(0).prod(dim=1)
    union = box_area(pred) + box_area(target) - intersection
    outer_start = torch.minimum(pred[:, :2], target[:, :2])
    outer_end = torch.maximum(pred[:, 2:], target[:, 2:])
    return intersection, union, outer_end - outer_start


def giou_loss(pred_boxes, gt_boxes):
    intersection, union, outer_size = _matched_geometry(pred_boxes.float(), gt_boxes.float())
    iou = intersection / union.clamp_min(1e-7)
    outer_area = outer_size.clamp_min(0).prod(dim=1)
    return (1 - iou + (outer_area - union) / outer_area.clamp_min(1e-7)).sum()


def ciou_loss(pred_boxes, gt_boxes):
    pred, target = pred_boxes.float(), gt_boxes.float()
    intersection, union, outer_size = _matched_geometry(pred, target)
    iou = intersection / union.clamp_min(1e-7)
    pred_center = (pred[:, :2] + pred[:, 2:]) / 2
    target_center = (target[:, :2] + target[:, 2:]) / 2
    distance = (pred_center - target_center).square().sum(dim=1)
    diagonal = outer_size.square().sum(dim=1).clamp_min(1e-7)
    pw, ph = (pred[:, 2:] - pred[:, :2]).clamp_min(1e-6).unbind(dim=1)
    tw, th = (target[:, 2:] - target[:, :2]).clamp_min(1e-6).unbind(dim=1)
    aspect = (4 / math.pi ** 2) * ((pw / ph).atan() - (tw / th).atan()).square()
    weight = aspect.detach() / (1 - iou.detach() + aspect.detach()).clamp_min(1e-7)
    return (1 - iou + distance / diagonal + weight * aspect).sum()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 alpha=0.25, gamma=2.5, reg_weight=2.0, reg_loss_type="giou",
                 smoothing=0.01, class_weights=None):
        super().__init__()
        if reg_loss_type not in ("giou", "ciou", "smooth_l1"):
            raise ValueError(f"Loss định vị không hỗ trợ: {reg_loss_type}")
        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.reg_loss_type = reg_loss_type
        self.smoothing = smoothing
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weights is not None and weights.shape != (num_classes,):
            raise ValueError("class_weights cần một giá trị cho mỗi lớp.")
        self.register_buffer("class_weights", weights)

    @torch.no_grad()
    def _match(self, anchors, gt_boxes):
        """Trả trạng thái anchor (1, 0, -1) và chỉ số hộp thật đã ghép."""
        if gt_boxes.numel() == 0 or anchors.numel() == 0:
            background = anchors.new_zeros(len(anchors), dtype=torch.long)
            return background, background.clone()

        overlaps = box_iou(anchors, gt_boxes)
        best_iou, matched = overlaps.max(dim=1)
        status = torch.full_like(matched, -1)
        status[best_iou < self.bg_iou_thresh] = 0
        status[best_iou >= self.fg_iou_thresh] = 1

        # Mỗi hộp thật đề cử anchor tốt nhất. Khi trùng anchor,
        # ưu tiên hộp thật ở vị trí sau như thứ tự gán của mẫu trên CPU.
        forced = overlaps.argmax(dim=0)
        gt_ids = torch.arange(len(gt_boxes), device=anchors.device)
        forced_matches = torch.full_like(matched, -1)
        forced_matches.scatter_reduce_(0, forced, gt_ids, reduce="amax")
        forced_positive = forced_matches >= 0
        status[forced_positive] = 1
        matched[forced_positive] = forced_matches[forced_positive]
        return status, matched

    def _regression_loss(self, anchors, deltas, gt_boxes):
        if self.reg_loss_type == "smooth_l1":
            target_deltas = encode_boxes(anchors, gt_boxes).clamp(-10, 10)
            return F.smooth_l1_loss(
                deltas.float(), target_deltas, beta=1 / 9, reduction="sum"
            )
        predicted = decode_boxes(anchors, deltas)
        if self.reg_loss_type == "ciou":
            return ciou_loss(predicted, gt_boxes)
        return giou_loss(predicted, gt_boxes)

    def forward(self, cls_logits, reg_preds, anchors, targets):
        """Trả focal loss và loss định vị đã nhân reg_weight."""
        cls_total = cls_logits.new_zeros((), dtype=torch.float32)
        reg_total = reg_preds.new_zeros((), dtype=torch.float32)
        positive_count = 0
        device = cls_logits.device

        for logits, deltas, target in zip(cls_logits, reg_preds, targets, strict=True):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            status, matched = self._match(anchors, gt_boxes)
            positives = torch.where(status == 1)[0]
            valid = status != -1
            positive_count += len(positives)
            class_targets = torch.zeros_like(logits, dtype=torch.float32)

            if positives.numel():
                gt_ids = matched[positives]
                class_targets[positives, gt_labels[gt_ids]] = 1
                box_loss = self._regression_loss(anchors[positives], deltas[positives], gt_boxes[gt_ids])
                if torch.isfinite(box_loss):
                    reg_total = reg_total + box_loss

            cls_total = cls_total + sigmoid_focal_loss(
                logits[valid], class_targets[valid], self.alpha, self.gamma,
                self.smoothing, self.class_weights,
            )

        normalizer = max(positive_count, 1)
        return cls_total / normalizer, self.reg_weight * reg_total / normalizer
