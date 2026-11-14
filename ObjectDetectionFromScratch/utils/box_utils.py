"""Các phép tính trên hộp [xmin, ymin, xmax, ymax] theo pixel."""

import torch
from torch import Tensor


BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def _geometry_tensor(boxes: Tensor) -> Tensor:
    # Diện tích hộp 512×512 đã vượt giới hạn biểu diễn của float16.
    if not boxes.is_floating_point() or boxes.dtype in (torch.float16, torch.bfloat16):
        return boxes.float()
    return boxes


def box_area(boxes: Tensor) -> Tensor:
    boxes = _geometry_tensor(boxes)
    return (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)


def _overlap(boxes_a: Tensor, boxes_b: Tensor) -> tuple[Tensor, Tensor]:
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    union = box_area(boxes_a)[:, None] + box_area(boxes_b)[None, :] - intersection
    return intersection, union


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Tính ma trận IoU (N, M) cho mọi cặp hộp."""
    intersection, union = _overlap(_geometry_tensor(boxes_a), _geometry_tensor(boxes_b))
    return intersection / union.clamp_min(1e-7)


def box_giou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Tính IoU có phạt phần trống trong hình chữ nhật bao cả hai hộp."""
    a, b = _geometry_tensor(boxes_a), _geometry_tensor(boxes_b)
    intersection, union = _overlap(a, b)
    outer_start = torch.minimum(a[:, None, :2], b[None, :, :2])
    outer_end = torch.maximum(a[:, None, 2:], b[None, :, 2:])
    outer_area = (outer_end - outer_start).clamp_min(0).prod(dim=-1)
    penalty = (outer_area - union) / outer_area.clamp_min(1e-7)
    return intersection / union.clamp_min(1e-7) - penalty


def _center_and_size(boxes: Tensor) -> tuple[Tensor, Tensor]:
    boxes = _geometry_tensor(boxes)
    size = boxes[:, 2:] - boxes[:, :2]
    return boxes[:, :2] + size / 2, size


def encode_boxes(anchors: Tensor, gt_boxes: Tensor, weights=BOX_WEIGHTS) -> Tensor:
    """Mã hóa mỗi hộp thật so với anchor cùng hàng."""
    anchor_center, anchor_size = _center_and_size(anchors)
    box_center, box_size = _center_and_size(gt_boxes)
    safe_size = anchor_size.clamp_min(1e-6)
    center_delta = (box_center - anchor_center) / safe_size
    size_delta = (box_size.clamp_min(1e-6) / safe_size).log()
    deltas = torch.cat((center_delta, size_delta), dim=1)
    return deltas * deltas.new_tensor(weights)


def decode_boxes(anchors: Tensor, deltas: Tensor, weights=BOX_WEIGHTS) -> Tensor:
    """Áp dụng độ lệch dự đoán để lấy tọa độ hộp."""
    anchor_center, anchor_size = _center_and_size(anchors)
    deltas = _geometry_tensor(deltas)
    offsets = deltas / deltas.new_tensor(weights)
    center = anchor_center + offsets[:, :2] * anchor_size
    # Giới hạn độ phóng đại trước exp, theo cấu hình bài mẫu.
    half_size = anchor_size * offsets[:, 2:].clamp_max(4.135).exp() / 2
    return torch.cat((center - half_size, center + half_size), dim=1)


def clip_boxes(boxes: Tensor, width: float, height: float) -> Tensor:
    """Giới hạn tọa độ theo biên ảnh và trả tensor mới."""
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    return torch.stack(
        (x1.clamp(0, width), y1.clamp(0, height), x2.clamp(0, width), y2.clamp(0, height)),
        dim=1,
    )
