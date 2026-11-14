"""Loại hộp trùng bằng IoU và điểm tin cậy."""

import torch

from .box_utils import _geometry_tensor, box_iou


def nms(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    selected = []
    while order.numel():
        selected.append(order[:1])
        remaining = order[1:]
        if remaining.numel() == 0:
            break
        overlap = box_iou(boxes[order[:1]], boxes[remaining])[0]
        order = remaining[overlap <= iou_threshold]
    return torch.cat(selected) if selected else order


def batched_nms(boxes, scores, labels, iou_threshold):
    """Dịch hộp theo lớp để dùng chung một lượt greedy NMS."""
    if boxes.shape[0] == 0:
        return boxes.new_empty((0,), dtype=torch.long)

    coordinates = _geometry_tensor(boxes)
    # Tách các lớp ngay cả khi đầu vào có tọa độ âm.
    lower_bound = coordinates.min().clamp_max(0)
    class_spacing = coordinates.max() - lower_bound + 1
    translated_boxes = coordinates + labels.to(coordinates)[:, None] * class_spacing
    return nms(translated_boxes, scores, iou_threshold)
