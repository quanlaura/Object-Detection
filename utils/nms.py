"""Khử hộp trùng bằng Greedy NMS tự triển khai."""

import torch

from .box_utils import box_iou


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Trả chỉ số hộp được giữ, theo điểm giảm dần."""
    if boxes.numel() == 0:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=boxes.device,
        )

    remaining = torch.argsort(scores, descending=True)
    selected = []

    while remaining.numel() > 0:
        best_index = remaining[0]
        selected.append(best_index)

        if remaining.numel() == 1:
            break

        candidates = remaining[1:]

        overlaps = box_iou(
            boxes[best_index].unsqueeze(0),
            boxes[candidates],
        )[0]

        # Chỉ loại khi IoU lớn hơn ngưỡng.
        remaining = candidates[overlaps <= iou_threshold]

    return torch.stack(selected)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """NMS theo lớp bằng cách dịch tọa độ các lớp khác nhau."""
    if boxes.numel() == 0:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=boxes.device,
        )

    class_spacing = boxes.max() + 1
    shifts = labels.to(boxes) * class_spacing
    separated_boxes = boxes + shifts[:, None]

    return nms(
        separated_boxes,
        scores,
        iou_threshold,
    )