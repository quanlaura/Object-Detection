"""Các phép tính hình học cho box [xmin, ymin, xmax, ymax].
"""

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Tính diện tích, trả về tensor [N]."""
    dimensions = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return dimensions.prod(dim=1)


def box_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Tính IoU của mọi cặp box, trả về [N, M]."""
    intersection_start = torch.maximum(
        boxes_a[:, None, :2],
        boxes_b[None, :, :2],
    )
    intersection_end = torch.minimum(
        boxes_a[:, None, 2:],
        boxes_b[None, :, 2:],
    )

    intersection = (
        intersection_end - intersection_start
    ).clamp(min=0).prod(dim=-1)

    union = (
        box_area(boxes_a)[:, None]
        + box_area(boxes_b)[None, :]
        - intersection
    )

    return intersection / union.clamp(min=1e-7)


def encode_boxes(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    weights=(10.0, 10.0, 5.0, 5.0),
) -> torch.Tensor:
    """Mã hóa từng box thật so với anchor tương ứng.

    Hai đầu vào có dạng [N, 4].
    Đầu ra là [dx, dy, dw, dh] có nhân weights.
    """
    anchor_size = anchors[:, 2:] - anchors[:, :2]
    anchor_center = anchors[:, :2] + anchor_size * 0.5

    target_size = gt_boxes[:, 2:] - gt_boxes[:, :2]
    target_center = gt_boxes[:, :2] + target_size * 0.5

    center_offset = (
        target_center - anchor_center
    ) / anchor_size.clamp(min=1e-6)

    size_offset = torch.log(
        target_size.clamp(min=1e-6)
        / anchor_size.clamp(min=1e-6)
    )

    factors = anchors.new_tensor(weights)

    return torch.cat(
        (center_offset, size_offset), dim=1
    ) * factors


def decode_boxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    weights=(10.0, 10.0, 5.0, 5.0),
) -> torch.Tensor:
    """Giải mã độ lệch thành box [xmin, ymin, xmax, ymax]."""
    anchor_size = anchors[:, 2:] - anchors[:, :2]
    anchor_center = anchors[:, :2] + anchor_size * 0.5

    factors = deltas.new_tensor(weights)
    offsets = deltas / factors

    predicted_center = (
        anchor_center + offsets[:, :2] * anchor_size
    )

    # Giữ đúng giới hạn của bài mẫu trước phép exp.
    predicted_size = (
        offsets[:, 2:].clamp(max=4.135).exp()
        * anchor_size
    )

    return torch.cat(
        (
            predicted_center - predicted_size * 0.5,
            predicted_center + predicted_size * 0.5,
        ),
        dim=1,
    )


def box_giou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Tính GIoU của mọi cặp box, trả về [N, M].

    Tính diện tích bằng float32 rồi trả về dtype của boxes_a.
    """
    original_dtype = boxes_a.dtype
    first = boxes_a.float()
    second = boxes_b.float()

    intersection_start = torch.maximum(
        first[:, None, :2],
        second[None, :, :2],
    )
    intersection_end = torch.minimum(
        first[:, None, 2:],
        second[None, :, 2:],
    )

    intersection = (
        intersection_end - intersection_start
    ).clamp(min=0).prod(dim=-1)

    union = (
        box_area(first)[:, None]
        + box_area(second)[None, :]
        - intersection
    )

    iou = intersection / union.clamp(min=1e-7)

    enclosing_start = torch.minimum(
        first[:, None, :2],
        second[None, :, :2],
    )
    enclosing_end = torch.maximum(
        first[:, None, 2:],
        second[None, :, 2:],
    )

    enclosing_area = (
        enclosing_end - enclosing_start
    ).clamp(min=0).prod(dim=-1)

    penalty = (
        enclosing_area - union
    ) / enclosing_area.clamp(min=1e-7)

    return (iou - penalty).to(original_dtype)


def clip_boxes(
    boxes: torch.Tensor,
    width: float,
    height: float,
) -> torch.Tensor:
    """Giới hạn box trong biên ảnh, trả về một tensor mới."""
    result = boxes.clone()
    result[:, 0::2].clamp_(min=0, max=width)
    result[:, 1::2].clamp_(min=0, max=height)
    return result