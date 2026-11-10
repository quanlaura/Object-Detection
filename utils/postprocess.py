"""Chuyển đầu ra detector thành các phát hiện sau NMS."""

import torch

from utils.box_utils import box_iou, decode_boxes


def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS cho một lớp, trả về chỉ số các hộp được giữ."""
    order = scores.argsort(descending=True)
    kept = []

    while order.numel() > 0:
        current = order[0]
        kept.append(current)

        if order.numel() == 1:
            break

        remaining = order[1:]

        overlaps = box_iou(
            boxes[current].unsqueeze(0),
            boxes[remaining],
        )[0]

        order = remaining[overlaps <= iou_threshold]

    if not kept:
        return torch.empty(
            (0,), dtype=torch.long, device=boxes.device
        )

    return torch.stack(kept)


def per_class_nms(boxes, scores, labels, iou_threshold=0.5):
    """Chạy NMS riêng cho từng lớp rồi sắp theo điểm."""
    kept_per_class = []

    for class_id in labels.unique():
        class_indices = torch.where(labels == class_id)[0]

        local_kept = nms(
            boxes[class_indices],
            scores[class_indices],
            iou_threshold=iou_threshold,
        )

        kept_per_class.append(class_indices[local_kept])

    if not kept_per_class:
        return torch.empty(
            (0,), dtype=torch.long, device=boxes.device
        )

    kept = torch.cat(kept_per_class)

    return kept[scores[kept].argsort(descending=True)]


@torch.no_grad()
def postprocess(
    outputs,
    original_sizes,
    input_size,
    confidence_threshold=0.05,
    nms_threshold=0.5,
    pre_nms_topk=1000,
    max_detections=100,
):
    """Xử lý một batch.

    original_sizes: danh sách (chiều cao gốc, chiều rộng gốc).
    input_size: (chiều cao đầu vào model, chiều rộng đầu vào model).
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Ngưỡng confidence phải nằm trong [0, 1].")
    if not 0 <= nms_threshold <= 1:
        raise ValueError("Ngưỡng NMS phải nằm trong [0, 1].")
    if pre_nms_topk < 1 or max_detections < 1:
        raise ValueError("Giới hạn số hộp phải lớn hơn 0.")

    class_logits = outputs["class_logits"]
    box_deltas = outputs["box_deltas"]
    anchors = outputs["anchors"]

    if len(original_sizes) != class_logits.shape[0]:
        raise ValueError("Cần kích thước gốc cho từng ảnh trong batch.")

    input_height, input_width = input_size
    results = []

    for image_index, original_size in enumerate(original_sizes):
        probabilities = class_logits[image_index].sigmoid()

        # Một anchor có thể là ứng viên của nhiều lớp.
        anchor_indices, labels = torch.where(
            probabilities > confidence_threshold
        )
        scores = probabilities[anchor_indices, labels]

        # Giới hạn số ứng viên trước NMS để giảm chi phí.
        if scores.numel() > pre_nms_topk:
            scores, top_indices = scores.topk(pre_nms_topk)
            anchor_indices = anchor_indices[top_indices]
            labels = labels[top_indices]

        boxes = decode_boxes(
            anchors[anchor_indices],
            box_deltas[image_index, anchor_indices],
        )

        # Giới hạn tọa độ trong ảnh đầu vào.
        boxes[:, 0::2].clamp_(0, input_width)
        boxes[:, 1::2].clamp_(0, input_height)

        valid = (
            torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(scores)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

        kept = per_class_nms(
            boxes,
            scores,
            labels,
            iou_threshold=nms_threshold,
        )[:max_detections]

        boxes = boxes[kept].clone()

        # Dataset hiện dùng resize trực tiếp.
        # Chia tỷ lệ riêng cho chiều ngang và chiều dọc.
        original_height, original_width = original_size

        boxes[:, 0::2] *= original_width / input_width
        boxes[:, 1::2] *= original_height / input_height

        results.append({
            "boxes": boxes,
            "scores": scores[kept],
            "labels": labels[kept],
        })

    return results