"""AP nội suy và mAP dùng để chọn checkpoint trên validation."""

from collections import defaultdict


def _iou(first, second):
    overlap_width = min(first[2], second[2]) - max(first[0], second[0])
    overlap_height = min(first[3], second[3]) - max(first[1], second[1])
    intersection = max(0.0, overlap_width) * max(0.0, overlap_height)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(true_positive_ranks, ground_truth_count):
    """Tích phân precision nội suy tại những hạng làm tăng recall."""
    precision_envelope = 0.0
    precision_sum = 0.0
    for count in range(len(true_positive_ranks), 0, -1):
        rank = true_positive_ranks[count - 1]
        precision_envelope = max(precision_envelope, count / rank)
        precision_sum += precision_envelope
    return precision_sum / ground_truth_count


def evaluate_map(predictions, gt_records, classes, iou_threshold=0.5):
    """Trả (mAP, AP theo lớp); chỉ lấy trung bình các lớp có hộp thật.

    Dự đoán dùng cấu trúc JSON của predict.py. Hộp thật lấy từ records
    của Dataset. Cả hai dùng tọa độ ảnh gốc.
    """
    truth_by_class = {name: defaultdict(list) for name in classes}
    for record in gt_records:
        boxes, labels = record["boxes"], record["labels"]
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        for box, label in zip(boxes, labels, strict=True):
            truth_by_class[classes[label]][record["id"]].append(box)

    predictions_by_class = {name: [] for name in classes}
    for entry in predictions:
        for box in entry["boxes"]:
            predictions_by_class[box["class"]].append(
                (box["confidence"], entry["image_id"], box["bbox"])
            )

    per_class = {}
    for name, truth_by_image in truth_by_class.items():
        truth_count = sum(len(boxes) for boxes in truth_by_image.values())
        if not truth_count:
            continue

        # Danh sách này được tạo lại mỗi lần chấm, nên pop không sửa nhãn gốc.
        ranked = sorted(predictions_by_class[name], key=lambda item: item[0], reverse=True)
        true_positive_ranks = []
        for rank, (_, image_id, box) in enumerate(ranked, start=1):
            candidates = truth_by_image.get(image_id, [])
            best_index, best_overlap = -1, 0.0
            for index, candidate in enumerate(candidates):
                overlap = _iou(box, candidate)
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index >= 0 and best_overlap >= iou_threshold:
                candidates.pop(best_index)
                true_positive_ranks.append(rank)

        per_class[name] = _average_precision(true_positive_ranks, truth_count)

    mean_ap = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return mean_ap, per_class
