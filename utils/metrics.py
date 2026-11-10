"""AP và mAP theo quy tắc matching của evaluator trong đề."""
from collections import defaultdict


def _iou(first, second):
    intersection_width = max(
        0.0, min(first[2], second[2]) - max(first[0], second[0])
    )
    intersection_height = max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _ap(recalls, precisions):
    if not recalls:
        return 0.0
    recall_points = [0.0, *recalls, 1.0]
    precision_envelope = [0.0, *precisions, 0.0]
    for index in reversed(range(len(precision_envelope) - 1)):
        precision_envelope[index] = max(
            precision_envelope[index], precision_envelope[index + 1]
        )
    return sum(
        (recall_points[index] - recall_points[index - 1])
        * precision_envelope[index]
        for index in range(1, len(recall_points))
        if recall_points[index] != recall_points[index - 1]
    )


def evaluate_map(predictions, gt_records, classes, iou_threshold=0.5):
    truth = {name: defaultdict(list) for name in classes}
    totals = {name: 0 for name in classes}
    detections = {name: [] for name in classes}

    for record in gt_records:
        boxes = record["boxes"]
        labels = record["labels"]
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        for box, label in zip(boxes, labels):
            name = classes[label]
            truth[name][record["id"]].append(box)
            totals[name] += 1

    for entry in predictions:
        for detection in entry["boxes"]:
            detections[detection["class"]].append(
                (entry["image_id"], detection["confidence"], detection["bbox"])
            )

    average_precisions = {}
    for name in classes:
        if totals[name] == 0:
            continue
        ordered = sorted(detections[name], key=lambda item: item[1], reverse=True)
        matched = set()
        true_count = 0
        false_count = 0
        recalls, precisions = [], []

        for image_id, _, predicted_box in ordered:
            best_overlap = 0.0
            best_index = -1
            for index, target_box in enumerate(truth[name].get(image_id, [])):
                if (image_id, index) in matched:
                    continue
                overlap = _iou(predicted_box, target_box)
                if overlap > best_overlap:
                    best_overlap, best_index = overlap, index
            if best_index >= 0 and best_overlap >= iou_threshold:
                matched.add((image_id, best_index))
                true_count += 1
            else:
                false_count += 1
            recalls.append(true_count / totals[name])
            precisions.append(true_count / (true_count + false_count))

        average_precisions[name] = _ap(recalls, precisions)

    mean_ap = (
        sum(average_precisions.values()) / len(average_precisions)
        if average_precisions else 0.0
    )
    return mean_ap, average_precisions
