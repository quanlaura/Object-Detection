"""Đối chiếu AP và mAP của dự án với bộ chấm đi kèm dữ liệu."""

import json
import runpy
from pathlib import Path

from utils.dataset import load_annotations
from utils.metrics import evaluate_map


def main():
    public = Path(__file__).resolve().parents[1] / "public"
    annotation_file = public / "annotations/val.json"
    with annotation_file.open(encoding="utf-8") as file:
        ground_truth = json.load(file)
    classes, records = load_annotations(annotation_file)
    grader = runpy.run_path(str(public / "tools/evaluate_predictions.py"))
    # Dự đoán hoàn hảo từ hộp thật, chỉ dùng để kiểm tra phép chấm.
    perfect = [
        {"image_id": record["id"], "boxes": [
            {"class": classes[label], "confidence": 1.0, "bbox": list(box)}
            for box, label in zip(record["boxes"], record["labels"], strict=True)
        ]}
        for record in records
    ]
    # Thêm hộp sai điểm cao và hộp trùng, đồng thời bỏ bớt hộp thật.
    noisy = [
        {"image_id": entry["image_id"], "boxes": [
            {"class": classes[index % len(classes)], "confidence": 0.99, "bbox": [0, 0, 1, 1]},
            *[
                {**box, "confidence": 0.8 - 0.1 * (index % 3)}
                for box in entry["boxes"][1:]
            ],
            *[{**box, "confidence": 0.2} for box in entry["boxes"][1:2]],
        ]}
        for index, entry in enumerate(perfect)
    ]
    images_by_id = {image["id"]: image for image in ground_truth["images"]}
    for label, predictions in (("Perfect", perfect), ("With errors", noisy)):
        mean_ap, per_class = evaluate_map(predictions, records, classes)
        normalized = grader["normalize_predictions"](predictions, classes, images_by_id, 100, True)
        expected = grader["evaluate"](ground_truth, normalized, classes, 0.5)
        assert round(mean_ap, 6) == expected["mAP@0.5"]
        for name, ap in per_class.items():
            assert round(ap, 6) == expected["per_class"][name]["ap"]
        if label == "Perfect":
            assert mean_ap == 1.0
        else:
            assert 0 < mean_ap < 1
        print(f"{label}: mAP@0.5={mean_ap:.6f}; matches grader")
    print("Hoan thanh buoc 4.3.")


if __name__ == "__main__":
    main()
