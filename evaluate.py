"""Chấm JSON dự đoán bằng evaluator chính thức của đề."""

import argparse
import json
import runpy
from pathlib import Path


DATA_DIR = Path(
    r"D:\DATA\Src VSCode\ObjectDetectionV1\final_public\public"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chấm mAP@0.5 bằng evaluator của đề."
    )

    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=DATA_DIR / "annotations" / "val.json",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=DATA_DIR / "tools" / "evaluate_predictions.py",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Chỉ chấm các ảnh xuất hiện trong JSON dự đoán.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for path in (
        args.ground_truth,
        args.evaluator,
        args.predictions,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if args.output.resolve() in {
        args.ground_truth.resolve(),
        args.evaluator.resolve(),
        args.predictions.resolve(),
    }:
        raise ValueError("File kết quả phải khác các file đầu vào.")

    # Đọc các hàm của evaluator, không chạy phần main của nó.
    evaluator = runpy.run_path(str(args.evaluator))

    ground_truth = evaluator["load_json"](args.ground_truth)
    prediction_json = evaluator["load_json"](args.predictions)

    # Kiểm tra cấu trúc nhãn thật bằng evaluator chính thức.
    classes, image_info = evaluator["validate_ground_truth"](
        ground_truth
    )

    if args.subset:
        if not isinstance(prediction_json, list):
            raise ValueError("JSON dự đoán phải là một danh sách.")

        image_ids = {
            entry["image_id"] for entry in prediction_json
        }

        if not image_ids:
            raise ValueError("JSON dự đoán không có ảnh.")

        unknown_ids = image_ids - set(image_info)

        if unknown_ids:
            raise ValueError(
                f"Ảnh không có trong ground truth: {unknown_ids}"
            )

        # Chỉ thay dữ liệu trong bộ nhớ, không sửa file nhãn gốc.
        ground_truth = {
            **ground_truth,
            "images": [
                image
                for image in ground_truth["images"]
                if image["id"] in image_ids
            ],
            "annotations": [
                annotation
                for annotation in ground_truth["annotations"]
                if annotation["image_id"] in image_ids
            ],
        }

        classes, image_info = evaluator["validate_ground_truth"](
            ground_truth
        )

    # Kiểm tra tên ảnh, lớp, tọa độ và độ đầy đủ của dự đoán.
    normalized_predictions = evaluator["normalize_predictions"](
        prediction_json,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=100,
        require_complete=True,
    )

    result = evaluator["evaluate"](
        ground_truth=ground_truth,
        predictions=normalized_predictions,
        classes=classes,
        iou_threshold=0.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("Phạm vi:", "Tập con" if args.subset else "Toàn bộ tập")
    print("Số ảnh được chấm:", len(ground_truth["images"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("File điểm:", args.output.resolve())
    print("\nHoàn thành bước 12.")


if __name__ == "__main__":
    main()