"""Kiểm tra môi trường và dữ liệu trước khi viết mô hình."""

import argparse
import json
from pathlib import Path

import PIL
from PIL import Image


CLASSES = ["person", "car", "dog", "cat", "chair"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_split(data_root: Path, split: str) -> None:
    annotation_path = data_root / "annotations" / f"{split}.json"
    with annotation_path.open(encoding="utf-8") as file:
        data = json.load(file)

    if data["classes"] != CLASSES:
        raise ValueError(f"{annotation_path}: thứ tự lớp không đúng yêu cầu.")

    images = data["images"]
    if not images:
        raise ValueError(f"{annotation_path}: danh sách ảnh trống.")

    images_by_id = {image["id"]: image for image in images}
    if len(images_by_id) != len(images):
        raise ValueError(f"{annotation_path}: có ID ảnh trùng.")

    missing = [
        image["file_name"]
        for image in images
        if not (data_root / image["file_name"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{split}: thiếu {len(missing)} ảnh, ví dụ {missing[0]}"
        )

    # Mỗi box phải trỏ đến một ảnh và một lớp có trong bộ dữ liệu.
    annotations = data["annotations"]
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in images_by_id or annotation["class"] not in CLASSES:
            raise ValueError(f"{split}: ảnh hoặc lớp không hợp lệ ở {image_id}.")
        x1, y1, x2, y2 = annotation["bbox"]
        image = images_by_id[image_id]
        if not (0 <= x1 < x2 <= image["width"] and 0 <= y1 < y2 <= image["height"]):
            raise ValueError(f"{split}: hộp bao không hợp lệ ở {image_id}.")

    example = images[0]
    with Image.open(data_root / example["file_name"]) as image:
        image.load()
        if image.size != (example["width"], example["height"]):
            raise ValueError(f"{split}: kích thước ảnh mẫu khác thông tin trong JSON.")

    print(f"{split}: {len(images):,} ảnh, {len(annotations):,} hộp, 0 ảnh thiếu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "public",
        help="Thư mục chứa dữ liệu gốc (mặc định: public trong dự án).",
    )
    parser.add_argument("--data-only", action="store_true", help="Chỉ kiểm tra ảnh và nhãn.")
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    if not (data_root / "tools" / "evaluate_predictions.py").is_file():
        parser.error(
            f"Thiếu công cụ chấm trong {data_root / 'tools'}. "
            "Giải nén final_public.zip vào thư mục dự án hoặc dùng --data-root."
        )

    print(f"Pillow {PIL.__version__}")
    print(f"Dữ liệu: {data_root}")
    print(f"Các lớp: {', '.join(CLASSES)}")
    for split in ("train", "val"):
        check_split(data_root, split)
    if args.data_only:
        print("Hoàn thành kiểm tra dữ liệu.")
        return

    import torch
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.tensor([1.0, 2.0, 3.0], device=device).sum().item()
    print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__}")
    print(f"Thiết bị: {device} | tính thử: {result:g}")
    print("Hoàn thành bước 1.")


if __name__ == "__main__":
    main()

