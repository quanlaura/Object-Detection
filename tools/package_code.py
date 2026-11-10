"""Đóng gói các file Python của pipeline mới."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "runs" / "ObjectDetectionV1_code.zip"

    relative_paths = [
        "models/__init__.py",
        "models/backbone.py",
        "models/tiny_backbone.py",
        "models/fpn.py",
        "models/head.py",
        "models/detector.py",
        "models/loss.py",
        "utils/__init__.py",
        "utils/dataset.py",
        "utils/box_utils.py",
        "utils/anchors.py",
        "utils/nms.py",
        "utils/metrics.py",
        "train.py",
        "predict.py",
    ]

    for relative in relative_paths:
        if not (project / relative).is_file():
            raise FileNotFoundError(f"Thiếu file: {relative}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for relative in relative_paths:
            archive.write(
                project / relative,
                arcname=relative,
            )

    print("Số file:", len(relative_paths))
    print("ZIP code:", output)
    print("Dung lượng:", round(output.stat().st_size / 1024, 1), "KB")


if __name__ == "__main__":
    main()