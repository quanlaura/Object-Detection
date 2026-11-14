"""Lưu ảnh gốc, ảnh resize và ảnh lật ngang để kiểm tra nhãn."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from utils.dataset import DetectionDataset, flip_sample, resize_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#33ddff", "#ffcf44", "#64ff9d", "#ff96df", "#ff8855")


def draw_boxes(image, target, classes):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box, label in zip(target["boxes"], target["labels"]):
        color = COLORS[label % len(COLORS)]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, box[1] + 3), classes[label], fill=color,
                  stroke_width=2, stroke_fill="black")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "public")
    parser.add_argument("--index", type=int, help="Chỉ số ảnh train, bắt đầu từ 0.")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    train = DetectionDataset(args.data_root / "annotations/train.json",
                             args.data_root / "train/images", args.image_size)
    val = DetectionDataset(args.data_root / "annotations/val.json",
                           args.data_root / "val/images", args.image_size)
    index = args.index
    if index is None:
        index = next(
            (i for i, record in enumerate(train.records) if len(record["boxes"]) >= 2),
            0,
        )
    if not 0 <= index < len(train):
        parser.error(f"--index phải nằm trong khoảng 0 đến {len(train) - 1}.")

    original, target = train.read_original(index)
    resized, resized_boxes = resize_sample(original, target["boxes"], args.image_size)
    flipped, flipped_boxes = flip_sample(resized, resized_boxes)
    panels = (
        ("Original", original, target),
        ("Resized", resized, {**target, "boxes": resized_boxes}),
        ("Horizontal flip", flipped, {**target, "boxes": flipped_boxes}),
    )
    panel_width = max(image.width for _, image, _ in panels)
    panel_height = max(image.height for _, image, _ in panels)
    preview = Image.new("RGB", (panel_width * len(panels), panel_height + 28), "#17202b")
    draw = ImageDraw.Draw(preview)
    for column, (title, image, annotations) in enumerate(panels):
        x = column * panel_width
        draw.text((x + 8, 7), title, fill="white")
        preview.paste(draw_boxes(image, annotations, train.classes), (x, 28))

    output_dir = PROJECT_ROOT / "runs"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "dataset_preview.png"
    preview.save(output)
    print(f"Train: {len(train):,} ảnh | Validation: {len(val):,} ảnh")
    print(f"Ảnh train [{index}]: {target['image_id']} | {len(target['boxes'])} hộp")
    print(f"Kích thước: {original.size} → {resized.size}")
    print(f"Ảnh kiểm tra: {output}")
    print("Hãy mở ảnh và kiểm tra hộp vẫn bám đúng vật thể ở cả ba cột.")


if __name__ == "__main__":
    main()
