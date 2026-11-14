"""Nạp checkpoint, nhận diện ảnh trong một thư mục và xuất JSON."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from models.detector import RetinaDetector
from utils.dataset import image_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--annotations", type=Path, help="Chỉ dự đoán các ảnh trong file split.")
    parser.add_argument("--img_size", type=int, help="Ghi đè kích thước trong checkpoint.")
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", action="store_true", help="Dự đoán thêm trên ảnh lật ngang.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_detections < 1:
        parser.error("batch_size/max_detections phải dương, num_workers phải không âm.")
    if args.img_size is not None and args.img_size < 1:
        parser.error("img_size phải dương.")
    if not 0 <= args.conf_thresh <= 1 or not 0 <= args.nms_thresh <= 1:
        parser.error("Ngưỡng confidence và NMS phải thuộc [0, 1].")
    return args


class InferenceDataset:
    def __init__(self, image_dir, img_size, annotations=None):
        self.files = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if annotations is not None:
            data = json.loads(annotations.read_text(encoding="utf-8"))
            by_name = {path.name: path for path in self.files}
            ids = [image["id"] for image in data["images"]]
            if len(ids) != len(set(ids)):
                raise ValueError("File split có ID ảnh trùng.")
            missing = [name for name in ids if name not in by_name]
            if missing:
                raise FileNotFoundError(f"Thiếu ảnh {missing[0]} trong {image_dir}.")
            self.files = [by_name[name] for name in ids]
        if not self.files:
            raise ValueError(f"Không tìm thấy ảnh trong {image_dir}.")
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as source:
            original_size = (source.height, source.width)
            image = source.convert("RGB").resize(
                (self.img_size, self.img_size), Image.Resampling.BILINEAR
            )
        return image_to_tensor(image), path.name, original_size


def collate_inference(batch):
    images, names, sizes = zip(*batch)
    return torch.stack(images), names, sizes


def format_prediction(name, detection, classes):
    boxes = []
    for box, score, label in zip(
        detection["boxes"].tolist(), detection["scores"].tolist(),
        detection["labels"].tolist(), strict=True,
    ):
        # Bỏ hộp quá nhỏ trước khi làm tròn để tránh hộp có diện tích 0.
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        boxes.append({
            "class": classes[label],
            "confidence": round(score, 6),
            "bbox": [round(value, 2) for value in box],
        })
    return {"image_id": name, "boxes": boxes}


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    classes = config["classes"]
    if len(classes) != config["num_classes"]:
        raise ValueError("Số lớp trong checkpoint không khớp.")
    img_size = args.img_size if args.img_size is not None else config["img_size"]
    dataset = InferenceDataset(args.image_dir, img_size, args.annotations)
    model = RetinaDetector(
        num_classes=len(classes), backbone=config["backbone"],
        fpn_channels=config["fpn_channels"], img_size=img_size, pretrained=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_inference, pin_memory=device.type == "cuda",
    )
    detect = model.detect_tta if args.tta else model.detect
    print(f"Device: {device} | Images: {len(dataset)} | Input: {img_size} | TTA: {args.tta}")
    predictions = []
    with torch.inference_mode():
        for images, names, sizes in loader:
            detections = detect(
                images.to(device, non_blocking=True), sizes,
                conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh,
                max_detections=args.max_detections,
            )
            for name, detection in zip(names, detections, strict=True):
                predictions.append(format_prediction(name, detection, classes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(predictions, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(args.output)
    box_count = sum(len(entry["boxes"]) for entry in predictions)
    print(f"Saved {len(predictions)} images, {box_count} boxes: {args.output}")


if __name__ == "__main__":
    main()
