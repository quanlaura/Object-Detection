"""Dự đoán thư mục ảnh bằng checkpoint của pipeline mới."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from models.detector import RetinaDetector


class ImageDataset(Dataset):
    def __init__(self, directory, size):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(directory)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        self.size = size

        if not self.paths:
            raise ValueError("Thư mục không có ảnh.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")

        width, height = image.size
        image = image.resize(
            (self.size, self.size),
            Image.Resampling.BILINEAR,
        )
        image = TF.normalize(
            TF.to_tensor(image),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return image, path.name, (height, width)


def collate(batch):
    images, names, sizes = zip(*batch)
    return torch.stack(images), names, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "models" / "best.pth")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Output không được trùng checkpoint.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size phải dương, num_workers không âm.")
    if not 0 <= args.conf_thresh <= 1 or not 0 <= args.nms_thresh <= 1:
        raise ValueError("Ngưỡng phải nằm trong [0, 1].")
    if args.max_detections < 1:
        raise ValueError("max_detections phải dương.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng; hãy dùng --device cpu.")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    classes = config["classes"]
    image_size = config["img_size"]

    if config["num_classes"] != len(classes):
        raise ValueError("Số lớp không khớp cấu hình.")

    model = RetinaDetector(
        num_classes=config["num_classes"],
        backbone=config["backbone"],
        pretrained=False,
        fpn_channels=config["fpn_channels"],
        img_size=image_size,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    del checkpoint

    dataset = ImageDataset(args.image_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    predictions = []
    print("Thiết bị:", device, flush=True)
    print("Số ảnh:", len(dataset), flush=True)

    with torch.inference_mode():
        for images, names, sizes in loader:
            results = model.detect(
                images.to(device, non_blocking=True),
                sizes,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh,
                max_detections=args.max_detections,
            )

            for name, result in zip(names, results):
                boxes = result["boxes"].detach().cpu().tolist()
                scores = result["scores"].detach().cpu().tolist()
                labels = result["labels"].detach().cpu().tolist()

                predictions.append({
                    "image_id": name,
                    "boxes": [
                        {
                            "class": classes[label],
                            "confidence": float(score),
                            "bbox": [float(v) for v in box],
                        }
                        for box, score, label in zip(
                            boxes, scores, labels
                        )
                    ],
                })

            print(
                f"Đã xử lý {len(predictions)}/{len(dataset)} ảnh",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(
            predictions,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(args.output)

    print("JSON kết quả:", args.output, flush=True)


if __name__ == "__main__":
    main()