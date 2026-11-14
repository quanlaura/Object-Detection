"""Đưa một ảnh validation qua backbone và kiểm tra ba đầu ra."""

import argparse
from pathlib import Path

import torch

from models.backbone import ResNetBackbone
from utils.dataset import DetectionDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="Nạp trọng số ImageNet.")
    args = parser.parse_args()
    data_root = Path(__file__).resolve().parents[1] / "public"
    dataset = DetectionDataset(data_root / "annotations/val.json", data_root / "val/images")
    image, _ = dataset[0]
    images = image.unsqueeze(0)
    model = ResNetBackbone("resnet34", pretrained=args.pretrained).eval()

    with torch.inference_mode():
        features = model(images)

    print("Ảnh đầu vào:", tuple(images.shape))
    for level, stride, channels, feature in zip(
        ("C3", "C4", "C5"), (8, 16, 32), model.out_channels, features
    ):
        expected = (1, channels, images.shape[2] // stride, images.shape[3] // stride)
        if feature.shape != expected or not torch.isfinite(feature).all():
            raise RuntimeError(f"{level}: kết quả không hợp lệ, cần kích thước {expected}.")
        print(f"{level}: {tuple(feature.shape)}")
    print("Hoàn thành bước 3.1.")


if __name__ == "__main__":
    main()
