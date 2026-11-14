"""Chạy một ảnh qua backbone và kiểm tra kích thước P3–P7."""

import argparse
from pathlib import Path

import torch

from models.backbone import ResNetBackbone
from models.fpn import FPN
from utils.dataset import DetectionDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="Nạp trọng số ImageNet.")
    args = parser.parse_args()
    public = Path(__file__).resolve().parents[1] / "public"
    dataset = DetectionDataset(public / "annotations/val.json", public / "val/images")
    image, _ = dataset[0]
    images = image.unsqueeze(0)
    backbone = ResNetBackbone(pretrained=args.pretrained).eval()
    fpn = FPN(backbone.out_channels).eval()

    with torch.inference_mode():
        pyramid = fpn(backbone(images))

    if len(pyramid) != 5:
        raise RuntimeError("FPN phải trả về năm tầng đặc trưng.")
    print("Input:", tuple(images.shape))
    for level, feature in enumerate(pyramid, start=3):
        stride = 2 ** level
        height, width = images.shape[-2:]
        expected = (1, 256, (height + stride - 1) // stride, (width + stride - 1) // stride)
        if tuple(feature.shape) != expected or not torch.isfinite(feature).all():
            raise RuntimeError(f"P{level}: cần kết quả hữu hạn với kích thước {expected}.")
        print(f"P{level}: {tuple(feature.shape)}")
    print("Hoan thanh buoc 3.2.")


if __name__ == "__main__":
    main()
