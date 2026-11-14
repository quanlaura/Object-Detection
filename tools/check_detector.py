"""Kiểm tra mô hình ghép trên ảnh validation."""

import argparse
from pathlib import Path

import torch

from models.detector import RetinaDetector
from utils.box_utils import decode_boxes
from utils.dataset import DetectionDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="Nạp trọng số ImageNet.")
    args = parser.parse_args()

    public = Path(__file__).resolve().parents[1] / "public"
    dataset = DetectionDataset(public / "annotations/val.json", public / "val/images")
    image, _ = dataset[0]
    images = image.unsqueeze(0)
    model = RetinaDetector(num_classes=len(dataset.classes), pretrained=args.pretrained).eval()

    with torch.inference_mode():
        logits, deltas = model(images)
        anchors = model.get_anchors(images.shape[-2:], images.device)
        decoded = decode_boxes(anchors, deltas[0])

    assert logits.shape == (1, len(anchors), len(dataset.classes))
    assert deltas.shape == (1, len(anchors), 4)
    assert decoded.shape == anchors.shape
    assert all(torch.isfinite(tensor).all() for tensor in (logits, deltas, decoded))
    assert model.get_anchors(images.shape[-2:], images.device) is anchors
    assert not torch.is_inference(anchors)
    print("Input:", tuple(images.shape))
    print("Class logits:", tuple(logits.shape))
    print("Box deltas:", tuple(deltas.shape))
    print("Anchors:", tuple(anchors.shape))
    print("Decoded boxes:", tuple(decoded.shape))
    print("Anchor cache: OK")
    print("Hoan thanh buoc 3.6.")


if __name__ == "__main__":
    main()
