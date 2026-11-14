"""Kiểm tra backbone → FPN → head trên một ảnh validation."""

import argparse
from pathlib import Path

import torch

from models.backbone import ResNetBackbone
from models.fpn import FPN
from models.head import DetectionHead, flatten_predictions
from utils.anchors import AnchorGenerator
from utils.dataset import DetectionDataset


def check_anchor_order():
    # Mã số duy nhất cho mỗi cặp (vị trí, anchor) trên lưới 2×3.
    raw = torch.arange(12).reshape(1, 2, 2, 3)
    expected = torch.tensor([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    torch.testing.assert_close(flatten_predictions(raw, 1).flatten(), expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="Nạp trọng số ImageNet.")
    args = parser.parse_args()
    public = Path(__file__).resolve().parents[1] / "public"
    dataset = DetectionDataset(public / "annotations/val.json", public / "val/images")
    image, _ = dataset[0]
    images = image.unsqueeze(0)

    generator = AnchorGenerator()
    anchors = generator.generate(images.shape[-2:])
    backbone = ResNetBackbone(pretrained=args.pretrained).eval()
    fpn = FPN(backbone.out_channels).eval()
    head = DetectionHead(256, len(dataset.classes), generator.num_anchors_per_loc).eval()
    with torch.inference_mode():
        pyramid = fpn(backbone(images))
        logits, deltas = head(pyramid)
        counts = generator.num_anchors_per_level(images.shape[-2:])
        for feature, count in zip(pyramid, counts, strict=True):
            level_logits, level_deltas = head([feature])
            assert level_logits.shape == (1, count, len(dataset.classes))
            assert level_deltas.shape == (1, count, 4)

    assert logits.shape == (1, len(anchors), len(dataset.classes))
    assert deltas.shape == (1, len(anchors), 4)
    assert torch.isfinite(logits).all() and torch.isfinite(deltas).all()
    initial_scores = head.classifier[-1].bias.sigmoid()
    torch.testing.assert_close(initial_scores, torch.full_like(initial_scores, 0.01))
    check_anchor_order()
    print("Input:", tuple(images.shape))
    print("Anchors:", tuple(anchors.shape))
    print("Class logits:", tuple(logits.shape))
    print("Box deltas:", tuple(deltas.shape))
    print("Anchor order: OK")
    print("Hoan thanh buoc 3.5.")


if __name__ == "__main__":
    main()
