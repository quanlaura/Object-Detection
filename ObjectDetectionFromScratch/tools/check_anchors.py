"""Đối chiếu số hộp neo với kích thước các tầng FPN."""

import torch

from models.fpn import FPN
from utils.anchors import AnchorGenerator


def main():
    image_size = (512, 512)
    fpn = FPN((128, 256, 512)).eval()
    backbone_features = [
        torch.zeros(1, channels, size, size)
        for channels, size in ((128, 64), (256, 32), (512, 16))
    ]
    with torch.inference_mode():
        pyramid = fpn(backbone_features)

    generator = AnchorGenerator()
    anchors = generator.generate(image_size)
    counts = generator.num_anchors_per_level(image_size)
    anchors_per_cell = generator.num_anchors_per_loc
    expected = [feature.shape[-2] * feature.shape[-1] * anchors_per_cell for feature in pyramid]
    if counts != expected or tuple(anchors.shape) != (sum(expected), 4):
        raise RuntimeError("So anchor khong khop voi cac tang FPN.")

    print("Anchors per location:", anchors_per_cell)
    for level, (feature, count) in enumerate(zip(pyramid, counts), start=3):
        h, w = feature.shape[-2:]
        print(f"P{level}: {h} x {w} x {anchors_per_cell} = {count}")
    print("Anchors:", tuple(anchors.shape))
    print("Hoan thanh buoc 3.3.")


if __name__ == "__main__":
    main()
