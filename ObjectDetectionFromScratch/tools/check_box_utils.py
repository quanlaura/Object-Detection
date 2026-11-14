"""Chạy ví dụ số cho IoU và mã hóa hộp."""

import torch

from utils.box_utils import box_iou, decode_boxes, encode_boxes


def main():
    anchor = torch.tensor([[0., 0., 100., 100.]])
    boxes = torch.tensor([
        [0., 0., 100., 100.],
        [110., 0., 210., 100.],
        [50., 0., 150., 100.],
    ])
    iou = box_iou(anchor, boxes)
    torch.testing.assert_close(iou, torch.tensor([[1., 0., 1 / 3]]))

    target = torch.tensor([[10., 20., 110., 120.]])
    encoded = encode_boxes(anchor, target)
    decoded = decode_boxes(anchor, encoded)
    torch.testing.assert_close(encoded, torch.tensor([[1., 2., 0., 0.]]))
    torch.testing.assert_close(decoded, target)

    print("IoU:", iou)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Hoan thanh buoc 3.4.")


if __name__ == "__main__":
    main()
