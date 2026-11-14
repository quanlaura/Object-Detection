"""Kiểm tra các trường hợp hình học và độ chính xác số."""

import unittest

import torch

from utils.box_utils import box_area, box_giou, box_iou, clip_boxes, decode_boxes, encode_boxes


class BoxUtilsTests(unittest.TestCase):
    def test_disjoint_boxes_have_negative_giou(self):
        a = torch.tensor([[0., 0., 2., 2.]])
        b = torch.tensor([[4., 0., 6., 2.]])
        torch.testing.assert_close(box_area(a), torch.tensor([4.]))
        torch.testing.assert_close(box_iou(a, b), torch.zeros(1, 1))
        torch.testing.assert_close(box_giou(a, b), torch.tensor([[-1 / 3]]))

    def test_round_trip_with_size_and_center_change(self):
        anchors = torch.tensor([[0., 0., 100., 100.], [50., 50., 150., 150.]])
        targets = torch.tensor([[10., 20., 110., 120.], [25., 50., 175., 150.]])
        deltas = encode_boxes(anchors, targets)
        self.assertGreater(deltas[1, 2].item(), 0)
        torch.testing.assert_close(decode_boxes(anchors, deltas), targets)

    def test_large_fp16_boxes_and_empty_input(self):
        boxes = torch.tensor([[0., 0., 512., 512.]], dtype=torch.float16)
        torch.testing.assert_close(box_iou(boxes, boxes), torch.ones(1, 1))
        torch.testing.assert_close(box_giou(boxes, boxes), torch.ones(1, 1))
        self.assertEqual(box_iou(boxes, boxes[:0]).shape, (1, 0))

    def test_clipping_preserves_input(self):
        boxes = torch.tensor([[-10., -20., 620., 520.]])
        result = clip_boxes(boxes, width=600, height=500)
        torch.testing.assert_close(result, torch.tensor([[0., 0., 600., 500.]]))
        torch.testing.assert_close(boxes, torch.tensor([[-10., -20., 620., 520.]]))
