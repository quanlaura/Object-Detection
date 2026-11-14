"""Kiểm tra giữ hộp theo điểm và tách các lớp trong NMS."""

import unittest

import torch

from utils.nms import batched_nms, nms


class NMSTests(unittest.TestCase):
    def test_greedy_nms_keeps_high_score_and_separate_box(self):
        boxes = torch.tensor([[1., 0., 11., 10.], [0., 0., 10., 10.], [20., 0., 30., 10.]])
        self.assertEqual(nms(boxes, torch.tensor([.7, .9, .8]), .5).tolist(), [1, 2])

    def test_negative_coordinates_do_not_mix_classes(self):
        boxes = torch.tensor([
            [0., 0., 9., 9.], [-10., -10., -1., -1.], [0., 0., 9., 9.],
        ], dtype=torch.float16)
        kept = batched_nms(boxes, torch.tensor([.9, .8, .7]), torch.tensor([0, 1, 0]), .5)
        self.assertEqual(kept.tolist(), [0, 1])

    def test_empty_boxes_return_long_indices(self):
        indices = batched_nms(torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long), .5)
        self.assertEqual(indices.shape, (0,))
        self.assertEqual(indices.dtype, torch.long)
