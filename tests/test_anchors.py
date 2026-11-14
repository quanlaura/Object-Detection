"""Kiểm tra hình học và thứ tự anchor."""

import unittest

import torch

from utils.anchors import AnchorGenerator


class AnchorTests(unittest.TestCase):
    def test_default_count_and_first_square(self):
        generator = AnchorGenerator()
        anchors = generator.generate((512, 512))
        self.assertEqual(anchors.shape, (81840, 4))
        # Scale đầu tiên và ratio 1 nằm ở vị trí thứ ba trong mỗi ô.
        torch.testing.assert_close(anchors[2], torch.tensor([-12., -12., 20., 20.]))

    def test_row_major_on_rectangular_image(self):
        generator = AnchorGenerator(strides=(8,), base_sizes=(8,), scales=(1,), ratios=(1,))
        anchors = generator.generate((9, 17))
        expected = torch.tensor([
            [0., 0., 8., 8.], [8., 0., 16., 8.], [16., 0., 24., 8.],
            [0., 8., 8., 16.], [8., 8., 16., 16.], [16., 8., 24., 16.],
        ])
        torch.testing.assert_close(anchors, expected)
