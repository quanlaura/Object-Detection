"""Kiểm tra ảnh và hộp bao được biến đổi cùng nhau."""

import unittest
from unittest.mock import patch

from PIL import Image

from utils.dataset import crop_sample, flip_sample, resize_sample


class DatasetGeometryTests(unittest.TestCase):
    def test_resize_scales_horizontal_and_vertical_axes(self):
        image = Image.new("RGB", (400, 200))
        resized, boxes = resize_sample(image, [(40, 20, 160, 80)], 200)
        self.assertEqual(resized.size, (200, 200))
        self.assertEqual(boxes, [(20, 20, 80, 80)])

    def test_flip_moves_pixels_and_boxes(self):
        image = Image.new("RGB", (200, 200))
        image.putpixel((20, 10), (255, 0, 0))
        flipped, boxes = flip_sample(image, [(20, 20, 80, 80)])
        self.assertEqual(flipped.getpixel((179, 10)), (255, 0, 0))
        self.assertEqual(boxes, [(120, 20, 180, 80)])

    def test_crop_clips_box_and_keeps_its_label(self):
        image = Image.new("RGB", (100, 100))
        with patch("utils.dataset.random.uniform", return_value=0.5), patch(
            "utils.dataset.random.randint", side_effect=[25, 20]
        ):
            cropped, boxes, labels = crop_sample(
                image, [(10, 20, 60, 80), (80, 80, 95, 95)], [0, 1]
            )
        self.assertEqual(cropped.size, (50, 50))
        self.assertEqual(boxes, [(0, 0, 35, 50)])
        self.assertEqual(labels, [0])
