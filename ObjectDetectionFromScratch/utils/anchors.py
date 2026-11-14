"""Tạo các hộp neo [xmin, ymin, xmax, ymax] cho tháp đặc trưng."""

from itertools import product

import torch


class AnchorGenerator:
    def __init__(
        self,
        strides=(8, 16, 32, 64, 128),
        base_sizes=(32, 64, 128, 256, 512),
        scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
        ratios=(0.33, 0.5, 1.0, 2.0, 3.0),
    ):
        if not strides or len(strides) != len(base_sizes):
            raise ValueError("Mỗi tầng cần một stride và một kích thước gốc.")
        if not scales or not ratios or any(
            value <= 0 for values in (strides, base_sizes, scales, ratios) for value in values
        ):
            raise ValueError("Các tham số của anchor phải dương.")

        self.strides = tuple(strides)
        self.base_sizes = tuple(base_sizes)
        self.scales = tuple(scales)
        self.ratios = tuple(ratios)
        self.num_anchors_per_loc = len(scales) * len(ratios)

        # ratio = width / height; scale thay đổi cả chiều rộng lẫn chiều cao.
        sizes = []
        for scale, ratio in product(self.scales, self.ratios):
            aspect = ratio ** 0.5
            sizes.append((scale * aspect, scale / aspect))
        half_size = torch.tensor(sizes, dtype=torch.float32) / 2
        self._unit_boxes = torch.cat((-half_size, half_size), dim=1)

    def _grid_shapes(self, image_size):
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError("Kích thước ảnh phải dương.")
        return [
            ((height + stride - 1) // stride, (width + stride - 1) // stride)
            for stride in self.strides
        ]

    def num_anchors_per_level(self, image_size):
        return [h * w * self.num_anchors_per_loc for h, w in self._grid_shapes(image_size)]

    def generate(self, image_size, device="cpu") -> torch.Tensor:
        """Sinh anchor theo thứ tự tầng → hàng → cột → scale → ratio."""
        templates = self._unit_boxes.to(device)
        pyramid_boxes = []
        for (height, width), stride, base_size in zip(
            self._grid_shapes(image_size), self.strides, self.base_sizes
        ):
            ys = torch.arange(height, dtype=templates.dtype, device=templates.device)
            xs = torch.arange(width, dtype=templates.dtype, device=templates.device)
            # Điểm (0, 0) của feature map có tâm tại nửa stride trên ảnh.
            yx = (torch.cartesian_prod(ys, xs) + 0.5) * stride
            xy = yx.flip(dims=(1,))
            centers = xy.repeat(1, 2)
            boxes = centers[:, None, :] + base_size * templates
            pyramid_boxes.append(boxes.flatten(0, 1))
        return torch.cat(pyramid_boxes, dim=0)
