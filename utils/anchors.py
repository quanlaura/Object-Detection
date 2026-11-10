"""Tạo anchor cho các mức FPN.

Tọa độ theo dạng [xmin, ymin, xmax, ymax].
Thứ tự là mức FPN → hàng → cột → scale → ratio.
"""

import math

import torch


class AnchorGenerator:
    def __init__(
        self,
        strides=(8, 16, 32, 64, 128),
        base_sizes=(32, 64, 128, 256, 512),
        scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
        ratios=(0.33, 0.5, 1.0, 2.0, 3.0),
    ):
        if len(strides) != len(base_sizes) or not strides:
            raise ValueError(
                "strides và base_sizes phải có cùng số phần tử, khác rỗng."
            )

        if not scales or not ratios:
            raise ValueError("scales và ratios không được rỗng.")

        if any(
            value <= 0
            for values in (strides, base_sizes, scales, ratios)
            for value in values
        ):
            raise ValueError("Các thông số anchor phải dương.")

        self.strides = tuple(strides)
        self.base_sizes = tuple(base_sizes)
        self.scales = tuple(scales)
        self.ratios = tuple(ratios)

        self.num_anchors_per_loc = (
            len(self.scales) * len(self.ratios)
        )

        # Lưu các hình dạng hộp quanh tâm (0, 0).
        self.cell_anchors = [
            self._cell_anchors(size)
            for size in self.base_sizes
        ]

    def _cell_anchors(self, base_size):
        """Tạo [A, 4] hộp quanh gốc tọa độ cho một mức."""
        dimensions = []

        for scale in self.scales:
            side = base_size * scale

            for ratio in self.ratios:
                width = side * math.sqrt(ratio)
                height = width / ratio
                dimensions.append((width, height))

        half_sizes = (
            torch.tensor(dimensions, dtype=torch.float32) / 2
        )

        return torch.cat(
            (-half_sizes, half_sizes),
            dim=1,
        )

    @torch.no_grad()
    def generate(
        self,
        image_size,
        device=torch.device("cpu"),
    ):
        """Tạo toàn bộ anchor cho đầu vào (height, width)."""
        height, width = image_size

        if height <= 0 or width <= 0:
            raise ValueError("Kích thước ảnh phải dương.")

        pyramid_anchors = []

        for stride, templates in zip(
            self.strides, self.cell_anchors
        ):
            rows = math.ceil(height / stride)
            columns = math.ceil(width / stride)

            x_centers = (
                torch.arange(
                    columns,
                    device=device,
                    dtype=torch.float32,
                ) + 0.5
            ) * stride

            y_centers = (
                torch.arange(
                    rows,
                    device=device,
                    dtype=torch.float32,
                ) + 0.5
            ) * stride

            grid_y, grid_x = torch.meshgrid(
                y_centers,
                x_centers,
                indexing="ij",
            )

            centers = torch.stack(
                (grid_x, grid_y),
                dim=-1,
            ).reshape(-1, 2)

            corner_offsets = torch.cat(
                (centers, centers),
                dim=1,
            )

            boxes = (
                corner_offsets[:, None, :]
                + templates.to(device)[None, :, :]
            )

            pyramid_anchors.append(boxes.reshape(-1, 4))

        return torch.cat(pyramid_anchors, dim=0)

    def num_anchors_per_level(self, image_size):
        """Trả số anchor ở từng mức FPN."""
        height, width = image_size

        return [
            math.ceil(height / stride)
            * math.ceil(width / stride)
            * self.num_anchors_per_loc
            for stride in self.strides
        ]