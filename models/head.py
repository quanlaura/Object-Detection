"""Hai nhánh dự đoán lớp và độ lệch bounding box."""

import math

import torch
from torch import nn


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """Chọn số nhóm lớn nhất chia hết số kênh."""
    limit = min(max_groups, max(1, channels // 4))

    while channels % limit != 0:
        limit -= 1

    return limit


class _DWSepConv(nn.Module):
    """Depthwise 3×3 → pointwise 1×1 → GroupNorm → GELU."""

    def __init__(self, channels: int):
        super().__init__()

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.gn = nn.GroupNorm(
            num_groups=_gn_groups(channels),
            num_channels=channels,
            eps=1e-3,
        )

        self.act = nn.GELU()

    def forward(self, features):
        features = self.dw(features)
        features = self.pw(features)
        features = self.gn(features)

        return self.act(features)


class _Subnet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_outputs,
        num_convs=3,
    ):
        super().__init__()

        self.body = nn.Sequential(
            *[
                _DWSepConv(in_channels)
                for _ in range(num_convs)
            ]
        )

        self.out = nn.Conv2d(
            in_channels,
            num_outputs,
            kernel_size=3,
            padding=1,
        )

    def forward(self, features):
        return self.out(self.body(features))


class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_anchors,
        prior=0.01,
        num_convs=4,
    ):
        super().__init__()

        if not 0 < prior < 1:
            raise ValueError("prior phải nằm giữa 0 và 1.")

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_subnet = _Subnet(
            in_channels=in_channels,
            num_outputs=num_anchors * num_classes,
            num_convs=num_convs,
        )

        self.reg_subnet = _Subnet(
            in_channels=in_channels,
            num_outputs=num_anchors * 4,
            num_convs=num_convs,
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01,
                )

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        foreground_bias = math.log(prior / (1 - prior))

        nn.init.constant_(
            self.cls_subnet.out.bias,
            foreground_bias,
        )

    def forward(self, features):
        classification_levels = []
        regression_levels = []

        for feature in features:
            batch_size = feature.shape[0]

            class_logits = self.cls_subnet(feature)
            box_deltas = self.reg_subnet(feature)

            # Thứ tự: hàng → cột → anchor → lớp/độ lệch.
            class_logits = class_logits.permute(
                0, 2, 3, 1
            ).reshape(
                batch_size, -1, self.num_classes
            )

            box_deltas = box_deltas.permute(
                0, 2, 3, 1
            ).reshape(
                batch_size, -1, 4
            )

            classification_levels.append(class_logits)
            regression_levels.append(box_deltas)

        return (
            torch.cat(classification_levels, dim=1),
            torch.cat(regression_levels, dim=1),
        )