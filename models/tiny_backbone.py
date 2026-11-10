import math

import torch
from torch import nn


def _gn_groups(out_ch: int, max_groups: int = 32) -> int:
    groups = min(max_groups, max(1, out_ch // 4))
    while out_ch % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=1, max_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.gn = nn.GroupNorm(
            _gn_groups(out_ch, max_groups), out_ch, eps=1e-3
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features):
        features = self.conv(features)
        features = self.gn(features)
        return self.relu(features)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, max_groups=32):
        super().__init__()
        self.conv1 = ConvBlock(
            in_ch, out_ch, stride=stride, max_groups=max_groups
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch, max_groups), out_ch, eps=1e-3),
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_gn_groups(out_ch, max_groups), out_ch, eps=1e-3),
            )

    def forward(self, features):
        residual = self.shortcut(features)
        transformed = self.conv2(self.conv1(features))
        return self.relu(transformed + residual)


def _make_stage(in_ch, out_ch, num_blocks, stride=2, max_groups=32):
    blocks = [
        ResidualBlock(in_ch, out_ch, stride=stride, max_groups=max_groups)
    ]
    blocks.extend(
        ResidualBlock(out_ch, out_ch, max_groups=max_groups)
        for _ in range(num_blocks - 1)
    )
    return nn.Sequential(*blocks)


class TinyBackbone(nn.Module):
    BASE_CHANNELS = (32, 64, 128, 256, 512)

    def __init__(self, width_multiplier=1.0, num_blocks=(2, 3, 3)):
        super().__init__()
        if not math.isfinite(width_multiplier) or width_multiplier <= 0:
            raise ValueError("width_multiplier phải hữu hạn và dương.")
        if len(num_blocks) != 3 or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in num_blocks
        ):
            raise ValueError("num_blocks phải chứa ba số nguyên dương.")

        c1, c2, c3, c4, c5 = [
            max(1, int(base * width_multiplier))
            for base in self.BASE_CHANNELS
        ]
        self.stage1 = nn.Sequential(
            ConvBlock(3, c1, max_groups=min(32, c1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(c1, c2, max_groups=min(32, c2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = _make_stage(
            c2, c3, num_blocks[0], max_groups=min(32, c3)
        )
        self.stage4 = _make_stage(
            c3, c4, num_blocks[1], max_groups=min(32, c4)
        )
        self.stage5 = _make_stage(
            c4, c5, num_blocks[2], max_groups=min(32, c5)
        )
        self._out_channels = (c3, c4, c5)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    layer.weight, a=0.1, mode="fan_in",
                    nonlinearity="leaky_relu",
                )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, images):
        features = self.stage2(self.stage1(images))
        c3 = self.stage3(features)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5
