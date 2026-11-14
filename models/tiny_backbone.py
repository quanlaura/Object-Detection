"""CNN nhẹ với các khối residual, trả về C3, C4, C5."""

import math

from torch import nn


def conv_norm(in_channels, out_channels, kernel_size=3, stride=1):
    groups = min(32, max(1, out_channels // 4))
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=kernel_size // 2, bias=False),
        nn.GroupNorm(groups, out_channels, eps=1e-3),
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.path = nn.Sequential(
            conv_norm(in_channels, out_channels, stride=stride),
            nn.LeakyReLU(0.1, inplace=True),
            conv_norm(out_channels, out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_norm(in_channels, out_channels, kernel_size=1, stride=stride)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activate(self.path(x) + self.shortcut(x))


class TinyBackbone(nn.Module):
    def __init__(self, width_multiplier=1.0, num_blocks=(2, 3, 3)):
        super().__init__()
        if not math.isfinite(width_multiplier) or width_multiplier <= 0:
            raise ValueError("width_multiplier phải là số dương hữu hạn.")
        if len(num_blocks) != 3 or any(count < 1 for count in num_blocks):
            raise ValueError("Cần ít nhất một khối residual cho mỗi tầng C3, C4, C5.")
        c1, c2, c3, c4, c5 = [
            max(1, int(channels * width_multiplier))
            for channels in (32, 64, 128, 256, 512)
        ]
        self.out_channels = (c3, c4, c5)
        self.stem = nn.Sequential(
            conv_norm(3, c1), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            conv_norm(c1, c2), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        self.stages = nn.ModuleList()
        for incoming, outgoing, depth in zip((c2, c3, c4), self.out_channels, num_blocks):
            blocks = [ResidualBlock(incoming, outgoing, stride=2)]
            blocks.extend(ResidualBlock(outgoing, outgoing) for _ in range(depth - 1))
            self.stages.append(nn.Sequential(*blocks))

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity="leaky_relu")

    def forward(self, images):
        c3 = self.stages[0](self.stem(images))
        c4 = self.stages[1](c3)
        c5 = self.stages[2](c4)
        return c3, c4, c5
