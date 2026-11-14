"""Kết hợp C3–C5 thành tháp đặc trưng P3–P7."""

from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        c3_channels, c4_channels, c5_channels = in_channels
        groups = min(32, max(1, out_channels // 4))
        while out_channels % groups != 0:
            groups -= 1

        self.project_c3 = nn.Sequential(
            nn.Conv2d(c3_channels, out_channels, 1),
            nn.GroupNorm(groups, out_channels, eps=1e-3),
        )
        self.project_c4 = nn.Sequential(
            nn.Conv2d(c4_channels, out_channels, 1),
            nn.GroupNorm(groups, out_channels, eps=1e-3),
        )
        self.project_c5 = nn.Sequential(
            nn.Conv2d(c5_channels, out_channels, 1),
            nn.GroupNorm(groups, out_channels, eps=1e-3),
        )

        # P3–P5 dùng conv 3×3 để xử lý sau khi ghép các tầng.
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels, eps=1e-3),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        )
        self.downsample = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.GroupNorm(groups, out_channels, eps=1e-3),
            )
            for _ in range(2)
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.refine[2](self.project_c5(c5))
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.refine[1](self.project_c4(c4) + p5_up)
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.refine[0](self.project_c3(c3) + p4_up)

        p6 = self.downsample[0](F.relu(p5))
        p7 = self.downsample[1](F.relu(p6))
        return [p3, p4, p5, p6, p7]
