"""FPN kết hợp C3–C5 thành năm mức đặc trưng P3–P7."""

from torch import nn
from torch.nn import functional as F


def _gn(channels: int) -> nn.GroupNorm:
    """Chọn tối đa 32 nhóm, số nhóm phải chia hết số kênh."""
    group_count = min(32, max(1, channels // 4))

    while channels % group_count != 0:
        group_count -= 1

    return nn.GroupNorm(
        num_groups=group_count,
        num_channels=channels,
        eps=1e-3,
    )


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        c3_channels, c4_channels, c5_channels = in_channels

        # Đưa ba mức về cùng số kênh.
        self.lateral3 = self._projection(
            c3_channels, out_channels
        )
        self.lateral4 = self._projection(
            c4_channels, out_channels
        )
        self.lateral5 = self._projection(
            c5_channels, out_channels
        )

        self.smooth3 = self._smoothing(out_channels)
        self.smooth4 = self._smoothing(out_channels)
        self.smooth5 = self._smoothing(out_channels)

        # Hai mức bổ sung giảm kích thước bằng stride 2.
        self.p6_conv = self._downsampling(out_channels)
        self.p7_conv = self._downsampling(out_channels)

        self._init_weights()

    @staticmethod
    def _projection(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
            ),
            _gn(output_channels),
        )

    @staticmethod
    def _smoothing(channels):
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            _gn(channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _downsampling(channels):
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            _gn(channels),
        )

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features):
        c3, c4, c5 = features

        lateral3 = self.lateral3(c3)
        lateral4 = self.lateral4(c4)
        lateral5 = self.lateral5(c5)

        p5 = self.smooth5(lateral5)

        upsampled5 = F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest",
        )
        p4 = self.smooth4(lateral4 + upsampled5)

        upsampled4 = F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest",
        )
        p3 = self.smooth3(lateral3 + upsampled4)

        p6 = self.p6_conv(F.relu(p5))
        p7 = self.p7_conv(F.relu(p6))

        return [p3, p4, p5, p6, p7]