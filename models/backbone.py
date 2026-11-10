"""Backbone ResNet cung cấp ba mức đặc trưng C3, C4, C5."""

from torch import nn
from torchvision import models


def _freeze_bn(module: nn.Module):
    """Giữ BatchNorm dùng running statistics và tắt gradient."""
    for layer in module.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()

            for parameter in layer.parameters():
                parameter.requires_grad_(False)


class ResNetBackbone(nn.Module):
    OUT_CHANNELS = {
        "resnet18": (128, 256, 512),
        "resnet34": (128, 256, 512),
        "resnet50": (512, 1024, 2048),
    }

    def __init__(
        self,
        name="resnet34",
        pretrained=True,
        freeze_stages=1,
    ):
        super().__init__()

        if name not in self.OUT_CHANNELS:
            supported = ", ".join(self.OUT_CHANNELS)
            raise ValueError(
                f"Backbone không được hỗ trợ: {name}. "
                f"Các lựa chọn: {supported}"
            )

        constructor = getattr(models, name)
        network = constructor(
            weights="DEFAULT" if pretrained else None
        )

        self.stem = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
        )

        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

        self.out_channels = self.OUT_CHANNELS[name]

        # Giữ đúng ý nghĩa freeze_stages của mẫu:
        # từ 1 trở lên đóng băng stem và layer1.
        if freeze_stages >= 1:
            for stage in (self.stem, self.layer1):
                for parameter in stage.parameters():
                    parameter.requires_grad_(False)

        _freeze_bn(self)

    def train(self, mode=True):
        # Chuyển các lớp sang mode yêu cầu trước.
        super().train(mode)

        # Sau đó đưa riêng BatchNorm về eval.
        _freeze_bn(self)

        return self

    def forward(self, images):
        features = self.stem(images)
        features = self.layer1(features)

        c3 = self.layer2(features)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5