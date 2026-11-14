"""Lấy ba mức đặc trưng từ ResNet của torchvision để đưa vào FPN."""

from torch import Tensor, nn
from torchvision.models import get_model, get_model_weights


class ResNetBackbone(nn.Module):
    OUT_CHANNELS = {
        "resnet18": (128, 256, 512),
        "resnet34": (128, 256, 512),
        "resnet50": (512, 1024, 2048),
    }
    def __init__(
        self, name: str = "resnet34", pretrained: bool = True, freeze_stages: int = 1
    ):
        super().__init__()
        if name not in self.OUT_CHANNELS:
            supported = ", ".join(self.OUT_CHANNELS)
            raise ValueError(f"Backbone '{name}' không được hỗ trợ. Chọn {supported}.")

        self.out_channels = self.OUT_CHANNELS[name]
        weights = get_model_weights(name).DEFAULT if pretrained else None
        resnet = get_model(name, weights=weights)
        self.early_features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4

        if freeze_stages >= 1:
            self.early_features.requires_grad_(False)

        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(False)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        # train() của mô hình cha sẽ bật lại chế độ train cho mọi tầng con.
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.early_features(images)
        c3 = self.stage3(features)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5
