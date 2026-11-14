"""Đầu dự đoán lớp và độ lệch hộp cho các anchor trên P3–P7."""

import math

import torch
from torch import nn


def make_branch(channels, outputs, depth):
    groups = min(32, max(1, channels // 4))
    while channels % groups:
        groups -= 1

    layers = []
    for _ in range(depth):
        layers.extend([
            # Conv 3×3 theo từng kênh, sau đó conv 1×1 trộn các kênh.
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels, eps=1e-3),
            nn.GELU(),
        ])
    layers.append(nn.Conv2d(channels, outputs, 3, padding=1))
    return nn.Sequential(*layers)


def flatten_predictions(prediction, values_per_anchor):
    """Từ [B, A*K, H, W] sang [B, H*W*A, K]."""
    # Duyệt hàng rồi cột, sau đó duyệt anchor trong mỗi vị trí.
    batch = prediction.shape[0]
    return prediction.permute(0, 2, 3, 1).reshape(batch, -1, values_per_anchor)


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors, prior=0.01, num_convs=4):
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError("Prior phải nằm giữa 0 và 1.")
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.classifier = make_branch(in_channels, num_anchors * num_classes, num_convs)
        self.box_regressor = make_branch(in_channels, num_anchors * 4, num_convs)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.constant_(self.classifier[-1].bias, math.log(prior / (1 - prior)))

    def forward(self, features):
        class_predictions, box_predictions = [], []
        for feature in features:
            logits = self.classifier(feature)
            deltas = self.box_regressor(feature)
            class_predictions.append(flatten_predictions(logits, self.num_classes))
            box_predictions.append(flatten_predictions(deltas, 4))
        return torch.cat(class_predictions, dim=1), torch.cat(box_predictions, dim=1)
