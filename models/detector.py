"""Detector RetinaNet với suy luận thường và TTA lật ngang."""

import re

import torch
from torch import nn

from utils.anchors import AnchorGenerator
from utils.box_utils import clip_boxes, decode_boxes
from utils.nms import batched_nms

from .backbone import ResNetBackbone
from .fpn import FPN
from .head import DetectionHead


class RetinaDetector(nn.Module):
    def __init__(
        self,
        num_classes=5,
        backbone="resnet34",
        pretrained=True,
        fpn_channels=256,
        img_size=512,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone_name = backbone

        if backbone.startswith("tiny"):
            # Chỉ cần module này khi chọn TinyBackbone.
            from .tiny_backbone import TinyBackbone

            match = re.fullmatch(
                r"tiny_?(\d+(?:\.\d+)?)?",
                backbone,
            )

            if match is None:
                raise ValueError(
                    "Tên TinyBackbone phải có dạng tiny hoặc tiny_0.5."
                )

            width_multiplier = (
                float(match.group(1))
                if match.group(1) is not None
                else 1.0
            )

            if width_multiplier <= 0:
                raise ValueError("width_multiplier phải dương.")

            self.backbone = TinyBackbone(
                width_multiplier=width_multiplier
            )

        elif backbone in ResNetBackbone.OUT_CHANNELS:
            self.backbone = ResNetBackbone(
                name=backbone,
                pretrained=pretrained,
            )

        else:
            raise ValueError(
                f"Backbone không được hỗ trợ: {backbone}"
            )

        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
        )

        self.anchor_gen = AnchorGenerator()

        self.head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=self.anchor_gen.num_anchors_per_loc,
        )

        self._anchor_cache = {}

    def get_anchors(self, image_size, device):
        """Tái sử dụng anchor cho cùng kích thước ảnh và thiết bị."""
        height, width = image_size
        key = (height, width, str(device))

        if key not in self._anchor_cache:
            self._anchor_cache[key] = self.anchor_gen.generate(
                (height, width),
                device=device,
            )

        return self._anchor_cache[key]

    def forward(self, images):
        backbone_features = self.backbone(images)
        pyramid = self.fpn(backbone_features)

        return self.head(pyramid)

    def _select_detections(
        self,
        boxes,
        class_scores,
        original_size,
        input_size,
        conf_thresh,
        nms_thresh,
        max_detections,
        candidate_limit,
    ):
        """Lọc ứng viên, NMS và đổi tọa độ cho một ảnh."""
        anchor_indices, class_indices = torch.where(
            class_scores > conf_thresh
        )

        if anchor_indices.numel() == 0:
            return self._empty_result(boxes.device)

        scores = class_scores[anchor_indices, class_indices]
        candidate_boxes = boxes[anchor_indices]

        if scores.numel() > candidate_limit:
            scores, selected = scores.topk(candidate_limit)
            candidate_boxes = candidate_boxes[selected]
            class_indices = class_indices[selected]

        kept = batched_nms(
            candidate_boxes,
            scores,
            class_indices,
            nms_thresh,
        )[:max_detections]

        output_boxes = candidate_boxes[kept].clone()

        input_height, input_width = input_size
        original_height, original_width = original_size

        output_boxes[:, 0::2] *= original_width / input_width
        output_boxes[:, 1::2] *= original_height / input_height

        output_boxes = clip_boxes(
            output_boxes,
            original_width,
            original_height,
        )

        return {
            "boxes": output_boxes,
            "scores": scores[kept],
            "labels": class_indices[kept],
        }

    @torch.no_grad()
    def detect(
        self,
        images,
        orig_sizes,
        conf_thresh=0.05,
        nms_thresh=0.5,
        max_detections=100,
        pre_nms_topk=1000,
    ):
        height, width = images.shape[-2:]
        anchors = self.get_anchors(
            (height, width),
            images.device,
        )

        class_logits, box_deltas = self(images)
        probabilities = class_logits.sigmoid()

        results = []

        for index in range(images.shape[0]):
            boxes = decode_boxes(
                anchors,
                box_deltas[index],
            )

            boxes = clip_boxes(boxes, width, height)

            result = self._select_detections(
                boxes=boxes,
                class_scores=probabilities[index],
                original_size=orig_sizes[index],
                input_size=(height, width),
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                max_detections=max_detections,
                candidate_limit=pre_nms_topk,
            )

            results.append(result)

        return results

    @torch.no_grad()
    def detect_tta(
        self,
        images,
        orig_sizes,
        conf_thresh=0.05,
        nms_thresh=0.5,
        max_detections=100,
        pre_nms_topk=1000,
    ):
        height, width = images.shape[-2:]
        anchors = self.get_anchors(
            (height, width),
            images.device,
        )

        original_logits, original_deltas = self(images)

        flipped_images = torch.flip(images, dims=(3,))
        flipped_logits, flipped_deltas = self(flipped_images)

        original_scores = original_logits.sigmoid()
        flipped_scores = flipped_logits.sigmoid()

        results = []

        for index in range(images.shape[0]):
            original_boxes = clip_boxes(
                decode_boxes(
                    anchors,
                    original_deltas[index],
                ),
                width,
                height,
            )

            flipped_boxes = clip_boxes(
                decode_boxes(
                    anchors,
                    flipped_deltas[index],
                ),
                width,
                height,
            )

            # Đưa tọa độ ảnh lật về hệ tọa độ ảnh đầu vào.
            restored_boxes = flipped_boxes.clone()
            restored_boxes[:, 0] = width - flipped_boxes[:, 2]
            restored_boxes[:, 2] = width - flipped_boxes[:, 0]

            combined_boxes = torch.cat(
                (original_boxes, restored_boxes),
                dim=0,
            )

            combined_scores = torch.cat(
                (
                    original_scores[index],
                    flipped_scores[index],
                ),
                dim=0,
            )

            result = self._select_detections(
                boxes=combined_boxes,
                class_scores=combined_scores,
                original_size=orig_sizes[index],
                input_size=(height, width),
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                max_detections=max_detections,
                candidate_limit=pre_nms_topk * 2,
            )

            results.append(result)

        return results

    @staticmethod
    def _empty_result(device):
        return {
            "boxes": torch.empty(
                (0, 4), device=device
            ),
            "scores": torch.empty(
                (0,), device=device
            ),
            "labels": torch.empty(
                (0,), dtype=torch.long, device=device
            ),
        }