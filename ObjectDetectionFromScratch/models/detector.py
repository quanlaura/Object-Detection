"""Ghép backbone, FPN và head thành mô hình RetinaNet."""

import torch
from torch import nn

from utils.anchors import AnchorGenerator
from utils.box_utils import clip_boxes, decode_boxes
from utils.nms import batched_nms

from .backbone import ResNetBackbone
from .fpn import FPN
from .head import DetectionHead
from .tiny_backbone import TinyBackbone


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
        self.backbone_name = backbone
        self.img_size = img_size
        if backbone in ResNetBackbone.OUT_CHANNELS:
            self.backbone = ResNetBackbone(backbone, pretrained=pretrained)
        elif backbone.startswith("tiny"):
            suffix = backbone[4:].removeprefix("_")
            width = float(suffix) if suffix else 1.0
            self.backbone = TinyBackbone(width_multiplier=width)
        else:
            raise ValueError(f"Backbone không hỗ trợ: {backbone}")
        self.fpn = FPN(self.backbone.out_channels, fpn_channels)
        self.anchor_gen = AnchorGenerator()
        self.head = DetectionHead(
            fpn_channels, num_classes, self.anchor_gen.num_anchors_per_loc
        )
        self._anchor_cache = {}

    def get_anchors(self, image_size, device):
        """Lấy anchor theo kích thước ảnh thực tế và thiết bị đang chạy."""
        size = tuple(image_size)
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (size, device)
        if key not in self._anchor_cache:
            # Cache tensor thường để có thể dùng lại khi tính gradient sau validation.
            with torch.inference_mode(False):
                self._anchor_cache[key] = self.anchor_gen.generate(size, device)
        return self._anchor_cache[key]

    def forward(self, images):
        features = self.backbone(images)
        pyramid = self.fpn(features)
        return self.head(pyramid)

    def _decode_batch(self, images):
        height, width = images.shape[-2:]
        logits, deltas = self(images)
        anchors = self.get_anchors((height, width), images.device)
        boxes = [clip_boxes(decode_boxes(anchors, row), width, height) for row in deltas]
        return boxes, logits.sigmoid()

    @staticmethod
    def _filter_image(boxes, scores, input_size, orig_size, conf_thresh,
                      nms_thresh, max_detections, pre_nms_topk):
        anchor_ids, labels = torch.where(scores > conf_thresh)
        confidence = scores[anchor_ids, labels]
        if confidence.numel() > pre_nms_topk:
            confidence, top = confidence.topk(pre_nms_topk)
            anchor_ids, labels = anchor_ids[top], labels[top]

        candidates = boxes[anchor_ids]
        keep = batched_nms(candidates, confidence, labels, nms_thresh)[:max_detections]
        height, width = input_size
        orig_height, orig_width = orig_size
        sx, sy = orig_width / width, orig_height / height
        scale = boxes.new_tensor((sx, sy, sx, sy))
        return {
            "boxes": clip_boxes(candidates[keep] * scale, orig_width, orig_height),
            "scores": confidence[keep],
            "labels": labels[keep],
        }

    @torch.no_grad()
    def detect(self, images, orig_sizes, conf_thresh=0.05, nms_thresh=0.5,
               max_detections=100, pre_nms_topk=1000):
        """Dự đoán mỗi ảnh và trả hộp theo tọa độ ảnh gốc."""
        boxes, scores = self._decode_batch(images)
        return [
            self._filter_image(b, s, images.shape[-2:], size, conf_thresh,
                               nms_thresh, max_detections, pre_nms_topk)
            for b, s, size in zip(boxes, scores, orig_sizes, strict=True)
        ]

    @torch.no_grad()
    def detect_tta(self, images, orig_sizes, conf_thresh=0.05, nms_thresh=0.5,
                   max_detections=100, pre_nms_topk=1000):
        """Gộp dự đoán trên ảnh gốc và ảnh lật ngang trước NMS."""
        boxes, scores = self._decode_batch(images)
        flipped_boxes, flipped_scores = self._decode_batch(images.flip(-1))
        width = images.shape[-1]
        results = []
        for b, s, flipped, fs, size in zip(
            boxes, scores, flipped_boxes, flipped_scores, orig_sizes, strict=True
        ):
            x1, y1, x2, y2 = flipped.unbind(dim=1)
            restored = torch.stack((width - x2, y1, width - x1, y2), dim=1)
            results.append(self._filter_image(
                torch.cat((b, restored)), torch.cat((s, fs)), images.shape[-2:],
                size, conf_thresh, nms_thresh, max_detections, pre_nms_topk * 2,
            ))
        return results
