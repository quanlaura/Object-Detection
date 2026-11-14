"""Chạy một lần cập nhật trọng số trên ảnh train thật."""

import argparse
import random
from pathlib import Path

import torch

from models.detector import RetinaDetector
from models.loss import DetectionLoss
from utils.dataset import DetectionDataset, collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="Nạp ResNet34 ImageNet.")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    public = Path(__file__).resolve().parents[1] / "public"
    # Dùng kích thước nhỏ và dữ liệu cố định để dễ kiểm tra một lần cập nhật.
    dataset = DetectionDataset(
        public / "annotations/train.json", public / "train/images", img_size=128
    )
    indices = [i for i, record in enumerate(dataset.records) if len(record["boxes"]) > 0][:2]
    if len(indices) != 2:
        raise RuntimeError("Cần hai ảnh train có nhãn để kiểm tra.")
    images, targets = collate_fn([dataset[i] for i in indices])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    model = RetinaDetector(
        num_classes=len(dataset.classes), pretrained=args.pretrained,
        fpn_channels=64, img_size=128,
    ).to(device).train()
    criterion = DetectionLoss(len(dataset.classes)).to(device)
    # Dùng LR nhỏ hơn cho backbone và FPN, như cấu hình bài mẫu.
    lr = 1e-4
    optimizer = torch.optim.AdamW(
        [
            {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
             "lr": lr * 0.05},
            {"params": model.fpn.parameters(), "lr": lr * 0.30},
            {"params": model.head.parameters(), "lr": lr},
        ],
        weight_decay=5e-4,
    )
    cls_weight = model.head.classifier[-1].weight
    box_weight = model.head.box_regressor[-1].weight
    frozen_weight = next(model.backbone.early_features.parameters())
    before_cls, before_box, before_frozen = (
        weight.detach().clone() for weight in (cls_weight, box_weight, frozen_weight)
    )

    optimizer.zero_grad(set_to_none=True)
    logits, deltas = model(images)
    anchors = model.get_anchors(images.shape[-2:], device)
    cls_loss, reg_loss = criterion(logits, deltas, anchors, targets)
    total_loss = cls_loss + reg_loss
    if not torch.isfinite(total_loss):
        raise RuntimeError("Loss phải hữu hạn trước khi cập nhật trọng số.")

    total_loss.backward()
    for weight in (cls_weight, box_weight):
        if weight.grad is None or not weight.grad.abs().any():
            raise RuntimeError("Một nhánh head chưa nhận được gradient khác 0.")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=5.0, error_if_nonfinite=True
    )
    optimizer.step()

    cls_change = (cls_weight.detach() - before_cls).abs().max().item()
    box_change = (box_weight.detach() - before_box).abs().max().item()
    if not (cls_change > 0 and box_change > 0 and torch.equal(frozen_weight, before_frozen)):
        raise RuntimeError("Trọng số sau cập nhật không đúng kỳ vọng.")

    print("Device:", device)
    print("Batch:", tuple(images.shape))
    print(f"Loss: cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f}, total={total_loss.item():.4f}")
    print(f"Gradient norm before clipping: {grad_norm.item():.4f}")
    print(f"Weight change: cls={cls_change:.2e}, box={box_change:.2e}")
    print("Frozen stem: OK")
    print("Hoan thanh buoc 4.2.")


if __name__ == "__main__":
    main()
