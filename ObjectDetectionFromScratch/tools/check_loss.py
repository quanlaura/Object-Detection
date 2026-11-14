"""Kiểm tra ghép anchor, loss và gradient bằng một batch nhỏ."""

import torch

from models.loss import DetectionLoss


def main():
    anchors = torch.tensor([
        [0., 0., 10., 10.],
        [0., 0., 10., 22.],
        [20., 20., 30., 30.],
    ])
    gt_boxes = torch.tensor([[0., 0., 10., 10.]])
    targets = [
        {"boxes": gt_boxes, "labels": torch.tensor([2])},
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
    ]
    criterion = DetectionLoss(5)
    status, _ = criterion._match(anchors, gt_boxes)
    assert status.tolist() == [1, -1, 0]
    print("Anchor status:", status.tolist(), "(positive, ignore, background)")
    for loss_type in ("giou", "ciou", "smooth_l1"):
        criterion = DetectionLoss(5, reg_loss_type=loss_type)
        logits = torch.zeros(2, 3, 5, requires_grad=True)
        deltas = torch.full((2, 3, 4), .2, requires_grad=True)
        cls_loss, reg_loss = criterion(logits, deltas, anchors, targets)
        (cls_loss + reg_loss).backward()
        assert torch.isfinite(cls_loss) and torch.isfinite(reg_loss)
        assert logits.grad.isfinite().all() and deltas.grad.isfinite().all()
        assert deltas.grad[0, 0].abs().sum() > 0 and deltas.grad[1].abs().sum() == 0
        assert logits.grad[0, 1].abs().sum() == 0
        print(f"{loss_type}: cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f}, gradients OK")

    _, empty_reg = criterion(logits[1:], deltas[1:], anchors, targets[1:])
    assert empty_reg.item() == 0
    print("Empty ground truth: OK")
    print("Hoan thanh buoc 4.1.")


if __name__ == "__main__":
    main()
