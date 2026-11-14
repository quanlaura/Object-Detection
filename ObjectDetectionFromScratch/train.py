"""Huấn luyện RetinaDetector và chọn checkpoint theo mAP@0.5."""

import argparse
import json
import random
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, get_ema_avg_fn
from torch.utils.data import DataLoader, Subset

from models.detector import RetinaDetector
from models.loss import DetectionLoss
from utils.dataset import DetectionDataset, collate_fn
from utils.metrics import evaluate_map


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models"))
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--fpn_channels", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--stop_after_epoch", type=int, default=0,
                        help="Dừng sau khi lưu epoch này, giữ lịch LR theo --epochs. 0 chạy hết.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--backbone_lr_factor", type=float, default=0.05)
    parser.add_argument("--fpn_lr_factor", type=float, default=0.30)
    parser.add_argument("--reg_loss_type", choices=("giou", "ciou", "smooth_l1"), default="giou")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--eval_conf", type=float, default=0.25)
    parser.add_argument("--eval_nms", type=float, default=0.4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 chạy hết tập train.")
    parser.add_argument("--eval_max_images", type=int, default=0, help="0 chấm toàn bộ validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Tiếp tục từ last.pth của dự án.")
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--ms_max_size", type=int, default=0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa_start", type=float, default=0.75)
    parser.add_argument("--reference_recipe", action="store_true",
                        help="Dùng quy tắc chấm và lịch LR của notebook Basic Object Detection.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.img_size, args.fpn_channels) <= 0:
        parser.error("epochs, batch_size, img_size và fpn_channels phải dương.")
    if not 0 <= args.stop_after_epoch <= args.epochs:
        parser.error("stop_after_epoch phải từ 0 đến epochs.")
    if min(args.max_batches, args.eval_max_images, args.num_workers, args.warmup_iters) < 0:
        parser.error("Các giới hạn, num_workers và warmup_iters phải không âm.")
    if args.ms_max_size < 0 or not 0 <= args.ema_decay < 1 or not 0 <= args.swa_start < 1:
        parser.error("ms_max_size phải không âm; ema_decay và swa_start phải thuộc [0, 1).")
    return args


def build_optimizer(model, args):
    groups = []
    for module, factor in (
        (model.backbone, args.backbone_lr_factor),
        (model.fpn, args.fpn_lr_factor),
        (model.head, 1.0),
    ):
        groups.append({
            "params": [parameter for parameter in module.parameters() if parameter.requires_grad],
            "lr": args.lr * factor,
        })
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_epoch(model, loader, criterion, optimizer, scheduler, device,
                epoch, steps, warmup_iters, ema, swa, swa_scheduler, reference_recipe):
    model.train()
    cls_sum, reg_sum = 0.0, 0.0
    updates, skipped = 0, 0
    for index, (images, targets) in enumerate(islice(loader, steps)):
        step = (epoch - 1) * steps + index
        if step < warmup_iters:
            fraction = (step + 1) / warmup_iters
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * fraction
        elif step == warmup_iters and not reference_recipe:
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr

        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, deltas = model(images)
        anchors = model.get_anchors(images.shape[-2:], images.device)
        cls_loss, reg_loss = criterion(logits, deltas, anchors, targets)
        loss = cls_loss + reg_loss
        if not torch.isfinite(loss):
            print(f"Epoch {epoch} | Batch {index + 1}: loss khong huu han, bo qua.")
            skipped += 1
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        if not torch.isfinite(grad_norm):
            print(f"Epoch {epoch} | Batch {index + 1}: gradient khong huu han, bo qua.")
            skipped += 1
            continue
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)
        if swa is not None:
            swa.update_parameters(model)
        if swa_scheduler is not None and (reference_recipe or step >= warmup_iters):
            swa_scheduler.step()
        if step >= warmup_iters and (reference_recipe or swa_scheduler is None):
            scheduler.step()

        updates += 1
        cls_sum += cls_loss.item()
        reg_sum += reg_loss.item()
        if index % 50 == 0 or index + 1 == steps:
            print(f"Epoch {epoch} | Batch {index + 1}/{steps} | Loss {loss.item():.4f}", flush=True)
    if not updates:
        raise RuntimeError(f"Epoch {epoch} không có batch nào cập nhật trọng số.")
    return cls_sum / updates, reg_sum / updates, skipped


@torch.inference_mode()
def validate(model, loader, dataset, records, device, args):
    model.eval()
    detect = model.detect_tta if args.use_tta else model.detect
    predictions = []
    for images, targets in loader:
        outputs = detect(
            images.to(device), [target["orig_size"] for target in targets],
            conf_thresh=args.eval_conf, nms_thresh=args.eval_nms,
        )
        for target, output in zip(targets, outputs, strict=True):
            boxes = []
            for box, score, label in zip(
                output["boxes"].tolist(), output["scores"].tolist(), output["labels"].tolist(),
                strict=True,
            ):
                if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                    boxes.append({"class": dataset.classes[label], "confidence": score, "bbox": box})
            predictions.append({"image_id": target["image_id"], "boxes": boxes})
    return evaluate_map(predictions, records, dataset.classes)


def save_checkpoint(state, path):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def multiscale_sizes(base, maximum):
    """Các cỡ cách nhau 32 pixel, giới hạn trên mặc định là img_size."""
    low = max(32, round(base * 0.75 / 32) * 32)
    high = min(maximum or base, round(base * 1.25 / 32) * 32)
    sizes = list(range(low, high + 1, 32))
    if not sizes:
        raise ValueError("ms_max_size quá nhỏ cho khoảng multi-scale.")
    return sizes


def write_history(history, output):
    columns = {
        "epoch": "epoch", "cls_loss": "classification_loss", "reg_loss": "regression_loss",
        "mAP": "mAP", "per_class_AP": "per_class_ap", "lr": "lr", "time_seconds": "time_seconds",
        "train_seconds": "train_seconds", "val_seconds": "val_seconds",
        "skipped_batches": "skipped_batches", "raw_mAP": "raw_mAP",
        "ema_mAP": "ema_mAP", "swa_mAP": "swa_mAP", "weight_source": "weight_source",
    }
    data = {name: [row.get(key) for row in history] for name, key in columns.items()}
    temp = output / "training_history.tmp"
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(output / "training_history.json")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_set = DetectionDataset(
        args.train_data, args.image_dir, img_size=args.img_size, train=True,
        mixup=not args.no_mixup,
    )
    val_set = DetectionDataset(args.val_data, args.val_image_dir, img_size=args.img_size)
    if not len(train_set) or not len(val_set) or train_set.classes != val_set.classes:
        raise ValueError("Train/val phải có ảnh và cùng thứ tự lớp.")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda",
        drop_last=len(train_set) >= args.batch_size,
        # Tạo lại worker mỗi epoch khi cần cập nhật kích thước ảnh.
        persistent_workers=args.num_workers > 0 and not args.multi_scale,
    )
    val_count = min(args.eval_max_images or len(val_set), len(val_set))
    val_loader = DataLoader(
        Subset(val_set, range(val_count)), batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    steps = min(args.max_batches or len(train_loader), len(train_loader))

    config = {
        "num_classes": len(train_set.classes), "classes": train_set.classes,
        "backbone": args.backbone, "fpn_channels": args.fpn_channels, "img_size": args.img_size,
    }
    model = RetinaDetector(
        num_classes=config["num_classes"], backbone=args.backbone,
        pretrained=not args.no_pretrained and args.resume is None,
        fpn_channels=args.fpn_channels, img_size=args.img_size,
    ).to(device)
    counts = Counter(label for record in train_set.records for label in record["labels"])
    most_frequent = max(counts.values(), default=1)
    weights = [min(3.0, most_frequent / max(counts[i], 1)) for i in range(len(train_set.classes))]
    criterion = DetectionLoss(
        len(train_set.classes), reg_loss_type=args.reg_loss_type, class_weights=weights,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3) * steps, eta_min=1e-6,
    )
    ema = AveragedModel(model, avg_fn=get_ema_avg_fn(args.ema_decay)) if args.ema else None
    swa = AveragedModel(model) if args.swa else None
    if ema is not None:
        ema.requires_grad_(False)
        if args.reference_recipe:
            # EMA của bài mẫu bắt đầu lấy trung bình ngay ở lần cập nhật đầu.
            ema.n_averaged.fill_(1)
    if swa is not None:
        swa.requires_grad_(False)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5) if args.swa else None
    sizes = multiscale_sizes(args.img_size, args.ms_max_size) if args.multi_scale else []

    output = args.checkpoint_dir
    output.mkdir(parents=True, exist_ok=True)
    end_epoch = args.stop_after_epoch or args.epochs
    print(f"Device: {device} | Train images: {len(train_set)} | Validation images: {val_count}")
    print(f"Workers: {args.num_workers} | Batches/epoch: {steps} | Stop after epoch: {end_epoch}")
    print(f"Output: {output.resolve()}")
    history = []
    best_map = float("-inf")
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Cấu hình mô hình hoặc thứ tự lớp khác checkpoint.")
        if not {"optimizer", "scheduler", "history"} <= checkpoint.keys():
            raise ValueError("Dùng last.pth để tiếp tục huấn luyện.")
        if checkpoint.get("steps", steps) != steps:
            raise ValueError("Số batch mỗi epoch khác lần huấn luyện trước.")
        if checkpoint.get("reference_recipe", False) != args.reference_recipe:
            raise ValueError("Cấu hình reference_recipe khác lần huấn luyện trước.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        for key, average in (("ema", ema), ("swa", swa), ("swa_scheduler", swa_scheduler)):
            if average is not None and key in checkpoint:
                average.load_state_dict(checkpoint[key])
        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint["best_map"]
        history = checkpoint["history"]
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        del checkpoint
        print(f"Resume: epoch {start_epoch} | best mAP@0.5={best_map:.4f}")
    if start_epoch > end_epoch:
        print("Checkpoint da dat so epoch yeu cau.")
        return

    for epoch in range(start_epoch, end_epoch + 1):
        started = time.perf_counter()
        if sizes:
            train_set._epoch_img_size = random.choice(sizes)
            print(f"Multi-scale: {train_set._epoch_img_size} x {train_set._epoch_img_size}")
        use_swa = swa is not None and epoch > int(args.epochs * args.swa_start)
        cls, reg, skipped = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, steps, args.warmup_iters, ema,
            swa if use_swa else None, swa_scheduler if use_swa else None, args.reference_recipe,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_started = time.perf_counter()
        train_seconds = validation_started - started
        print(f"Epoch {epoch}: train {train_seconds:.1f}s, bat dau validation.", flush=True)
        candidates = [("raw", model)]
        if ema is not None and not args.reference_recipe:
            candidates.append(("ema", ema.module))
        if use_swa and not args.reference_recipe:
            candidates.append(("swa", swa.module))
        results = {
            source: validate(candidate, val_loader, val_set, val_set.records[:val_count], device, args)
            for source, candidate in candidates
        }
        if args.reference_recipe and use_swa and epoch == args.epochs:
            results["swa"] = validate(
                swa.module, val_loader, val_set, val_set.records[:val_count], device, args,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
        val_seconds = finished - validation_started
        score_source, selected = max(candidates, key=lambda entry: results[entry[0]][0])
        mean_ap, per_class = results[score_source]
        source = score_source
        if args.reference_recipe and ema is not None:
            source, selected = "ema", ema.module
        history.append({
            "epoch": epoch, "classification_loss": cls, "regression_loss": reg,
            "train_loss": cls + reg, "mAP": mean_ap, "per_class_ap": per_class,
            "lr": optimizer.param_groups[-1]["lr"], "time_seconds": finished - started,
            "train_seconds": train_seconds, "val_seconds": val_seconds,
            "skipped_batches": skipped, "weight_source": score_source,
            **{f"{name}_mAP": score for name, (score, _) in results.items()},
        })
        improved = mean_ap > best_map or (args.reference_recipe and mean_ap == best_map)
        best_map = max(best_map, mean_ap)
        state = {
            "model": model.state_dict(), "config": config, "epoch": epoch,
            "best_map": best_map, "weight_source": "raw",
        }
        if improved:
            save_checkpoint({
                **state, "model": selected.state_dict(), "weight_source": source,
                "score_source": score_source,
            }, output / "best.pth")
        if use_swa and epoch == args.epochs:
            save_checkpoint({
                **state, "model": swa.module.state_dict(),
                "best_map": results["swa"][0], "weight_source": "swa",
            }, output / "swa.pth")
        last = {
            **state, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history,
            "steps": steps, "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "reference_recipe": args.reference_recipe,
        }
        if device.type == "cuda":
            last["cuda_rng"] = torch.cuda.get_rng_state(device)
        for key, average in (("ema", ema), ("swa", swa), ("swa_scheduler", swa_scheduler)):
            if average is not None:
                last[key] = average.state_dict()
        save_checkpoint(last, output / "last.pth")
        write_history(history, output)
        scores = " | ".join(f"{name} mAP={score:.4f}" for name, (score, _) in results.items())
        print(f"Epoch {epoch}: loss={cls + reg:.4f} | {scores} | best={best_map:.4f} ({score_source} this epoch)")
        print(f"Train {train_seconds:.1f}s | Validation {val_seconds:.1f}s | "
              f"Total {finished - started:.1f}s | Da luu last.pth", flush=True)

    print("Hoan thanh buoc 4.4.")


if __name__ == "__main__":
    main()
