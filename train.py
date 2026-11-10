import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from models.detector import RetinaDetector
from models.loss import DetectionLoss
from utils.dataset import DetectionDataset, collate_fn
from utils.metrics import evaluate_map


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_data", "val_data", "image_dir", "val_image_dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint_dir", default="./models/")
    parser.add_argument(
        "--backbone", default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "tiny", "tiny_0.25",
                 "tiny_0.5", "tiny_0.75", "tiny_1.0", "tiny_1.25", "tiny_1.5"],
    )
    parser.add_argument("--fpn_channels", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval_conf", type=float, default=0.25)
    parser.add_argument("--eval_nms", type=float, default=0.4)
    parser.add_argument("--eval_max_images", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--ms_max_size", type=int, default=0)
    parser.add_argument(
        "--reg_loss_type", default="giou", choices=["smooth_l1", "giou", "ciou"]
    )
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.05)
    parser.add_argument("--fpn_lr_factor", type=float, default=0.30)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa_start", type=float, default=0.75)
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Khởi tạo ResNet ngẫu nhiên. Tiny luôn tự khởi tạo.",
    )
    parser.add_argument(
        "--max_batches", type=int, default=0,
        help="Giới hạn batch mỗi epoch để kiểm tra; 0 dùng toàn bộ.",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.fpn_channels) < 1:
        parser.error("epochs, batch_size và fpn_channels phải dương.")
    if args.img_size < 128 or args.img_size % 128:
        parser.error("img_size phải là bội số của 128.")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr phải dương, weight_decay không âm.")
    if min(args.backbone_lr_factor, args.fpn_lr_factor) <= 0:
        parser.error("Các hệ số learning rate phải dương.")
    if min(args.num_workers, args.warmup_iters, args.eval_max_images,
           args.max_batches, args.ms_max_size) < 0:
        parser.error("Các giới hạn và num_workers không được âm.")
    if not 0 <= args.eval_conf <= 1 or not 0 <= args.eval_nms <= 1:
        parser.error("Ngưỡng confidence và NMS phải trong [0, 1].")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.swa_start < 1:
        parser.error("ema_decay và swa_start phải trong [0, 1).")
    if args.resume and not Path(args.resume).is_file():
        parser.error("Không tìm thấy checkpoint resume.")
    return args


def _build_ms_sizes(base, ms_max):
    lower = max(256, round(base * 0.75 / 32) * 32)
    upper = min(ms_max if ms_max > 0 else base, round(base * 1.25 / 32) * 32)
    if upper < lower:
        raise ValueError("ms_max_size nhỏ hơn cận dưới multi-scale.")
    return list(range(lower, upper + 1, 32))


def _build_param_groups(model, base_lr, backbone_factor, fpn_factor, weight_decay):
    grouped = []
    for module, factor in (
        (model.backbone, backbone_factor), (model.fpn, fpn_factor), (model.head, 1.0)
    ):
        parameters = [parameter for parameter in module.parameters()
                      if parameter.requires_grad]
        if parameters:
            grouped.append({
                "params": parameters, "lr": base_lr * factor,
                "weight_decay": weight_decay,
            })
    return grouped


@torch.no_grad()
def evaluate(model, loader, dataset, args, use_tta=False):
    base = model.module if isinstance(model, AveragedModel) else model
    base.eval()
    predictions = []
    seen = set()
    infer = base.detect_tta if use_tta else base.detect

    for images, targets in loader:
        if args.eval_max_images:
            remaining = args.eval_max_images - len(predictions)
            if remaining <= 0:
                break
            images = images[:remaining]
            targets = targets[:remaining]

        images = images.to(args.device, non_blocking=True)
        results = infer(
            images, [target["orig_size"] for target in targets],
            conf_thresh=args.eval_conf, nms_thresh=args.eval_nms,
        )
        for target, result in zip(targets, results):
            detections = []
            for box, score, label in zip(
                result["boxes"].cpu().tolist(),
                result["scores"].cpu().tolist(),
                result["labels"].cpu().tolist(),
            ):
                if not all(math.isfinite(value) for value in [*box, score]):
                    continue
                if box[2] - box[0] <= 1 or box[3] - box[1] <= 1:
                    continue
                detections.append({
                    "class": dataset.classes[label], "confidence": score, "bbox": box
                })
            predictions.append({"image_id": target["image_id"], "boxes": detections})
            seen.add(target["image_id"])

    records = [record for record in dataset.records if record["id"] in seen]
    return evaluate_map(predictions, records, dataset.classes)


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.ema._anchor_cache.clear()
        self.decay = decay
        self.ema.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if value.is_floating_point():
                value.mul_(self.decay).add_(
                    source[name].detach().to(value), alpha=1 - self.decay
                )
            else:
                value.copy_(source[name])


def _save(path, checkpoint):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _save_history(path, history):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(path)


def _inference_checkpoint(model, config, epoch, score, kind):
    return {
        "model": model.state_dict(), "config": config,
        "epoch": epoch, "best_map": score, "weight_source": kind,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng.")

    destination = Path(args.checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume else None
    )
    train_dataset = DetectionDataset(
        args.train_data, args.image_dir, img_size=args.img_size, train=True,
        multi_scale=args.multi_scale, mixup=not args.no_mixup,
    )
    val_dataset = DetectionDataset(
        args.val_data, args.val_image_dir, img_size=args.img_size, train=False
    )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Thứ tự lớp train/validation khác nhau.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
        pin_memory=device.type == "cuda", persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    if not len(train_loader) or not len(val_dataset):
        raise ValueError("Train hoặc validation không đủ ảnh.")

    config = {
        "num_classes": len(train_dataset.classes), "classes": train_dataset.classes,
        "backbone": args.backbone, "fpn_channels": args.fpn_channels,
        "img_size": args.img_size,
    }
    # Các tùy chọn này ảnh hưởng trạng thái tối ưu và cách đọc batch khi resume.
    settings = {
        name: getattr(args, name)
        for name in (
            "batch_size", "lr", "weight_decay", "warmup_iters",
            "backbone_lr_factor", "fpn_lr_factor", "reg_loss_type",
            "no_mixup", "multi_scale", "ms_max_size", "ema", "ema_decay",
            "swa", "swa_start", "max_batches", "no_pretrained",
        )
    }
    if checkpoint is not None:
        if checkpoint.get("config") != config:
            raise ValueError("Cấu hình model không khớp checkpoint.")
        if checkpoint.get("training_settings") != settings:
            raise ValueError("Thiết lập train không khớp checkpoint resume.")
        if "optimizer" not in checkpoint:
            raise ValueError("Resume cần last.pth chứa trạng thái optimizer.")

    model = RetinaDetector(
        num_classes=len(train_dataset.classes), backbone=args.backbone,
        pretrained=checkpoint is None and not args.no_pretrained,
        fpn_channels=args.fpn_channels, img_size=args.img_size,
    ).to(device)

    counts = [0] * len(train_dataset.classes)
    for record in train_dataset.records:
        for label in record["labels"]:
            counts[label] += 1
    largest = max(max(counts), 1)
    class_weights = [
        min(largest / max(count, 1), 3.0) for count in counts
    ]
    criterion = DetectionLoss(
        num_classes=len(counts), reg_loss_type=args.reg_loss_type,
        class_weights=class_weights,
    ).to(device)

    groups = _build_param_groups(
        model, args.lr, args.backbone_lr_factor,
        args.fpn_lr_factor, args.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups)
    base_lrs = [group["lr"] for group in groups]
    total_batches = len(train_loader)
    cycle_steps = max(1, args.epochs // 3) * total_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cycle_steps, T_mult=1, eta_min=1e-6
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    swa_model = AveragedModel(model) if args.swa else None
    swa_scheduler = (
        SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5, anneal_strategy="cos")
        if args.swa else None
    )

    start_epoch = 0
    global_step = 0
    best_map = -1.0
    best_source = None
    history = {name: [] for name in (
        "epoch", "cls_loss", "reg_loss", "mAP", "per_class_AP",
        "lr", "time_seconds", "raw_mAP", "ema_mAP", "skipped_batches"
    )}
    swa_start_epoch = int(args.epochs * args.swa_start)
    lr_multiplier = 1.0

    if checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_map = checkpoint["best_map"]
        best_source = checkpoint["best_source"]
        history = checkpoint["history"]
        base_lrs = checkpoint["base_lrs"]
        swa_start_epoch = checkpoint["swa_start_epoch"]
        lr_multiplier = checkpoint.get("lr_multiplier", 1.0)
        if ema is not None:
            ema.ema.load_state_dict(checkpoint["ema"])
        if swa_model is not None:
            swa_model.load_state_dict(checkpoint["swa"])
            swa_scheduler.load_state_dict(checkpoint["swa_scheduler"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint

    if start_epoch >= args.epochs:
        raise ValueError("epochs phải lớn hơn số epoch đã hoàn thành.")

    ms_sizes = _build_ms_sizes(args.img_size, args.ms_max_size) if args.multi_scale else []
    # MaxPool của Tiny làm tròn xuống: dùng bội 128 để khớp anchor.
    if args.backbone.startswith("tiny") and ms_sizes:
        ms_sizes = [size for size in ms_sizes if size % 128 == 0]
        if not ms_sizes:
            raise ValueError("Multi-scale Tiny cần ít nhất một kích thước chia hết 128.")
    train_dataset.multi_scale_sizes = ms_sizes

    print(f"Train: {len(train_dataset)}, validation: {len(val_dataset)}", flush=True)
    print("Class weights:", dict(zip(train_dataset.classes, class_weights)), flush=True)
    print("Multi-scale:", ms_sizes, flush=True)
    consecutive_invalid = 0

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        train_dataset._epoch_img_size = (
            random.choice(ms_sizes) if args.multi_scale else None
        )
        if args.multi_scale:
            print("Kích thước epoch:", train_dataset._epoch_img_size, flush=True)
        model.train()
        sums = [0.0, 0.0]
        successful = 0
        skipped = 0
        swa_active = swa_model is not None and epoch >= swa_start_epoch

        for iteration, (images, targets) in enumerate(train_loader):
            if args.max_batches and iteration >= args.max_batches:
                break
            # Chỉ một scheduler điều khiển LR. Warmup có ưu tiên cao nhất.
            if not swa_active:
                if global_step < args.warmup_iters:
                    factor = (global_step + 1) / max(args.warmup_iters, 1)
                    rates = [rate * factor for rate in base_lrs]
                else:
                    scheduler.step(global_step - args.warmup_iters)
                    rates = scheduler.get_last_lr()
                for group, rate in zip(optimizer.param_groups, rates):
                    group["lr"] = rate * lr_multiplier

            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, deltas = model(images)
            anchors = model.get_anchors(images.shape[-2:], device)
            cls_loss, reg_loss = criterion(logits, deltas, anchors, targets)
            loss = cls_loss + reg_loss
            valid_update = bool(torch.isfinite(loss).item())

            if valid_update:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0, error_if_nonfinite=False
                )
                valid_update = bool(torch.isfinite(norm).item())

            global_step += 1
            if not valid_update:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                consecutive_invalid += 1
                print(f"Bỏ batch không hữu hạn: epoch {epoch + 1}, batch {iteration + 1}",
                      flush=True)
                if consecutive_invalid >= 10:
                    lr_multiplier *= 0.5
                    for group in optimizer.param_groups:
                        group["lr"] = max(group["lr"] * 0.5, 1e-7)
                    consecutive_invalid = 0
                continue

            consecutive_invalid = 0
            optimizer.step()
            if ema is not None:
                ema.update(model)
            if swa_active:
                swa_model.update_parameters(model)
            sums[0] += cls_loss.item()
            sums[1] += reg_loss.item()
            successful += 1
            if iteration % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs}, batch {iteration + 1}/{total_batches}, "
                    f"cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f}, "
                    f"lr_head={optimizer.param_groups[-1]['lr']:.2e}", flush=True
                )

        if not successful:
            raise RuntimeError("Epoch không có cập nhật hợp lệ, dừng để kiểm tra.")
        if swa_active:
            swa_scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        raw_map, raw_ap = evaluate(model, val_loader, val_dataset, args, args.use_tta)
        candidates = [("raw", model, raw_map, raw_ap)]
        ema_map = None
        if ema is not None:
            ema_map, ema_ap = evaluate(ema.ema, val_loader, val_dataset, args, args.use_tta)
            candidates.append(("ema", ema.ema, ema_map, ema_ap))
        kind, selected_model, selected_map, per_class = max(
            candidates, key=lambda item: item[2]
        )
        if selected_map >= best_map:
            best_map, best_source = selected_map, kind
            _save(
                destination / "best.pth",
                _inference_checkpoint(selected_model, config, epoch, selected_map, kind),
            )

        record = {
            "epoch": epoch, "cls_loss": sums[0] / successful,
            "reg_loss": sums[1] / successful, "mAP": selected_map,
            "per_class_AP": per_class, "lr": optimizer.param_groups[-1]["lr"],
            "time_seconds": time.perf_counter() - started,
            "raw_mAP": raw_map, "ema_mAP": ema_map, "skipped_batches": skipped,
        }
        for name, value in record.items():
            history[name].append(value)

        state = {
            "model": model.state_dict(), "config": config, "epoch": epoch,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_map": best_map, "best_source": best_source, "history": history,
            "global_step": global_step, "base_lrs": base_lrs,
            "training_settings": settings, "swa_start_epoch": swa_start_epoch,
            "lr_multiplier": lr_multiplier, "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "ema": ema.ema.state_dict() if ema is not None else None,
            "swa": swa_model.state_dict() if swa_model is not None else None,
            "swa_scheduler": swa_scheduler.state_dict() if swa_scheduler is not None else None,
        }
        _save(destination / "last.pth", state)
        _save_history(destination / "training_history.json", history)
        print(
            f"Hoàn thành epoch {epoch + 1}: raw mAP={raw_map:.4f}, "
            f"EMA={ema_map}, best={best_map:.4f} ({best_source}), "
            f"{record['time_seconds']:.0f}s", flush=True,
        )

    if swa_model is not None and int(swa_model.n_averaged.item()) > 0:
        # Frozen BN và GroupNorm không cần update_bn.
        swa_map, swa_ap = evaluate(swa_model, val_loader, val_dataset, args, args.use_tta)
        swa_checkpoint = _inference_checkpoint(
            swa_model.module, config, args.epochs - 1, swa_map, "swa"
        )
        swa_checkpoint["per_class_AP"] = swa_ap
        _save(destination / "swa.pth", swa_checkpoint)
        if swa_map > best_map:
            best_map, best_source = swa_map, "swa"
            _save(destination / "best.pth", swa_checkpoint)
            state["best_map"] = best_map
            state["best_source"] = best_source
            _save(destination / "last.pth", state)
        print(f"SWA mAP={swa_map:.4f}", flush=True)
    print(f"Hoàn thành. Best mAP={best_map:.4f}, nguồn={best_source}", flush=True)


if __name__ == "__main__":
    main()
