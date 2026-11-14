"""Đọc ảnh và nhãn, biến đổi đồng thời ảnh và hộp bao."""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


Box = tuple[float, float, float, float]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_to_tensor(image):
    """Đổi ảnh RGB sang tensor CHW và chuẩn hóa ImageNet."""
    import torch

    pixels = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


def resize_sample(image: Image.Image, boxes: list[Box], size: int):
    """Resize về ảnh vuông và đổi tọa độ box theo từng trục."""
    scale_x = size / image.width
    scale_y = size / image.height
    resized_boxes = [
        (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        for x1, y1, x2, y2 in boxes
    ]
    return image.resize((size, size), Image.Resampling.BILINEAR), resized_boxes


def flip_sample(image: Image.Image, boxes: list[Box]):
    """Lật ngang ảnh và phản chiếu box theo trục dọc ở giữa ảnh."""
    width = image.width
    flipped_boxes = [(width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes]
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), flipped_boxes


def load_annotations(json_path):
    """Đọc nhãn và nhóm box theo ID ảnh, kể cả ảnh không có box."""
    with Path(json_path).open(encoding="utf-8") as file:
        data = json.load(file)
    classes = data["classes"]
    class_indices = {name: index for index, name in enumerate(classes)}
    records = {
        image["id"]: {**image, "boxes": [], "labels": []}
        for image in data["images"]
    }
    for annotation in data["annotations"]:
        record = records[annotation["image_id"]]
        record["boxes"].append(tuple(map(float, annotation["bbox"])))
        record["labels"].append(class_indices[annotation["class"]])
    return classes, list(records.values())


def crop_sample(image, boxes, labels):
    """Cắt ngẫu nhiên và giữ các vật thể có tâm nằm trong vùng cắt."""
    if not boxes:
        return image, boxes, labels
    for _ in range(15):
        scale = random.uniform(0.5, 1.0)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        left = random.randint(0, image.width - width)
        top = random.randint(0, image.height - height)
        right, bottom = left + width, top + height
        selected_boxes, selected_labels = [], []
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            if left <= (x1 + x2) / 2 < right and top <= (y1 + y2) / 2 < bottom:
                selected_boxes.append((
                    max(x1, left) - left, max(y1, top) - top,
                    min(x2, right) - left, min(y2, bottom) - top,
                ))
                selected_labels.append(label)
        if selected_boxes:
            return image.crop((left, top, right, bottom)), selected_boxes, selected_labels
    return image, boxes, labels


class DetectionDataset:
    """Đọc mẫu theo chỉ số; DataLoader gọi __len__ và __getitem__."""

    def __init__(self, json_path, image_dir, img_size=512, train=False,
                 color_jitter=True, random_crop=True, hflip=True,
                 multi_scale=False, mixup=True):
        if img_size <= 0:
            raise ValueError("Kích thước ảnh phải dương.")
        self.classes, self.records = load_annotations(json_path)
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.train = train
        self.color_jitter = color_jitter and train
        self.random_crop = random_crop and train
        self.hflip = hflip and train
        self.multi_scale = multi_scale and train
        self.mixup = mixup and train
        self.multi_scale_sizes = list(range(320, 705, 32))
        self._epoch_img_size = None

    def __len__(self):
        return len(self.records)

    def read_original(self, index):
        record = self.records[index]
        name = Path(record.get("file_name", record["id"]))
        candidates = (
            self.image_dir / name.name,
            self.image_dir / name,
            self.image_dir / Path(record["id"]).name,
        )
        path = next((path for path in candidates if path.is_file()), candidates[0])
        with Image.open(path) as source:
            image = source.convert("RGB")
        return image, {
            "image_id": record["id"],
            "orig_size": (image.height, image.width),
            "boxes": list(record["boxes"]),
            "labels": list(record["labels"]),
        }

    def _prepare(self, image, target, size):
        import torch

        boxes, labels = target["boxes"], target["labels"]
        if self.color_jitter and random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
            hue, saturation, value = image.convert("HSV").split()
            shift = int(random.uniform(-0.1, 0.1) * 255)
            hue = hue.point([(level + shift) % 256 for level in range(256)])
            image = Image.merge("HSV", (hue, saturation, value)).convert("RGB")
            if random.random() < 0.1:
                image = ImageOps.grayscale(image).convert("RGB")
        if self.random_crop and random.random() < 0.5:
            image, boxes, labels = crop_sample(image, boxes, labels)
        if self.hflip and random.random() < 0.5:
            image, boxes = flip_sample(image, boxes)
        if self.train and random.random() < 0.15:
            height = image.height
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            boxes = [(x1, height - y2, x2, height - y1) for x1, y1, x2, y2 in boxes]

        image, boxes = resize_sample(image, boxes, size)
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp_(0, size)
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        valid = ((box_tensor[:, 2:] - box_tensor[:, :2]) > 1).all(dim=1)
        tensor = image_to_tensor(image)
        if self.train and random.random() < 0.2:
            erase_patch(tensor)
        return tensor, box_tensor[valid], label_tensor[valid]

    def _mosaic(self, image, target, size):
        import torch
        from torch.nn.functional import interpolate

        cut_x = random.randint(size // 4, size * 3 // 4)
        cut_y = random.randint(size // 4, size * 3 // 4)
        bounds = ((0, 0, cut_x, cut_y), (cut_x, 0, size, cut_y),
                  (0, cut_y, cut_x, size), (cut_x, cut_y, size, size))
        samples = [(image, target)]
        samples.extend(self.read_original(random.randrange(len(self))) for _ in range(3))
        canvas = torch.zeros((3, size, size))
        all_boxes, all_labels = [], []
        for (sample, annotation), (left, top, right, bottom) in zip(samples, bounds):
            width, height = right - left, bottom - top
            tile_size = max(32, min(width, height))
            tile, boxes, labels = self._prepare(sample, annotation, tile_size)
            canvas[:, top:bottom, left:right] = interpolate(
                tile[None], size=(height, width), mode="bilinear", align_corners=False
            )[0]
            boxes[:, 0::2].mul_(width / tile_size).add_(left).clamp_(left, right)
            boxes[:, 1::2].mul_(height / tile_size).add_(top).clamp_(top, bottom)
            valid = ((boxes[:, 2:] - boxes[:, :2]) > 2).all(dim=1)
            all_boxes.append(boxes[valid])
            all_labels.append(labels[valid])
        return canvas, torch.cat(all_boxes), torch.cat(all_labels)

    def __getitem__(self, index):
        import torch

        image, target = self.read_original(index)
        size = self._epoch_img_size
        if size is None:
            size = random.choice(self.multi_scale_sizes) if self.multi_scale else self.img_size
        if self.train and size >= 4 and len(self) >= 4 and random.random() < 0.4:
            tensor, boxes, labels = self._mosaic(image, target, size)
        else:
            tensor, boxes, labels = self._prepare(image, target, size)
        if self.mixup and len(self) > 1 and random.random() < 0.15:
            second_image, second_target = self.read_original(random.randrange(len(self)))
            second_tensor, second_boxes, second_labels = self._prepare(
                second_image, second_target, size
            )
            weight = random.uniform(0.4, 0.6)
            tensor = tensor * weight + second_tensor * (1 - weight)
            boxes = torch.cat((boxes, second_boxes))
            labels = torch.cat((labels, second_labels))
        return tensor, {**target, "boxes": boxes, "labels": labels}


def erase_patch(tensor):
    """Đặt một vùng nhỏ về 0 sau chuẩn hóa (màu trung bình ImageNet)."""
    _, height, width = tensor.shape
    for _ in range(10):
        area = height * width * random.uniform(0.02, 0.2)
        ratio = random.uniform(0.3, 1 / 0.3)
        patch_height = round((area * ratio) ** 0.5)
        patch_width = round((area / ratio) ** 0.5)
        if 0 < patch_height < height and 0 < patch_width < width:
            top = random.randint(0, height - patch_height)
            left = random.randint(0, width - patch_width)
            tensor[:, top:top + patch_height, left:left + patch_width] = 0
            return


def collate_fn(batch):
    """Gom ảnh cùng kích thước và giữ riêng danh sách nhãn từng ảnh."""
    import torch

    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
