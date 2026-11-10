
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_annotations(json_path):
    with Path(json_path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    classes = document["classes"]
    indices = {name: number for number, name in enumerate(classes)}
    records = {
        item["id"]: {
            "id": item["id"],
            "file_name": item.get("file_name", item["id"]),
            "width": item["width"],
            "height": item["height"],
            "boxes": [],
            "labels": [],
        }
        for item in document["images"]
    }
    for annotation in document.get("annotations", []):
        record = records.get(annotation["image_id"])
        if record is not None:
            record["boxes"].append([float(x) for x in annotation["bbox"]])
            record["labels"].append(indices[annotation["class"]])
    return classes, list(records.values())


def _random_erase(image, sl=0.02, sh=0.2, r1=0.3):
    _, height, width = image.shape
    for _ in range(10):
        area = random.uniform(sl, sh) * height * width
        ratio = random.uniform(r1, 1 / r1)
        erase_height = round((area * ratio) ** 0.5)
        erase_width = round((area / ratio) ** 0.5)
        if 0 < erase_height < height and 0 < erase_width < width:
            left = random.randint(0, width - erase_width)
            top = random.randint(0, height - erase_height)
            result = image.clone()
            result[:, top:top + erase_height, left:left + erase_width] = 0
            return result
    return image


class DetectionDataset(Dataset):
    def __init__(self, json_path, image_dir, img_size=512, train=False,
                 color_jitter=True, random_crop=True, hflip=True,
                 multi_scale=False, mixup=True):
        self.classes, self.records = load_annotations(json_path)
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.train = train
        self.color_jitter = train and color_jitter
        self.random_crop = train and random_crop
        self.hflip = train and hflip
        self.multi_scale = train and multi_scale
        self.mixup = train and mixup
        self.multi_scale_sizes = list(range(320, 705, 32))
        self._epoch_img_size = None

    def __len__(self):
        return len(self.records)

    def _get_img_size(self):
        if self._epoch_img_size is not None:
            return self._epoch_img_size
        if self.multi_scale:
            return random.choice(self.multi_scale_sizes)
        return self.img_size

    def _resolve_path(self, record):
        filename = Path(record["file_name"])
        options = (
            self.image_dir / filename.name,
            self.image_dir / filename,
            self.image_dir / Path(record["id"]).name,
        )
        return next((path for path in options if path.exists()), options[0])

    def _random_crop(self, image, boxes, labels):
        if boxes.shape[0] == 0:
            return image, boxes, labels
        width, height = image.size
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        for _ in range(15):
            fraction = random.uniform(0.5, 1)
            crop_width = max(1, round(width * fraction))
            crop_height = max(1, round(height * fraction))
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            keep = (
                (centers[:, 0] >= left)
                & (centers[:, 0] < left + crop_width)
                & (centers[:, 1] >= top)
                & (centers[:, 1] < top + crop_height)
            )
            if keep.any():
                cropped_boxes = boxes[keep].clone()
                cropped_boxes[:, 0::2] = (
                    cropped_boxes[:, 0::2] - left
                ).clamp(0, crop_width)
                cropped_boxes[:, 1::2] = (
                    cropped_boxes[:, 1::2] - top
                ).clamp(0, crop_height)
                return (
                    image.crop((left, top, left + crop_width, top + crop_height)),
                    cropped_boxes,
                    labels[keep].clone(),
                )
        return image, boxes, labels

    def _load_and_augment(self, record, size):
        with Image.open(self._resolve_path(record)) as source:
            image = source.convert("RGB")
        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(record["labels"], dtype=torch.long)

        if self.color_jitter and random.random() < 0.8:
            image = vision.adjust_brightness(image, random.uniform(0.5, 1.5))
            image = vision.adjust_contrast(image, random.uniform(0.5, 1.5))
            image = vision.adjust_saturation(image, random.uniform(0.5, 1.5))
            image = vision.adjust_hue(image, random.uniform(-0.1, 0.1))
            if random.random() < 0.1:
                image = vision.to_grayscale(image, num_output_channels=3)

        if self.random_crop and random.random() < 0.5:
            image, boxes, labels = self._random_crop(image, boxes, labels)
        if self.hflip and random.random() < 0.5:
            image = vision.hflip(image)
            boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
        if self.train and random.random() < 0.15:
            image = vision.vflip(image)
            boxes[:, [1, 3]] = image.height - boxes[:, [3, 1]]

        width, height = image.size
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        boxes *= torch.tensor([size / width, size / height] * 2)
        boxes.clamp_(0, size)
        valid = ((boxes[:, 2:] - boxes[:, :2]) > 1).all(dim=1)
        boxes, labels = boxes[valid], labels[valid]
        tensor = vision.normalize(
            vision.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD
        )
        if self.train and random.random() < 0.2:
            tensor = _random_erase(tensor)
        return tensor, boxes, labels

    def _mosaic(self, index, size):
        selected = [index] + random.choices(range(len(self)), k=3)
        cut_x = random.randint(size // 4, 3 * size // 4)
        cut_y = random.randint(size // 4, 3 * size // 4)
        rectangles = [
            (0, 0, cut_x, cut_y),
            (cut_x, 0, size, cut_y),
            (0, cut_y, cut_x, size),
            (cut_x, cut_y, size, size),
        ]
        canvas = torch.zeros((3, size, size))
        box_parts, label_parts = [], []
        for source_index, (left, top, right, bottom) in zip(selected, rectangles):
            width, height = right - left, bottom - top
            tile_size = max(min(width, height), 32)
            tile, boxes, labels = self._load_and_augment(
                self.records[source_index], tile_size
            )
            canvas[:, top:bottom, left:right] = functional.interpolate(
                tile.unsqueeze(0), size=(height, width),
                mode="bilinear", align_corners=False,
            )[0]
            boxes = boxes.clone()
            boxes[:, 0::2] = (
                boxes[:, 0::2] * width / tile_size + left
            ).clamp(left, right)
            boxes[:, 1::2] = (
                boxes[:, 1::2] * height / tile_size + top
            ).clamp(top, bottom)
            valid = ((boxes[:, 2:] - boxes[:, :2]) > 2).all(dim=1)
            box_parts.append(boxes[valid])
            label_parts.append(labels[valid])
        return canvas, torch.cat(box_parts), torch.cat(label_parts)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self._resolve_path(record)) as original:
            original_width, original_height = original.size
        size = self._get_img_size()
        if self.train and len(self) > 3 and random.random() < 0.4:
            tensor, boxes, labels = self._mosaic(index, size)
        else:
            tensor, boxes, labels = self._load_and_augment(record, size)

        if self.mixup and len(self) > 1 and random.random() < 0.15:
            other = self.records[random.randrange(len(self))]
            other_tensor, other_boxes, other_labels = self._load_and_augment(other, size)
            weight = random.uniform(0.4, 0.6)
            tensor = weight * tensor + (1 - weight) * other_tensor
            boxes = torch.cat((boxes, other_boxes))
            labels = torch.cat((labels, other_labels))
        return tensor, {
            "boxes": boxes,
            "labels": labels,
            "image_id": record["id"],
            "orig_size": (original_height, original_width),
        }


def collate_fn(batch):
    images, targets = zip(*batch)
    if len({tuple(image.shape) for image in images}) != 1:
        raise ValueError(
            "Multi-scale cần cùng kích thước trong một batch. "
            "Đặt dataset._epoch_img_size trước khi tạo iterator DataLoader."
        )
    return torch.stack(images), list(targets)
