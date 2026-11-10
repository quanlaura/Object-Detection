"""Đọc dữ liệu object detection và chuẩn bị batch cho PyTorch."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# Thông số chuẩn hóa đầu vào cho backbone ResNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DetectionDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        image_size=512,
        augment=False,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        annotation_path = (
            self.data_dir / "annotations" / f"{split}.json"
        )

        with annotation_path.open(encoding="utf-8") as file:
            data = json.load(file)

        self.images = data["images"]
        self.classes = data["classes"]

        # Ví dụ: "person" -> 0, "car" -> 1.
        self.class_to_index = {
            name: index
            for index, name in enumerate(self.classes)
        }

        # Gom các vật thể theo ảnh.
        self.annotations_by_image = {}

        for annotation in data["annotations"]:
            image_id = annotation["image_id"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(annotation)

    def __len__(self):
        """Cho PyTorch biết dataset có bao nhiêu ảnh."""
        return len(self.images)

    def __getitem__(self, index):
        """Đọc và xử lý một ảnh."""
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = self.data_dir / image_info["file_name"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        original_width, original_height = image.size
        annotations = self.annotations_by_image.get(image_id, [])

        # Dữ liệu này lưu box dạng [x1, y1, x2, y2].
        boxes = torch.tensor(
            [annotation["bbox"] for annotation in annotations],
            dtype=torch.float32,
        ).reshape(-1, 4)

        labels = torch.tensor(
            [
                self.class_to_index[annotation["class"]]
                for annotation in annotations
            ],
            dtype=torch.long,
        )

        # Resize ảnh về kích thước cố định.
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        # Đổi tọa độ box theo kích thước ảnh mới.
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height

        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y

        # Augmentation chỉ bật khi huấn luyện.
        # Khi lật ảnh, phải lật cả box.
        if self.augment and torch.rand(()).item() < 0.5:
            image = TF.hflip(image)

            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()

            boxes[:, 0] = self.image_size - old_x2
            boxes[:, 2] = self.image_size - old_x1

        # Giới hạn box trong biên ảnh.
        boxes.clamp_(min=0, max=self.image_size)

        # Bỏ box có chiều rộng hoặc chiều cao bằng 0.
        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes = boxes[valid]
        labels = labels[valid]

        # PIL image -> tensor [3, H, W], giá trị từ 0 đến 1.
        image_tensor = TF.to_tensor(image)

        # Chuẩn hóa từng kênh màu.
        image_tensor = TF.normalize(
            image_tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "original_size": (original_height, original_width),
        }

        return image_tensor, target


def collate_fn(batch):
    """Ghép nhiều ảnh, giữ riêng nhãn của từng ảnh."""
    images, targets = zip(*batch)

    images = torch.stack(images, dim=0)
    targets = list(targets)

    return images, targets