import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

CLASS_MAP = {"person": 0, "car": 1, "dog": 2, "cat": 3, "chair": 4}

class ObjectDetectionDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=512):
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.images = {img['id']: img for img in data['images']}
        self.annotations = data.get('annotations', [])
        
        # Nhóm các hộp bao theo ID của ảnh
        self.img_to_anns = {img_id: [] for img_id in self.images.keys()}
        for ann in self.annotations:
            if ann['image_id'] in self.img_to_anns:
                self.img_to_anns[ann['image_id']].append(ann)
                
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        anns = self.img_to_anns[img_id]

        img_path = os.path.join(self.img_dir, os.path.basename(img_info['file_name']))
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # 1. Letterbox resize (giữ tỷ lệ, viền xám)
        scale = self.img_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        pad_left, pad_top = pad_w // 2, pad_h // 2
        pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top
        
        # Điền viền màu xám (114) giống cấu trúc RetinaNet chuẩn
        image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=114)
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 2. Cập nhật lại tọa độ hộp bao cho khớp với ảnh đã đệm viền
        target = []
        for ann in anns:
            xmin, ymin, xmax, ymax = ann['bbox']
            # Chỉnh theo tỷ lệ
            xmin, ymin = xmin * scale, ymin * scale
            xmax, ymax = xmax * scale, ymax * scale
            # Dịch chuyển theo padding
            xmin, xmax = xmin + pad_left, xmax + pad_left
            ymin, ymax = ymin + pad_top, ymax + pad_top
            
            label_idx = CLASS_MAP[ann['class']]
            target.append([label_idx, xmin, ymin, xmax, ymax, 0]) # Số 0 là difficult flag

        # Tensor chứa target có dạng [N, 6]
        target_tensor = torch.tensor(target, dtype=torch.float32) if len(target) > 0 else torch.zeros((0, 6))

        return image_tensor, target_tensor

def collate_fn(batch):
    """Giúp PyTorch đóng gói các ảnh có số lượng đối tượng (hộp bao) khác nhau vào chung một lô (batch)"""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets