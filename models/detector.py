import sys
import os
import torch
import torch.nn as nn

# Thêm đường dẫn gốc để gọi các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone import ResNetBackbone
from models.fpn import FPN
from models.head import DetectionHead

class RetinaDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 1. Khởi tạo Đôi mắt (Backbone)
        self.backbone = ResNetBackbone()
        
        # 2. Khởi tạo Tháp đặc trưng (FPN)
        self.fpn = FPN(in_channels_list=self.backbone.out_channels, out_channels=256)
        
        # 3. Khởi tạo Đầu dự đoán (Head)
        self.head = DetectionHead(in_channels=256, num_anchors=9, num_classes=num_classes)

    def forward(self, images):
        """
        Luồng đi của dữ liệu khi gọi mô hình
        images: Tensor ảnh có kích thước [Batch, 3, Height, Width]
        """
        # Bước 1: Trích xuất đặc trưng C3, C4, C5
        c3, c4, c5 = self.backbone(images)
        
        # Bước 2: Trộn đặc trưng qua tháp P3 -> P7
        fpn_features = self.fpn([c3, c4, c5])
        
        # Bước 3: Đưa ra dự đoán cuối cùng
        cls_preds, reg_preds = self.head(fpn_features)
        
        return cls_preds, reg_preds

if __name__ == '__main__':
    # Kiểm tra đóng gói toàn bộ mô hình
    model = RetinaDetector(num_classes=5)
    
    # Giả lập 2 bức ảnh (Batch = 2) đưa vào cùng lúc
    dummy_images = torch.randn(2, 3, 512, 512)
    
    # Bấm nút chạy cỗ máy
    cls_preds, reg_preds = model(dummy_images)
    
    print("Lắp ráp cỗ máy RetinaDetector hoàn tất!")
    print(f"Kích thước đầu vào: {dummy_images.shape}")
    print(f"Kích thước tensor Phân lớp (Class): {cls_preds.shape}")
    print(f"Kích thước tensor Hộp bao (Box): {reg_preds.shape}")