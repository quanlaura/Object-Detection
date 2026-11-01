import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9, num_classes=5):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # 1. Nhánh dự đoán Phân lớp (Classification)
        # Sử dụng các lớp tích chập liên tiếp để học cách nhận diện đối tượng
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Lớp xuất kết quả: Tại mỗi điểm ảnh có (9 anchor * 5 class) = 45 con số dự đoán
        self.cls_out = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

        # 2. Nhánh dự đoán Hộp bao (Regression/Bounding Box)
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Lớp xuất kết quả: Tại mỗi điểm ảnh có (9 anchor * 4 tọa độ) = 36 con số dự đoán (dx, dy, dw, dh)
        self.reg_out = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, fpn_features):
        """
        fpn_features: danh sách chứa các bản đồ đặc trưng [P3, P4, P5, P6, P7]
        """
        cls_outputs = []
        reg_outputs = []

        # Chạy từng tầng của tháp đặc trưng qua Đầu dự đoán
        for feature in fpn_features:
            # Dự đoán phân lớp
            cls_x = self.cls_convs(feature)
            cls_pred = self.cls_out(cls_x)  # Kích thước: [Batch, 45, H, W]

            # Dự đoán hộp bao
            reg_x = self.reg_convs(feature)
            reg_pred = self.reg_out(reg_x)  # Kích thước: [Batch, 36, H, W]

            # --- BƯỚC ĐỊNH DẠNG LẠI DỮ LIỆU ---
            # Để dễ tính loss, ta cần duỗi phẳng kích thước chiều H, W thành 1 hàng dài.
            # Ví dụ: Từ [Batch, 45, H, W] -> chuyển thành [Batch, H*W*9, 5]
            Batch, _, H, W = cls_pred.shape
            
            # Hàm permute đổi chỗ các chiều: [Batch, H, W, 45]
            # Hàm reshape gộp H, W và tách 5 class ra riêng: [Batch, H * W * 9, 5]
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(Batch, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(Batch, -1, 4)

            cls_outputs.append(cls_pred)
            reg_outputs.append(reg_pred)

        # Nối kết quả của tất cả các tầng (P3 -> P7) lại thành 1 Tensor duy nhất
        # Trả về: (Toàn bộ dự đoán lớp, Toàn bộ dự đoán tọa độ)
        return torch.cat(cls_outputs, dim=1), torch.cat(reg_outputs, dim=1)

if __name__ == '__main__':
    # Chạy thử nối cả Backbone + FPN + Head
    from backbone import ResNetBackbone
    from fpn import FPN
    
    backbone = ResNetBackbone()
    fpn = FPN(in_channels_list=backbone.out_channels, out_channels=256)
    head = DetectionHead(in_channels=256, num_anchors=9, num_classes=5)
    
    dummy_image = torch.randn(1, 3, 512, 512)
    
    # 3 bước quy trình chuẩn của mô hình
    c3, c4, c5 = backbone(dummy_image)
    features = fpn([c3, c4, c5])
    cls_preds, reg_preds = head(features)
    
    print("Dự đoán thành công!")
    print(f"Tổng số hộp neo (anchors) dự đoán trên toàn ảnh: {cls_preds.shape[1]}")
    print(f"Kích thước tensor Class: {cls_preds.shape} -> (Batch, Tổng số anchors, Số lớp)")
    print(f"Kích thước tensor Box: {reg_preds.shape} -> (Batch, Tổng số anchors, 4 tọa độ)")
