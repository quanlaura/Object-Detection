import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256):
        super().__init__()
        # 1. Các lớp Lateral (Conv 1x1): Ép tất cả C3, C4, C5 về chung 256 kênh
        self.lat_c3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lat_c5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

        # 2. Các lớp Smooth (Conv 3x3): Khử nhiễu ranh giới sau khi phóng to ảnh và cộng gộp
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 3. Các lớp mở rộng cho RetinaNet (P6, P7): Dùng để dò tìm vật thể cực kỳ lớn
        self.p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        c3, c4, c5 = features

        # Bước A: Chiếu mọi thứ về 256 kênh
        p5_lat = self.lat_c5(c5)
        p4_lat = self.lat_c4(c4)
        p3_lat = self.lat_c3(c3)

        # Bước B: Truyền thông tin từ trên (ảnh nhỏ) xuống dưới (ảnh to)
        # F.interpolate dùng để phóng to ảnh cho bằng kích thước với tầng dưới trước khi cộng
        p5 = p5_lat
        p4 = p4_lat + F.interpolate(p5, size=p4_lat.shape[-2:], mode='nearest')
        p3 = p3_lat + F.interpolate(p4, size=p3_lat.shape[-2:], mode='nearest')

        # Bước C: Làm mịn ảnh
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        # Bước D: Tạo P6 và P7 từ P5
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        # Output cuối cùng là 5 bản đồ đặc trưng P3 -> P7
        return [p3, p4, p5, p6, p7]

if __name__ == '__main__':
    # Chạy thử ghép nối Backbone và FPN
    from backbone import ResNetBackbone
    
    # Tạo backbone và fpn
    backbone = ResNetBackbone()
    fpn = FPN(in_channels_list=backbone.out_channels, out_channels=256)
    
    # Mô phỏng 1 ảnh 512x512
    dummy_image = torch.randn(1, 3, 512, 512)
    
    # Chạy thử
    backbone_features = backbone(dummy_image)
    fpn_features = fpn(backbone_features)
    
    print("Xây dựng tháp đặc trưng thành công!")
    for i, feature in enumerate(fpn_features):
        # Kích thước sẽ giảm dần một nửa ở mỗi tầng
        print(f"Tầng P{i+3} có kích thước (Batch, Channel, Height, Width): {feature.shape}")