import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Tải cấu trúc mạng ResNet34 đã được huấn luyện sẵn trên ImageNet
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Tách phần gốc (stem) và layer 1 của ResNet (những phần này trích xuất đặc trưng chung nhất)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        
        # Tách các khối trích xuất đặc trưng quan trọng: C3, C4, C5
        self.layer2 = resnet.layer2  # C3: Nhìn chi tiết nhỏ
        self.layer3 = resnet.layer3  # C4: Nhìn chi tiết vừa
        self.layer4 = resnet.layer4  # C5: Nhìn tổng thể toàn cục
        
        # Số lượng kênh (channels) đầu ra của C3, C4, C5 trên ResNet34
        self.out_channels = [128, 256, 512]

    def forward(self, x):
        # Đưa ảnh qua các lớp của ResNet
        x = self.stem(x)
        x = self.layer1(x)
        
        c3 = self.layer2(x) # Giảm kích thước ảnh đi 8 lần
        c4 = self.layer3(c3) # Giảm kích thước ảnh đi 16 lần
        c5 = self.layer4(c4) # Giảm kích thước ảnh đi 32 lần
        
        # Trả về 3 bộ đặc trưng này để lát nữa FPN xử lý
        return [c3, c4, c5]

if __name__ == '__main__':
    # Chạy thử nghiệm mạng Backbone
    model = ResNetBackbone()
    
    # Tạo một ảnh giả (tensor) với kích thước [Batch=1, Kênh màu=3, Rộng=512, Cao=512]
    dummy_image = torch.randn(1, 3, 512, 512)
    
    # Cho ảnh giả đi qua đôi mắt Backbone
    features = model(dummy_image)
    
    print("Trích xuất đặc trưng thành công!")
    for i, feature in enumerate(features):
        print(f"Đặc trưng C{i+3} có kích thước: {feature.shape}")