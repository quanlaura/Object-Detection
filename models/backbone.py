"""Lấy ba mức đặc trưng từ ResNet của torchvision để đưa vào FPN."""

from torch import Tensor, nn
from torchvision.models import get_model, get_model_weights

# kế thừa nn.Module để tạo một backbone ResNet tùy chỉnh, lấy các đặc trưng từ các tầng của ResNet
class ResNetBackbone(nn.Module):
    # kênh đầu ra của C3,4,5
    OUT_CHANNELS = {
        "resnet18": (128, 256, 512),
        "resnet34": (128, 256, 512),
        "resnet50": (512, 1024, 2048),
    }
    # tạo backbone ResNet tùy chỉnh
    def __init__(
        # sd trọng số pretrained, freeze_stages để đóng băng các tầng đầu tiên của ResNet
        self, name: str = "resnet34", pretrained: bool = True, freeze_stages: int = 1
    ):
        # gọi hàm khởi tạo của lớp cha nn.Module
        super().__init__()
        if name not in self.OUT_CHANNELS:
            # ", ".join(...) ghép các tên lại thành một chuỗi
            supported = ", ".join(self.OUT_CHANNELS)
            raise ValueError(f"Backbone '{name}' không được hỗ trợ. Chọn {supported}.")

        # lưu kênh đầu ra của C3,4,5 để FPN dùng sau vì FPN cần các feature map từ các tầng này
        self.out_channels = self.OUT_CHANNELS[name]
        # trọng số
        weights = get_model_weights(name).DEFAULT if pretrained else None
        # tạo mô hình ResNet từ các trọng số đã chọn
        resnet = get_model(name, weights=weights)
        # nn.Sequential để lấy các tầng đầu tiên của ResNet (conv1, bn1, relu, maxpool, layer1) làm early_features
        self.early_features = nn.Sequential(
            # conv1 : tích chập ban đầu để trích xuất đặc trưng cơ bản như cạnh, đường nét từ ảnh đầu vào
            resnet.conv1,
            # bn1 : batch normalization để chuẩn hóa đầu ra của conv1, giúp mô hình học ổn định hơn
            resnet.bn1,
            # relu : đưa các giá trị âm về 0 giúp mô hình phi tuyến hơn và học các đặc trưng phức tạp hơn
            resnet.relu,
            # maxpool: giảm kích thước feature map, nhờ đó các tầng sau cần xử lý ít vị trí hơn
            resnet.maxpool,
            # residual block đầu tiên, tiếp tục trích xuất các đặc trưng phức tạp hơn từ ảnh đầu vào
            resnet.layer1
        )
        # feature map C3
        self.stage3 = resnet.layer2
        # lấy layer 3 của Resnet làm c4
        self.stage4 = resnet.layer3
        # C5 có độ phân giải không gian thấp nhất nhưng đặc trưng sâu hơn
        self.stage5 = resnet.layer4

        # có yc đóng băng -> đóng băng các tầng đầu của resnet để k học thêm trọng số, giảm overfitting
        if freeze_stages >= 1:
            self.early_features.requires_grad_(False)


        for layer in self.modules():
            # ktra module hiện tại có phải là nn.BatchNorm2d (tầng chuẩn hóa trong CNN), nếu đúng thì đóng băng các tham số của nó
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(False)
        self.train()

    def train(self, mode: bool = True):
        # gọi train của lớp cha để đặt mô hình vào chế độ huấn luyện hoặc đánh giá
        super().train(mode)
        for layer in self.modules():
            # duyệt qua toàn module và chuyển các tầng BatchNorm sang eval mode để k cập nhật thông số thống kê trong quá trình huấn luyện
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    # forward xác định đường đi của dữ liệu qua mô hình, nhận vào ảnh và trả về các feature map C3, C4, C5
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.early_features(images)
        c3 = self.stage3(features)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        # các feature map sẽ đc đưa vào fpn
        return c3, c4, c5
