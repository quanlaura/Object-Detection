"""CNN nhẹ với các khối residual, trả về C3, C4, C5."""

import math

from torch import nn

# stride là số bước nhảy của bộ lọc trong tích chập, kernel_size là kích thước bộ lọc
def conv_norm(in_channels, out_channels, kernel_size=3, stride=1):
    # chọn tối đa 32 nhóm GroupNorm sao cho số kênh chia hết cho số nhóm
    groups = min(32, max(1, out_channels // 4))
    while out_channels % groups:
        # giảm số nhóm đến khi chia hết số kênh đầu ra
        groups -= 1
    return nn.Sequential(
        # Input → Conv2d → GroupNorm → Output
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=kernel_size // 2, bias=False),
        nn.GroupNorm(groups, out_channels, eps=1e-3),
    )
# số bộ lọc x số kênh x kích thước kênh 2D
# mỗi bộ lọc tạo ra 1 feature map đầu ra

# cho dữ liệu đi qua 2 đường : đường chính gồm 2 lớp conv và đường shortcut để giữ hoặc chỉnh thông tin đầu vào
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.path = nn.Sequential( # ghép 3 tp thành 1 đường xử lý tuần tự
            conv_norm(in_channels, out_channels, stride=stride),
            nn.LeakyReLU(0.1, inplace=True),
            conv_norm(out_channels, out_channels),
        )
        # nếu stride=1 và in_channels=out_channels thì dùng nn.Identity() để giữ nguyên đầu vào
        # nếu kích thước hoặc số kênh khác, dùng conv_norm 1x1 để hai nhánh có shape khớp nhau
        # self.path(x) + self.shortcut(x)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_norm(in_channels, out_channels, kernel_size=1, stride=stride)
        # hàm kích hoạt LeakyReLU
        # f(x) = x khi x > 0, 0.1x khi x < 0
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # hàm kích hoạt LeakyReLU ở trên
        return self.activate(self.path(x) + self.shortcut(x))

# tạo một mạng CNN nhỏ sd các block residual để trích xuất 3 mức đặc trưng c3,4,5 từ ảnh
class TinyBackbone(nn.Module):
    # width_multiplier: hệ số nhân để tăng/giảm số kênh của các tầng,
    # num_blocks: số lượng khối residual cho mỗi tầng C3, C4, C5
    def __init__(self, width_multiplier=1.0, num_blocks=(2, 3, 3)):
        super().__init__()
        if not math.isfinite(width_multiplier) or width_multiplier <= 0:
            raise ValueError("width_multiplier phải là số dương hữu hạn.")
        if len(num_blocks) != 3 or any(count < 1 for count in num_blocks):
            raise ValueError("Cần ít nhất một khối residual cho mỗi tầng C3, C4, C5.")
        c1, c2, c3, c4, c5 = [
            max(1, int(channels * width_multiplier))
            for channels in (32, 64, 128, 256, 512)
        ]
        self.out_channels = (c3, c4, c5)
        # self.stem là một chuỗi các lớp conv, ReLU và maxpool để giảm kích thước ảnh đầu vào và trích xuất đặc trưng cơ bản trước khi đi vào các tầng residual
        self.stem = nn.Sequential(
            conv_norm(3, c1), nn.LeakyReLU(0.1, inplace=True),
            # MaxPool để giảm kích thước map trước khi đưa tiếp qua mạng, k đổi số kênh
            nn.MaxPool2d(2, ceil_mode=True),
            conv_norm(c1, c2), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        # stages là danh sách các tầng residual, mỗi tầng gồm nhiều khối, số khối = num_blocks[i] cho tầng i
        self.stages = nn.ModuleList()
        for incoming, outgoing, depth in zip((c2, c3, c4), self.out_channels, num_blocks):
            # tạo block đầu tiên, nếu stride=2 thì giảm kích thước map, tăng số kênh từ incoming → outgoing
            blocks = [ResidualBlock(incoming, outgoing, stride=2)]
            # thêm các block còn lại, stride=1 để giữ nguyên kích thước map, số kênh = outgoing
            blocks.extend(ResidualBlock(outgoing, outgoing) for _ in range(depth - 1))
            # gom các block của 1 tầng thành 1 chuỗi rồi thêm tầng đó và stages
            self.stages.append(nn.Sequential(*blocks))

        # duyệt qua từng module, nếu là lớp conv2d thì khởi tạo lại trọng số
        # maxpool, leaky, groupnorm k cần vì có pp khởi tạo mặc định
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # Kaiming khởi tạo trọng số Conv phù hợp với LeakyReLU
                nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity="leaky_relu")

    def forward(self, images):
        # images đi qua stem trước, sau đó đi qua từng stage để trích xuất các đặc trưng C3, C4, C5
        c3 = self.stages[0](self.stem(images))
        c4 = self.stages[1](c3)
        c5 = self.stages[2](c4)
        return c3, c4, c5
