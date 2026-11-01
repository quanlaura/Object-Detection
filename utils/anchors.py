import torch
import math

class AnchorGenerator:
    def __init__(self, strides=[8, 16, 32, 64, 128], base_sizes=[32, 64, 128, 256, 512],
                 scales=[1.0, 1.2599, 1.5874], # Tương đương 2^0, 2^(1/3), 2^(2/3)
                 ratios=[0.5, 1.0, 2.0]):    # Tỷ lệ 1:2, 1:1, 2:1
        self.strides = strides
        self.base_sizes = base_sizes
        self.scales = scales
        self.ratios = ratios

    def generate_base_anchors(self, base_size):
        """Tạo 9 hộp neo cơ bản nằm ngay tại tâm (0,0)"""
        base_anchors = []
        for ratio in self.ratios:
            # Tính chiều cao (h) và chiều rộng (w) dựa trên tỷ lệ
            h = base_size / math.sqrt(ratio)
            w = base_size * math.sqrt(ratio)
            for scale in self.scales:
                w_s = w * scale
                h_s = h * scale
                # Định dạng hộp bao: [xmin, ymin, xmax, ymax]
                base_anchors.append([-w_s/2, -h_s/2, w_s/2, h_s/2])
        return torch.tensor(base_anchors, dtype=torch.float32)

    def generate_anchors_for_feature_map(self, base_anchors, feat_height, feat_width, stride):
        """Sao chép 9 hộp neo cơ bản mang đi rải đều khắp các điểm ảnh trên lưới"""
        # Tính tọa độ tâm (cx, cy) của từng điểm ảnh trên ảnh gốc
        shifts_x = (torch.arange(0, feat_width) + 0.5) * stride
        shifts_y = (torch.arange(0, feat_height) + 0.5) * stride
        
        # Tạo lưới tọa độ 2D
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # Duỗi thẳng ra thành 1 hàng dọc
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        # Ma trận dịch chuyển: [Số điểm ảnh, 4]
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        
        # Cộng tọa độ lưới vào 9 hộp neo gốc
        # base_anchors: [9, 4], shifts: [Số điểm ảnh, 4] => anchors: [Số điểm ảnh, 9, 4]
        anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
        
        # Trải phẳng lại: [Số điểm ảnh * 9, 4]
        return anchors.reshape(-1, 4)

    def __call__(self, feature_maps):
        """Tự động tạo hộp neo cho tất cả 5 bản đồ đặc trưng (P3 -> P7)"""
        all_anchors = []
        for i, feat in enumerate(feature_maps):
            # Lấy chiều cao và rộng của tầng FPN hiện tại
            _, _, feat_h, feat_w = feat.shape
            
            base_anchors = self.generate_base_anchors(self.base_sizes[i])
            anchors = self.generate_anchors_for_feature_map(base_anchors, feat_h, feat_w, self.strides[i])
            all_anchors.append(anchors)
        
        # Nối tất cả các tầng lại thành 1 Tensor khổng lồ
        return torch.cat(all_anchors, dim=0)

if __name__ == '__main__':
    # Tạo mô phỏng kích thước 5 tầng FPN cho ảnh gốc 512x512
    # P3(64x64), P4(32x32), P5(16x16), P6(8x8), P7(4x4)
    mock_features = [
        torch.empty(1, 256, 64, 64),
        torch.empty(1, 256, 32, 32),
        torch.empty(1, 256, 16, 16),
        torch.empty(1, 256, 8, 8),
        torch.empty(1, 256, 4, 4)
    ]
    
    generator = AnchorGenerator()
    anchors = generator(mock_features)
    
    print("Khởi tạo mạng lưới Hộp neo (Anchors) thành công!")
    print(f"Tổng số lượng hộp neo được tạo ra: {anchors.shape[0]}")
    
    print("\nVí dụ Hộp neo đầu tiên (nằm ở góc trên bên trái ảnh):")
    print([round(x, 1) for x in anchors[0].tolist()])