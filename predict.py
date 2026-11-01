import os
import torch
import cv2
import argparse
import numpy as np
from torchvision import transforms

from models.detector import RetinaDetector
from utils.anchors import AnchorGenerator
from utils.box_utils import decode_boxes
from torchvision.ops import nms

# Nếu bạn có tên lớp cụ thể, hãy sửa mảng này (ví dụ: ['cho', 'meo', ...])
CLASS_NAMES = ['person', 'car', 'dog', 'cat', 'chair']
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy dự đoán trên: {device}")

    # 1. Khởi tạo cỗ máy và nạp trí não (trọng số)
    model = RetinaDetector(num_classes=5).to(device)
    if os.path.exists(args.weight_path):
        model.load_state_dict(torch.load(args.weight_path, map_location=device))
        print("Đã nạp trọng số thành công!")
    else:
        print("Lỗi: Không tìm thấy file trọng số.")
        return
    
    model.eval() # Chuyển sang chế độ dự đoán (tắt các cập nhật gradient)

    # Khởi tạo bộ tạo hộp neo
    anchor_generator = AnchorGenerator()
    mock_features = [torch.empty(1, 1, 512//s, 512//s) for s in [8, 16, 32, 64, 128]]
    anchors = anchor_generator(mock_features).to(device)

    # 2. Xử lý ảnh đầu vào
    original_img = cv2.imread(args.image_path)
    if original_img is None:
        print("Không đọc được ảnh!")
        return
    
    orig_h, orig_w = original_img.shape[:2]
    img = cv2.resize(original_img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Chuyển ảnh thành Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device) # Thêm chiều Batch

    # 3. Chạy mô hình dự đoán
    with torch.no_grad():
        cls_preds, reg_preds = model(img_tensor)

    # Chuyển xác suất phân lớp qua hàm sigmoid
    cls_preds = torch.sigmoid(cls_preds[0]) # Lấy ảnh đầu tiên trong batch [49104, 5]
    reg_preds = reg_preds[0] # [49104, 4]

    # Giải mã "độ lệch" thành tọa độ thật
    pred_boxes = decode_boxes(reg_preds, anchors)

    # 4. Lọc kết quả (Loại bỏ các hộp có xác suất thấp)
    # Lấy điểm số cao nhất và nhãn tương ứng cho từng hộp neo
    scores, labels = torch.max(cls_preds, dim=1)
    
    # Chỉ giữ lại những hộp tự tin trên ngưỡng (ví dụ: > 0.4)
    keep_idx = scores > args.conf_thresh
    filtered_boxes = pred_boxes[keep_idx]
    filtered_scores = scores[keep_idx]
    filtered_labels = labels[keep_idx]

    # 5. Non-Maximum Suppression (NMS) - Khử trùng lặp
    # Nếu nhiều hộp cùng chỉ vào 1 vật thể, NMS sẽ chỉ giữ lại hộp tốt nhất
    nms_idx = nms(filtered_boxes, filtered_scores, args.nms_thresh)
    
    final_boxes = filtered_boxes[nms_idx].cpu().numpy()
    final_scores = filtered_scores[nms_idx].cpu().numpy()
    final_labels = filtered_labels[nms_idx].cpu().numpy()

    # 6. Vẽ hộp lên ảnh gốc
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        score = final_scores[i]
        label = final_labels[i]

        # Phục hồi tọa độ từ ảnh 512x512 về ảnh gốc
        x1 = int(box[0] * orig_w / 512)
        y1 = int(box[1] * orig_h / 512)
        x2 = int(box[2] * orig_w / 512)
        y2 = int(box[3] * orig_h / 512)

        color = COLORS[label % len(COLORS)]
        class_name = CLASS_NAMES[label]

        # Vẽ hình chữ nhật
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
        
        # In nhãn và điểm số
        text = f"{class_name}: {score:.2f}"
        cv2.putText(original_img, text, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Lưu kết quả
    cv2.imwrite(args.output_path, original_img)
    print(f"Đã lưu ảnh kết quả tại: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Đường dẫn ảnh cần test')
    parser.add_argument('--weight_path', type=str, default='./models/best.pth', help='Đường dẫn file trọng số')
    parser.add_argument('--output_path', type=str, default='result.jpg', help='Nơi lưu ảnh đầu ra')
    parser.add_argument('--conf_thresh', type=float, default=0.4, help='Ngưỡng tin cậy (0.0 -> 1.0)')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='Ngưỡng khử trùng lặp (IoU)')
    args = parser.parse_args()
    
    predict(args)