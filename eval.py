import os
import json
import torch
import cv2
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.detector import RetinaDetector
from utils.anchors import AnchorGenerator
from utils.box_utils import decode_boxes
from torchvision.ops import nms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang khởi động cỗ máy chấm điểm trên: {device}")

    # 1. Nạp mô hình và trọng số tốt nhất
    model = RetinaDetector(num_classes=5).to(device)
    model.load_state_dict(torch.load('./models/best.pth', map_location=device))
    model.eval()

    anchor_generator = AnchorGenerator()
    mock_features = [torch.empty(1, 1, 512//s, 512//s) for s in [8, 16, 32, 64, 128]]
    anchors = anchor_generator(mock_features).to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    # Khởi tạo bộ đo chuẩn COCO
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    # 2. Đọc file đáp án (Ground Truth) từ tập Validation
    val_json_path = './final_public/public/annotations/val.json'
    val_image_dir = './final_public/public/val/images'
    
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Gom nhóm đáp án theo từng bức ảnh
    img_to_anns = {}
    for ann in val_data.get('annotations', []):
        img_id = str(ann['image_id']) # Ép về dạng chuỗi (string) để chống lỗi lệch kiểu ID
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    print(f"Bắt đầu chấm điểm {len(val_data['images'])} ảnh trong tập Validation...")

    # 3. Quét từng ảnh, dự đoán và so khớp đáp án
    for img_info in tqdm(val_data['images']):
        img_id = str(img_info['id'])
        
        # Chỉ lấy tên file cuối cùng, tránh lỗi đường dẫn lồng nhau trong JSON
        img_name = os.path.basename(img_info['file_name']) 
        img_path = os.path.join(val_image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"\n[Lỗi đường dẫn] Không thấy ảnh: {img_path}")
            continue

        # --- LẤY ĐÁP ÁN THẬT ---
        gt_boxes = []
        gt_labels = []
        
        # Danh sách này phải khớp thứ tự 100% với code dự đoán
        CLASS_NAMES = ['person', 'car', 'dog', 'cat', 'chair']
        
        for ann in img_to_anns.get(img_id, []):
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            
            class_name = ann.get('class', '')
            if class_name in CLASS_NAMES:
                cat_id = CLASS_NAMES.index(class_name)
            else:
                cat_id = 0
                
            gt_labels.append(cat_id)

        if len(gt_boxes) == 0:
            continue

        # ---> ĐÂY CHÍNH LÀ ĐOẠN KHỞI TẠO BIẾN TARGET BỊ THIẾU <---
        target = [{
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32).to(device),
            'labels': torch.tensor(gt_labels, dtype=torch.int64).to(device)
        }]
        
        # --- CHẠY DỰ ĐOÁN ---
        original_img = cv2.imread(img_path)
        orig_h, orig_w = original_img.shape[:2]
        
        img = cv2.resize(original_img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            cls_preds, reg_preds = model(img_tensor)
        
        cls_preds = torch.sigmoid(cls_preds[0])
        reg_preds = reg_preds[0]
        pred_boxes = decode_boxes(reg_preds, anchors)

        # Mẹo: Giữ ngưỡng tin cậy thấp (0.05) để thuật toán vẽ được trọn vẹn biểu đồ Precision-Recall
        scores, labels = torch.max(cls_preds, dim=1)
        keep_idx = scores > 0.05 
        filtered_boxes = pred_boxes[keep_idx]
        filtered_scores = scores[keep_idx]
        filtered_labels = labels[keep_idx]

        nms_idx = nms(filtered_boxes, filtered_scores, 0.5)
        final_boxes = filtered_boxes[nms_idx]
        final_scores = filtered_scores[nms_idx]
        final_labels = filtered_labels[nms_idx]

        # Phóng to tọa độ dự đoán về kích thước ảnh gốc để so khớp chuẩn với đáp án
        scale_tensor = torch.tensor([orig_w/512, orig_h/512, orig_w/512, orig_h/512]).to(device)
        final_boxes = final_boxes * scale_tensor

        prediction = [{
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }]

        # Đưa vào sổ điểm
        metric.update(prediction, target)

    # 4. Tính toán và in Bảng Điểm
    print("\nĐang tổng hợp điểm số...")
    result = metric.compute()
    
    print("\n" + "="*50)
    print(f" BẢNG ĐIỂM MÔ HÌNH RETINANET (TẬP VALIDATION) ")
    print("="*50)
    print(f" mAP@0.5       : {result['map_50'].item() * 100:.2f}%")
    print(f" mAP@0.5:0.95  : {result['map'].item() * 100:.2f}%")
    print(f" Recall        : {result['mar_100'].item() * 100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()