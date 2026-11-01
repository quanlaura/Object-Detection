import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kéo đường dẫn thư mục gốc vào hệ thống để file này gọi được utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.box_utils import match_anchors, encode_boxes, decode_boxes

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_preds, reg_preds, anchors, targets):
        batch_size = cls_preds.shape[0]
        cls_losses = []
        reg_losses = []

        for i in range(batch_size):
            cls_pred = cls_preds[i] # [49104, 5]
            reg_pred = reg_preds[i] # [49104, 4]
            target = targets[i]     # [Số vật thể, 6] (label, x1, y1, x2, y2, diff)

            if target.shape[0] == 0:
                # Nếu ảnh trống trơn, mọi hộp neo đều là nền
                loss_cls = self._calc_focal_loss(cls_pred, torch.zeros_like(cls_pred))
                cls_losses.append(loss_cls)
                reg_losses.append(torch.tensor(0.0, device=cls_pred.device))
                continue

            gt_boxes = target[:, 1:5]
            gt_labels = target[:, 0].long()

            # 1. Áp hộp neo vào hộp thật
            mask, argmax_ious = match_anchors(anchors, gt_boxes)

            # Lọc ra các hộp neo chứa vật thể (pos) và hộp hợp lệ (valid: pos + neg)
            pos_idx = mask == 1.0
            valid_idx = mask >= 0.0
            num_pos = max(1.0, pos_idx.sum().item()) # Chuẩn hóa số chia

            # 2. FOCAL LOSS (Phân lớp)
            target_labels = torch.zeros_like(cls_pred)
            target_labels[pos_idx, gt_labels[argmax_ious[pos_idx]]] = 1.0
            
            # Chỉ tính loss cho các anchor hợp lệ (bỏ qua những cái lấp lửng ở IoU 0.4 - 0.5)
            loss_cls = self._calc_focal_loss(cls_pred[valid_idx], target_labels[valid_idx])
            cls_losses.append(loss_cls / num_pos)

            # 3. GIOU LOSS (Hộp bao - Chỉ tính phạt cho những hộp đang chứa vật thể)
            if pos_idx.sum() > 0:
                pos_reg_pred = reg_pred[pos_idx]
                pos_anchors = anchors[pos_idx]
                pos_gt_boxes = gt_boxes[argmax_ious[pos_idx]]
                
                # Giải mã để lấy tọa độ hộp
                pred_boxes = decode_boxes(pos_reg_pred, pos_anchors)
                
                # Tính GIoU (Generalized IoU)
                lt = torch.max(pred_boxes[:, :2], pos_gt_boxes[:, :2])
                rb = torch.min(pred_boxes[:, 2:], pos_gt_boxes[:, 2:])
                wh = (rb - lt).clamp(min=0)
                inter = wh[:, 0] * wh[:, 1]
                
                area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                area_gt = (pos_gt_boxes[:, 2] - pos_gt_boxes[:, 0]) * (pos_gt_boxes[:, 3] - pos_gt_boxes[:, 1])
                union = area_pred + area_gt - inter
                iou = inter / (union + 1e-7)
                
                # Tính khung chứa lồi nhỏ nhất (Convex hull)
                lt_c = torch.min(pred_boxes[:, :2], pos_gt_boxes[:, :2])
                rb_c = torch.max(pred_boxes[:, 2:], pos_gt_boxes[:, 2:])
                wh_c = (rb_c - lt_c).clamp(min=0)
                area_c = wh_c[:, 0] * wh_c[:, 1]
                
                giou = iou - (area_c - union) / (area_c + 1e-7)
                
                # Loss = 1 - GIoU
                reg_loss = (1.0 - giou).sum()
                reg_losses.append(reg_loss / num_pos)
            else:
                reg_losses.append(torch.tensor(0.0, device=cls_pred.device))

        total_cls_loss = torch.stack(cls_losses).mean()
        total_reg_loss = torch.stack(reg_losses).mean()
        return total_cls_loss + total_reg_loss, total_cls_loss, total_reg_loss

    def _calc_focal_loss(self, preds, targets):
        """Tính toán Focal Loss bằng tay để hiểu rõ ngọn ngành thuật toán"""
        p = torch.sigmoid(preds)
        ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        
        # Công thức điều chỉnh tỷ trọng
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        return loss.sum()

if __name__ == '__main__':
    print("Đang chạy luồng kiểm tra Hàm Loss (GIoU & Focal Loss)...")
    loss_fn = DetectionLoss()
    
    # 1. Tạo giả 49,104 dự đoán của mô hình
    cls_preds = torch.randn(1, 49104, 5)
    reg_preds = torch.randn(1, 49104, 4)
    
    # 2. Tạo giả 49,104 hộp neo ngẫu nhiên
    anchors = torch.rand(49104, 4) * 512
    anchors[:, 2:] += anchors[:, :2]
    
    # 3. Tạo giả dữ liệu thực tế (1 người [nhãn 0] đứng ở tọa độ 100,100 -> 200,200)
    targets = [torch.tensor([[0, 100, 100, 200, 200, 0]], dtype=torch.float32)]
    
    total_loss, cls_loss, reg_loss = loss_fn(cls_preds, reg_preds, anchors, targets)
    
    print("\n[THÀNH CÔNG] Dòng chảy đạo hàm đã được khơi thông!")
    print(f"Tổng Loss tính toán: {total_loss.item():.4f}")
    print(f"  - Sai số Phân lớp (Focal Loss): {cls_loss.item():.4f}")
    print(f"  - Sai số Hộp bao (GIoU Loss)  : {reg_loss.item():.4f}")