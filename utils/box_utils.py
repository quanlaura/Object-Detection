import torch

def box_iou(boxes1, boxes2):
    """Tính tỷ lệ diện tích giao nhau (Intersection over Union - IoU) giữa 2 tập hợp hộp bao"""
    # Tính diện tích từng hộp
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Tìm tọa độ phần giao nhau
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # Tọa độ góc trái trên của phần giao
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # Tọa độ góc phải dưới của phần giao

    wh = (rb - lt).clamp(min=0)  # Kích thước phần giao (nếu không giao nhau thì = 0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # Diện tích phần giao

    union = area1[:, None] + area2 - inter # Diện tích hợp
    return inter / union

def encode_boxes(gt_boxes, anchors):
    """Mã hóa (Encode): Chuyển tọa độ hộp thực tế thành 'độ lệch' so với hộp neo để mô hình học"""
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    an_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    an_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    an_w = anchors[:, 2] - anchors[:, 0]
    an_h = anchors[:, 3] - anchors[:, 1]

    # Tính độ lệch (Deltas)
    tx = (gt_cx - an_cx) / an_w
    ty = (gt_cy - an_cy) / an_h
    tw = torch.log(gt_w / an_w)
    th = torch.log(gt_h / an_h)

    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(deltas, anchors):
    """Giải mã (Decode): Chuyển 'độ lệch' mà mô hình dự đoán ngược lại thành tọa độ trên ảnh gốc"""
    an_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    an_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    an_w = anchors[:, 2] - anchors[:, 0]
    an_h = anchors[:, 3] - anchors[:, 1]

    pred_cx = deltas[:, 0] * an_w + an_cx
    pred_cy = deltas[:, 1] * an_h + an_cy
    pred_w = torch.exp(deltas[:, 2]) * an_w
    pred_h = torch.exp(deltas[:, 3]) * an_h

    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2

    return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

def match_anchors(anchors, gt_boxes, iou_thresh_low=0.4, iou_thresh_high=0.5):
    """
    So khớp hộp neo với hộp thực tế. 
    Trả về: mask (-1: bỏ qua, 0: nền/background, 1: có vật thể) và vị trí hộp thực tế tương ứng.
    """
    ious = box_iou(anchors, gt_boxes) # [49104, Số lượng vật thể]
    max_ious, argmax_ious = ious.max(dim=1)
    
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.float32)
    labels[max_ious < iou_thresh_low] = 0.0 # Negative
    labels[max_ious >= iou_thresh_high] = 1.0 # Positive
    
    # Ép buộc: Mỗi hộp thực tế phải có ít nhất 1 hộp neo nhận diện được nó
    gt_max_ious, gt_argmax_ious = ious.max(dim=0)
    for i in range(gt_boxes.shape[0]):
        if gt_max_ious[i] > 0:
            labels[gt_argmax_ious[i]] = 1.0
            argmax_ious[gt_argmax_ious[i]] = i

    return labels, argmax_ious

if __name__ == '__main__':
    # Thử nghiệm tính IoU
    box_a = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    box_b = torch.tensor([[30, 30, 70, 70]], dtype=torch.float32)
    iou = box_iou(box_a, box_b)
    
    print("Công cụ xử lý Hộp bao (Box Utils) đã sẵn sàng!")
    print(f"IoU giữa Box A và Box B là: {iou.item():.4f}")