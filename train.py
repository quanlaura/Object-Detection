import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.anchors import AnchorGenerator
from models.detector import RetinaDetector
from models.loss import DetectionLoss

def train(args):
    # Tự động chọn GPU nếu có, nếu không sẽ dùng CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Chuẩn bị Dữ liệu
    print("Đang nạp dữ liệu...")
    train_dataset = ObjectDetectionDataset(args.train_data, args.image_dir, img_size=512)
    # Gom nhóm (batch) 4 ảnh mỗi lần để máy tính xử lý. 
    # Nếu máy báo lỗi hết RAM/VRAM, bạn có thể giảm batch_size xuống 2.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 2. Khởi tạo Cỗ máy, Hộp neo và Hàm Loss
    model = RetinaDetector(num_classes=5).to(device)
    loss_fn = DetectionLoss(num_classes=5).to(device)
    
    anchor_generator = AnchorGenerator()
    # Tạo lưới hộp neo 49,104 khung 1 lần duy nhất cho kích thước ảnh 512x512
    mock_features = [torch.empty(1, 1, 512//s, 512//s) for s in [8, 16, 32, 64, 128]]
    anchors = anchor_generator(mock_features).to(device)

    # 3. Khởi tạo bộ Tối ưu hóa (AdamW giúp trọng số hội tụ mượt mà hơn)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 4. Vòng lặp Huấn luyện
    epochs = 10 # Chạy thử 10 vòng lớn
    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            # Xóa các tính toán của bước trước
            optimizer.zero_grad()

            # Mô hình dự đoán
            cls_preds, reg_preds = model(images)

            # Tính điểm phạt (Loss)
            loss, cls_loss, reg_loss = loss_fn(cls_preds, reg_preds, anchors, targets)

            # Lan truyền ngược để cập nhật trọng số
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            
            # Cứ mỗi 5 batch, in ra màn hình để theo dõi
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} (Phân lớp: {cls_loss.item():.4f}, Hộp bao: {reg_loss.item():.4f})")

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"==> Hết Epoch {epoch+1} | Trung bình Loss: {avg_loss:.4f}")

        # Lưu lại mô hình tốt nhất
        save_path = os.path.join(args.checkpoint_dir, "best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Đã lưu trọng số tại {save_path}\n")

if __name__ == '__main__':
    # Các tham số truyền vào từ dòng lệnh theo đúng yêu cầu của đề bài
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--val_image_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    args = parser.parse_args()
    
    train(args)