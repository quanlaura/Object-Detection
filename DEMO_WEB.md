# Demo nhận diện bằng Gradio

Ứng dụng chạy cục bộ, dùng models/best.pth. Không cần train lại.

## Mở demo trong VS Code

Tại thư mục ObjectDetectionV1, chạy PowerShell:

```powershell
.\.venv\Scripts\python.exe web_demo.py
```

Chờ thông báo địa chỉ http://127.0.0.1:7860. Trình duyệt sẽ mở tự động.
Lần đầu cần nạp thư viện và mô hình nên có thể chờ một lúc.
Giữ terminal mở khi demo. Dừng bằng Ctrl+C trong terminal.
Nếu cổng đang dùng, mở lại trang hiện có hoặc chạy `--port 7861`.

## Cách dùng

1. Chọn ảnh JPG/PNG hoặc ảnh mẫu, có thể dán ảnh từ clipboard.
2. Giữ confidence 0.25 và NMS 0.4 để dùng ngưỡng đã đánh giá.
3. Bấm Nhận diện ảnh, chờ xử lý.
4. Xem hộp bao, bảng lớp/confidence/tọa độ; tải PNG và JSON nếu cần.
5. Bấm Làm mới để chọn ảnh khác.

Chỉ nhận diện person, car, dog, cat, chair. Confidence không phải accuracy.
Tăng confidence giúp bớt dự đoán yếu nhưng có thể bỏ sót; giảm NMS loại hộp trùng mạnh hơn.
Điểm 78.61% là mAP@0.5 trên validation, không phải cam kết chính xác trên mọi ảnh.
Ảnh quá 24 megapixel cần giảm kích thước trước; upload giới hạn 15 MB.
Ảnh có hướng EXIF được xoay theo hướng hiển thị trước khi nhận diện.
CPU chạy được; ứng dụng tự dùng CUDA nếu PyTorch hỗ trợ.

## File đầu ra

Ảnh có hộp bao và JSON lưu trong `runs/web_demo/<mã phiên>/`.
JSON giữ cấu trúc của đề; image_id là `uploaded_image` vì ảnh được gửi từ trình duyệt.
Không sử dụng JSON demo này để thay cho kết quả chấm cả thư mục ảnh; dùng predict.py cho mục đích đó.
Gradio sử dụng file tạm phục vụ upload/download. Ứng dụng không tạo public share link.
Giao diện mặc định chỉ lắng nghe trên 127.0.0.1; không gửi ảnh đến dịch vụ nhận diện bên ngoài.

## Cài trên môi trường mới

```powershell
python -m pip install -r requirements-web.txt
```

Gradio được tách khỏi requirements.txt của bài nộp để không bắt buộc cài giao diện khi chấm tự động.
Checkpoint, train.py và predict.py không bị thay đổi bởi ứng dụng web.
Nguồn hướng dẫn Gradio: https://www.gradio.app/docs
