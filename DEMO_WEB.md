# Mở website demo RetinaNet

Website dùng checkpoint EMA của **ResNet34 ở epoch 38** để nhận diện vật thể ngay trên máy chạy Python. Ảnh được xử lý cục bộ.

## 1. Khởi động từ VS Code

1. Mở thư mục `ObjectDetectionFromScratch` trong VS Code.
2. Chọn **Terminal → New Terminal**. Terminal cần ở thư mục chứa `web_demo.py`.
3. Chạy lệnh sau trong PowerShell.

```powershell
.\.venv\Scripts\python.exe -X utf8 web_demo.py
```

4. Chờ dòng `Demo RetinaNet: http://127.0.0.1:7861`, rồi mở [trang demo](http://127.0.0.1:7861) trên trình duyệt của máy đó. Giữ Terminal mở trong lúc dùng. Nhấn **Ctrl+C** trong Terminal để tắt demo.

## 2. Trình bày và kiểm tra mô hình

1. Khi mở trang, mô hình tự chạy trên ảnh **Trên phố** nếu dữ liệu test có sẵn. Ảnh đầu vào nằm bên trái, hộp dự đoán nằm bên phải. Bật **Hiện nhãn thật** để xem hộp gốc ở bên trái.
2. Sao chép ảnh rồi bấm **Dán ảnh**, hoặc nhấn **Ctrl+V** ngay trên trang. Demo sẽ tự nhận diện ảnh vừa dán. Bạn cũng có thể chọn ảnh trong **Ảnh từ tập test** hoặc bấm **Tải ảnh**. Ứng dụng nhận JPG, PNG, WebP và BMP, tối đa 12 MiB và 32 megapixel.
3. Chỉnh **Độ tin cậy**, **Ngưỡng NMS** hoặc bật **TTA** để thử thêm ảnh lật ngang. Sau khi đổi thiết lập hoặc tải một file ảnh, bấm **Chạy nhận diện**. Dùng **Tải JSON** nếu cần xem tọa độ và điểm tin cậy của từng hộp.
4. Mở mục **Đánh giá** để xem mAP, precision và recall đã lưu. Checkpoint của dự án đạt **70,49% mAP@0.5 trên 1.500 ảnh test** với confidence 0,25, NMS 0,40 và TTA tắt. Mục **Kiến trúc** cho thấy luồng từ ảnh đến hộp dự đoán.

## 3. Khi không mở được trang

| Dấu hiệu | Cách xử lý |
| --- | --- |
| PowerShell báo thiếu `.venv\Scripts\python.exe` | Tạo môi trường bằng `python -m venv .venv`, rồi cài thư viện bằng `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. |
| Terminal báo thiếu checkpoint | Đặt `best.pth` tại `resnet34_full/best.pth`. |
| Cổng 7861 đang được dùng | Chạy `.\.venv\Scripts\python.exe -X utf8 web_demo.py --port 7862`, rồi mở [cổng 7862](http://127.0.0.1:7862). |
| Trình duyệt báo không kết nối | Kiểm tra Terminal còn chạy và mở đúng địa chỉ mà Terminal in ra. Địa chỉ `127.0.0.1` dùng trên máy đang chạy Python. |
| Nút **Dán ảnh** không đọc được clipboard | Bấm vào trang rồi nhấn **Ctrl+V**. Hãy sao chép nội dung ảnh từ trình duyệt hoặc công cụ chụp màn hình. |
| Trang thiếu ảnh test để chọn | Kiểm tra `splits/test.json` và `public/train/images/`. Bạn vẫn có thể tải ảnh riêng lên để thử mô hình. |

Demo tự chọn GPU hoặc CPU. Để chạy bằng CPU, thêm `--device cpu` vào lệnh khởi động.

## 4. Chạy demo bằng Docker

1. Mở Docker Desktop, hoặc khởi động Docker Engine nếu đã có Docker Compose.
2. Lấy mã nguồn và đặt checkpoint của dự án tại `resnet34_full/best.pth`. Nếu được gửi checkpoint riêng, hãy giữ đúng tên `best.pth`.
3. Mở Terminal tại thư mục chứa `compose.yaml` và chạy lệnh sau.

```sh
docker compose up --build
```

4. Khi Terminal in dòng `Demo RetinaNet`, mở [trang demo](http://127.0.0.1:7861) trên chính máy đó. Bấm **Dán ảnh** hoặc **Tải ảnh** để thử. Nhấn **Ctrl+C** để dừng.

Docker dùng CPU và tải thư viện khi build lần đầu. Checkpoint được gắn ở chế độ chỉ đọc. Các chỉ số test và validation có sẵn trong mục **Đánh giá**. Bản Docker không có bộ ảnh test mẫu, nên hãy dùng ảnh của bạn để nhận diện.

Nếu cổng 7861 đang bận, tạo tệp `.env` cạnh `compose.yaml` với nội dung `DEMO_PORT=7862`, chạy lại lệnh trên và mở [cổng 7862](http://127.0.0.1:7862).
