# ObjectDetectionV1 — Phát hiện đối tượng theo kiến trúc RetinaNet

Pipeline phát hiện đối tượng cho 5 lớp **person, car, dog, cat, chair**, sử dụng backbone ResNet34 của torchvision và các module FPN, anchors, prediction heads, loss, giải mã box và NMS trong mã nguồn dự án. Không sử dụng bộ phát hiện hoàn chỉnh như YOLO, Detectron2, MMDetection hay torchvision Faster R-CNN/SSD.

**Kết quả đã kiểm chứng: mAP@0.5 = 0.786071 (78,61%) trên 1.500 ảnh validation.** Trọng số tốt nhất thuộc epoch thứ 39 trong lượt huấn luyện 40 epoch, nguồn `raw`.

Backbone của lượt này được khởi tạo từ ImageNet. Việc sử dụng backbone pretrained cần được giảng viên cho phép theo đề bài. Dự án có tham khảo sử dụng hỗ trợ AI trong quá trình phát triển; xem mục [Nguồn tham khảo](#nguồn-tham-khảo-và-phạm-vi-triển-khai).

## Quick Start — Chạy và chấm bài

Các lệnh bên dưới được thực hiện tại thư mục chứa `train.py` và `predict.py`. Lệnh được viết trên một dòng để chạy được trong cả PowerShell và terminal Linux.

### 1. Chuẩn bị môi trường

```bash
python -m venv .venv
```

Trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên Linux/Colab/Kaggle, sử dụng môi trường Python tương ứng. PyTorch và torchvision cần là hai phiên bản tương thích; bản CPU vẫn chạy được suy luận. Với CUDA, cài bản phù hợp từ [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/).

### 2. Đặt trọng số và dữ liệu

- Trọng số suy luận: `models/best.pth`, khoảng **96,78 MiB**, gồm trọng số và metadata, không có optimizer. Checkpoint không được commit vào repository GitHub; cần lấy từ gói kết quả/bài nộp được cung cấp riêng và đặt đúng đường dẫn này trước khi chạy.
- Thư mục dữ liệu trên máy phát triển: `final_public/public/`.
- Khi chấm bài, đặt bộ dữ liệu tại `public/` theo quy ước đề. Không cần nộp lại dữ liệu.
- Không cần tải backbone ImageNet hoặc truy cập kho trọng số trực tuyến khi suy luận với checkpoint đã có.

### 3. Lệnh suy luận bắt buộc

```bash
python predict.py --image_dir /path/to/images --output predictions.json
```

Script tự tìm `models/best.pth` dựa trên vị trí của file `predict.py` và tự chọn CUDA nếu có, ngược lại dùng CPU.

Ví dụ đánh giá dữ liệu hiện có trên máy:

```bash
python predict.py --image_dir ./final_public/public/val/images --output val_predictions.json
python ./final_public/public/tools/evaluate_predictions.py --ground_truth ./final_public/public/annotations/val.json --predictions val_predictions.json --output val_score.json
```

Nếu dữ liệu đặt ở `public/`, thay `./final_public/public/` bằng `./public/` trong hai lệnh.

Trích kết quả chấm chính thức của checkpoint nộp:

```json
{
  "mAP@0.5": 0.786071,
  "performance_points": 20,
  "iou_threshold": 0.5,
  "num_ground_truth_boxes": 2021,
  "num_predictions": 5335,
  "micro_precision": 0.33477,
  "micro_recall": 0.883721
}
```

`performance_points = 20` là kết quả công cụ trên validation; không khẳng định điểm cuối trên hidden test hoặc điểm sau chuẩn hóa toàn lớp.

### Tham số suy luận

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--image_dir` | Bắt buộc | Thư mục ảnh cần nhận diện, không cần nhãn |
| `--output` | Bắt buộc | JSON đầu ra |
| `--checkpoint` | `models/best.pth` cạnh mã nguồn | Checkpoint có `model` và `config` |
| `--batch_size` | 8 | Số ảnh mỗi batch |
| `--num_workers` | 2 | Số worker đọc ảnh |
| `--conf_thresh` | 0.25 | Giữ dự đoán có confidence vượt ngưỡng |
| `--nms_thresh` | 0.4 | Ngưỡng IoU cho NMS theo lớp |
| `--max_detections` | 100 | Số hộp tối đa mỗi ảnh |
| `--device` | `auto` | `auto`, `cpu` hoặc `cuda` |

Ví dụ cấu hình ít tài nguyên:

```bash
python predict.py --image_dir ./demo_images --output demo_predictions.json --device cpu --batch_size 1 --num_workers 0
```

## Yêu cầu đề bài → Implementation map

| Yêu cầu | File | Cách thực hiện |
|---|---|---|
| Đọc ảnh và nhãn 5 lớp | `utils/dataset.py` | `DetectionDataset`, JSON theo định dạng đề |
| Resize, normalize, nhiều đối tượng | `utils/dataset.py` | Pillow, tensor ảnh và dictionary target |
| Tăng cường dữ liệu | `utils/dataset.py` | Lật ngang, crop, màu sắc, mosaic, mixup và các phép bổ sung |
| CNN trích xuất đặc trưng | `models/backbone.py` | torchvision ResNet; có lựa chọn Tiny trong `tiny_backbone.py` |
| Đặc trưng đa tỷ lệ | `models/fpn.py` | FPN P3–P7, GroupNorm |
| Sinh anchors | `utils/anchors.py` | 3 scales × 5 ratios = 15 anchors/vị trí |
| Phân lớp và định vị | `models/head.py` | Hai subnet dùng chung trên các mức FPN |
| Matching và loss | `models/loss.py` | IoU matching, focal loss, GIoU/CIoU/Smooth L1 |
| Encode/decode box, IoU | `utils/box_utils.py` | Tọa độ góc, độ lệch tâm/kích thước |
| Confidence, NMS, đổi tọa độ | `models/detector.py`, `utils/nms.py` | `detect()` và NMS tự triển khai theo lớp |
| Train và chọn best | `train.py` | Chọn checkpoint theo mAP validation |
| Định dạng JSON | `predict.py` | Một mục cho mỗi tên file ảnh |
| Chấm mAP | `utils/metrics.py`, công cụ đề | AP nội bộ và bộ chấm chính thức |
| Demo trình duyệt | `web_demo.py` | Gradio, tách khỏi entry point chấm bài |

## 1. Quy trình xử lý dữ liệu — Data Pipeline

### 1.1. Bộ đọc dữ liệu

Dataset sử dụng đúng JSON của đề, không phải nhãn COCO `[x, y, width, height]`:

```json
{
  "classes": ["person", "car", "dog", "cat", "chair"],
  "images": [
    {"id": "example.jpg", "file_name": "train/images/example.jpg", "width": 500, "height": 375}
  ],
  "annotations": [
    {"image_id": "example.jpg", "class": "dog", "bbox": [48, 72, 210, 356]}
  ]
}
```

`load_annotations()` nhóm các annotation theo ảnh. Nhãn lớp được ánh xạ sang chỉ số `0…4` theo `classes`; ảnh không có annotation vẫn được giữ. Ảnh được đọc bằng Pillow và chuyển sang RGB.

Quy ước box xuyên suốt: **`[xmin, ymin, xmax, ymax]`**, theo pixel. `image_dir` có thể trỏ trực tiếp tới thư mục `train/images` hoặc `val/images`.

### 1.2. Resize và chuẩn hóa

Ảnh được resize trực tiếp thành hình vuông bằng bilinear, mặc định `512 × 512`; không dùng letterbox/padding. Box được nhân riêng theo tỷ lệ chiều ngang và chiều dọc:

```text
x_mới = x_gốc × 512 / width_gốc
y_mới = y_gốc × 512 / height_gốc
```

Pixel chuyển thành float trong `[0, 1]`, sau đó normalize:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
normalized = (pixel - mean) / std
```

Box được cắt vào biên ảnh và loại nếu chiều rộng hoặc chiều cao không vượt 1 pixel sau bước resize thông thường. Các phép augmentation hình học cập nhật box tương ứng.

### 1.3. Xử lý batch nhiều đối tượng

Mỗi mẫu trả về ảnh và dictionary:

```python
target = {
    "boxes": boxes,             # [N, 4], float32
    "labels": labels,           # [N], int64
    "image_id": image_id,
    "orig_size": (height, width),
}
```

`collate_fn()` stack ảnh thành `[B, 3, H, W]`, giữ targets dưới dạng list vì mỗi ảnh có số vật thể khác nhau. Không thêm lớp background vào danh sách 5 lớp; anchor nền có target phân lớp bằng 0 trước smoothing.

### 1.4. Tăng cường dữ liệu

Các phép xử lý được viết bằng Pillow, torchvision transforms functional và PyTorch.

| Phép biến đổi | Cấu hình trong code |
|---|---|
| Màu sắc | Xác suất 0.8; brightness/contrast/saturation trong 0.5–1.5, hue ±0.1 |
| Grayscale | Xác suất 0.1 bên trong nhánh thay đổi màu |
| Crop ngẫu nhiên | Xác suất 0.5; giữ box có tâm trong vùng crop, cắt box theo vùng |
| Lật ngang | Xác suất 0.5 |
| Lật dọc | Xác suất 0.15 |
| Random erase | Xác suất 0.2 sau normalize trong nhánh tải/augment ảnh |
| Mosaic | Xác suất 0.4 khi train và dataset có trên 3 ảnh; ghép 4 ảnh |
| Mixup | Xác suất 0.15 khi bật; trọng số pha trong 0.4–0.6, gộp danh sách box |

Các xác suất áp dụng trong từng nhánh xử lý; mosaic gọi augmentation cho từng ảnh thành phần. Validation và suy luận không sử dụng augmentation ngẫu nhiên.

### 1.5. Multi-scale

Code hỗ trợ `--multi_scale`; train chọn một kích thước chung cho mỗi epoch và đặt `dataset._epoch_img_size` trước khi lấy batch. Danh sách kích thước được tạo bởi `_build_ms_sizes()` và có thể giới hạn bằng `--ms_max_size`.

**Lượt train đạt 78,61% không bật multi-scale**, dùng cố định 512. Việc có hỗ trợ tính năng không có nghĩa tính năng đó đã được dùng trong kết quả báo cáo.

## 2. Kiến trúc mô hình — Model Architecture

```text
Ảnh [B, 3, 512, 512]
        │
        ├── ResNet34 → C3, C4, C5 (stride 8, 16, 32)
        │
        ├── FPN → P3, P4, P5, P6, P7 (256 kênh)
        │
        ├── Anchors: 15 / vị trí, tổng 81.840 / ảnh
        │
        ├── Classification head → [B, 81840, 5]
        └── Regression head     → [B, 81840, 4]
                    │
              Decode → confidence → NMS theo lớp
                    │
              Đổi tọa độ ảnh gốc → JSON
```

### 2.1. Backbone

| Lựa chọn | Kênh C3 | Kênh C4 | Kênh C5 |
|---|---:|---:|---:|
| ResNet18 | 128 | 256 | 512 |
| ResNet34 — mô hình đã chấm | 128 | 256 | 512 |
| ResNet50 | 512 | 1024 | 2048 |
| Tiny, width multiplier 1.0 | 128 | 256 | 512 |

ResNet bỏ phần phân loại cuối, dùng stem và layer1–layer4. Stem và layer1 được đóng băng ở cấu hình mặc định; các BatchNorm backbone được giữ ở chế độ eval khi train và ngừng cập nhật tham số affine.

TinyBackbone dùng Conv + GroupNorm + LeakyReLU, hai tầng MaxPool đầu và các residual stage `(2, 3, 3)` khối. Tiny luôn khởi tạo ngẫu nhiên, hỗ trợ width multiplier. Chưa có kết quả validation đầy đủ cho các backbone thay thế trong lượt báo cáo này.

### 2.2. FPN

Ba lateral convolution `1 × 1` chiếu C3/C4/C5 về 256 kênh, kèm GroupNorm. Nhánh top-down dùng nearest-neighbor upsampling và phép cộng:

```text
P5 = Smooth(Lateral(C5))
P4 = Smooth(Lateral(C4) + Upsample(P5))
P3 = Smooth(Lateral(C3) + Upsample(P4))
P6 = Conv3x3_stride2 + GN, đầu vào ReLU(P5)
P7 = Conv3x3_stride2 + GN, đầu vào ReLU(P6)
```

`Smooth` là Conv `3 × 3` + GroupNorm + ReLU. GroupNorm không sử dụng thống kê theo batch như BatchNorm.

### 2.3. Anchor Generator

Scales chung: `1`, `2^(1/3)`, `2^(2/3)`. Ratios theo quy ước `width / height`: `0.33, 0.5, 1, 2, 3`.

| Level | Stride | Base size | Feature map tại input 512 | Số anchors |
|---|---:|---:|---|---:|
| P3 | 8 | 32 | 64 × 64 | 61.440 |
| P4 | 16 | 64 | 32 × 32 | 15.360 |
| P5 | 32 | 128 | 16 × 16 | 3.840 |
| P6 | 64 | 256 | 8 × 8 | 960 |
| P7 | 128 | 512 | 4 × 4 | 240 |
| Tổng | | | | **81.840** |

Với stride `s`, tâm anchor nằm tại `((j + 0.5)s, (i + 0.5)s)`. Với base size `b`, scale `k`, ratio `r`:

```text
w = b × k × sqrt(r)
h = b × k / sqrt(r)
```

Thứ tự sinh anchor khớp thứ tự head: level → hàng → cột → scale → ratio.

### 2.4. Detection Heads

Hai subnet độc lập cho classification và regression, mỗi subnet gồm **4 block depthwise-separable Conv + GroupNorm + GELU** và một convolution đầu ra `3 × 3`. Trọng số subnet được chia sẻ giữa các level FPN.

- Classification có `15 × 5 = 75` kênh đầu ra, reshape thành `[B, A, 5]`.
- Regression có `15 × 4 = 60` kênh đầu ra, reshape thành `[B, A, 4]`.
- Có đúng 5 điểm sigmoid, không có kênh background thứ sáu.
- Bias nhánh phân lớp khởi tạo theo prior 0.01: `-log((1-prior)/prior)`.

### 2.5. RetinaDetector và TTA

`forward(images)` trả về `(class_logits, box_deltas)`. Anchors được sinh và cache qua `get_anchors()`.

`detect(images, orig_sizes, ...)` thực hiện decode, lọc confidence, top-k trước NMS, NMS và đổi tọa độ. Code có thêm `detect_tta()` cho ảnh gốc và ảnh lật ngang: đưa box lật về đúng tọa độ, **ghép các dự đoán rồi NMS**. Không lấy trung bình trực tiếp hai danh sách box chưa ghép cặp.

TTA không được bật trong điểm 78,61%; CLI `predict.py` hiện tại cũng chưa cung cấp cờ `--tta`.

## 3. Hàm mất mát và quy trình huấn luyện

### 3.1. Anchor matching

`DetectionLoss._match()` tính ma trận IoU giữa anchors và ground truth:

- IoU tốt nhất `>= 0.5`: positive.
- IoU tốt nhất `< 0.4`: negative.
- Khoảng `[0.4, 0.5)`: ignore, không tham gia loss.
- Mỗi ground truth đề cử anchor IoU cao nhất làm positive, kể cả dưới ngưỡng.

Nếu nhiều ground truth chọn cùng anchor, phép gán có thể ghi đè; không khẳng định rằng mỗi ground truth luôn có một anchor riêng. Ảnh không có vật thể được xử lý như ảnh nền.

### 3.2. Focal loss — Classification

```text
FL = -alpha_t × (1 - p_t)^gamma × log(p_t)
```

| Thông số | Giá trị hiện tại |
|---|---:|
| Alpha | 0.25 |
| Gamma | 2.5 |
| Label smoothing | 0.01 |
| Target sau smoothing | `target × 0.99 + 0.005` |

Loss dùng BCE-with-logits, focal factor và alpha balancing trên cả positive và negative anchors; ignore bị loại. Tổng được chuẩn hóa bằng `max(tổng số anchors positive trong batch, 1)`.

Class weights do `train.py` tính từ số vật thể: `min(max_count / class_count, 3)`. Lượt đã chạy có vector `[1, 3, 3, 3, 3]`. Trong code, vector này điều chỉnh trọng số từng anchor thông qua target phân lớp; không phải áp dụng một cross-entropy có softmax.

### 3.3. Regression loss

Lượt có kết quả báo cáo dùng **GIoU** trên các cặp box predicted–ground truth của positive anchors:

```text
GIoU = IoU - (area(C) - union) / area(C)
L_GIoU = 1 - GIoU
```

`C` là hình chữ nhật bao cả hai box. Tính hình học cho loss bằng float32. Code cũng hỗ trợ CIoU và Smooth L1 qua `--reg_loss_type`, nhưng chưa có phép so sánh đầy đủ cho hai lựa chọn này trong báo cáo.

### 3.4. Box encoding/decoding

Độ lệch tâm và log-kích thước sử dụng weights `(10, 10, 5, 5)`:

```text
dx = 10 × (gt_cx - anchor_cx) / anchor_w
dy = 10 × (gt_cy - anchor_cy) / anchor_h
dw =  5 × log(gt_w / anchor_w)
dh =  5 × log(gt_h / anchor_h)
```

Decode chia ngược weights, cập nhật tâm và dùng `exp` cho kích thước. Độ lệch log-kích thước sau chia weights được giới hạn tối đa 4.135 trước `exp`.

### 3.5. Tổng loss và optimizer

```text
L = L_classification + 2 × L_regression_chưa_nhân_hệ_số
```

`DetectionLoss` trả regression loss đã nhân hệ số 2; train chỉ cộng hai giá trị trả về, không nhân thêm lần nữa.

| Thành phần | Cấu hình lượt train đã chạy |
|---|---|
| Backbone | ResNet34, khởi tạo ImageNet |
| Input / FPN channels | 512 / 256 |
| Batch size / Epochs | 8 / 40 |
| Optimizer | AdamW |
| LR head | 0.0001 |
| Hệ số LR backbone / FPN | 0.05 / 0.30 |
| Weight decay | 0.0005 |
| Warmup | Linear, 1.000 bước |
| Scheduler | CosineAnnealingWarmRestarts, T_mult=1, eta_min=1e-6 |
| Chu kỳ khi khởi tạo với 40 epoch | 13 × số batch/epoch |
| Gradient clipping | max_norm=5.0 |
| Tính toán train | Float32; script hiện tại không dùng AMP/GradScaler |
| Multi-scale / EMA / SWA | Đều không bật trong lượt báo cáo |

Code bỏ qua cập nhật khi loss/gradient không hữu hạn, ghi `skipped_batches` và giảm hệ số LR khi gặp nhiều batch không hữu hạn liên tiếp. EMA và SWA có trong code như tùy chọn thử nghiệm, không phải nguồn checkpoint nộp.

### 3.6. Lệnh huấn luyện

Lệnh theo đề bài, khi dữ liệu nằm ở `public/`:

```bash
python train.py --train_data ./public/annotations/train.json --val_data ./public/annotations/val.json --image_dir ./public/train/images --val_image_dir ./public/val/images --checkpoint_dir ./models/
```

Khi train thử nghiệm, dùng checkpoint directory mới để bảo vệ trọng số nộp:

```bash
python train.py --train_data ./final_public/public/annotations/train.json --val_data ./final_public/public/annotations/val.json --image_dir ./final_public/public/train/images --val_image_dir ./final_public/public/val/images --checkpoint_dir ./runs/new_experiment --backbone resnet34 --img_size 512 --fpn_channels 256 --epochs 40 --batch_size 8 --num_workers 2 --lr 0.0001 --weight_decay 0.0005 --warmup_iters 1000 --backbone_lr_factor 0.05 --fpn_lr_factor 0.30 --reg_loss_type giou --eval_conf 0.25 --eval_nms 0.4 --seed 42
```

Các cờ bổ sung: `--multi_scale`, `--ms_max_size`, `--no_mixup`, `--ema`, `--ema_decay`, `--swa`, `--swa_start`, `--use_tta` (đánh giá trong train), `--no_pretrained`, `--resume`, `--max_batches`, `--eval_max_images`.

`--no_pretrained` cho phép khởi tạo ResNet ngẫu nhiên; không được gán kết quả 78,61% cho cấu hình này. Tiny cũng chưa có điểm đầy đủ để báo cáo.

### 3.7. Checkpoints và resume

| File | Nội dung / cách dùng |
|---|---|
| `best.pth` | Trọng số tốt nhất, config, epoch, best_map, weight_source; dùng dự đoán |
| `last.pth` | Trọng số và trạng thái optimizer, scheduler, RNG, history để tiếp tục train |
| `training_history.json` | Loss, LR, mAP, AP từng lớp, thời gian và thông tin cập nhật |

Thêm `--resume /path/to/last.pth` vào lệnh train với cấu hình tương thích. `--epochs 40` là tổng mốc epoch cần đạt, không phải train thêm 40 epoch. Checkpoint chỉ lưu sau khi hoàn thành epoch; phần epoch đang chạy dở phải chạy lại nếu ngắt phiên.

Không dùng `best.pth` làm full resume vì nó không chứa optimizer. Lượt báo cáo chạy qua Colab và Kaggle, dùng một T4 mỗi lượt; code không tự dùng cả hai GPU khi môi trường có hai T4.

## 4. Suy luận — Inference

```text
Ảnh RGB → resize vuông → normalize ImageNet → forward
→ sigmoid lớp + decode box → confidence → pre-NMS top-k
→ NMS theo lớp → giới hạn 100 box → scale về ảnh gốc → JSON
```

### 4.1. Nạp mô hình

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
config = checkpoint["config"]
# config chứa num_classes, classes, backbone, fpn_channels, img_size
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

Mô hình được dựng với `pretrained=False` khi suy luận; toàn bộ trọng số đã được nạp từ checkpoint. Điều này không có nghĩa lượt train trước đó không dùng ImageNet.

### 4.2. Confidence và NMS

Sigmoid trả điểm riêng cho mỗi lớp. Mỗi cặp anchor–class vượt ngưỡng được xem là ứng viên; không chỉ lấy một lớp argmax cho mỗi anchor. Trước NMS giới hạn tối đa 1.000 ứng viên/ảnh ở đường `detect()` mặc định.

Greedy NMS sắp xếp score giảm dần, giữ hộp cao nhất rồi loại các hộp cùng lớp có IoU vượt ngưỡng. `batched_nms()` dịch tọa độ theo lớp để hộp khác lớp không loại lẫn nhau. Vì thế một vùng có thể còn cả hộp `dog` và `cat` nếu mô hình nhầm lớp.

Tăng confidence giảm dự đoán yếu nhưng có thể làm giảm recall. Thay ngưỡng không tương đương huấn luyện lại mô hình; cần chấm lại trước khi công bố điểm cho cấu hình mới.

### 4.3. Tọa độ ảnh gốc

Vì dùng resize trực tiếp, tọa độ được đổi theo hai hệ số độc lập:

```text
x_gốc = x_dự_đoán × width_gốc / width_đầu_vào
y_gốc = y_dự_đoán × height_gốc / height_đầu_vào
```

Box được clip về `[0, width] × [0, height]`. Không có bước trừ padding vì pipeline không dùng letterbox.

### 4.4. Định dạng JSON

```json
[
  {
    "image_id": "example.jpg",
    "boxes": [
      {"class": "dog", "confidence": 0.91, "bbox": [48.0, 72.0, 210.0, 356.0]}
    ]
  },
  {"image_id": "no_detection.jpg", "boxes": []}
]
```

`image_id` là tên file ảnh. Confidence trong `[0,1]`. Ảnh không phát hiện vẫn được ghi một mục. Suy luận thư mục không đọc annotation và không phụ thuộc vào vị trí tập validation. Script ghi file tạm rồi thay thế JSON đích sau khi hoàn tất.

## 5. Kết quả — Results

### 5.1. Kết quả chính thức trên validation

| Chỉ số | Giá trị |
|---|---:|
| Số ảnh validation | 1.500 |
| Ground-truth boxes | 2.021 |
| Predicted boxes | 5.335 |
| True positives | 1.786 |
| False positives | 3.549 |
| False negatives | 235 |
| **mAP@0.5** | **78,6071%** |
| Precision micro | 33,4770% |
| Recall micro | 88,3721% |

Đánh giá dùng confidence 0.25, NMS 0.4, IoU matching 0.5 và tối đa 100 hộp/ảnh. Precision/recall là tại ngưỡng này; AP tích hợp đường precision–recall theo thứ hạng điểm. mAP là trung bình AP năm lớp, không phải accuracy.

### 5.2. Per-class breakdown

| Lớp | AP@0.5 | Precision | Recall | TP / GT |
|---|---:|---:|---:|---:|
| person | 83,6002% | 43,0700% | 88,8268% | 954 / 1.074 |
| car | 77,7655% | 29,0210% | 87,9859% | 249 / 283 |
| dog | 87,8166% | 40,5983% | 92,2330% | 190 / 206 |
| cat | 89,9858% | 49,8489% | 93,7500% | 165 / 176 |
| chair | 53,8673% | 15,5844% | 80,8511% | 228 / 282 |

`chair` là lớp yếu nhất. Mô hình vẫn có thể nhầm chó–mèo, tạo nhiều hộp khác lớp trên cùng vật thể hoặc bỏ sót các vật thể đứng gần nhau. Ảnh tải từ Internet có thể khác phân phối dữ liệu train; không bảo đảm nhận diện đúng mọi ảnh.

### 5.3. Tiến trình huấn luyện của dự án

| Epoch thứ | mAP@0.5 |
|---|---:|
| 1 | 11,27% |
| 3 | 50,11% |
| 10 | 72,67% |
| 20 | 76,42% |
| 26 | 78,58% |
| 30 | 75,51% |
| 39 — best | **78,61%** |
| 40 — cuối | 78,37% |

Các epoch trong `training_history.json` đánh số từ 0; `epoch=38` là epoch thứ 39. File lịch sử của bài mẫu đạt 80,74% không phải lịch sử của lượt train này.

`time_seconds` của lượt này khoảng 510–538 giây/epoch giai đoạn Colab và khoảng 441–443 giây/epoch ở phần lớn giai đoạn Kaggle sau đó. Đây là thời gian ghi nhận trên môi trường đã chạy, không phải cam kết tốc độ trên máy khác.

## 6. Website demo cục bộ — Gradio

Cài thư viện giao diện riêng:

```bash
python -m pip install -r requirements-web.txt
python web_demo.py
```

Trong VS Code trên Windows có thể gọi trực tiếp môi trường dự án:

```powershell
.\.venv\Scripts\python.exe web_demo.py
```

Mở **http://127.0.0.1:7860**. Chọn hoặc dán ảnh, chỉnh confidence/NMS, bấm **Nhận diện ảnh**. Giao diện trả ảnh có box, bảng phát hiện, thời gian xử lý và file PNG/JSON để tải. Trên máy phát triển có ảnh thử trong `demo_images/`. Thư mục này không được commit lên GitHub; có thể tự thêm ảnh JPG để hiện ví dụ trên giao diện.

```bash
python web_demo.py --port 7861 --device cpu
```

Lệnh trên dùng khi cổng 7860 đã bị chiếm. Máy hiện tại có thể chạy CPU; ứng dụng tự chọn CUDA khi môi trường hỗ trợ. Upload giới hạn 15 MB, ảnh tối đa 24 megapixel. Kết quả được lưu trong `runs/web_demo/<mã phiên>/`.

Ứng dụng mặc định chỉ lắng nghe trên `127.0.0.1`, không bật public share link. Truy cập từ máy ở nơi khác cần triển khai/chia sẻ riêng, chưa được cấu hình trong bản này. Giữ tiến trình chạy trong lúc dùng, nhấn Ctrl+C tại terminal để dừng. Xem thêm [DEMO_WEB.md](DEMO_WEB.md).

Website là công cụ demo một ảnh; dùng `predict.py` cho đầu ra chấm chính thức. Gradio không nằm trong `requirements.txt` cốt lõi nên môi trường chấm không phải cài giao diện.

## 7. Cấu trúc mã nguồn — Project Structure

```text
ObjectDetectionV1/
├── train.py                   # Huấn luyện, validation, checkpoint
├── predict.py                 # Suy luận thư mục, xuất JSON
├── web_demo.py                # Demo Gradio độc lập
├── README.md
├── DEMO_WEB.md
├── requirements.txt           # torch, torchvision, Pillow, numpy
├── requirements-web.txt       # Thêm Gradio
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── tiny_backbone.py
│   ├── fpn.py
│   ├── head.py
│   ├── detector.py
│   ├── loss.py
│   └── best.pth
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── anchors.py
│   ├── box_utils.py
│   ├── metrics.py
│   └── nms.py
├── tools/
│   └── package_code.py
├── demo_images/               # Ảnh thử giao diện
├── training_history.json      # Lịch sử lượt train 40 epoch
├── final_public/public/       # Dữ liệu cung cấp, không cần nộp
└── runs/                      # Thử nghiệm, sao lưu, bài đóng gói, kết quả web
```

Thư mục phát triển có thể còn các script và checkpoint cũ; các file chính của pipeline hiện tại là những file liệt kê trên. Khi nộp, chỉ lấy các module cần thiết, `train.py`, `predict.py`, README, requirements và `models/best.pth`. Không đưa `.venv`, ảnh dữ liệu hoặc checkpoint thử nghiệm vào ZIP.

## 8. Kiểm tra — Tests / Verification

Các kiểm tra đã thực hiện:

- Nạp checkpoint tốt nhất bằng `strict=True`, khớp kiến trúc hiện tại.
- Lệnh suy luận chỉ có `--image_dir` và `--output` chạy được trên CPU trong bản bài nộp đã chuẩn bị.
- JSON của lượt validation được bộ chấm chính thức chấp nhận, mAP khớp metadata best.
- Train smoke bằng Tiny, một batch và hai ảnh validation chạy được; chỉ kiểm tra pipeline, không dùng điểm smoke làm kết quả mô hình.
- Web: nhận diện ảnh thử, tải PNG/JSON, trường hợp không có phát hiện và chưa chọn ảnh.

Ví dụ chạy smoke trong thư mục checkpoint riêng:

```bash
python train.py --train_data ./final_public/public/annotations/train.json --val_data ./final_public/public/annotations/val.json --image_dir ./final_public/public/train/images --val_image_dir ./final_public/public/val/images --checkpoint_dir ./runs/smoke_readme --backbone tiny_0.25 --fpn_channels 32 --img_size 128 --epochs 1 --batch_size 2 --num_workers 0 --max_batches 1 --eval_max_images 2 --no_pretrained --device cpu
```

Không có tuyên bố "80 tests passed" hoặc coverage tự động cho dự án này. Đánh giá lại đầy đủ dùng hai lệnh validation trong Quick Start.

## 9. FAQ

**Vì sao mAP 78,61% nhưng một ảnh demo vẫn sai?**

mAP là chỉ số tổng hợp trên tập validation. Nó không bảo đảm từng ảnh hoặc mỗi vật thể đều được nhận đúng. Confidence từng box cũng không phải accuracy của toàn mô hình.

**Vì sao cùng vị trí có cả box dog và cat?**

Đây là dự đoán nhầm lớp còn vượt ngưỡng. NMS thực hiện theo lớp nên không tự xóa hộp khác lớp. Tăng confidence có thể loại hộp yếu nhưng cũng có thể làm mất phát hiện đúng.

**Lỗi `KeyError: classes` khi chạy predict?**

Kiểm tra đang dùng đúng bản `predict.py` hiện tại. Checkpoint mới lưu classes và img_size bên trong `checkpoint["config"]`; không đọc ở cấp ngoài cùng. Đồng bộ đúng bản code, không train lại chỉ để sửa lỗi định dạng checkpoint.

**CUDA không khả dụng?**

Suy luận có thể dùng `--device cpu`. Có GPU vật lý chưa đủ: bản PyTorch cũng cần hỗ trợ CUDA. CPU và GPU có thể khác tốc độ và sai khác số học nhỏ.

**Không mở được website hoặc cổng 7860 đã được dùng?**

Thử trang đang chạy ở http://127.0.0.1:7860 trước. Nếu cần phiên khác, chạy `python web_demo.py --port 7861`. Không mở đồng thời nhiều phiên không cần thiết vì mỗi phiên nạp một mô hình.

**Mở bằng máy khác không được?**

127.0.0.1 trỏ về chính máy đang truy cập. Bản hiện tại phục vụ cục bộ; máy khác không truy cập qua địa chỉ đó.

**Có cần train lại để demo?**

Không. Chỉ cần code, thư viện và `models/best.pth`. Muốn resume train thì cần `last.pth` đầy đủ trạng thái.

**Có Docker không?**

Bản dự án hiện tại chạy bằng Python/venv; chưa có Dockerfile được chuẩn bị và kiểm chứng cho bài này. Không sử dụng lệnh Docker hoặc các con số môi trường của README bài mẫu làm hướng dẫn cho dự án này.

## Nguồn tham khảo và phạm vi triển khai

- Có sử dụng hỗ trợ AI trong quá trình viết, giải thích, gỡ lỗi, thử nghiệm và chuẩn bị tài liệu; sinh viên cần hiểu và có khả năng bảo vệ mã nguồn.
- Backbone ResNet dùng torchvision, pretrained ImageNet trong lượt báo cáo. Các module phát hiện nằm trong mã nguồn dự án; không dùng detector hoàn chỉnh đóng gói sẵn.
- ResNet: [Deep Residual Learning](https://arxiv.org/abs/1512.03385).
- FPN: [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144).
- RetinaNet / focal loss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
- GIoU: [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630).
- Gradio: [Tài liệu chính thức](https://www.gradio.app/docs).

README mô tả trạng thái code và kết quả thực tế của ObjectDetectionV1. Các tính năng tùy chọn, kết quả của bài mẫu và kết quả hidden test được phân biệt với lượt validation đã kiểm chứng.
