"""Website nhận diện cục bộ: python web_demo.py."""

import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms import functional as TF

from models.detector import RetinaDetector

ROOT = Path(__file__).resolve().parent
COLORS = ["#2563eb", "#c2410c", "#15803d", "#9333ea", "#be123c"]
CSS = """
.gradio-container {max-width:1250px !important; margin:auto;}
#hero {padding:26px 30px; border-radius:18px; background:linear-gradient(115deg,#102b3f,#125d68);color:white;margin-bottom:18px;}
#hero h1,#hero p {color:white !important;}
#hero h1 {font-size:30px;margin:0 0 10px;}
#hero p {opacity:.9;margin:0;line-height:1.65;}
#result-status {border-radius:12px;padding:12px 18px;background:#edf7f5;}
footer {display:none !important;}
"""


class DetectorService:
    def __init__(self, checkpoint_path, device="auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng. Chọn --device cpu.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        self.classes = config["classes"]
        self.size = config["img_size"]
        self.best_map = checkpoint.get("best_map")
        self.model = RetinaDetector(
            num_classes=config["num_classes"], backbone=config["backbone"],
            pretrained=False, fpn_channels=config["fpn_channels"], img_size=self.size,
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, image, confidence, nms):
        if image is None:
            raise ValueError("Bạn hãy chọn ảnh trước khi nhận diện.")
        if image.width * image.height > 24_000_000:
            raise ValueError("Ảnh quá lớn. Vui lòng dùng ảnh dưới 24 megapixel.")
        if not 0 <= confidence <= 1 or not 0 <= nms <= 1:
            raise ValueError("Ngưỡng phải nằm trong [0, 1].")
        with self.lock:
            started = time.perf_counter()
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            resized = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            tensor = TF.normalize(TF.to_tensor(resized),
                                  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            result = self.model.detect(
                tensor.unsqueeze(0).to(self.device), [(height, width)],
                conf_thresh=confidence, nms_thresh=nms, max_detections=100,
            )[0]
            detections = []
            for box, score, label in zip(result["boxes"].cpu().tolist(),
                                         result["scores"].cpu().tolist(),
                                         result["labels"].cpu().tolist()):
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detections.append({"class": self.classes[label],
                                   "confidence": float(score), "bbox": box})
            elapsed = time.perf_counter() - started
        return image, detections, elapsed


def annotate(image, detections, classes):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(14, min(36, round(min(image.size) / 32)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    thickness = max(2, round(min(image.size) / 200))
    for detection in detections:
        box = detection["bbox"]
        color = COLORS[classes.index(detection["class"]) % len(COLORS)]
        draw.rectangle(box, outline=color, width=thickness)
        label = f'{detection["class"]}  {detection["confidence"]:.0%}'
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(box[0], image.width - text_width - 10))
        y = max(0, box[1] - text_height - 12)
        draw.rectangle((x, y, x + text_width + 10, y + text_height + 10), fill=color)
        draw.text((x + 5, y + 4 - bounds[1]), label, font=font, fill="white")
    return canvas


def build_demo(service):
    def run(image, confidence, nms):
        try:
            original, detections, elapsed = service.predict(image, confidence, nms)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        output_image = annotate(original, detections, service.classes)
        folder = ROOT / "runs" / "web_demo" / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        image_path, json_path = folder / "detections.png", folder / "detections.json"
        output_image.save(image_path)
        payload = {"image_id": "uploaded_image", "boxes": detections}
        json_path.write_text(json.dumps([payload], ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        rows = [[i + 1, d["class"], round(d["confidence"] * 100, 2),
                 *[round(v, 1) for v in d["bbox"]]] for i, d in enumerate(detections)]
        message = (f"**{len(detections)} vật thể** · {elapsed:.2f} giây · "
                   f"{original.width} × {original.height} px · {service.device.type.upper()}")
        if not detections:
            message += "\n\nKhông có phát hiện ở ngưỡng này. Bạn có thể giảm ngưỡng hoặc thử ảnh khác."
        return output_image, rows, message, str(image_path), str(json_path)

    with gr.Blocks(title="ObjectDetectionV1 | Nhận diện ảnh", analytics_enabled=False,
                   delete_cache=(3600, 86400)) as demo:
        gr.HTML('<div id="hero"><h1>Nhận diện đối tượng trong ảnh</h1><p>Chọn một bức ảnh và khám phá những gì mô hình nhìn thấy.<br>5 lớp: person · car · dog · cat · chair</p></div>')
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(type="pil", sources=["upload", "clipboard"],
                                  label="1. Chọn ảnh", height=380)
                with gr.Accordion("Điều chỉnh nhận diện", open=True):
                    confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                           label="Ngưỡng tin cậy", info="Tăng để giữ các dự đoán chắc chắn hơn.")
                    nms = gr.Slider(0.1, 0.8, value=0.4, step=0.05,
                                   label="Ngưỡng NMS", info="Giảm để loại hộp trùng mạnh hơn.")
                with gr.Row():
                    button = gr.Button("Nhận diện ảnh", variant="primary", size="lg")
                    clear = gr.Button("Làm mới", size="lg")
            with gr.Column(scale=1):
                result = gr.Image(type="pil", label="2. Kết quả nhận diện", height=380,
                                  interactive=False, format="png")
                status = gr.Markdown("Chưa có kết quả. Chọn ảnh rồi bấm **Nhận diện ảnh**.",
                                     elem_id="result-status")
                with gr.Row():
                    image_download = gr.File(label="Tải ảnh có hộp bao", interactive=False)
                    json_download = gr.File(label="Tải kết quả JSON", interactive=False)
        table = gr.Dataframe(headers=["STT", "Lớp", "Tin cậy (%)", "xmin", "ymin", "xmax", "ymax"],
                             datatype=["number", "str", "number", "number", "number", "number", "number"],
                             label="Chi tiết phát hiện", interactive=False, value=[])
        examples = sorted((ROOT / "demo_images").glob("*.jpg"))
        if examples:
            gr.Examples(examples=[[str(p)] for p in examples[:4]], inputs=[source],
                        label="Thử ảnh mẫu")
        gr.Markdown("Mô hình chỉ nhận diện 5 lớp đã học và có thể dự đoán sai. "
                    "**mAP@0.5 validation: 78,61%** là điểm trên tập kiểm định, không phải độ tin cậy của mỗi ảnh. "
                    "Ảnh và kết quả được xử lý cục bộ; các file kết quả lưu trong `runs/web_demo/`.")
        button.click(run, [source, confidence, nms],
                     [result, table, status, image_download, json_download],
                     concurrency_limit=1, api_name="detect")
        clear.click(lambda: (None, None, [], "Chọn ảnh để bắt đầu.", None, None),
                    outputs=[source, result, table, status, image_download, json_download], queue=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/best.pth")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads phải dương")
    torch.set_num_threads(args.threads)
    service = DetectorService(args.checkpoint, args.device)
    print(f"Model ready: {service.device}; checkpoint={args.checkpoint}", flush=True)
    demo = build_demo(service)
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1", server_port=args.port, share=False,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(primary_hue="teal", font=["Arial", "sans-serif"]), css=CSS,
        allowed_paths=[str(ROOT / "runs/web_demo"), str(ROOT / "demo_images")],
        max_file_size="15mb", show_error=False,
    )


if __name__ == "__main__":
    main()
