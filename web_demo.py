"""Mở giao diện suy luận RetinaNet trên máy chạy checkpoint."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from predict import format_prediction, load_detector
from utils.dataset import image_to_tensor


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
RESULTS = ROOT / "resnet34_full"
MAX_UPLOAD = 12 * 1024 * 1024
MAX_PIXELS = 32_000_000
SAMPLE_NAMES = {
    "img_0550469e61f4.jpg": "Trên phố",
    "img_30408b37cc12.jpg": "Xe đua",
    "img_642e889105ce.jpg": "Mèo",
    "img_1f8f83da5303.jpg": "Chó",
    "img_05a632a6489b.jpg": "Ghế",
}
ASSETS = {
    "/": (WEB / "index.html", "text/html; charset=utf-8"),
    "/styles.css": (WEB / "styles.css", "text/css; charset=utf-8"),
    "/demo.js": (WEB / "demo.js", "text/javascript; charset=utf-8"),
}


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


class Demo:
    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        self.model, config, self.image_size = load_detector(checkpoint, self.device)
        self.classes = config["classes"]
        self.backbone = config["backbone"]
        self.lock = threading.Lock()
        data = read_json(ROOT / "splits/test.json", {"images": [], "annotations": []})
        self.samples = {}
        for item in data["images"]:
            name = item["id"]
            if name in SAMPLE_NAMES:
                path = ROOT / "public/train/images" / name
                if path.is_file():
                    self.samples[name] = (path, [])
        for annotation in data["annotations"]:
            sample = self.samples.get(annotation["image_id"])
            if sample is not None:
                sample[1].append({"class": annotation["class"], "bbox": annotation["bbox"]})

    def info(self):
        history = read_json(RESULTS / "training_history.json", {})
        test = read_json(RESULTS / "test_score.json")
        return {
            "backbone": self.backbone,
            "device": str(self.device),
            "image_size": self.image_size,
            "classes": self.classes,
            "epochs": len(history.get("epoch", [])),
            "test_count": len(read_json(ROOT / "splits/test.json", {"images": []})["images"]),
            "test": test,
            "validation": read_json(RESULTS / "val_score.json"),
            "samples": [
                {"id": name, "title": title}
                for name, title in SAMPLE_NAMES.items() if name in self.samples
            ],
        }

    def infer(self, image_bytes, sample_id, confidence, nms, tta):
        original = self.samples[sample_id][0] if sample_id else BytesIO(image_bytes)
        with Image.open(original, formats=["JPEG", "PNG", "WEBP", "BMP"]) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Ảnh vượt quá 32 megapixel. Hãy giảm kích thước rồi thử lại.")
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor = image_to_tensor(image).unsqueeze(0)
        detect = self.model.detect_tta if tta else self.model.detect
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            result, = detect(tensor.to(self.device), [(height, width)], conf_thresh=confidence,
                             nms_thresh=nms)
            prediction = format_prediction(sample_id, result, self.classes)
            elapsed = time.perf_counter() - started
        return {
            "width": width, "height": height, "latency_ms": round(elapsed * 1000),
            "boxes": prediction["boxes"],
            "ground_truth": self.samples[sample_id][1] if sample_id else None,
        }


class DemoHandler(BaseHTTPRequestHandler):
    def reply(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.reply(status, "application/json; charset=utf-8", data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ASSETS:
            file, content_type = ASSETS[path]
            self.reply(HTTPStatus.OK, content_type, file.read_bytes())
        elif path == "/api/project":
            self.send_json(HTTPStatus.OK, self.server.demo.info())
        elif path.startswith("/samples/") and path.removeprefix("/samples/") in self.server.demo.samples:
            file, _ = self.server.demo.samples[path.removeprefix("/samples/")]
            self.reply(HTTPStatus.OK, "image/jpeg", file.read_bytes())
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy nội dung."})

    def do_POST(self):
        request = urlsplit(self.path)
        if request.path != "/api/detect":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy nội dung."})
            return
        query = parse_qs(request.query)
        sample_id = query.get("sample", [None])[0]
        try:
            confidence = float(query.get("confidence", ["0.25"])[0])
            nms = float(query.get("nms", ["0.4"])[0])
            tta = query.get("tta", ["0"])[0] == "1"
            if not 0 <= confidence <= 1 or not 0 <= nms <= 1:
                raise ValueError("Confidence và NMS phải nằm từ 0 đến 1.")
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_UPLOAD:
                self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Ảnh tối đa 12 MiB."})
                return
            if sample_id and sample_id not in self.server.demo.samples:
                raise ValueError("Ảnh mẫu không có trong tập test.")
            if length < 0 or (not sample_id and not length):
                raise ValueError("Hãy chọn một ảnh trước khi nhận diện.")
            self.connection.settimeout(20)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Ảnh tải lên chưa đầy đủ.")
            result = self.server.demo.infer(payload, sample_id, confidence, nms, tta)
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            message = str(error) if type(error) is ValueError else "Hãy chọn ảnh JPG, PNG, WebP hoặc BMP hợp lệ."
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": message})
            return
        except (OSError, socket.timeout):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Không đọc được ảnh. Hãy thử một ảnh khác."})
            return
        except RuntimeError as error:
            self.log_error("Inference: %s", error)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Mô hình gặp lỗi khi xử lý ảnh."})
            return
        self.send_json(HTTPStatus.OK, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ lắng nghe của máy chủ.")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--checkpoint", type=Path, default=RESULTS / "best.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Thiếu checkpoint {args.checkpoint}")
    if args.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    with ThreadingHTTPServer((args.host, args.port), DemoHandler) as server:
        server.demo = Demo(args.checkpoint, args.device)
        display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"Demo RetinaNet: http://{display_host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
