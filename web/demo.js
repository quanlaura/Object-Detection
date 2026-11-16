"use strict";

const byId = (id) => document.getElementById(id);
const colors = { person: "#a9e681", car: "#86d9fe", dog: "#ffca8c", cat: "#e3b6fc", chair: "#f9a2ad" };
const percent = new Intl.NumberFormat("vi", { style: "percent", maximumFractionDigits: 2 });
const number = new Intl.NumberFormat("vi");
const decimal = new Intl.NumberFormat("vi", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const upload = byId("upload");
const detect = byId("detect");
const save = byId("export");
const paste = byId("paste-image");
const resultStage = byId("result-stage");
const truth = byId("truth");
const maxUpload = 12 * 1024 * 1024;
const clipboardExtensions = new Map([
  ["image/png", "png"], ["image/jpeg", "jpg"], ["image/webp", "webp"],
  ["image/bmp", "bmp"], ["image/x-ms-bmp", "bmp"],
]);
let modelReady = false;
let selection = null;
let result = null;
let activeRequest = null;

function text(tag, value, className) {
  const element = document.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  return element;
}

function setStatus(message, isError = false) {
  const status = byId("status");
  status.textContent = message;
  status.hidden = !message;
  status.classList.toggle("error", isError);
}

function classLabel(name) {
  const label = text("span", name, "label-dot");
  label.style.setProperty("--class-color", colors[name] || "#6eae7e");
  label.prepend(document.createElement("i"));
  return label;
}

function fitPhoto(photo) {
  const image = photo.querySelector("img");
  if (photo.hidden || !image.naturalWidth) return;
  const stage = photo.parentElement;
  const scale = Math.min(stage.clientWidth / image.naturalWidth, stage.clientHeight / image.naturalHeight);
  photo.style.width = image.naturalWidth * scale + "px";
  photo.style.height = image.naturalHeight * scale + "px";
}

function placeImage(side, src) {
  const photo = byId(side + "-photo");
  const image = byId(side + "-image");
  photo.hidden = !src;
  byId(side + "-placeholder").hidden = Boolean(src);
  if (!src) {
    image.removeAttribute("src");
    return;
  }
  image.src = src;
  fitPhoto(photo);
}

function drawBoxes(layer, boxes, width, height, ground = false) {
  layer.replaceChildren();
  for (const box of boxes) {
    const [left, top, right, bottom] = box.bbox;
    const item = text("div", "", ground ? "detection-box ground" : "detection-box");
    item.style.setProperty("--class-color", colors[box.class] || "#a9e681");
    item.style.left = left / width * 100 + "%";
    item.style.top = top / height * 100 + "%";
    item.style.width = (right - left) / width * 100 + "%";
    item.style.height = (bottom - top) / height * 100 + "%";
    const label = ground ? box.class : box.class + " " + Math.round(box.confidence * 100) + "%";
    const tag = text("span", label, top / height < 0.08 ? "detection-tag inside" : "detection-tag");
    item.append(tag);
    layer.append(item);
  }
}

function showTruth() {
  const layer = byId("truth-boxes");
  if (truth.checked && result?.ground_truth) {
    drawBoxes(layer, result.ground_truth, result.width, result.height, true);
  } else {
    layer.replaceChildren();
  }
}

function selectImage(next) {
  activeRequest?.abort();
  activeRequest = null;
  if (selection?.file) URL.revokeObjectURL(selection.url);
  selection = next;
  result = null;
  truth.checked = false;
  byId("truth-label").hidden = true;
  byId("truth-boxes").replaceChildren();
  byId("predicted-boxes").replaceChildren();
  byId("prediction-details").hidden = true;
  byId("image-name").textContent = next.title || next.name;
  byId("detection-count").textContent = "Chưa chạy nhận diện";
  byId("runtime").textContent = "";
  resultStage.setAttribute("aria-busy", "false");
  placeImage("input", next.url);
  placeImage("result", null);
  save.disabled = true;
  detect.disabled = !modelReady;
  detect.firstChild.textContent = "Chạy nhận diện ";
  for (const button of byId("samples").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.id === next.sample));
  }
  setStatus("");
}

function selectFile(file) {
  if (!file) return false;
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
    setStatus("Hãy chọn ảnh JPG, PNG, WebP hoặc BMP.", true);
    return false;
  }
  if (file.size > maxUpload) {
    setStatus("Ảnh tối đa 12 MiB.", true);
    return false;
  }
  selectImage({ file, name: file.name, url: URL.createObjectURL(file) });
  return true;
}

function usePastedImage(blob, mime) {
  const extension = clipboardExtensions.get(mime);
  if (blob && extension && selectFile(new File([blob], "anh-da-dan." + extension, { type: mime }))) {
    runDetection();
  }
}

async function readClipboardImage() {
  const previousSelection = selection;
  paste.disabled = true;
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const mime = item.types.find((type) => clipboardExtensions.has(type));
      if (!mime) continue;
      const blob = await item.getType(mime);
      if (selection === previousSelection) usePastedImage(blob, mime);
      return;
    }
    if (selection === previousSelection) {
      setStatus("Hãy sao chép một ảnh JPG, PNG, WebP hoặc BMP rồi bấm Dán ảnh.", true);
    }
  } catch {
    if (selection === previousSelection) {
      setStatus("Hãy bấm vào trang rồi nhấn Ctrl+V để dán ảnh (⌘+V trên Mac).", true);
    }
  } finally {
    paste.disabled = false;
  }
}

function displayResult(data, current) {
  result = data;
  placeImage("result", current.url);
  drawBoxes(byId("predicted-boxes"), data.boxes, data.width, data.height);
  byId("truth-label").hidden = data.ground_truth === null;
  showTruth();
  byId("detection-count").textContent = data.boxes.length
    ? number.format(data.boxes.length) + " hộp dự đoán"
    : "Không có phát hiện ở ngưỡng này";
  byId("runtime").textContent = number.format(data.latency_ms) + " ms";
  const rows = byId("prediction-rows");
  rows.replaceChildren();
  for (const box of data.boxes) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.append(classLabel(box.class));
    row.append(label, text("td", percent.format(box.confidence)), text("td", box.bbox.join(", ")));
    rows.append(row);
  }
  byId("prediction-details").hidden = !data.boxes.length;
  save.disabled = false;
}

async function runDetection() {
  if (!modelReady || !selection || activeRequest) return;
  const current = selection;
  const controller = new AbortController();
  activeRequest = controller;
  const params = new URLSearchParams({
    confidence: byId("confidence").value,
    nms: byId("nms").value,
    tta: byId("tta").checked ? "1" : "0",
  });
  if (current.sample) params.set("sample", current.sample);
  detect.disabled = true;
  detect.firstChild.textContent = "Đang nhận diện ";
  resultStage.setAttribute("aria-busy", "true");
  setStatus("Đang xử lý ảnh…");
  try {
    const response = await fetch("/api/detect?" + params, {
      method: "POST",
      body: current.file || null,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    if (activeRequest !== controller) return;
    displayResult(data, current);
    setStatus("");
  } catch (error) {
    if (activeRequest === controller) setStatus(error.message || "Không nhận được kết quả.", true);
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      resultStage.setAttribute("aria-busy", "false");
      detect.disabled = false;
      detect.firstChild.textContent = "Chạy nhận diện ";
    }
  }
}

function populateReport(info) {
  const test = info.test;
  byId("hero-map").textContent = test ? percent.format(test["mAP@0.5"]) : "—";
  byId("hero-test-count").textContent = info.test_count ? number.format(info.test_count) + " ảnh test" : "Kết quả đã lưu";
  byId("training-summary").textContent = info.epochs ? "ResNet34 · " + info.epochs + " epoch" : "";
  for (const [label, score] of [["Validation", info.validation], ["Test", test]]) {
    const row = document.createElement("tr");
    const keys = ["mAP@0.5", "micro_precision", "micro_recall"];
    row.append(text("td", label), ...keys.map((key) => text("td", score ? percent.format(score[key]) : "—")));
    byId("metric-rows").append(row);
  }
  if (test) {
    for (const [name, scores] of Object.entries(test.per_class)) {
      const row = text("div", "", "ap-row");
      const bar = text("div", "", "bar");
      const fill = document.createElement("span");
      fill.style.width = scores.ap * 100 + "%";
      bar.append(fill);
      row.append(classLabel(name), bar, text("b", percent.format(scores.ap)));
      byId("class-bars").append(row);
    }
  } else {
    byId("class-bars").append(text("p", "Chưa có điểm test đã lưu.", "note"));
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/project");
    if (!response.ok) throw new Error("Không đọc được cấu hình.");
    const info = await response.json();
    const device = byId("device");
    device.lastChild.textContent = info.backbone.toUpperCase() + " · " + info.device.toUpperCase();
    for (const name of info.classes) byId("class-legend").append(classLabel(name));
    populateReport(info);
    modelReady = true;
    const sampleChoices = info.samples.map((sample) => ({
      sample: sample.id, title: sample.title, name: sample.id, url: "/samples/" + encodeURIComponent(sample.id),
    }));
    for (const example of sampleChoices) {
      const button = text("button", "", "sample");
      button.type = "button";
      button.dataset.id = example.sample;
      button.setAttribute("aria-pressed", "false");
      const image = document.createElement("img");
      image.src = example.url;
      image.alt = "";
      button.append(image, text("span", example.title));
      button.addEventListener("click", () => {
        selectImage(example);
        runDetection();
      });
      byId("samples").append(button);
    }
    byId("sample-strip").hidden = !sampleChoices.length;
    if (!selection && sampleChoices.length) {
      selectImage(sampleChoices[0]);
      await runDetection();
    } else {
      detect.disabled = !selection;
    }
  } catch {
    const device = byId("device");
    device.classList.add("failed");
    device.lastChild.textContent = "Mất kết nối";
    setStatus("Không kết nối được mô hình. Hãy kiểm tra máy chủ demo rồi tải lại trang.", true);
  }
}

for (const side of ["input", "result"]) {
  const photo = byId(side + "-photo");
  byId(side + "-image").addEventListener("load", () => fitPhoto(photo));
  new ResizeObserver(() => fitPhoto(photo)).observe(photo.parentElement);
}
for (const id of ["confidence", "nms"]) {
  byId(id).addEventListener("input", () => {
    byId(id + "-value").textContent = decimal.format(Number(byId(id).value));
  });
}
upload.addEventListener("change", () => {
  selectFile(upload.files[0]);
  upload.value = "";
});
paste.addEventListener("click", readClipboardImage);
document.addEventListener("paste", (event) => {
  const clipboard = event.clipboardData;
  if (!clipboard) return;
  const image = Array.from(clipboard.items).find(
    (item) => item.kind === "file" && clipboardExtensions.has(item.type),
  );
  const itemFile = image?.getAsFile();
  const file = itemFile ||
    Array.from(clipboard.files).find((item) => clipboardExtensions.has(item.type));
  if (file) {
    event.preventDefault();
    usePastedImage(file, itemFile ? image.type : file.type);
  }
});
const dropArea = byId("drop-area");
dropArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropArea.classList.add("dragging");
});
dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragging"));
dropArea.addEventListener("drop", (event) => {
  event.preventDefault();
  dropArea.classList.remove("dragging");
  selectFile(event.dataTransfer.files[0]);
});
truth.addEventListener("change", showTruth);
detect.addEventListener("click", runDetection);
save.addEventListener("click", () => {
  if (!result || !selection) return;
  const data = [{ image_id: selection.name, boxes: result.boxes }];
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "predictions.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
initialize();
