"""Tạo split train/val/test theo ảnh từ bộ dữ liệu gốc."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def split_dataset(source, test_count, seed):
    images = source["images"]
    if not 0 < test_count < len(images):
        raise ValueError("Số ảnh test phải nằm giữa 0 và tổng số ảnh train.")

    labels_by_image = defaultdict(set)
    for box in source["annotations"]:
        labels_by_image[box["image_id"]].add(box["class"])
    groups = defaultdict(list)
    for image in images:
        key = tuple(sorted(labels_by_image[image["id"]]))
        groups[key].append(image["id"])

    # Chia gần cùng tỷ lệ trong từng nhóm ảnh có cùng tổ hợp lớp.
    keys = sorted(groups)
    quotas = {key: divmod(len(groups[key]) * test_count, len(images)) for key in keys}
    remaining = test_count - sum(count for count, _ in quotas.values())
    extra = set(sorted(keys, key=lambda key: -quotas[key][1])[:remaining])
    rng = random.Random(seed)
    test_ids = set()
    for key in keys:
        ids = sorted(groups[key])
        rng.shuffle(ids)
        count = quotas[key][0] + int(key in extra)
        test_ids.update(ids[:count])

    parts = {
        name: {"classes": source["classes"], "images": [], "annotations": []}
        for name in ("train", "test")
    }
    for image in images:
        parts["test" if image["id"] in test_ids else "train"]["images"].append(image)
    for annotation in source["annotations"]:
        name = "test" if annotation["image_id"] in test_ids else "train"
        parts[name]["annotations"].append(annotation)
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("public"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--test-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    annotations = args.data_root / "annotations"
    train = json.loads((annotations / "train.json").read_text(encoding="utf-8"))
    val = json.loads((annotations / "val.json").read_text(encoding="utf-8"))
    if train["classes"] != val["classes"]:
        raise ValueError("Thứ tự lớp giữa train và val khác nhau.")
    if args.output.resolve() == annotations.resolve():
        raise ValueError("Chọn thư mục đầu ra khác thư mục nhãn gốc.")

    parts = split_dataset(train, args.test_count, args.seed)
    parts["val"] = val
    seen = set()
    for name, data in parts.items():
        ids = {image["id"] for image in data["images"]}
        if len(ids) != len(data["images"]) or seen & ids:
            raise ValueError(f"Tập {name} có ID ảnh trùng.")
        seen.update(ids)
        counts = Counter()
        for box in data["annotations"]:
            if box["image_id"] not in ids:
                raise ValueError(f"Tập {name} có nhãn thuộc ảnh ngoài tập.")
            counts[box["class"]] += 1
        if set(counts) != set(data["classes"]):
            raise ValueError(f"Tập {name} cần đủ các lớp.")
        for image in data["images"]:
            if not (args.data_root / image["file_name"]).is_file():
                raise FileNotFoundError(image["file_name"])
        print(f"{name}: {len(ids)} ảnh, {sum(counts.values())} hộp | {dict(counts)}")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in parts.items():
        path = args.output / f"{name}.json"
        content = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} đã có split khác. Chọn --output mới.")
        path.write_text(content, encoding="utf-8")
    print("Split:", args.output.resolve())


if __name__ == "__main__":
    main()
