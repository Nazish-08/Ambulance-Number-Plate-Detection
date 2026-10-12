
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
COCO_JSON = ROOT / "COCO_Export" / "annotations" / "instances_default.json"
SOURCE_IMAGES = ROOT / "COCO_Export" / "images"
OUTPUT = ROOT / "Ambulance_YOLO_Annotated_Only"

random.seed(42)


def main():
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        c["id"]: c["name"] for c in coco["categories"]
    }
    plate_ids = {
        cid for cid, name in categories.items()
        if "plate" in name.lower()
    }

    if not plate_ids:
        raise ValueError("number_plate category not found")

    images = {img["id"]: img for img in coco["images"]}
    anns = defaultdict(list)

    for ann in coco["annotations"]:
        if ann["category_id"] in plate_ids:
            anns[ann["image_id"]].append(ann)

    # Only images with at least one number plate annotation
    records = []
    for image_id, image in images.items():
        if not anns[image_id]:
            continue

        filename = Path(image["file_name"]).name
        matches = list(SOURCE_IMAGES.rglob(filename))

        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one image for {filename}, found {len(matches)}"
            )

        w, h = image["width"], image["height"]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid dimensions: {filename}")

        labels = []
        for ann in anns[image_id]:
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)

            if x2 <= x1 or y2 <= y1:
                continue

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            labels.append(
                f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
            )

        if labels:
            records.append((matches[0], filename, labels))

    random.shuffle(records)

    n = len(records)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename it before rerunning."
        )

    try:
        for split, items in splits.items():
            img_dir = OUTPUT / "images" / split
            lbl_dir = OUTPUT / "labels" / split
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            for source, filename, labels in items:
                shutil.copy2(source, img_dir / filename)
                (lbl_dir / f"{Path(filename).stem}.txt").write_text(
                    "\n".join(labels) + "\n",
                    encoding="utf-8"
                )

        (OUTPUT / "data.yaml").write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            "  0: number_plate\n",
            encoding="utf-8"
        )

    except Exception:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise

    print("Annotated-only dataset created!")
    print("Annotated images:", n)
    for split, items in splits.items():
        print(f"{split}: {len(items)} images")
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
