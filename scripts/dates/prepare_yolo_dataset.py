"""
Converts a Label Studio JSON export (rectanglelabels annotations) of date
regions into an Ultralytics YOLO-format detection dataset: images/ +
labels/ directories plus a data.yaml, ready for `yolo train` or
train_date_detector.py.

Stage 1 of the two-stage date-extraction approach: pools every date type
into a single "date" class by default (--per-class opts out), since
detecting *where* dates are needs far fewer labeled instances per class
than classifying *which type* does - with a handful of annotated pages,
pooling is what makes stage 1 trainable at all right now. Run with
--report-only first to see current per-class counts; don't attempt
--per-class training until the sparse/missing classes it lists have real
counts (stage 2 - a separate classifier over crops from this detector's
boxes - is where the per-type taxonomy actually gets used).

Label Studio's x/y/width/height are independently normalized percentages
(0-100) of the image's own width/height (x,width relative to width; y,height
relative to height), with (x, y) as the top-left corner - this converts
directly to YOLO's (x_center, y_center, width, height), all as 0-1
fractions, without needing the image's actual pixel dimensions for the
coordinate math itself (only for resolving/copying the image file).

Usage:
    # see per-class counts only, no images touched:
    python scripts/dates/prepare_yolo_dataset.py \\
        --label-studio-json data/labels/dates_label_studio/project-6-*.json --report-only

    python scripts/dates/prepare_yolo_dataset.py \\
        --label-studio-json data/labels/dates_label_studio/project-6-*.json \\
        --image-root /path/to/your/page/images --out-dir data/dates_yolo

    # keep the full type taxonomy instead of pooling (see docstring above -
    # not recommended yet given current per-class counts):
    python scripts/dates/prepare_yolo_dataset.py \\
        --label-studio-json data/labels/dates_label_studio/project-6-*.json \\
        --image-root /path/to/your/page/images --out-dir data/dates_yolo_typed --per-class
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def strip_label_studio_prefix(filename: str) -> str:
    """Label Studio prepends an 8-hex-char + '-' prefix to uploaded
    filenames to avoid collisions (e.g. 'a537088e-page_0002.png' ->
    'page_0002.png') - strips it if present, to recover the filename as it
    likely exists in your own image directory."""
    prefix, sep, rest = filename.partition("-")
    if sep and len(prefix) == 8 and all(c in "0123456789abcdef" for c in prefix.lower()):
        return rest
    return filename


def resolve_image(image_root: Path, file_upload: str) -> Path | None:
    for name in (file_upload, strip_label_studio_prefix(file_upload)):
        matches = list(image_root.rglob(name))
        if matches:
            return matches[0]
    return None


def load_annotations(json_path: Path) -> list[dict]:
    """[{file_upload, boxes: [(label, x_center, y_center, width, height)]}] -
    coordinates already converted to YOLO's 0-1-fraction, center-anchored
    convention."""
    data = json.loads(json_path.read_text())
    items = []
    for entry in data:
        boxes = []
        for annotation in entry.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                v = result["value"]
                label = v["rectanglelabels"][0]
                x_center = (v["x"] + v["width"] / 2) / 100
                y_center = (v["y"] + v["height"] / 2) / 100
                width = v["width"] / 100
                height = v["height"] / 100
                boxes.append((label, x_center, y_center, width, height))
        items.append({"file_upload": entry["file_upload"], "boxes": boxes})
    return items


def write_dataset(items: list[dict], image_root: Path, out_dir: Path,
                   class_to_idx: dict[str, int], class_names: list[str],
                   val_fraction: float, seed: int) -> None:
    resolved, unresolved = [], []
    for item in items:
        image_path = resolve_image(image_root, item["file_upload"])
        (resolved if image_path else unresolved).append((image_path, item["boxes"]) if image_path else item["file_upload"])

    if unresolved:
        raise SystemExit(
            f"{len(unresolved)}/{len(items)} annotated image(s) not found under --image-root (tried both the "
            f"Label Studio upload name and that name with its hex prefix stripped): "
            f"{unresolved[:5]}{'...' if len(unresolved) > 5 else ''}\n"
            f"Fix --image-root, or these filenames won't line up with your real page images."
        )

    rng = random.Random(seed)
    rng.shuffle(resolved)
    n_val = max(1, round(len(resolved) * val_fraction)) if len(resolved) > 1 else 0
    splits = {"val": resolved[:n_val], "train": resolved[n_val:]}
    note = " (too few images for a meaningful val split yet - treat val metrics as informal at this scale)" if len(resolved) < 10 else ""
    print(f"\nsplit: {len(splits['train'])} train / {len(splits['val'])} val{note}")

    for split, split_items in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path, boxes in split_items:
            shutil.copy(image_path, out_dir / "images" / split / image_path.name)
            lines = [f"{class_to_idx[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for label, xc, yc, w, h in boxes]
            (out_dir / "labels" / split / (image_path.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = out_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    data_yaml.write_text(f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n")
    print(f"Wrote {len(resolved)} image(s) + labels to {out_dir}, and {data_yaml}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-studio-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None,
                         help="required unless --report-only - searched recursively for each annotated image")
    parser.add_argument("--out-dir", type=Path, default=None, help="required unless --report-only")
    parser.add_argument("--per-class", action="store_true",
                         help="keep the full type taxonomy as separate YOLO classes, instead of pooling into "
                              "one 'date' class (the default) - see module docstring before using this")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true",
                         help="just print per-class counts and exit - no --image-root/--out-dir needed")
    args = parser.parse_args()

    items = load_annotations(args.label_studio_json)
    total_boxes = sum(len(item["boxes"]) for item in items)
    class_counts = Counter(label for item in items for label, *_ in item["boxes"])
    print(f"{len(items)} annotated image(s), {total_boxes} box(es), {len(class_counts)} distinct class(es) present")
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")

    if args.report_only:
        return
    if not args.image_root or not args.out_dir:
        raise SystemExit("--image-root and --out-dir are required unless --report-only")

    if args.per_class:
        class_names = sorted(class_counts)
        class_to_idx = {c: i for i, c in enumerate(class_names)}
    else:
        class_names = ["date"]
        class_to_idx = {c: 0 for c in class_counts}

    write_dataset(items, args.image_root, args.out_dir, class_to_idx, class_names, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
