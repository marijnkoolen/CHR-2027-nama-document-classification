"""
Trains stage 1 of the date-extraction pipeline: a YOLO detector over the
dataset prepare_yolo_dataset.py produces (pooled single "date" class by
default, or the full per-type taxonomy if you passed --per-class there -
see that script's docstring for why pooling is the current recommendation).

Thin wrapper around Ultralytics - a plain `yolo train` CLI call would do
the same thing; this documents the settings this project's data actually
needs and keeps output under the same runs/ convention as the rest of the
pipeline.

Starts from a COCO-pretrained checkpoint (transfer learning), not from
scratch: with a few dozen annotated images total, training from scratch
wouldn't learn anything useful - transfer learning from COCO's general
"what does an object look like" prior is what makes a result possible at
this scale at all. Expect this to need meaningfully more annotated
instances (order of a few hundred, as a rough rule of thumb, not the ~43
currently in data/labels/dates_label_studio) before its real precision/
recall is worth trusting for anything beyond a pipeline sanity check -
watch val mAP trend as you add more annotated rounds, not this run's
number in isolation.

--imgsz defaults higher than YOLO's usual 640: date regions are small
relative to a full page scan (typically ~2-10% of page width, ~1-3% of
page height, going by the current annotations), and small-object detection
benefits from higher input resolution - at proportionally higher compute
cost, so lower it if that's a binding constraint.

Usage:
    python scripts/dates/train_date_detector.py --data data/dates_yolo/data.yaml --out-dir runs/dates_stage1
    python scripts/dates/train_date_detector.py --data data/dates_yolo/data.yaml --out-dir runs/dates_stage1 --device cpu --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="a prepare_yolo_dataset.py data.yaml")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dates_stage1"))
    parser.add_argument("--model", default="yolo11n.pt",
                         help="starting checkpoint - COCO-pretrained, transfer learning (see module docstring). "
                              "'n' (nano) is the smallest/fastest Ultralytics variant - reasonable given how "
                              "little data there currently is to fine-tune with; consider 's' or 'm' once the "
                              "annotated set has grown substantially.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280, help="see module docstring re: small date regions")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', 0 (first GPU) - default: let ultralytics auto-pick")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        project=str(args.out_dir.parent), name=args.out_dir.name, exist_ok=True,
        device=args.device,
    )


if __name__ == "__main__":
    main()
