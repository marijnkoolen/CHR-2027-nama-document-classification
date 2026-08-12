"""Regenerates <run-dir>/eval_report.html from whatever's currently under
<run-dir>: start_page/summary.tsv, doc_type/summary.tsv (written by
evaluate_models.py), and every pipeline*/summary.tsv found directly under
<run-dir> (written by evaluate_pipeline.py - one directory per sweep, e.g.
pipeline_top3_e2e/, pipeline_latefusion_check/; all of them get merged into
one pipeline leaderboard, ranked by LAS).

Injects the resulting JSON into report_template.html (this directory) via
its __REPORT_DATA_JSON__ placeholder and writes the result to
<run-dir>/eval_report.html. See that template for the actual page
(structure/styling/JS) - this script only builds the DATA it consumes.

Deliberately only reports the honest, macro-averaged, end-to-end metrics
(start_macro_f1, not the positive-class-only start_f1 that
evaluate_predictions() also computes internally; doc_macro_f1_e2e, not any
detection-conditional variant) - see the "positive-class-only metrics"
project memory for why, and evaluate_pipeline.py's own module docstring for
which fields it does/doesn't expose for exactly this reason.

Usage:
    python scripts/classification/build_report.py --run-dir runs/per_task
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PRETTY_BACKBONE = {
    "facebook/dinov2-small": "DINOv2-S",
    "microsoft/dit-large-finetuned-rvlcdip": "DiT-Large",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet-multilingual",
    "bert-base-uncased": "BERT-base",
    "vgg16": "VGG16",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet": "EfficientNet-B0",
}


def pretty_backbone(b: str) -> str:
    b = b.replace("__", "/")
    return PRETTY_BACKBONE.get(b, b)


def parse_model(name: str) -> tuple[str, str, list[str]]:
    if name.startswith("knn-"):
        regime, rest, algo = "baseline", name[len("knn-"):], "KNN"
    elif name.startswith("xgboost-"):
        regime, rest, algo = "baseline", name[len("xgboost-"):], "XGBoost"
    elif name.startswith("lstm-"):
        regime, rest, algo = "sequence", name[len("lstm-"):], "LSTM"
    elif name.startswith("early-fusion-"):
        regime, rest, algo = "fusion", name[len("early-fusion-"):], "Early fusion"
    elif name.startswith("late-fusion-"):
        regime, rest, algo = "fusion", name[len("late-fusion-"):], "Late fusion"
    elif name.startswith("bert-ft-"):
        regime, rest, algo = "finetune", name[len("bert-ft-"):], "BERT fine-tune"
    elif name.endswith("-ft"):
        regime, rest, algo = "finetune", name[:-3], "Fine-tune"
    else:
        regime, rest, algo = "other", name, ""
    backbones = [pretty_backbone(b) for b in rest.split("+")]
    return regime, algo, backbones


def load_task(run_dir: Path, task: str) -> list[dict]:
    summary_path = run_dir / task / "summary.tsv"
    if not summary_path.exists():
        print(f"  skip (missing): {summary_path}")
        return []
    df = pd.read_csv(summary_path, sep="\t").dropna(how="all")
    rows = []
    for _, r in df.iterrows():
        regime, algo, backbones = parse_model(r["model"])
        rows.append({
            "model": r["model"], "regime": regime, "algo": algo, "backbones": backbones,
            "accuracy": round(float(r["accuracy"]), 4),
            "macro_f1": round(float(r["macro_f1"]), 4),
            "weighted_f1": round(float(r["weighted_f1"]), 4),
        })
    rows.sort(key=lambda x: -x["macro_f1"])
    return rows


def load_pipeline(run_dir: Path, start_page: list[dict]) -> list[dict]:
    summaries = sorted(run_dir.glob("pipeline*/summary.tsv"))
    if not summaries:
        print("  no pipeline*/summary.tsv found under run-dir - skipping pipeline table")
        return []
    pipe_df = pd.concat([pd.read_csv(p, sep="\t").dropna(how="all") for p in summaries], ignore_index=True)

    pipeline = []
    for _, r in pipe_df.iterrows():
        pair = r["pair"]
        start_m = next((m["model"] for m in start_page if pair.startswith(m["model"] + "__")), None)
        doc_m = pair[len(start_m) + 2:] if start_m else None
        pipeline.append({
            "start_model": start_m,
            "doc_model": doc_m,
            "start_macro_f1": round(float(r["start_macro_f1"]), 4),
            "doc_macro_f1_e2e": round(float(r["doc_macro_f1_e2e"]), 4),
            "pq": round(float(r["pq"]), 4),
            "uas": round(float(r["uas"]), 4),
            "las": round(float(r["las"]), 4),
        })
    # LAS ties happen (e.g. two doc_type models that behave identically once
    # start_page noise is folded in) - break them by doc_macro_f1_e2e
    # descending, then by doc_model name for full determinism when even that
    # ties (rather than leaving the final order to glob()'s filesystem-
    # dependent ordering, which isn't a meaningful ranking criterion).
    pipeline.sort(key=lambda x: (-x["las"], -x["doc_macro_f1_e2e"], x["doc_model"]))
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/per_task"))
    parser.add_argument("--template", type=Path, default=Path(__file__).parent / "report_template.html")
    parser.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/eval_report.html")
    args = parser.parse_args()

    print("Loading per-task leaderboards …")
    start_page = load_task(args.run_dir, "start_page")
    doc_type = load_task(args.run_dir, "doc_type")
    print(f"  start_page: {len(start_page)} models, doc_type: {len(doc_type)} models")

    print("Loading pipeline combinations …")
    pipeline = load_pipeline(args.run_dir, start_page)
    print(f"  {len(pipeline)} pipeline combinations")

    data = {"start_page": start_page, "doc_type": doc_type, "pipeline": pipeline}
    data_json = json.dumps(data, indent=None)

    template = args.template.read_text()
    if "__REPORT_DATA_JSON__" not in template:
        raise SystemExit(f"{args.template} has no __REPORT_DATA_JSON__ placeholder - was it edited by hand?")
    html = template.replace("__REPORT_DATA_JSON__", data_json)

    out_path = args.out or (args.run_dir / "eval_report.html")
    out_path.write_text(html)
    print(f"Wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
