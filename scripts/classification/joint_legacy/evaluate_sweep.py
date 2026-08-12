"""
Evaluates every trained config in sweep_configs.py (as written by
train_sweep.py to <runs-dir>/<config-name>/) on its test split, and writes
ONE combined, machine-readable TSV covering every config: precision,
recall, macro and weighted F1 (+ support) per classification task
(document_type/layout_type/functional_category), precision/recall/F1 for
start_page (binary), Panoptic Quality (PQ/SQ/RQ + TP/FP/FN), and
Labeled/Unlabeled Attachment Score (LAS/UAS) for every task - see
evaluate_segmentation.py's module docstring for exactly what each of these
measures and which paper's protocol it follows.

Deliberately separate from train_sweep.py (see that script's docstring) -
rerun this alone any time a metric changes or you just want to regenerate
the table, without retraining anything.

Long/tidy output format (one row per config x metric), not one wide table
with a different column set per metric type (precision/recall/F1/support
for classification vs. TP/FP/FN for PQ vs. UAS/LAS) - this keeps every row
the same shape, which is what makes the file trivially filterable/
pivotable in pandas, Excel, or R, rather than needing a different reader
for different sections of the same file:

    config | modality | task | metric | average | value | support

Usage:
    python scripts/classification/evaluate_sweep.py --runs-dir runs/from_embeddings \\
        --out runs/from_embeddings/metrics.tsv

    # only re-evaluate a subset:
    python scripts/classification/evaluate_sweep.py --runs-dir runs/from_embeddings \\
        --out runs/from_embeddings/metrics.tsv --configs vision_efficient text_bert_base_uncased
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from evaluate_segmentation import (
    attachment_scores, macro_f1_report, majority_label_from_segments,
    panoptic_quality, segments_from_id_col, segments_from_start_col, start_page_report,
)
from sweep_configs import SWEEP_CONFIGS
from train_sweep import select_configs

PDF_COL, PAGE_COL = "pdf_name", "page_num"
CLASSIFICATION_TASKS = ["document_type", "layout_type", "functional_category"]
PRED_COL = {t: f"predicted_{t}" for t in CLASSIFICATION_TASKS}


def classification_rows(config_name: str, modality: str, df: pd.DataFrame, task: str, classes: list[str]) -> list[dict]:
    r = macro_f1_report(df, task, PRED_COL[task], classes)
    rows = [{
        "config": config_name, "modality": modality, "task": task,
        "metric": "accuracy", "average": "overall", "value": r["accuracy"], "support": r["n_pages"],
    }]
    for average in ("macro", "weighted"):
        for metric in ("precision", "recall", "f1"):
            rows.append({
                "config": config_name, "modality": modality, "task": task,
                "metric": metric, "average": average,
                "value": r[f"{average}_{metric}"], "support": r["n_pages"],
            })
    return rows


def start_page_rows(config_name: str, modality: str, df: pd.DataFrame) -> list[dict]:
    r = start_page_report(df, "start_page", "predicted_start_page")
    return [
        {
            "config": config_name, "modality": modality, "task": "start_page",
            "metric": metric, "average": "binary", "value": r[metric], "support": len(df),
        }
        for metric in ("precision", "recall", "f1")
    ]


def pq_rows(config_name: str, modality: str, true_segments: dict, pred_segments: dict) -> list[dict]:
    r = panoptic_quality(true_segments, pred_segments)
    rows = [
        {
            "config": config_name, "modality": modality, "task": "segmentation",
            "metric": metric, "average": "", "value": r[metric], "support": r["n_streams"],
        }
        for metric in ("pq", "sq", "rq", "tp", "fp", "fn")
    ]
    return rows


def attachment_rows(
    config_name: str, modality: str, task: str, df: pd.DataFrame, true_segments: dict, pred_segments: dict,
) -> list[dict]:
    true_labels = majority_label_from_segments(df, PDF_COL, PAGE_COL, task, true_segments)
    pred_labels = majority_label_from_segments(df, PDF_COL, PAGE_COL, PRED_COL[task], pred_segments)
    r = attachment_scores(true_segments, pred_segments, true_labels, pred_labels)
    return [
        {
            "config": config_name, "modality": modality, "task": task,
            "metric": metric, "average": "", "value": r[metric], "support": r["n_pages"],
        }
        for metric in ("uas", "las")
    ]


def evaluate_config(run_dir: Path, config, force: bool) -> list[dict]:
    predictions_path = run_dir / "test_predictions.tsv"
    checkpoint_path = run_dir / "sequence_model.pt"
    up_to_date = (
        predictions_path.exists() and checkpoint_path.exists()
        and predictions_path.stat().st_mtime >= checkpoint_path.stat().st_mtime
    )
    if force or not up_to_date:
        subprocess.run(
            [
                sys.executable, str(Path(__file__).parent / "predict_from_embeddings.py"),
                "--run-dir", str(run_dir), "--cached-embeddings", config.cached_embeddings,
                "--split", "test", "--out", str(predictions_path),
            ],
            check=True,
        )
    else:
        print(f"  {predictions_path} already up to date with the checkpoint - skipping prediction (--force to redo)")

    df = pd.read_csv(predictions_path, sep="\t")
    classes = json.loads((run_dir / "classes.json").read_text())

    rows = []
    for task in CLASSIFICATION_TASKS:
        rows += classification_rows(config.name, config.modality, df, task, classes[task])
    rows += start_page_rows(config.name, config.modality, df)

    true_segments = segments_from_start_col(df, PDF_COL, PAGE_COL, "start_page")
    pred_segments = segments_from_id_col(df, PDF_COL, PAGE_COL, "predicted_segment_id")
    rows += pq_rows(config.name, config.modality, true_segments, pred_segments)
    for task in CLASSIFICATION_TASKS:
        rows += attachment_rows(config.name, config.modality, task, df, true_segments, pred_segments)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/from_embeddings"),
                         help="a train_sweep.py --out-dir, containing one subdirectory per config")
    parser.add_argument("--configs", nargs="+", default=None,
                         help="only evaluate these configs by name (default: all - see sweep_configs.py)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                         help="re-run prediction even if test_predictions.tsv looks up to date")
    args = parser.parse_args()

    configs = select_configs(args.configs)
    all_rows = []
    for i, config in enumerate(configs):
        run_dir = args.runs_dir / config.name
        print(f"\n[{i + 1}/{len(configs)}] EVALUATE: {config.name} ({run_dir})")
        if not (run_dir / "sequence_model.pt").exists():
            print(f"  no checkpoint at {run_dir} - skipping (run train_sweep.py first)")
            continue
        all_rows += evaluate_config(run_dir, config, args.force)

    table = pd.DataFrame(all_rows, columns=["config", "modality", "task", "metric", "average", "value", "support"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep="\t", index=False)
    print(f"\n{len(table)} rows ({table['config'].nunique()} configs) written to {args.out}")


if __name__ == "__main__":
    main()
