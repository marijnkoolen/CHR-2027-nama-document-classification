"""Evaluates the realistic two-stage pipeline: predict start_page on every
test page, then predict document_type only on the pages a start_page model
actually predicted as start pages (not the true ones - there's no oracle in
a real setting), then carry that document_type forward to every other page
of the same predicted document. This is a different, harder question than
either evaluate_models.py answers alone: doc_type models there are always
scored against *true* start pages, but in a real pipeline every start_page
false positive/negative distorts what document_type sees downstream.

Runs this for the top --top-n models of each task (ranked by their already-
computed test macro_f1 - run evaluate_models.py --task start_page and
--task doc_type first) paired exhaustively (top-N x top-N), rather than
every trained model against every other one: with ~10+ models per task that
full cross product is rarely worth the compute, and the best individual
models are also the most informative pipeline candidates in practice.
--top-n 3 (the default) means 3 start_page predictions + 9 document_type
predictions - a good balance for a first look; raise it if you want a
broader sweep, or pass --start-models/--doc-models to name specific models
directly instead of ranking.

late-fusion-<vision>+<text> IS a valid pipeline candidate: it has no
checkpoint of its own, but its model_config.json (written by
evaluate_models.py's compute_late_fusion) names its two constituent
finetune checkpoints, and lib/predict.py's predict_late_fusion reloads and
runs both of those - on whatever custom page subset this script needs,
not just their original --task test split - then averages. Worth
checking explicitly: a late-fusion model that's merely competitive
standalone can still end up the best *pipeline* choice, the same way
bert-ft-bert-base-uncased (a modest #4 standalone) pairs into the overall
best pipeline combination once start_page noise is accounted for.

Feature caches: every backbone used by ANY selected model - start_page or
doc_type - must have been extracted via extract_features.py. Its cache
always covers every annotated page in the corpus (see its module
docstring), not just true start pages, so doc_type models here running on
*predicted* start pages (which can include pages that were never true
start pages) are always covered - no special-casing needed between the two
tasks' models.

Like evaluate_models.py, every model prediction runs in its own subprocess
(one start_page model, or one doc_type model, never both at once) for the
same reason: multiple BERT loads or any torch+xgboost mix in one long-lived
process was observed to segfault on macOS.

Usage:
    # first, for both tasks:
    python scripts/classification/sequential/evaluate_models.py --task start_page --data-root data --run-dir runs
    python scripts/classification/sequential/evaluate_models.py --task doc_type --data-root data --run-dir runs

    # then:
    python scripts/classification/sequential/evaluate_pipeline.py \\
        --data-root data --run-dir runs --top-n 3

Writes, per (start_model, doc_model) pair, under <run-dir>/pipeline/<start_model>__<doc_model>/:
    predictions.tsv  (dossier, page_num, start_page, predicted_start_page,
                       document_type, predicted_document_type, predicted_segment_id)
    metrics.json      (start_page macro_precision/macro_recall/macro_f1/accuracy - the
                        mean-of-both-classes numbers, matching the standalone start_page
                        leaderboard's own macro_f1 in evaluate_models.py; document_type
                        accuracy_e2e/macro_f1_e2e/weighted_f1_e2e, the honest end-to-end
                        reading over every true start page (a start_page miss counts as
                        wrong); panoptic quality PQ/SQ/RQ; UAS/LAS. Deliberately NOT
                        reported anywhere in this script: positive-class-only start_page
                        precision/recall/f1, or any document_type accuracy/F1 conditional
                        on start_page detection succeeding - both silently score an
                        easier, differently-scoped question than the real inference task
                        and have been mistaken for the honest number downstream, including
                        for document-type count corrections (see the "positive-class-only
                        metrics" project memory). evaluate_predictions() (lib/evaluate.py)
                        still computes the positive-class-only start_page fields
                        internally - they're just never read into this dict.
    per_class_metrics_start_page.tsv (both classes; no scoping issue - keep as-is)
    per_class_metrics_document_type_e2e.tsv
        (the honest, end-to-end reading - every true start page, a
        start_page miss counts as wrong; see write_joint_doctype_per_class -
        the only document_type per-class file this script writes)
    confusion_matrix_start_page.png
and, once every pair is done:
    <run-dir>/pipeline/summary.tsv  summary.json  classifier_comparison.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import load_config
from evaluate import evaluate_predictions
from labels import load_labels
from predict import predict_with_checkpoint
from results_io import load_all_results
from segmentation_metrics import attachment_scores, head_page_lookup, majority_label_from_segments, panoptic_quality, segments_from_start_col
from sklearn.metrics import classification_report, f1_score
from summarize import write_summary
from tasks import get_task

DEFAULT_TOP_N = 3


# --------------------------------------------------------------------------
# Model ranking + joint test data
# --------------------------------------------------------------------------

def rank_models(run_dir: Path, task_name: str, top_n: int) -> list[str]:
    results = load_all_results(run_dir, task_name)
    if not results:
        raise SystemExit(
            f"no evaluated models found under {run_dir / task_name} - run "
            f"`evaluate_models.py --task {task_name}` first"
        )
    ranked = sorted(results.items(), key=lambda kv: kv[1]["metrics"].get("macro_f1", -1.0), reverse=True)
    return [name for name, _ in ranked[:top_n]]


def load_joint_test_df(
    data_root: Path, random_seed: int, split_source: str | None, allow_missing_files: bool = False
) -> pd.DataFrame:
    """Every test-split page (not just start pages), with both the true
    start_page label and the true document_type - the ground-truth TSV
    already carries document_type on every page (propagated from its
    segment's start page when the labels were built), not just start-page
    rows, so no extra work is needed to get full-page ground truth."""
    task_start = get_task("start_page")
    label_data = load_labels(
        task_start, data_root, random_seed=random_seed, split_source=split_source,
        allow_missing_files=allow_missing_files,
    )
    df = label_data.df[label_data.df["split"] == "test"].copy().reset_index(drop=True)
    df = df.rename(columns={"label": "start_page"})
    df["start_page"] = df["start_page"].map({1: "yes", 0: "no"})

    raw = pd.read_csv(data_root / task_start.labels_tsv, sep="\t")
    raw = raw.rename(columns={task_start.dossier_column: "dossier"})
    df = df.merge(raw[["dossier", "page_num", "document_type"]], on=["dossier", "page_num"], how="left")
    missing = int(df["document_type"].isna().sum())
    if missing:
        print(f"warning: {missing}/{len(df)} test pages have no document_type in {task_start.labels_tsv}")
    return df


# --------------------------------------------------------------------------
# Workers - each loads exactly one model
# --------------------------------------------------------------------------

def _ctx(run_dir, task_name, cache_dir, device, batch_size) -> dict:
    return {
        "run_dir": run_dir, "task_name": task_name,
        "cache_dir": cache_dir, "device": device, "batch_size": batch_size,
    }


def _pick_device_if_needed(config, device_arg):
    if config["model_family"] == "baseline":
        return None
    from common import pick_device

    return pick_device(device_arg)


def worker_predict_start(args) -> None:
    joint_df = load_joint_test_df(args.data_root, args.random_seed, args.split_source, args.allow_missing_files)
    config = load_config(args.run_dir, "start_page", args._worker_start)
    device = _pick_device_if_needed(config, args.device)
    cache_dir = args.cache_dir or (args.data_root / "embeddings")
    ctx = _ctx(args.run_dir, "start_page", cache_dir, device, args.batch_size)

    predict_df = joint_df[["dossier", "page_num", "img_path", "text_path"]]
    print(f"Predicting start_page with {args._worker_start} on {len(predict_df)} pages …")
    keys, preds, probs = predict_with_checkpoint(ctx, args._worker_start, config, predict_df)

    out = pd.DataFrame(keys, columns=["dossier", "page_num"])
    out["pred_start"] = preds
    out["pred_start_prob"] = probs[:, 1] if probs is not None and probs.shape[1] == 2 else np.nan
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")


def worker_predict_doctype(args) -> None:
    heads_df = pd.read_csv(args.heads_file, sep="\t")
    heads_df["img_path"] = heads_df["img_path"].map(Path)
    heads_df["text_path"] = heads_df["text_path"].map(Path)

    config = load_config(args.run_dir, "doc_type", args._worker_doctype)
    device = _pick_device_if_needed(config, args.device)
    cache_dir = args.cache_dir or (args.data_root / "embeddings")
    ctx = _ctx(args.run_dir, "doc_type", cache_dir, device, args.batch_size)

    print(f"Predicting document_type with {args._worker_doctype} on {len(heads_df)} predicted head pages …")
    keys, preds_idx, _ = predict_with_checkpoint(ctx, args._worker_doctype, config, heads_df)
    class_names = config["class_names"]

    out = pd.DataFrame(keys, columns=["dossier", "page_num"])
    out["pred_document_type"] = [class_names[i] for i in preds_idx]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def _worker_cmd(args, extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--data-root", str(args.data_root), "--run-dir", str(args.run_dir),
        "--batch-size", str(args.batch_size), "--random-seed", str(args.random_seed),
        *extra,
    ]
    if args.cache_dir is not None:
        cmd += ["--cache-dir", str(args.cache_dir)]
    if args.split_source is not None:
        cmd += ["--split-source", args.split_source]
    if args.allow_missing_files:
        cmd += ["--allow-missing-files"]
    if args.device is not None:
        cmd += ["--device", args.device]
    return cmd


def _joint_doctype_labels(df: pd.DataFrame, class_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
    """y_true/y_pred for the honest, end-to-end document_type reading, shared
    by compute_joint_doctype_metrics (aggregate) and
    write_joint_doctype_per_class (per-class) below - every TRUE start page
    is scored, not just the ones the start_page model actually flagged
    (that's what doc_accuracy/doc_macro_f1 in compute_pair_metrics do, via
    their 'keep' mask: a true start page the start_page model missed is
    silently dropped from that score entirely, never counted as a failure).
    Here, a true start page only counts as correct if it was BOTH predicted
    as a start page AND given the right document_type - a miss (wrong or
    absent start_page prediction) counts as wrong for whatever document_type
    that page actually was.

    An unpredicted/wrongly-typed page gets a sentinel class index (-1, never
    a real class 0..K-1) in y_pred. Passing an explicit labels=range(K) to
    whatever sklearn metric consumes this still counts -1 as a miss against
    whichever true class that row belongs to (increasing that class's false
    negatives) without creating a phantom extra class in the macro average -
    verified against sklearn's actual behavior, not assumed from the docs."""
    SENTINEL = -1
    is_true_start = df["start_page"].astype(str).str.lower().eq("yes")
    is_pred_start = df["predicted_start_page"].astype(str).str.lower().eq("yes")

    scored = df.loc[is_true_start & df["document_type"].isin(class_to_idx)]
    y_true = scored["document_type"].map(class_to_idx).to_numpy()

    pred_class = scored["predicted_document_type"].map(class_to_idx)  # NaN if missing/unrecognised
    correctly_scoped = is_pred_start.loc[scored.index].to_numpy() & pred_class.notna().to_numpy()
    y_pred = np.where(correctly_scoped, pred_class.fillna(SENTINEL).to_numpy(), SENTINEL).astype(int)
    return y_true, y_pred


def compute_joint_doctype_metrics(df: pd.DataFrame, class_to_idx: dict, doc_class_names: list[str]) -> dict:
    """Same denominator (every true start page) as the standalone doc_type
    evaluation's own macro_f1 (see evaluate_models.py), so this number is
    directly comparable to it in a way doc_macro_f1 isn't - and it's the
    number that reflects what actually happens to a document if its start
    page goes undetected: nothing predicts its type at all. See
    _joint_doctype_labels above for how y_true/y_pred are built."""
    y_true, y_pred = _joint_doctype_labels(df, class_to_idx)
    labels = list(range(len(doc_class_names)))
    return {
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "n_true_start_pages": int(len(y_true)),
    }


def write_joint_doctype_per_class(df: pd.DataFrame, class_to_idx: dict, doc_class_names: list[str], out_dir: Path) -> None:
    """Per-class precision/recall/f1/support for the same honest, end-to-end
    reading as compute_joint_doctype_metrics's aggregate numbers above - the
    only document_type per-class breakdown this script writes. An earlier
    version of this function also wrote a conditional-on-detection variant
    (scored only on pages the start_page model actually flagged, dropping any
    true start page it missed rather than counting it as a failure) - that
    was removed because per-class numbers are exactly what a corpus-count
    correction would use, and the conditional version silently answers an
    easier, differently-scoped question that isn't comparable to the
    standalone doc_type per-class table in evaluate_models.py or valid input
    for count imputation (see the "positive-class-only metrics" project
    memory)."""
    y_true, y_pred = _joint_doctype_labels(df, class_to_idx)
    labels = list(range(len(doc_class_names)))
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=doc_class_names, output_dict=True, zero_division=0
    )
    report_dict.pop("accuracy", None)  # a bare float in this dict, not a precision/recall/f1/support row
    per_class_df = pd.DataFrame(report_dict).T
    per_class_df.index.name = "class"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_df.to_csv(out_dir / "per_class_metrics_document_type_e2e.tsv", sep="\t")


def compute_pair_metrics(df: pd.DataFrame, class_to_idx: dict, doc_class_names: list[str], out_dir: Path) -> dict:
    true_segments = segments_from_start_col(df, "dossier", "page_num", "start_page")
    pred_segments = segments_from_start_col(df, "dossier", "page_num", "predicted_start_page")

    pq = panoptic_quality(true_segments, pred_segments)

    true_labels = majority_label_from_segments(df, "dossier", "page_num", "document_type", true_segments)
    pred_labels = majority_label_from_segments(df, "dossier", "page_num", "predicted_document_type", pred_segments)
    att = attachment_scores(true_segments, pred_segments, true_labels, pred_labels)

    y_true_start = df["start_page"].astype(str).str.lower().eq("yes").astype(int).values
    y_pred_start = df["predicted_start_page"].astype(str).str.lower().eq("yes").astype(int).values
    start_metrics = evaluate_predictions(
        "start_page", y_true_start, y_pred_start, num_classes=2,
        class_names=["not-start", "start"], split="start_page", out_dir=out_dir,
    )
    # Only the macro_* (both-classes) fields feed the returned dict below - see
    # lib/evaluate.py's evaluate_predictions docstring for why the positive-
    # class-only precision/recall/f1 it also computes are never surfaced here.

    # n_doctype_pages_scored is purely descriptive (pipeline coverage - how many
    # predicted head pages got an in-vocabulary document_type prediction at all),
    # not a score - no conditional-on-detection document_type accuracy/F1 is
    # computed here anymore. That number silently scores an easier, start_page-
    # model-selected subset than the real inference task and has been mistaken
    # for the honest end-to-end reading downstream (see the "positive-class-only
    # metrics" project memory) - compute_joint_doctype_metrics/
    # write_joint_doctype_per_class below are the only document_type scores this
    # function reports, both end-to-end over every true start page.
    keep = (
        df["predicted_document_type"].notna()
        & df["document_type"].isin(class_to_idx)
        & df["predicted_document_type"].isin(class_to_idx)
    )

    joint_metrics = compute_joint_doctype_metrics(df, class_to_idx, doc_class_names)
    print(
        f"\n  document_type [end-to-end, every true start page]\n"
        f"  accuracy: {joint_metrics['accuracy']:.4f}  macro_f1: {joint_metrics['macro_f1']:.4f}  "
        f"weighted_f1: {joint_metrics['weighted_f1']:.4f}  (n={joint_metrics['n_true_start_pages']})"
    )
    write_joint_doctype_per_class(df, class_to_idx, doc_class_names, out_dir)

    return {
        "start_macro_precision": start_metrics["macro_precision"],
        "start_macro_recall": start_metrics["macro_recall"],
        "start_macro_f1": start_metrics["macro_f1"],
        "start_accuracy": start_metrics["accuracy"],
        "start_roc_auc": start_metrics.get("roc_auc"),
        "doc_accuracy_e2e": joint_metrics["accuracy"], "doc_macro_f1_e2e": joint_metrics["macro_f1"],
        "doc_weighted_f1_e2e": joint_metrics["weighted_f1"],
        "pq": pq["pq"], "sq": pq["sq"], "rq": pq["rq"],
        "pq_tp": pq["tp"], "pq_fp": pq["fp"], "pq_fn": pq["fn"], "pq_n_streams": pq["n_streams"],
        "uas": att["uas"], "las": att["las"], "n_pages_scored": att["n_pages"],
        "n_doctype_pages_scored": int(keep.sum()), "n_true_start_pages": joint_metrics["n_true_start_pages"],
    }


HEADLINE_METRICS = [
    "start_macro_precision", "start_macro_recall", "start_macro_f1", "start_accuracy",
    "doc_accuracy_e2e", "doc_macro_f1_e2e", "doc_weighted_f1_e2e",
    "pq", "sq", "rq", "uas", "las",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <run-dir>/pipeline")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="see module docstring for the tradeoff")
    parser.add_argument(
        "--start-models", nargs="*", default=None,
        help="use these start_page models instead of ranking the top --top-n by macro_f1",
    )
    parser.add_argument(
        "--doc-models", nargs="*", default=None,
        help="use these doc_type models instead of ranking the top --top-n by macro_f1",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="only matters under --split-source computed (start_page's split is derived from it); this "
             "single split, from start_page's own label loading, defines the entire joint test set both "
             "tasks are evaluated over here",
    )
    parser.add_argument("--split-source", choices=["computed", "tsv_column"], default=None)
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="don't error on a missing image (or transcription) file - drop rows with a missing image and "
             "continue instead (missing text always falls back to empty text, regardless). Off by default: "
             "a wrong --data-root or a path-formula mistake should fail fast, before any heavy lifting, not "
             "silently shrink the test set.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--_worker-start", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-doctype", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--heads-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker_start:
        worker_predict_start(args)
        return
    if args._worker_doctype:
        worker_predict_doctype(args)
        return

    out_dir = args.out_dir or (args.run_dir / "pipeline")
    cache_root = out_dir / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    start_models = args.start_models or rank_models(args.run_dir, "start_page", args.top_n)
    doc_models = args.doc_models or rank_models(args.run_dir, "doc_type", args.top_n)
    print(f"start_page candidates ({len(start_models)}): {start_models}")
    print(f"doc_type candidates ({len(doc_models)}): {doc_models}")

    doc_class_names = load_config(args.run_dir, "doc_type", doc_models[0])["class_names"]
    class_to_idx = {c: i for i, c in enumerate(doc_class_names)}
    joint_df = load_joint_test_df(args.data_root, args.random_seed, args.split_source, args.allow_missing_files)

    # Pass 1: one start_page prediction per start model - reused for every doc_model it's paired with.
    start_preds: dict[str, pd.DataFrame] = {}
    for sm in start_models:
        print()
        sm_out = cache_root / f"start__{sm}.tsv"
        result = subprocess.run(_worker_cmd(args, ["--_worker-start", sm, "--out", str(sm_out)]))
        if result.returncode != 0:
            print(f"** {sm} failed to predict start_page (exit {result.returncode}) - dropping it **")
            continue
        start_preds[sm] = pd.read_csv(sm_out, sep="\t")

    if not start_preds:
        raise SystemExit("every start_page model failed to predict - nothing to pair with doc_type models")

    pair_metrics: dict[str, dict] = {}
    for sm, sm_preds in start_preds.items():
        merged = joint_df.merge(sm_preds, on=["dossier", "page_num"], how="left")
        merged["predicted_start_page"] = merged["pred_start"].map({1: "yes", 0: "no"})

        pred_segments = segments_from_start_col(merged, "dossier", "page_num", "predicted_start_page")
        head_lookup = head_page_lookup(pred_segments)
        head_keys = pd.DataFrame(
            [(pdf, head) for pdf, segs in pred_segments.items() for head in segs], columns=["dossier", "page_num"]
        )
        heads_df = merged.merge(head_keys, on=["dossier", "page_num"], how="inner")
        heads_df = heads_df[["dossier", "page_num", "img_path", "text_path"]].sort_values(["dossier", "page_num"])
        heads_path = cache_root / f"heads__{sm}.tsv"
        heads_df.to_csv(heads_path, sep="\t", index=False)
        print(f"\n{sm}: {len(heads_df)} predicted head pages across {heads_df['dossier'].nunique()} dossiers")

        for dm in doc_models:
            pair_name = f"{sm}__{dm}"
            print()
            doc_out = cache_root / f"doctype__{pair_name}.tsv"
            result = subprocess.run(_worker_cmd(args, ["--_worker-doctype", dm, "--heads-file", str(heads_path), "--out", str(doc_out)]))
            if result.returncode != 0:
                print(f"** {pair_name} failed to predict document_type (exit {result.returncode}) - skipping **")
                continue
            doc_preds = pd.read_csv(doc_out, sep="\t")
            head_to_doctype = {
                (row["dossier"], row["page_num"]): row["pred_document_type"] for _, row in doc_preds.iterrows()
            }

            pair_df = merged.copy()
            heads_for_row = pair_df.apply(lambda r: head_lookup.get(r["dossier"], {}).get(r["page_num"]), axis=1)
            pair_df["predicted_segment_id"] = heads_for_row
            pair_df["predicted_document_type"] = [
                head_to_doctype.get((dossier, head)) for dossier, head in zip(pair_df["dossier"], heads_for_row)
            ]

            pair_dir = out_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            cols = [
                "dossier", "page_num", "start_page", "predicted_start_page",
                "document_type", "predicted_document_type", "predicted_segment_id",
            ]
            pair_df[cols].to_csv(pair_dir / "predictions.tsv", sep="\t", index=False)

            print(f"--- {pair_name} ---")
            metrics = compute_pair_metrics(pair_df, class_to_idx, doc_class_names, pair_dir)
            with open(pair_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"wrote {pair_dir / 'metrics.json'}")

            pair_metrics[pair_name] = {k: metrics[k] for k in HEADLINE_METRICS}

    if not pair_metrics:
        raise SystemExit("no (start_model, doc_model) pairs evaluated successfully")

    summary_df = pd.DataFrame(pair_metrics).T.round(4)
    summary_df = summary_df.sort_values("pq", ascending=False)
    summary_df.index.name = "pair"
    print()
    print(summary_df.to_string())
    write_summary(summary_df, out_dir, "pipeline")


if __name__ == "__main__":
    main()
