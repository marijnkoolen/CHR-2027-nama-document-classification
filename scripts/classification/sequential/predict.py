"""Runs one trained start_page model, then one trained doc_type model, over
a new, unlabeled corpus - the sequential pipeline's counterpart to
../joint/predict.py.

Chains the two models the same way sequential/evaluate_pipeline.py does:
predicts start_page on every page, then predicts document_type only on the
pages predicted as a start page (not the whole corpus), then carries that
prediction forward to every other page of the same predicted document. This
is the actual point of the sequential approach over running two independent
full-corpus passes: doc_type inference only runs on however many pages the
start_page model actually flags - typically a small minority of a corpus -
not on every page.

Takes a --manifest listing the pages to predict on (column names for
dossier/page/image/text are configurable - see --pdf-col etc., matching
../joint/predict.py's flags) rather than a labels TSV: unlike every other
script in this pipeline, this one is for genuinely unlabeled data, where
nothing analogous to labels/dossier_labels_merged_pdf12_stratified.tsv
exists.

Fail-fast path validation applies here too, same as every other script in
this pipeline (see lib/labels.py's module docstring): a missing image or
text file raises by default, before any model is even loaded - pass
--allow-missing-files to instead drop rows with a missing image and
continue (missing text always falls back to empty text, regardless).

--start-model/--doc-model choices from the baseline/sequence/fusion
families need this corpus's features already cached for whichever
backbone(s) they use - run extract_features.py against this --manifest
first (its --allow-missing-files/corpus-wide-cache behavior is exactly the
same regardless of whether the corpus is labeled). finetune choices read
raw images/text directly and need no cache, but cost far more per page - a
full forward pass through the whole backbone, on CPU. Across 114K pages
that's the difference between minutes (vgg16, efficientnet_b0, a cached
baseline/sequence/fusion model) and potentially days (microsoft/dit-large-
finetuned-rvlcdip) - use --limit-dossiers to time a small sample first if
you're not sure which side of that line your chosen models fall on.

Every prediction runs in its own subprocess (one for --start-model, one for
--doc-model), matching every other script in this pipeline: a baseline
checkpoint (KNN/XGBoost) and a torch-family checkpoint (finetune/sequence/
fusion) in the same long-lived process was observed to segfault on macOS if
xgboost isn't the first of the two imported - trivial to get right by hand
for one predict_baseline() call, easy to get backwards if --start-model and
--doc-model come from different families and whichever prediction happens
to run first varies by invocation, so this sidesteps the question entirely
the same way evaluate_models.py/evaluate_pipeline.py do.

Usage:
    python scripts/classification/sequential/predict.py \\
        --manifest new_corpus_manifest.tsv --image-root /path/to/new_corpus \\
        --run-dir runs --start-model knn-vgg16 --doc-model xgboost-vgg16 \\
        --out predictions.tsv

    # time a small sample before committing to the full corpus
    python scripts/classification/sequential/predict.py \\
        --manifest new_corpus_manifest.tsv --image-root /path/to/new_corpus \\
        --run-dir runs --start-model vgg16-ft --doc-model bert-ft-bert-base-uncased \\
        --out sample_predictions.tsv --limit-dossiers 20

Writes one row per input page to --out: every original --manifest column,
plus predicted_start_page/start_page_confidence,
predicted_document_type/document_type_confidence (both propagated from the
page's own predicted segment - see module docstring - not just its own,
independent prediction), and predicted_segment_id (the head/start page
number of the segment this page was assigned to, matching
evaluate_pipeline.py's own convention for this column).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import load_config  # noqa: E402
from common import load_page_manifest  # noqa: E402

# lib/ is on sys.path ahead of this file's own directory (see the insert
# above), so this resolves to lib/predict.py, not to this file - despite
# the identical name. See lib/segmentation_metrics.py's docstring for the
# collision this project has already had to design around once before.
from predict import predict_with_checkpoint  # noqa: E402
from segmentation_metrics import head_page_lookup, segments_from_start_col  # noqa: E402


def load_manifest(args) -> pd.DataFrame:
    return load_page_manifest(
        args.manifest, args.pdf_col, args.page_col, args.image_col, args.text_col,
        image_root=args.image_root, limit_dossiers=args.limit_dossiers, allow_missing_files=args.allow_missing_files,
    )


def _ctx(run_dir, task_name, cache_dir, device, batch_size) -> dict:
    return {"run_dir": run_dir, "task_name": task_name, "cache_dir": cache_dir, "device": device, "batch_size": batch_size}


def _pick_device_if_needed(config, device_arg):
    if config["model_family"] == "baseline":
        return None
    from common import pick_device

    return pick_device(device_arg)


def worker_predict_start(args) -> None:
    manifest = load_manifest(args)
    config = load_config(args.run_dir, "start_page", args._worker_start)
    device = _pick_device_if_needed(config, args.device)
    ctx = _ctx(args.run_dir, "start_page", args.cache_dir, device, args.batch_size)

    predict_df = manifest[["dossier", "page_num", "img_path", "text_path"]]
    print(f"Predicting start_page with {args._worker_start} on {len(predict_df)} pages …")
    keys, preds, probs = predict_with_checkpoint(ctx, args._worker_start, config, predict_df)

    out = pd.DataFrame(keys, columns=["dossier", "page_num"])
    out["pred_start"] = preds
    out["pred_start_prob"] = probs[:, 1] if probs is not None and probs.shape[1] == 2 else float("nan")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")


def worker_predict_doctype(args) -> None:
    heads_df = pd.read_csv(args.heads_file, sep="\t")
    heads_df["img_path"] = heads_df["img_path"].map(Path)
    heads_df["text_path"] = heads_df["text_path"].map(Path)

    config = load_config(args.run_dir, "doc_type", args._worker_doctype)
    device = _pick_device_if_needed(config, args.device)
    ctx = _ctx(args.run_dir, "doc_type", args.cache_dir, device, args.batch_size)

    print(f"Predicting document_type with {args._worker_doctype} on {len(heads_df)} predicted head pages …")
    keys, preds_idx, probs = predict_with_checkpoint(ctx, args._worker_doctype, config, heads_df)
    class_names = config["class_names"]

    out = pd.DataFrame(keys, columns=["dossier", "page_num"])
    out["pred_document_type"] = [class_names[i] for i in preds_idx]
    out["pred_document_type_prob"] = [
        float(probs[i, idx]) if probs is not None else float("nan") for i, idx in enumerate(preds_idx)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")


def _worker_cmd(args, extra: list[str]) -> list[str]:
    """extra carries this worker's own --_worker-start/--_worker-doctype and
    --out (and --heads-file, for the doctype worker) - every other flag a
    worker needs to independently rebuild the manifest is forwarded here."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--manifest", str(args.manifest), "--run-dir", str(args.run_dir),
        "--pdf-col", args.pdf_col, "--page-col", args.page_col,
        "--image-col", args.image_col, "--text-col", args.text_col,
        "--image-root", str(args.image_root), "--batch-size", str(args.batch_size),
        *extra,
    ]
    if args.cache_dir is not None:
        cmd += ["--cache-dir", str(args.cache_dir)]
    if args.limit_dossiers:
        cmd += ["--limit-dossiers", str(args.limit_dossiers)]
    if args.allow_missing_files:
        cmd += ["--allow-missing-files"]
    if args.device is not None:
        cmd += ["--device", args.device]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                         help="TSV/CSV listing the pages to predict on - one row per page")
    parser.add_argument("--pdf-col", default="pdf_name")
    parser.add_argument("--page-col", default="page_num")
    parser.add_argument("--image-col", default="img_path")
    parser.add_argument("--text-col", default="text_path")
    parser.add_argument(
        "--image-root", type=Path, default=Path(""),
        help="joined onto every --image-col/--text-col value; leave unset if --manifest already has "
             "fully-resolved paths",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="where --start-model/--doc-model live")
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="needed only if --start-model/--doc-model is a baseline/sequence/fusion checkpoint - run "
             "extract_features.py against this corpus first; not needed for finetune checkpoints, which "
             "read raw images/text directly (see module docstring for the cost tradeoff)",
    )
    # Not argparse-required: the worker subprocess invocation (--_worker-start/
    # --_worker-doctype) never receives these, only the top-level run does - see
    # the explicit check below, after the worker dispatch.
    parser.add_argument("--start-model", default=None, help="a trained model under <run-dir>/start_page/")
    parser.add_argument("--doc-model", default=None, help="a trained model under <run-dir>/doc_type/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit-dossiers", type=int, default=None,
        help="predict on only the first N dossiers in --manifest order - for timing a sample before "
             "committing to the full corpus, see module docstring",
    )
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="don't error on a missing image (or transcription) file - drop rows with a missing image and "
             "continue instead (missing text always falls back to empty text, regardless). Off by default: "
             "a wrong --image-root or a wrong --*-col mistake should fail fast, before any heavy lifting, "
             "not silently shrink the corpus.",
    )
    parser.add_argument("--device", default="cpu", help="see train_all.py's module docstring for why cpu is the default")
    parser.add_argument("--_worker-start", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-doctype", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--heads-file", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker_start:
        worker_predict_start(args)
        return
    if args._worker_doctype:
        worker_predict_doctype(args)
        return

    if not args.start_model or not args.doc_model:
        raise SystemExit("--start-model and --doc-model are required")

    CACHE_FAMILIES = {"baseline", "sequence_lstm", "fusion_early"}
    for task_name, model_name in (("start_page", args.start_model), ("doc_type", args.doc_model)):
        config = load_config(args.run_dir, task_name, model_name)
        if config["model_family"] in CACHE_FAMILIES and args.cache_dir is None:
            raise SystemExit(
                f"--{'start' if task_name == 'start_page' else 'doc'}-model {model_name!r} is a "
                f"{config['model_family']} checkpoint, which reads pre-extracted features - pass --cache-dir "
                f"pointing at wherever extract_features.py wrote this corpus's embeddings (run it first if you "
                f"haven't yet). finetune_image/finetune_text checkpoints are the only ones that don't need this."
            )

    manifest = load_manifest(args)
    print(f"{len(manifest)} pages, {manifest['dossier'].nunique()} dossiers to predict\n")

    out_dir = args.out.parent if args.out.parent != Path("") else Path(".")
    cache_root = out_dir / "_predict_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sm_out = cache_root / "start_preds.tsv"
    result = subprocess.run(_worker_cmd(args, ["--_worker-start", args.start_model, "--out", str(sm_out)]))
    if result.returncode != 0:
        raise SystemExit(f"{args.start_model} failed to predict start_page (exit {result.returncode})")
    start_preds = pd.read_csv(sm_out, sep="\t")
    print(f"start_page prediction: {time.time() - t0:.0f}s\n")

    merged = manifest.merge(start_preds, on=["dossier", "page_num"], how="left")
    merged["predicted_start_page"] = merged["pred_start"].map({1: "yes", 0: "no"})
    merged["start_page_confidence"] = merged["pred_start_prob"].map(
        lambda p: max(p, 1 - p) if pd.notna(p) else float("nan")
    )

    pred_segments = segments_from_start_col(merged, "dossier", "page_num", "predicted_start_page")
    head_lookup = head_page_lookup(pred_segments)
    head_keys = pd.DataFrame(
        [(pdf, head) for pdf, segs in pred_segments.items() for head in segs], columns=["dossier", "page_num"]
    )
    heads_df = merged.merge(head_keys, on=["dossier", "page_num"], how="inner")
    heads_df = heads_df[["dossier", "page_num", "img_path", "text_path"]].sort_values(["dossier", "page_num"])
    heads_path = cache_root / "heads.tsv"
    heads_df.to_csv(heads_path, sep="\t", index=False)
    print(f"{len(heads_df)} predicted start pages across {heads_df['dossier'].nunique()} dossiers\n")

    t1 = time.time()
    doc_out = cache_root / "doc_preds.tsv"
    result = subprocess.run(
        _worker_cmd(args, ["--_worker-doctype", args.doc_model, "--heads-file", str(heads_path), "--out", str(doc_out)])
    )
    if result.returncode != 0:
        raise SystemExit(f"{args.doc_model} failed to predict document_type (exit {result.returncode})")
    doc_preds = pd.read_csv(doc_out, sep="\t")
    print(f"document_type prediction: {time.time() - t1:.0f}s\n")

    head_to_doctype = {(r["dossier"], r["page_num"]): r["pred_document_type"] for _, r in doc_preds.iterrows()}
    head_to_conf = {(r["dossier"], r["page_num"]): r["pred_document_type_prob"] for _, r in doc_preds.iterrows()}

    heads_for_row = merged.apply(lambda r: head_lookup.get(r["dossier"], {}).get(r["page_num"]), axis=1)
    merged["predicted_segment_id"] = heads_for_row
    merged["predicted_document_type"] = [
        head_to_doctype.get((dossier, head)) for dossier, head in zip(merged["dossier"], heads_for_row)
    ]
    merged["document_type_confidence"] = [
        head_to_conf.get((dossier, head)) for dossier, head in zip(merged["dossier"], heads_for_row)
    ]

    keep_cols = list(manifest.columns) + [
        "predicted_start_page", "start_page_confidence",
        "predicted_document_type", "document_type_confidence", "predicted_segment_id",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols].to_csv(args.out, sep="\t", index=False)

    print(f"\n{len(merged)} pages, {len(heads_df)} predicted documents")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
