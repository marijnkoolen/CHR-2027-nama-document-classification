# NAMA document classification

Code and results from our project studying migration-dossier documents held
by the National Archives of Australia (NAA): a page-classification
pipeline, document-composition/dossier-size analyses, annotation tooling,
and an OCR/transcription pipeline. The underlying scans, transcriptions,
and raw personal data are not included anywhere in this repo - only code,
plots/summary results, and one anonymised labels dataset (see "What's
here" below). The restricted data is stored locally and can be view on request.

Although the code was developed specifically for the analysis of dossiers of
Dutch migrants travelling under the NAMA agreement, many parts of the pipeline
can be reused and repurposed for similar page stream segmentation and document
classification tasks.

The classification pipeline itself covers two tasks:

- **start_page** (binary): is this scan the first page of a document within
  a larger PDF/dossier?
- **document_type** (multi-class, start pages only): what kind of document
  is it?

Several training regimes are supported (KNN/XGBoost on frozen backbone
features, an LSTM over a document's page sequence, early/late multimodal
fusion, and end-to-end finetuning), across configurable vision and text
backbones, chained into a two-stage pipeline: predict start_page over every
page, then predict document_type only on the pages flagged as document
heads. See [`runs/per_task/eval_report.md`](runs/per_task/eval_report.md)
(or `eval_report.html` for the interactive version) for this project's own
results.

## Setup

```bash
# classification pipeline
pip install -r scripts/classification/requirements.txt
pip install -r scripts/classification/sequential/requirements.txt
# only if you need joint_legacy/'s standalone scripts directly:
pip install -r scripts/classification/joint_legacy/requirements.txt

# dossier_composition/ and dossier_size_model/ analyses (pymc/arviz/etc -
# see "A gap in scripts/dossier_composition/requirements-gpu.txt" below)
pip install -r scripts/dossier_composition/requirements-gpu.txt
```

## Description of the data

The ground truth labels are available in `data/labels/dossier_labels_merged_pdf12_stratified.tsv` with, per page: a dossier
identifier, page number, `document_type`, `start_page` (yes/no),
`img_path`, `text_path`, and (optionally) a `split` column - see
`scripts/classification/lib/labels.py`'s module docstring for the exact
column names and how splits are assigned if you don't provide one. Point
`DATA_ROOT` at the directory containing it.
`data/labels/dossier_labels_merged_pdf12_stratified.tsv` (this project's
own anonymised labels) is a real example of that shape, minus the actual
`img_path`/`text_path` targets - those point at scans this repo doesn't
ship.

## Pipeline stages

`make help` lists these; override `DATA_ROOT`/`RUN_DIR`/`DEVICE`/
`VISION_BACKBONES`/`TEXT_BACKBONES` on the command line as needed.

1. `make extract-features` - cache per-backbone embeddings (needed by the
   baseline/sequence/fusion regimes; finetune reads raw images/text
   directly).
2. `make train` - train every regime x backbone x modality x task
   combination that exists (idempotent, safe to re-run/resume).
3. `make evaluate-models` - score every trained model on its own task's
   held-out test set.
4. `make evaluate-pipeline` - chain the top start_page/doc_type models
   (including late-fusion candidates - see `evaluate_pipeline.py`'s module
   docstring) and score end-to-end.
5. `make predict-corpus` - run a chosen pipeline combination over an
   unlabeled corpus (`START_MODEL`/`DOC_MODEL`/`MANIFEST`/`IMAGE_ROOT`/
   `CACHE_DIR`/`PREDICT_OUT` - see the Makefile).
6. `make report` - build `eval_report.html` (and its markdown sibling
   `eval_report.md`) from whatever's under `RUN_DIR`.

`make all` runs stages 1-4.

### A metrics rule worth keeping

Only report **macro-averaged, end-to-end** metrics from this pipeline -
`start_macro_f1` (mean of both start_page classes), not the positive-
class-only F1 sklearn's binary mode computes by default; and the `*_e2e`
document_type metrics (every true start page scored, a start_page miss
counts as wrong), not a reading conditioned on start_page detection having
succeeded. Both excluded numbers score an easier, differently-scoped
question than the real two-stage inference task, and are easy to mistake
for the honest number if you're using this pipeline's output to estimate
document-type counts across a corpus. See `lib/evaluate.py`'s
`evaluate_predictions()` and `evaluate_pipeline.py`'s module docstring for
exactly which fields are computed internally but not surfaced, and why.

## Layout

```
scripts/classification/
  lib/          shared library (labels, features, models, training loops,
                 checkpointing, evaluation, segmentation metrics, ...)
  sequential/   the pipeline described above
  joint_legacy/ an older pipeline variant, superseded by sequential/ for
                start_page/doc_type - kept because a few of its files
                (precompute_embeddings.py, evaluate_segmentation.py,
                flag_prediction_errors.py) are still genuine dependencies
                of sequential/. No Makefile targets of its own.
scripts/dossier_composition/, scripts/dossier_size_model/
  Separate analyses (document co-occurrence/ordering/dispersion within a
  dossier; Bayesian modelling of dossier size, doc-type counts, and their
  temporal trends). Each has its own Makefile (`make help` inside either
  directory). Outputs land in data/dossier_composition/,
  data/dossier_size_model/ - see their `REPORT.md`/`report.md` for the
  full narrative write-up (or `report.html` for the same content with
  interactive plots), and "What's not here" below for what's excluded
  from those directories.
scripts/labels/
  Annotation label merging/splitting: merge_annotations.py combines raw
  per-annotator exports, merge_rare_doctypes.py folds low-count document
  types into "Other (...)" buckets, assign_stratified_splits.py builds the
  train/val/test split. Produces data/labels/dossier_labels_merged_pdf12_
  stratified.tsv (an anonymised version of which ships in this repo - see
  below).
scripts/ocr/
  OCR/transcription tooling used ahead of the classification pipeline -
  run_qwen_vl.py (Qwen2-VL-based transcription), run_got_ocr2.py
  (GOT-OCR2 alternative), benchmark_vllm.py, form_registration.py,
  visualize_detected_text.py.
docs/model_equations.tex
  The Bayesian model specifications behind dossier_size_model/'s analyses.
runs/per_task/
  Evaluation results: eval_report.html (and its markdown sibling
  eval_report.md, for reading the same leaderboards directly on GitHub),
  model_tables.tex, and per-model/per-pipeline-combination metrics.json/
  per_class_metrics/confusion matrices - the classification pipeline's
  actual results on this project's data, not a synthetic example run.
data/labels/dossier_labels_merged_pdf12_stratified.tsv
  The anonymised ground-truth labels (start_page/document_type per page)
  this project's classification results were trained/evaluated on. Dossier
  IDs have their name segment replaced with a synthetic sequential token
  (a2478-d0000123-1234567) - the NAA numeric identifier is kept as-is,
  since NAA's own finding aid already makes it publicly searchable outside
  the EU/GDPR (see scripts/classification's originating private repo's
  anonymise_dossier_ids.py for the full reasoning, not included here).
```

### What's not here

- **Trained model weights** (`model.pt`/`model.pkl`, ~6.8G across every
  regime/backbone/task) and **raw per-page prediction arrays**
  (`preds_test.npy`/`probs_test.npy`) under `runs/per_task/` - only the
  human-readable results (metrics, per-class breakdowns, confusion matrix
  images) are included.
- **Bayesian inference traces** (`*.nc` files under `data/dossier_composition/`
  and `data/dossier_size_model/`, ~2.3G total, some individual files over
  250M) and two large intermediate `.npy` arrays in
  `data/dossier_composition/` - only the plots/summary CSVs/reports derived
  from them are included. Re-running the relevant Makefile target
  regenerates them locally.
- **Raw page images/text/PDFs** - never included anywhere in this project's
  public code, at any point.

### A gap in scripts/dossier_composition/requirements-gpu.txt

`scripts/dossier_composition/order.py` imports `statsmodels`, which isn't
in `requirements-gpu.txt` - that file's own header explains it's a
carefully pinned closure (to avoid two specific version-conflict issues
already hit once).
Also note: `scripts/dossier_size_model/` uses the same pymc/arviz/
matplotlib/numpy/pandas/scipy stack as `dossier_composition/` but has no
`requirements.txt` of its own - install `dossier_composition`'s
`requirements-gpu.txt` before running either.
