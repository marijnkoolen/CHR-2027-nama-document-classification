# NAMA document classification

Page-level classification pipeline for scanned archival document
collections: two tasks,

- **start_page** (binary): is this scan the first page of a document within
  a larger PDF/dossier?
- **document_type** (multi-class, start pages only): what kind of document
  is it?

Several training regimes are supported (KNN/XGBoost on frozen backbone
features, an LSTM over a document's page sequence, early/late multimodal
fusion, and end-to-end finetuning), across configurable vision and text
backbones, chained into a two-stage pipeline: predict start_page over every
page, then predict document_type only on the pages flagged as document
heads.

This is the code from the classification component of a larger project
studying migration-dossier documents held by the National Archives of
Australia; the annotated data itself is not included here (it's personal
data, kept private) - point this pipeline at your own similarly-shaped
corpus (see "Bringing your own data" below).

## Setup

```bash
pip install -r scripts/classification/requirements.txt
pip install -r scripts/classification/sequential/requirements.txt
# only if you need joint_legacy/'s standalone scripts directly:
pip install -r scripts/classification/joint_legacy/requirements.txt
```

## Bringing your own data

Your corpus needs a labels TSV with, per page: a dossier/document
identifier, page number, `document_type`, `start_page` (yes/no),
`img_path`, `text_path`, and (optionally) a `split` column - see
`scripts/classification/lib/labels.py`'s module docstring for the exact
column names and how splits are assigned if you don't provide one. Point
`DATA_ROOT` at the directory containing it.

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
6. `make report` - build `eval_report.html` from whatever's under
   `RUN_DIR`.

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
```
