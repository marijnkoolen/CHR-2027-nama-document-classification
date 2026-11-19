"""Plots publication-ready confusion matrices for one or more pipeline pairs
written by evaluate_pipeline.py, straight from each pair's predictions.tsv -
no models or feature caches needed, so this runs on the shipped results
under runs/per_task/pipeline_*/ as-is.

Two plots per pair:

- segmentation (start_page): every test page, true vs predicted start page.
- document_type, end-to-end: every TRUE start page, the same scoping as
  evaluate_pipeline.py's *_e2e metrics. A true start page the start_page
  model missed gets no document_type prediction of its own (it inherits the
  previous predicted segment's type), so instead of crediting/blaming that
  inherited type it lands in an extra "Missed start page" column - the
  sentinel that _joint_doctype_labels() counts as wrong. Predicted start
  pages that aren't true start pages have no true document_type row here;
  they show up as false positives in the segmentation plot instead.

Labels are shortened through DOC_TYPE_LABELS / SEGMENT_LABELS below - edit
those dicts to change what appears on the axes. The dict order is the axis
order; a class missing from DOC_TYPE_LABELS keeps its original name and is
appended at the end. Mapping several classes to the same short label merges
them into one row/column.

Cell colour is the row-normalised share (recall per true class), so small
classes stay as readable as large ones; cell text is the raw count by
default, or that share with --values row.

Usage:
    python scripts/classification/sequential/plot_confusion_matrices.py \\
        runs/per_task/pipeline_top3_e2e/lstm-facebook__dinov2-small__bert-ft-bert-base-uncased

    # every pair in a pipeline run, written next to each predictions.tsv:
    python scripts/classification/sequential/plot_confusion_matrices.py runs/per_task/pipeline_top3_e2e/*/

Writes, per pair (into the pair dir, or --out-dir/<pair name>/):
    confusion_matrix_segmentation.<fmt>
    confusion_matrix_document_type_e2e.<fmt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Editable label mappings
# --------------------------------------------------------------------------

# Full document_type class name -> axis label. Order here = axis order.
DOC_TYPE_LABELS = {
    "NAA cover": "NAA cover",
    "NAMA agreement": "NAMA agreement",
    "Approval notice": "Approval notice",
    "D.1": "D.1",
    "D.2": "D.2",
    "DM.1": "DM.1",
    "Registration card": "Registration card",
    "Photo": "Photo",
    "Report of selection and medical officers": "Report of selection and medical officers",
    "Judicial and political background check": "Judicial and political background check",
    "Letter about procedure": "Letter about procedure",
    "Testimonial labour (Qualification & Employment Proof)": "Testimonial labour",
    "Testimonial medical form (Medical & Health Documents)": "Testimonial medical form",
    "Testimonial medical letter (Medical & Health Documents)": "Testimonial medical letter",
    "Testimonial status (Application Documents)": "Testimonial status",
    # The "Other (<layout type>, <functional category>)" buckets would all
    # collapse into one "Other" without their parentheses, so they keep a
    # compact form of both. Map them all to "Other" to merge them instead.
    "Other (Letter, Application Documents)": "Other letter (application)",
    "Other (Letter, Other)": "Other letter",
    "Other (Structured Form, Administrative & Internal Processing Documents)": "Other form (administrative)",
    "Other (Structured Form, Application Documents)": "Other form (application)",
    "Other (Structured Form, Medical & Health Documents)": "Other form (medical)",
    "Other (Structured Form, Qualification & Employment Proof)": "Other form (employment)",
    "Other (Structured Form, Other)": "Other form",
    "Other (Other, Other)": "Other",
}

# Extra predicted-side column for true start pages the start_page model missed.
MISSED_START_LABEL = "Missed start page"

# start_page value -> axis label. Order here = axis order.
SEGMENT_LABELS = {
    "yes": "Start page",
    "no": "Continuation page",
}

AXIS_TITLES = {"true": "True", "pred": "Predicted"}
TITLES = {
    "segmentation": "Document segmentation (start page)",
    "document_type": "Document type (end-to-end)",
}

# Font sizes (pt) per plot. "large" is the document_type matrix, "small" the
# 2x2 segmentation one. Cells are sized just big enough for their numbers, so
# raising these makes the text bigger relative to the whole figure - which is
# what counts once it's scaled to the paper's column/text width. The longest
# axis labels set the figure's width, so shortening those in DOC_TYPE_LABELS
# helps legibility too.
FONT_SIZES = {
    "large": {"ticks": 14, "cells": 13, "axis": 15, "title": 16, "colorbar": 12},
    "small": {"ticks": 12, "cells": 13, "axis": 13, "title": 14, "colorbar": 10},
}

# --------------------------------------------------------------------------


def _is_yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("yes")


def segmentation_confusion(df: pd.DataFrame) -> pd.DataFrame:
    true = np.where(_is_yes(df["start_page"]), "yes", "no")
    pred = np.where(_is_yes(df["predicted_start_page"]), "yes", "no")
    cm = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="pred"))
    order = list(SEGMENT_LABELS)
    cm = cm.reindex(index=order, columns=order, fill_value=0)
    return cm.rename(index=SEGMENT_LABELS, columns=SEGMENT_LABELS)


def _short(name: str) -> str:
    return DOC_TYPE_LABELS.get(name, name)


def document_type_confusion(df: pd.DataFrame, keep_empty: bool = False) -> pd.DataFrame:
    scored = df.loc[_is_yes(df["start_page"]) & df["document_type"].notna()]
    pred_is_start = _is_yes(scored["predicted_start_page"])
    true = scored["document_type"].map(_short)
    pred = scored["predicted_document_type"].where(pred_is_start & scored["predicted_document_type"].notna())
    pred = pred.map(_short, na_action="ignore").fillna(MISSED_START_LABEL)

    cm = pd.crosstab(true.rename("true"), pred.rename("pred"))

    # Axis order follows DOC_TYPE_LABELS (deduplicated after merging), then any
    # unmapped classes seen in this pair, alphabetically.
    known = list(dict.fromkeys(DOC_TYPE_LABELS.values()))
    seen = set(cm.index) | set(cm.columns)
    seen.discard(MISSED_START_LABEL)
    classes = known + sorted(seen - set(known))
    if not keep_empty:
        classes = [c for c in classes if c in seen]
    cm = cm.reindex(index=classes, columns=classes + [MISSED_START_LABEL], fill_value=0)
    return cm


def plot_confusion(cm: pd.DataFrame, title: str | None, out_path: Path, values: str = "counts") -> None:
    counts = cm.to_numpy()
    row_sums = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, row_sums, out=np.zeros(counts.shape, dtype=float), where=row_sums > 0)

    n_rows, n_cols = counts.shape
    fs = FONT_SIZES["large" if n_cols > 4 else "small"]
    max_label = max(len(str(lbl)) for lbl in [*cm.index, *cm.columns])
    # Cells sized to fit a two-digit number; label margin ~ average glyph width x longest label.
    cell = 0.026 * fs["cells"] if n_cols > 4 else 0.9
    label_space = 0.0075 * fs["ticks"] * max_label
    fig, ax = plt.subplots(figsize=(n_cols * cell + label_space + 1.2, n_rows * cell + label_space + 0.6))

    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(n_rows):
        for j in range(n_cols):
            if counts[i, j] == 0:
                continue
            text = f"{counts[i, j]}" if values == "counts" else f"{share[i, j]:.2f}".lstrip("0")
            ax.text(
                j, i, text, ha="center", va="center", fontsize=fs["cells"],
                color="white" if share[i, j] > 0.55 else "#1a1a1a",
            )

    if n_cols > 4:
        ax.set_xticks(range(n_cols), labels=cm.columns, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks(range(n_cols), labels=cm.columns)
    ax.set_yticks(range(n_rows), labels=cm.index)
    ax.tick_params(labelsize=fs["ticks"], length=0)
    ax.set_xlabel(AXIS_TITLES["pred"], fontsize=fs["axis"])
    ax.set_ylabel(AXIS_TITLES["true"], fontsize=fs["axis"])
    # Thin white gridlines between cells; no outer frame.
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=fs["title"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Share of true class (row)", fontsize=fs["colorbar"])
    cbar.ax.tick_params(labelsize=fs["colorbar"] - 1)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair_dirs", type=Path, nargs="+", help="pipeline pair dirs, each containing predictions.tsv")
    parser.add_argument("--out-dir", type=Path, default=None, help="write to <out-dir>/<pair name>/ instead of the pair dir")
    parser.add_argument("--values", choices=["counts", "row"], default="counts", help="cell text: raw counts or row share")
    parser.add_argument("--format", default="png", help="output file format (png, pdf, svg, ...)")
    parser.add_argument("--keep-empty", action="store_true", help="keep classes with no true or predicted pages in this pair")
    parser.add_argument("--no-title", action="store_true", help="omit plot titles (e.g. when the paper's caption names them)")
    args = parser.parse_args()

    for pair_dir in args.pair_dirs:
        preds_path = pair_dir / "predictions.tsv"
        if not preds_path.exists():
            print(f"skipping {pair_dir}: no predictions.tsv")
            continue
        df = pd.read_csv(preds_path, sep="\t")
        out = (args.out_dir / pair_dir.name) if args.out_dir else pair_dir

        seg_cm = segmentation_confusion(df)
        plot_confusion(
            seg_cm, None if args.no_title else TITLES["segmentation"],
            out / f"confusion_matrix_segmentation.{args.format}", args.values,
        )
        doc_cm = document_type_confusion(df, keep_empty=args.keep_empty)
        plot_confusion(
            doc_cm, None if args.no_title else TITLES["document_type"],
            out / f"confusion_matrix_document_type_e2e.{args.format}", args.values,
        )


if __name__ == "__main__":
    main()
