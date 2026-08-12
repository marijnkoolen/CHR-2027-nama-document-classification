"""Unified evaluation - a superset of the two notebooks' separate
evaluate_model() (binary: precision/recall/f1/roc_auc for the positive
class) and evaluate() (multi-class: accuracy/macro-F1) helpers. accuracy,
macro_f1 and weighted_f1 are always computed; the binary-only metrics are
added automatically when num_classes == 2, so this one function replaces
both notebook cell 18-equivalents.

Runs as a script, not a notebook, so confusion matrices are saved to disk
(out_dir) instead of plt.show()-n - hence the Agg backend, which needs no
display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
    probs: np.ndarray | None = None, class_names: list[str] | None = None,
    split: str = "test", out_dir: Path | None = None, max_classes_to_plot: int = 30,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if num_classes == 2:
        # precision/recall/f1 here are positive-class-only (sklearn's binary-average
        # default) - macro_precision/macro_recall are the mean-of-both-classes
        # counterparts to macro_f1 above, and are the ones safe to use anywhere a
        # number from this function might feed document-count corrections/imputation:
        # positive-class-only metrics silently describe an easier, differently-scoped
        # question and have caused wrong count estimates when mistaken for the honest
        # both-classes number (see the "positive-class-only metrics" project memory).
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        if probs is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))

    print(f"\n{'-' * 60}\n  {name} [{split}]\n{'-' * 60}")
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = None
    if class_names is not None:
        target_names = [(c[:28] + "…" if len(c) > 28 else c) for c in (class_names[i] for i in present)]
    print(classification_report(y_true, y_pred, labels=present, target_names=target_names, zero_division=0))
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_dict = classification_report(
            y_true, y_pred, labels=present, target_names=target_names, output_dict=True, zero_division=0
        )
        report_dict.pop("accuracy", None)  # a bare float in this dict, not a precision/recall/f1/support row
        per_class_df = pd.DataFrame(report_dict).T
        per_class_df.index.name = "class"
        per_class_df.to_csv(out_dir / f"per_class_metrics_{split}.tsv", sep="\t")

    if out_dir is not None and num_classes <= max_classes_to_plot:
        out_dir.mkdir(parents=True, exist_ok=True)
        labels_range = list(range(num_classes))
        display_labels = class_names if class_names is not None else [str(i) for i in labels_range]
        cm = confusion_matrix(y_true, y_pred, labels=labels_range)
        fig, ax = plt.subplots(figsize=(max(4, num_classes * 0.5), max(3, num_classes * 0.45)))
        ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} — {split}")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        fig.savefig(out_dir / f"confusion_matrix_{split}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return metrics
