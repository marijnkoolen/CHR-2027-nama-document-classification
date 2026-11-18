"""Builds the combined comparison table + bar chart across every model
that's been evaluated for a task - shared by summarize_results.py (a thin
standalone CLI over this) and evaluate_models.py (which calls this once at
the end of a full evaluation run, so a single evaluate_models.py invocation
produces the combined report too, without a second command).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_io import load_all_results


def build_summary(run_dir: Path, task: str) -> pd.DataFrame:
    results = load_all_results(run_dir, task)
    if not results:
        raise SystemExit(f"no results found under {run_dir / task} - run evaluate_models.py first")

    metrics_by_model = {name: r["metrics"] for name, r in results.items()}
    summary_df = pd.DataFrame(metrics_by_model).T.round(4)
    sort_col = "macro_f1" if "macro_f1" in summary_df.columns else summary_df.columns[0]
    summary_df = summary_df.sort_values(sort_col, ascending=False)
    summary_df.index.name = "model"
    return summary_df


def write_summary(summary_df: pd.DataFrame, out_dir: Path, task: str) -> tuple[Path, Path, Path]:
    """Writes summary.tsv, summary.json (records-oriented - one object per
    model, easiest to consume from most tooling) and classifier_comparison.png
    to out_dir. Returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = out_dir / "summary.tsv"
    summary_df.to_csv(tsv_path, sep="\t")

    json_path = out_dir / "summary.json"
    summary_df.reset_index().to_json(json_path, orient="records", indent=2)

    plot_metrics = [c for c in summary_df.columns if summary_df[c].notna().any()]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], plot_metrics):
        vals = summary_df[metric].astype(float)
        colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(vals)))
        bars = ax.bar(vals.index, vals.values, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
        for bar, v in zip(bars, vals.values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(f"Test-set performance — {task}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = out_dir / "classifier_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return tsv_path, json_path, plot_path


def build_and_write_summary(run_dir: Path, task: str, out_dir: Path | None = None) -> pd.DataFrame:
    summary_df = build_summary(run_dir, task)
    tsv_path, json_path, plot_path = write_summary(summary_df, out_dir or (run_dir / task), task)
    print(summary_df.to_string())
    print(f"\nWrote {tsv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    return summary_df
