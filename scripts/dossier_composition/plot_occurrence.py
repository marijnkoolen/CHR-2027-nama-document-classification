"""
Plot naive vs. classifier-corrected occurrence statistics from occurrence.py:
prevalence (left) and mean instance-count given present (right), per tracked
document type, ordered by corrected prevalence.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OTHER, OUT_DIR

BLUE = "#2a78d6"    # corrected
ORANGE = "#eb6834"  # naive
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

TICK_FONTSIZE = 16
LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 16
NOTE_FONTSIZE = 14


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=SECONDARY_INK, labelsize=TICK_FONTSIZE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    table = pd.read_csv(f"{args.out_dir}/occurrence_summary.csv")
    table = table[table["doc_type"] != OTHER].sort_values("corrected_prevalence_mean", ascending=True)
    types = table["doc_type"].tolist()
    types = [t.split(' (')[0] for t in types]
    print(types)
    y = np.arange(len(types))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))

    ax = axes[0]
    offset = 0.17
    ax.hlines(y + offset, table["corrected_prevalence_hdi_3"], table["corrected_prevalence_hdi_97"],
               color=BLUE, linewidth=2, zorder=2)
    ax.scatter(table["corrected_prevalence_mean"], y + offset, color=BLUE, s=32, zorder=3,
               label="corrected (94% HDI)", edgecolor="white", linewidth=0.5)
    ax.scatter(table["naive_prevalence"], y - offset, color=ORANGE, s=32, zorder=3, marker="D", label="naive")
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace(" (", "\n(") for t in types], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("prevalence (fraction of dossiers with >=1 instance)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Occurrence: naive vs. corrected", loc="left", color=INK, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, labelcolor=SECONDARY_INK, loc="upper left")
    #ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, labelcolor=SECONDARY_INK, loc="lower right")

    ax = axes[1]
    ax.hlines(y + offset, table["corrected_mean_count_given_present_hdi_3"],
               table["corrected_mean_count_given_present_hdi_97"], color=BLUE, linewidth=2, zorder=2)
    ax.scatter(table["corrected_mean_count_given_present_mean"], y + offset, color=BLUE, s=32, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.scatter(table["naive_mean_count_given_present"], y - offset, color=ORANGE, s=32, zorder=3, marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("mean instance count, given present", fontsize=LABEL_FONTSIZE)
    ax.set_title("Typical multiplicity when present", loc="left", color=INK, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.axvline(1, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    style_axes(ax)

    fig.tight_layout()
    path = f"{args.out_dir}/occurrence_naive_vs_corrected.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
