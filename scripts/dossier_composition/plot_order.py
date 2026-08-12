"""
Order forest plot: Bradley-Terry rank theta (left -- more negative = earlier
in the typical sequence, from pairwise first-occurrence comparisons) and
mean normalized position (right -- 0 = start of dossier, 1 = end), naive vs.
classifier-corrected, both ordered by corrected rank.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT_DIR

BLUE = "#2a78d6"    # corrected
ORANGE = "#eb6834"  # naive
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    table = pd.read_csv(f"{args.out_dir}/order_summary.csv")
    table = table.sort_values("corrected_rank_theta_mean", ascending=False)
    types = table["doc_type"].tolist()
    y = np.arange(len(types))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    ax = axes[0]
    offset = 0.17
    ax.hlines(y + offset, table["corrected_rank_theta_hdi_3"], table["corrected_rank_theta_hdi_97"],
               color=BLUE, linewidth=2, zorder=2)
    ax.scatter(table["corrected_rank_theta_mean"], y + offset, color=BLUE, s=32, zorder=3,
               label="corrected (94% HDI)", edgecolor="white", linewidth=0.5)
    ax.scatter(table["naive_rank_theta"], y - offset, color=ORANGE, s=32, zorder=3, marker="D", label="naive")
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xlabel("Bradley-Terry rank (earlier <- 0 -> later)")
    ax.set_title("Typical sequence order", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")

    ax = axes[1]
    ax.hlines(y + offset, table["corrected_mean_position_hdi_3"], table["corrected_mean_position_hdi_97"],
               color=BLUE, linewidth=2, zorder=2)
    ax.scatter(table["corrected_mean_position_mean"], y + offset, color=BLUE, s=32, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.scatter(table["naive_mean_position"], y - offset, color=ORANGE, s=32, zorder=3, marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean normalized position (0=start, 1=end)")
    ax.set_title("Absolute position in dossier", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)

    fig.tight_layout()
    path = f"{args.out_dir}/order_naive_vs_corrected.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
