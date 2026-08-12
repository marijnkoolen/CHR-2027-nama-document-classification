"""
Dispersion forest plot: per-type z-score (left, deviation from the exact
runs-test null of random placement -- negative = more clustered than
chance, positive = more scattered) and mean normalized contiguity score
(right, 0 = always one contiguous block, 1 = maximally scattered), naive vs.
classifier-corrected, ordered by corrected z-score.
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

    table = pd.read_csv(f"{args.out_dir}/dispersion_summary.csv")
    table = table.sort_values("corrected_z_score_mean", ascending=True)
    types = table["doc_type"].tolist()
    y = np.arange(len(types))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    ax = axes[0]
    offset = 0.17
    ax.hlines(y + offset, table["corrected_z_score_hdi_3"], table["corrected_z_score_hdi_97"],
               color=BLUE, linewidth=2, zorder=2)
    ax.scatter(table["corrected_z_score_mean"], y + offset, color=BLUE, s=32, zorder=3,
               label="corrected (94% HDI)", edgecolor="white", linewidth=0.5)
    ax.scatter(table["naive_z_score"], y - offset, color=ORANGE, s=32, zorder=3, marker="D", label="naive")
    for yi, low_support in zip(y, table["low_support"]):
        if low_support:
            ax.text(-0.02, yi, "*", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=11, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xlabel("z-score (< 0 clustered  |  0  |  scattered > 0)")
    ax.set_title("Dispersion vs. random-placement null", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")

    ax = axes[1]
    ax.hlines(y + offset, table["corrected_contiguity_hdi_3"], table["corrected_contiguity_hdi_97"],
               color=BLUE, linewidth=2, zorder=2)
    ax.scatter(table["corrected_contiguity_mean"], y + offset, color=BLUE, s=32, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.scatter(table["naive_contiguity"], y - offset, color=ORANGE, s=32, zorder=3, marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean contiguity score (0=clustered, 1=scattered)")
    ax.set_title("Descriptive contiguity", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)

    fig.text(0.01, 0.01, "* low support (< 15 qualifying dossiers on average)", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    path = f"{args.out_dir}/dispersion_naive_vs_corrected.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
