"""
Heatmap of pairwise co-occurrence (log odds ratio) between document types,
using the richness-controlled (Mantel-Haenszel) estimate from
co_occurrence.py -- the raw/unstratified version is confounded by dossiers
varying a lot in how many types they contain at all (1-11), which inflates
every pair's raw co-occurrence uniformly. Diverging blue (co-occur less than
chance) / red (more than chance) color, neutral gray at zero. Cells whose
94% HDI includes 0 (not credible) are shown at reduced opacity rather than
omitted, so the reader sees both the point estimate and how much to trust
it -- though in this dataset every pair turns out credibly positive even
after the richness control (see script docstring in co_occurrence.py).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from common import OUT_DIR, TRACKED_TYPES

BLUE = "#2a78d6"    # co-occur less than chance
RED = "#e34948"     # co-occur more than chance
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"

FADED_ALPHA = 0.30
CLIP = 4.0  # log-odds clipping for color scale


def short_label(t: str) -> str:
    return t.split(" (")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    table = pd.read_csv(f"{args.out_dir}/co_occurrence_summary.csv")
    types = TRACKED_TYPES
    n = len(types)
    idx = {t: i for i, t in enumerate(types)}

    log_odds = np.full((n, n), np.nan)
    credible = np.zeros((n, n), dtype=bool)
    for _, row in table.iterrows():
        i, j = idx[row["type_a"]], idx[row["type_b"]]
        log_odds[i, j] = log_odds[j, i] = row["corrected_mh_log_odds_mean"]
        credible[i, j] = credible[j, i] = row["mh_credible"]

    log_odds_clipped = np.clip(log_odds, -CLIP, CLIP)

    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    norm = TwoSlopeNorm(vmin=-CLIP, vcenter=0, vmax=CLIP)
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])

    for i in range(n):
        for j in range(n):
            if i == j:
                color = MUTED
                alpha = 0.15
            else:
                color = cmap(norm(log_odds_clipped[i, j]))
                alpha = 1.0 if credible[i, j] else FADED_ALPHA
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=color, alpha=alpha, edgecolor="white", linewidth=1))
            if i != j and not np.isnan(log_odds[i, j]):
                text_color = INK if not credible[i, j] else ("white" if abs(log_odds_clipped[i, j]) > CLIP * 0.55 else INK)
                ax.text(j + 0.5, n - 1 - i + 0.5, f"{log_odds[i, j]:.1f}", ha="center", va="center",
                        fontsize=7.5, color=text_color, alpha=1.0 if credible[i, j] else 0.55)

    labels = [short_label(t) for t in types]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, color=SECONDARY_INK)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=8.5, color=SECONDARY_INK)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log odds ratio (co-occur less <- 0 -> co-occur more)", fontsize=9, color=SECONDARY_INK)
    cbar.ax.tick_params(labelsize=8, colors=SECONDARY_INK)

    ax.set_title(
        "Pairwise co-occurrence of document types (faded = not credible at 94% HDI)",
        loc="left", color=INK, fontsize=12, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    path = f"{args.out_dir}/co_occurrence_heatmap.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
