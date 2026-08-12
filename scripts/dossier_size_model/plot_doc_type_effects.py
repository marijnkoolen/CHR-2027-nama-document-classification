"""
Forest plot of per-document-type adult/minor effects from model_doc_types.py.
Reads data/dossier_size_model/idata_doc_types.nc.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model"

BLUE = "#2a78d6"    # adult
ORANGE = "#eb6834"  # minor
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


def pct_effect(beta_draws):
    return (np.exp(beta_draws) - 1) * 100


def main():
    idata = az.from_netcdf(f"{OUT_DIR}/idata_doc_types.nc")
    post = idata.posterior

    order = (
        post["beta_adult"].mean(dim=("chain", "draw")).to_pandas().sort_values(ascending=True).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(order))
    offset = 0.17

    for series_offset, (var, color, label) in zip(
        [offset, -offset],
        [("beta_adult", BLUE, "per adult"), ("beta_minor", ORANGE, "per minor")],
    ):
        means, lo, hi = [], [], []
        for t in order:
            draws = pct_effect(post[var].sel(type=t).values.flatten())
            means.append(draws.mean())
            lo.append(np.percentile(draws, 3))
            hi.append(np.percentile(draws, 97))
        means, lo, hi = np.array(means), np.array(lo), np.array(hi)
        yy = y + series_offset
        ax.hlines(yy, lo, hi, color=color, linewidth=2, zorder=2)
        ax.scatter(means, yy, color=color, s=36, zorder=3, label=label, edgecolor="white", linewidth=0.6)

    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9.5, color=INK)
    ax.set_xlabel("% change in expected document count per additional person (94% HDI)")
    ax.set_title(
        "Which document types are driven by adults vs. minors?",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_ylim(-0.7, len(order) - 0.3)

    fig.tight_layout()
    path = f"{OUT_DIR}/doc_type_adult_minor_effects.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
