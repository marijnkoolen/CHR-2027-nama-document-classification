"""
Three forest plots (full period, pre-1956, post-1956) of the minor/pre-adult
/adult per-document-type effects from model_doc_types_three_groups.py.
Same doc-type ordering across all three panels (by full-period pre-adult
effect) so a type can be tracked across periods.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model"

BLUE = "#2a78d6"     # adult
ORANGE = "#eb6834"   # pre-adult (16-17)
AQUA = "#1baf7a"      # minor
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

PERIOD_LABELS = {
    "full": "Full period (1952-1965, n=1307)",
    "pre1956": "Pre-1956 (1952-1955, n=646)",
    "post1956": "Post-1956 (1956-1965, n=661)",
}


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


def plot_period(ax, idata, order, show_legend):
    post = idata.posterior
    y = np.arange(len(order))
    offset = 0.24

    for series_offset, (var, color, label) in zip(
        [offset, 0, -offset],
        [("beta_adult", BLUE, "per adult (18+)"),
         ("beta_preadult", ORANGE, "per pre-adult (16-17)"),
         ("beta_minor", AQUA, "per minor (<16)")],
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
        ax.scatter(means, yy, color=color, s=30, zorder=3, label=label if show_legend else None,
                   edgecolor="white", linewidth=0.5)

    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("% change in expected document count per additional person (94% HDI)")
    style_axes(ax)
    ax.set_ylim(-0.7, len(order) - 0.3)


def main():
    idatas = {p: az.from_netcdf(f"{OUT_DIR}/idata_doc_types_3group_{p}.nc") for p in PERIOD_LABELS}

    order = (
        idatas["full"].posterior["beta_preadult"].mean(dim=("chain", "draw"))
        .to_pandas().sort_values(ascending=True).index.tolist()
    )

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True)
    for ax, (period, label) in zip(axes, PERIOD_LABELS.items()):
        plot_period(ax, idatas[period], order, show_legend=(period == "full"))
        label = label.replace('Pre-1956', 'Pre-policy change').replace('Post-1956', 'Post-policy change')
        print(f"subplot title label: {label}")
        ax.set_title(label, loc="left", color=INK, fontsize=11, fontweight="bold")

    axes[0].legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    fig.suptitle(
        "Which document types are driven by adults vs. pre-adults (16-17) vs. minors -- and does it shift at 1956?",
        fontsize=12.5, fontweight="bold", color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = f"{OUT_DIR}/doc_type_3group_effects_by_period.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
