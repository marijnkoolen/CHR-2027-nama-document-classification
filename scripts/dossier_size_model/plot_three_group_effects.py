"""
Corrected pre/post-1956 effect plot: replaces era_effect_pre_post_1956.png.
That earlier plot showed a shrinking "adult" effect at 1956 -- based on
num_adults switching definition (18+ -> 16+) at that year. Per domain-expert
correction, the adult threshold never moved; the 16-17 (pre-adult) group's
own paperwork requirement changed instead. This plots the corrected story
from Model C3: adult and minor effects constant across 1956, pre-adult
effect drops.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

OUT_DIR = "data/dossier_size_model"

BLUE = "#2a78d6"     # pre-adult, pre-1956
ORANGE = "#eb6834"   # pre-adult, post-1956
AQUA = "#1baf7a"      # adult (18+), constant
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
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    idata = az.from_netcdf(f"{OUT_DIR}/idata_three_group_C3.nc")
    post = idata.posterior

    adult = (np.exp(post["beta_adult"].values.flatten()) - 1) * 100
    minor = (np.exp(post["beta_minor"].values.flatten()) - 1) * 100
    preadult_pre = (np.exp(post["beta_preadult"].values.flatten()) - 1) * 100
    preadult_post = (np.exp(post["beta_preadult"].values.flatten()
                            + post["beta_preadult_era"].values.flatten()) - 1) * 100

    fig, ax = plt.subplots(figsize=(9.5, 5))

    all_vals = np.concatenate([adult, minor, preadult_pre, preadult_post])
    x_grid = np.linspace(all_vals.min() - 2, all_vals.max() + 2, 600)

    series = [
        (preadult_pre, BLUE, "pre-adult (16-17), pre-1956", "-"),
        (preadult_post, ORANGE, "pre-adult (16-17), post-1956", "-"),
        (adult, AQUA, "adult (18+), both eras (unchanged)", "--"),
    ]
    for values, color, label, ls in series:
        kde_y = gaussian_kde(values)(x_grid)
        kde_y = kde_y / kde_y.max()
        ax.fill_between(x_grid, kde_y, color=color, alpha=0.2 if ls == "-" else 0.0, linewidth=0, zorder=2)
        ax.plot(x_grid, kde_y, color=color, linewidth=2, linestyle=ls, label=label, zorder=3)
        mean_val = values.mean()
        ax.axvline(mean_val, color=color, linewidth=1, linestyle=":", zorder=2, ymax=0.87)
        ax.annotate(f"{mean_val:+.0f}%", xy=(mean_val, 0.9), ha="center", va="bottom",
                    color=color, fontsize=9, fontweight="bold")

    minor_y = gaussian_kde(minor)(x_grid)
    minor_y = minor_y / minor_y.max()
    ax.plot(x_grid, minor_y, color=MUTED, linewidth=1.5, linestyle=":", label="minor (<16), both eras (reference)",
            zorder=1)

    ax.set_ylim(0, 1.15)
    ax.set_xlabel("% change in expected number of documents per additional person")
    ax.set_ylabel("posterior density (normalized to peak)")
    ax.set_title(
        #"Corrected: it's the pre-adult (16-17) effect that shrinks at 1956, not the adult effect",
        "Contributions of different age groups to the number of documents in a dossier",
        loc="left", color=INK, fontsize=11.5, fontweight="bold",
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    path = f"{OUT_DIR}/three_group_era_effect_corrected.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
