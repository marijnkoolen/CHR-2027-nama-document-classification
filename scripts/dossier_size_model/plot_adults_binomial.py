"""
Fitted per-year adult probability p (num_adults ~ Binomial(num_persons, p))
vs. the raw observed proportion, from model_num_adults_binomial.py.

Uses the "iid" (unconstrained per-year) mean structure for the same reason
as the other temporal plots: LOO didn't decisively prefer a structured
(trend/step) form here either, so an imposed line would overstate the
evidence -- and there's a known confound at 1956 (the adult-age threshold
itself changed from 18+ to 16+), which would produce a level shift in p
mechanically, independent of any real demographic change.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from model_dossier_size import DATA_PATH, load_data
from model_dossier_size_temporal import prep_data

OUT_DIR = "data/dossier_size_model"

BLUE = "#2a78d6"
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
    df = prep_data(load_data(DATA_PATH))
    years = np.sort(df["travel_year"].unique())

    idata = az.from_netcdf(f"{OUT_DIR}/idata_adults_binom_temporal_mean_iid.nc")
    post = idata.posterior
    alpha = post["alpha"].values.flatten()
    alpha_year = post["alpha_year"].values.reshape(-1, len(years))
    p = 1 / (1 + np.exp(-(alpha[:, None] + alpha_year)))  # (n_draws, n_years)

    p_mean = p.mean(axis=0)
    p_lo, p_hi = np.percentile(p, [3, 97], axis=0)

    raw = df.groupby("travel_year").apply(
        lambda g: g["num_adults"].sum() / g["num_persons"].sum(), include_groups=False
    ).reindex(years)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.vlines(years, p_lo, p_hi, color=BLUE, linewidth=2, zorder=2)
    ax.scatter(years, p_mean, color=BLUE, s=32, zorder=3, label="fitted per-year p (adult probability)")
    ax.scatter(years, raw.values, color=INK, s=24, zorder=3, label="observed proportion adult")
    ax.axvline(1956, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.annotate("adult threshold\n18+ -> 16+", xy=(1956, ax.get_ylim()[0]), xytext=(1956.2, 0.3),
                fontsize=8, color=SECONDARY_INK)
    ax.set_xlabel("travel year")
    ax.set_ylabel("P(person is adult)")
    ax.set_title(
        "Adult proportion of unit size, per year (Binomial model)",
        loc="left", color=INK, fontsize=11, fontweight="bold",
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")

    fig.tight_layout()
    path = f"{OUT_DIR}/adults_binomial_p_per_year.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
