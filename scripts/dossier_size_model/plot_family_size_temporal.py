"""
Per-year (unconstrained) posterior estimates of mean and spread for
num_persons and num_adults, from model_family_size_temporal.py.

Neither outcome's mean-structure comparison gave a decisive win to a
directional (trend/step) form over plain year-to-year noise, and dispersion
consistently favored unstructured per-year variation over a smooth trend --
so, as with num_docs, this plots the honest per-year (iid) estimates rather
than a fitted line that would overstate the evidence for a trend.
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

OUTCOMES = ["num_persons", "num_adults"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def per_year_mean(idata_mean_iid, years):
    post = idata_mean_iid.posterior
    alpha = post["alpha"].values.flatten()
    alpha_year = post["alpha_year"].values.reshape(-1, len(years))
    mu = np.exp(alpha[:, None] + alpha_year)  # (n_draws, n_years)
    return mu.mean(axis=0), np.percentile(mu, 3, axis=0), np.percentile(mu, 97, axis=0)


def per_year_sd(idata_disp_iid, years):
    post = idata_disp_iid.posterior
    alpha = post["alpha"].values.flatten()
    alpha_year = post["alpha_year"].values.reshape(-1, len(years))
    year_c = years - 1956

    mean_trend = post["mean_trend"].values.flatten() if "mean_trend" in post else np.zeros_like(alpha)
    mean_step = post["mean_step"].values.flatten() if "mean_step" in post else np.zeros_like(alpha)
    mean_trend_pre = post["mean_trend_pre"].values.flatten() if "mean_trend_pre" in post else None
    mean_trend_post = post["mean_trend_post"].values.flatten() if "mean_trend_post" in post else None

    log_alpha_year_disp = post["log_alpha_year_disp"].values.reshape(-1, len(years))

    n_draws = alpha.shape[0]
    sd = np.empty((n_draws, len(years)))
    for j, yc in enumerate(years - 1956):
        era = 1.0 if yc >= 0 else 0.0
        if mean_trend_pre is not None:
            mean_term = mean_step * era + mean_trend_pre * yc * (1 - era) + mean_trend_post * yc * era
        else:
            mean_term = mean_trend * yc + mean_step * era
        mu = np.exp(alpha + alpha_year[:, j] + mean_term)
        alpha_nb = np.exp(log_alpha_year_disp[:, j])
        sd[:, j] = np.sqrt(mu + mu**2 / alpha_nb)

    return sd.mean(axis=0), np.percentile(sd, 3, axis=0), np.percentile(sd, 97, axis=0)


def main():
    df = prep_data(load_data(DATA_PATH))
    years = np.sort(df["travel_year"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for row, outcome in enumerate(OUTCOMES):
        idata_mean_iid = az.from_netcdf(f"{OUT_DIR}/idata_{outcome}_temporal_mean_iid.nc")
        raw = df.groupby("travel_year")[outcome].agg(["mean", "std", "var"]).reindex(years)

        mean_mean, mean_lo, mean_hi = per_year_mean(idata_mean_iid, years)

        ax = axes[row, 0]
        ax.vlines(years, mean_lo, mean_hi, color=BLUE, linewidth=2, zorder=2)
        ax.scatter(years, mean_mean, color=BLUE, s=30, zorder=3, label="fitted per-year mean")
        ax.scatter(years, raw["mean"], color=INK, s=22, zorder=3, label="observed yearly mean")
        ax.axvline(1956, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.set_xlabel("travel year")
        ax.set_ylabel(outcome)
        ax.set_title(f"{outcome}: mean, per year", loc="left", color=INK, fontsize=11, fontweight="bold")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK, loc="best")

        ax = axes[row, 1]
        if outcome == "num_adults":
            # NB dispersion is structurally invalid here: variance/mean < 1 in every
            # single year (under-dispersed), which NB (variance >= mean, always)
            # cannot represent -- so plot the diagnostic that shows why, not a fitted
            # NB spread curve that would misrepresent the data.
            var_to_mean = raw["var"] / raw["mean"]
            ax.scatter(years, var_to_mean, color=BLUE, s=30, zorder=3, label="observed variance / mean")
            ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle="--", zorder=1, label="Poisson reference (=1)")
            ax.axvline(1956, color=MUTED, linewidth=1, linestyle=":", zorder=1)
            ax.set_xlabel("travel year")
            ax.set_ylabel("variance / mean")
            ax.set_title(
                "num_adults: under-dispersed every year -- NB spread invalid",
                loc="left", color=INK, fontsize=11, fontweight="bold",
            )
            style_axes(ax)
            ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK, loc="best")
            continue

        idata_disp_iid = az.from_netcdf(f"{OUT_DIR}/idata_{outcome}_temporal_disp_iid.nc")
        sd_mean, sd_lo, sd_hi = per_year_sd(idata_disp_iid, years)
        ax.vlines(years, sd_lo, sd_hi, color=BLUE, linewidth=2, zorder=2)
        ax.scatter(years, sd_mean, color=BLUE, s=30, zorder=3, label="fitted per-year SD")
        ax.scatter(years, raw["std"], color=INK, s=22, zorder=3, label="observed yearly SD")
        ax.axvline(1956, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.set_xlabel("travel year")
        ax.set_ylabel(f"SD of {outcome}")
        ax.set_title(f"{outcome}: spread, per year", loc="left", color=INK, fontsize=11, fontweight="bold")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK, loc="best")

    fig.tight_layout()
    path = f"{OUT_DIR}/family_size_temporal_trend.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
