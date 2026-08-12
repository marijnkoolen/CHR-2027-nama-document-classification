"""
Plot the fitted temporal trend in dossier size (mean) and spread (implied SD,
from the NB dispersion trend) for a "typical" dossier (num_adults/num_minors
held at their sample means), against raw per-year statistics.

Reads data/dossier_size_model/temporal_winner.txt to find which mean
structure (iid/trend/step/step_trend) actually won the LOO comparison in
model_dossier_size_temporal.py -- NOT hardcoded to "trend". An earlier
version of this script hardcoded idata_temporal_mean_trend.nc and a
`mean_trend` variable name, which broke outright (KeyError) the moment a
rerun on better-corrected data made "step" win instead -- LOO comparisons
between these structures are close (see model_dossier_size_temporal.py's
docstring), so which one wins can legitimately change with the data, and
the plot needs to follow that, not assume it.

Reads idata_temporal_mean_{winner}.nc for the mean panel and
idata_temporal_disp_iid.nc for the spread panel -- the unconstrained
per-year dispersion model, which beat both constant and trend dispersion on
LOO in model_dossier_size_temporal.py, so it's the honest choice for the
spread panel rather than a smooth (and unsupported) trend line.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from model_dossier_size import DATA_PATH, load_data

OUT_DIR = "data/dossier_size_model"


def mean_structure_term(post, structure: str, year_c: np.ndarray, era: np.ndarray) -> np.ndarray:
    """(n_draws, n_years) array, mirroring model_dossier_size_temporal.py's
    add_structured_term but evaluated over posterior draws instead of built
    into a PyMC model. `era` here is per-YEAR (0/1, year >= 1956), matching
    year_c's shape -- the model's own era is per-OBSERVATION, but every
    observation in a given year shares the same era, so this is equivalent."""
    n_draws = post["alpha"].values.flatten().shape[0]
    n_years = len(year_c)
    if structure == "iid":
        return np.zeros((n_draws, n_years))
    if structure == "trend":
        beta = post["mean_trend"].values.flatten()
        return beta[:, None] * year_c[None, :]
    if structure == "step":
        beta = post["mean_step"].values.flatten()
        return beta[:, None] * era[None, :]
    if structure == "step_trend":
        beta_step = post["mean_step"].values.flatten()
        beta_pre = post["mean_trend_pre"].values.flatten()
        beta_post = post["mean_trend_post"].values.flatten()
        return (
            beta_step[:, None] * era[None, :]
            + beta_pre[:, None] * (year_c[None, :] * (1 - era[None, :]))
            + beta_post[:, None] * (year_c[None, :] * era[None, :])
        )
    raise ValueError(structure)

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
    df = load_data(DATA_PATH)

    with open(f"{OUT_DIR}/temporal_winner.txt") as f:
        winner = f.read().strip()
    print(f"Using winning mean structure: '{winner}'")

    idata_mean = az.from_netcdf(f"{OUT_DIR}/idata_temporal_mean_{winner}.nc")
    # dispersion structure: "iid" per-year model -- beat both "constant" and "trend"
    # dispersion by 26-57 elpd, so a smooth trend line would misrepresent the spread story
    idata_disp = az.from_netcdf(f"{OUT_DIR}/idata_temporal_disp_iid.nc")

    years = np.sort(df["travel_year"].unique())
    year_c = years - 1956
    era = (years >= 1956).astype(float)
    mean_adults = df["num_adults"].mean()
    mean_minors = df["num_minors"].mean()

    post_m = idata_mean.posterior
    alpha = post_m["alpha"].values.flatten()
    beta_adult = post_m["beta_adult"].values.flatten()
    beta_minor = post_m["beta_minor"].values.flatten()
    mean_term = mean_structure_term(post_m, winner, year_c, era)  # (n_draws, n_years)
    alpha_year = post_m["alpha_year"].values.reshape(-1, len(years))  # (n_draws, n_years)

    log_mu = (
        alpha[:, None]
        + alpha_year
        + mean_term
        + (beta_adult * mean_adults)[:, None]
        + (beta_minor * mean_minors)[:, None]
    )
    mu_typical = np.exp(log_mu)

    mu_mean = mu_typical.mean(axis=0)
    mu_lo, mu_hi = np.percentile(mu_typical, [3, 97], axis=0)

    # spread: use the SAME "typical dossier" mu from the winning mean-structure model,
    # combined with the iid model's own per-year mu (its mean structure is the SAME
    # winner, refit jointly with iid dispersion) and its per-year alpha_nb, so mu and
    # alpha_nb come from one consistent posterior rather than mixing two separate fits
    post_d = idata_disp.posterior
    alpha_d = post_d["alpha"].values.flatten()
    beta_adult_d = post_d["beta_adult"].values.flatten()
    beta_minor_d = post_d["beta_minor"].values.flatten()
    mean_term_d = mean_structure_term(post_d, winner, year_c, era)
    alpha_year_d = post_d["alpha_year"].values.reshape(-1, len(years))
    log_alpha_year_disp = post_d["log_alpha_year_disp"].values.reshape(-1, len(years))

    log_mu_d = (
        alpha_d[:, None]
        + alpha_year_d
        + mean_term_d
        + (beta_adult_d * mean_adults)[:, None]
        + (beta_minor_d * mean_minors)[:, None]
    )
    mu_d = np.exp(log_mu_d)
    alpha_nb_year = np.exp(log_alpha_year_disp)
    sd_typical = np.sqrt(mu_d + mu_d**2 / alpha_nb_year)

    sd_mean = sd_typical.mean(axis=0)
    sd_lo, sd_hi = np.percentile(sd_typical, [3, 97], axis=0)

    raw = df.groupby("travel_year")["num_docs"].agg(["mean", "std", "count"])
    raw = raw.reindex(years)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.fill_between(years, mu_lo, mu_hi, color=BLUE, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(years, mu_mean, color=BLUE, linewidth=2, zorder=2, label="fitted mean (typical dossier)")
    ax.scatter(years, raw["mean"], color=INK, s=22, zorder=3, label="observed yearly mean")
    ax.axvline(1956, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.set_xlabel("travel year")
    ax.set_ylabel("num_docs")
    ax.set_title("Mean dossier size over time", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")

    ax = axes[1]
    # per-year (unpooled-shape) estimate, NOT a smooth trend -- a monotonic trend on
    # dispersion lost decisively to this unconstrained per-year model on LOO (elpd
    # difference 56.6), so drawing a fitted line here would overstate the evidence
    ax.vlines(years, sd_lo, sd_hi, color=BLUE, linewidth=2, zorder=2)
    ax.scatter(years, sd_mean, color=BLUE, s=30, zorder=3, label="fitted SD, per-year (typical dossier)")
    ax.scatter(years, raw["std"], color=INK, s=22, zorder=3, label="observed yearly SD")
    ax.axvline(1956, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.set_xlabel("travel year")
    ax.set_ylabel("SD of num_docs")
    ax.set_title("Spread of dossier size, per year (no smooth trend)", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")

    fig.tight_layout()
    path = f"{OUT_DIR}/dossier_size_temporal_trend.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
