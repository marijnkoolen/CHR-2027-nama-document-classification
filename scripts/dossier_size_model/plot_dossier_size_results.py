"""
Posterior predictive check and pre/post-1956 adult-effect plot for the
negative-binomial dossier-size model (Model C: num_adults + num_minors +
adult_x_era). Requires model_dossier_size.py to have been run first (reads
its saved idata_C_adults_minors_era.nc trace).

SUPERSEDED (see model_dossier_size.py docstring): Model C's era interaction
is on an era-dependent "num_adults" that turned out to be mis-specified --
use plot_three_group_effects.py for the corrected pre/post-1956 effect plot.
Kept for reproducibility of the original result.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from model_dossier_size import DATA_PATH, load_data, fit_model  # noqa: F401 (fit_model kept for parity)

OUT_DIR = "data/dossier_size_model"

# dataviz reference palette (references/palette.md)
BLUE = "#2a78d6"      # categorical slot 1 -> pre-1956
ORANGE = "#eb6834"    # categorical slot 2 -> post-1956
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


def build_model_C(df):
    years = np.sort(df["travel_year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years)}
    year_idx = df["travel_year"].map(year_to_idx).to_numpy()
    docs = df["num_docs"].to_numpy()
    adults = df["num_adults"].to_numpy()
    minors = df["num_minors"].to_numpy()
    era = df["era"].to_numpy()

    coords = {"year": years, "obs": df.index}
    with pm.Model(coords=coords) as model:
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        alpha = pm.Normal("alpha", mu=np.log(docs.mean()), sigma=1.5)
        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
        z_year = pm.Normal("z_year", mu=0.0, sigma=1.0, dims="year")
        alpha_year = pm.Deterministic("alpha_year", z_year * sigma_year, dims="year")

        adult_data = pm.Data("num_adults", adults, dims="obs")
        minor_data = pm.Data("num_minors", minors, dims="obs")
        era_data = pm.Data("era", era, dims="obs")

        beta_adults = pm.Normal("beta_num_adults", mu=0.0, sigma=1.0)
        beta_minors = pm.Normal("beta_num_minors", mu=0.0, sigma=1.0)
        beta_adult_era = pm.Normal("beta_adult_era", mu=0.0, sigma=1.0)

        log_mu = (
            alpha
            + alpha_year[year_idx_data]
            + beta_adults * adult_data
            + beta_minors * minor_data
            + beta_adult_era * adult_data * era_data
        )
        mu = pm.math.exp(log_mu)
        alpha_nb = pm.Gamma("alpha_nb", alpha=2.0, beta=0.1)
        pm.NegativeBinomial("num_docs_obs", mu=mu, alpha=alpha_nb, observed=docs, dims="obs")

    return model


def plot_ppc(idata, df, path):
    ppc = idata.posterior_predictive["num_docs_obs"].values.reshape(-1, len(df))
    observed = df["num_docs"].to_numpy()

    rng = np.random.default_rng(0)
    draw_idx = rng.choice(ppc.shape[0], size=200, replace=False)
    bins = np.arange(0, observed.max() + 5, 3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # left: density overlay
    ax = axes[0]
    for i in draw_idx:
        ax.hist(ppc[i], bins=bins, density=True, histtype="step", color=BLUE, alpha=0.05, linewidth=1, zorder=1)
    ax.hist(
        observed, bins=bins, density=True, histtype="step", color=INK, linewidth=2, zorder=3, label="observed"
    )
    ax.plot([], [], color=BLUE, alpha=0.6, linewidth=1.5, label="posterior predictive draws")
    ax.set_xlabel("num_docs")
    ax.set_ylabel("density")
    ax.set_title("Posterior predictive check", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)

    # right: predicted mean (94% HDI) vs observed, sorted by observed
    ax = axes[1]
    pred_mean = ppc.mean(axis=0)
    pred_lo = np.percentile(ppc, 3, axis=0)
    pred_hi = np.percentile(ppc, 97, axis=0)
    order = np.argsort(observed)

    x = np.arange(len(df))
    ax.fill_between(
        x, pred_lo[order], pred_hi[order], color=BLUE, alpha=0.15, linewidth=0, label="94% predictive interval",
        zorder=1,
    )
    ax.plot(x, pred_mean[order], color=BLUE, linewidth=1.5, label="predicted mean", zorder=2)
    ax.scatter(x, observed[order], color=INK, s=6, zorder=3, label="observed")
    ax.set_xlabel("dossiers, sorted by observed num_docs")
    ax.set_ylabel("num_docs")
    ax.set_title("Predicted vs. observed", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_era_effect(idata, path):
    post = idata.posterior
    pct_per_adult_pre = (np.exp(post["beta_num_adults"]) - 1) * 100
    pct_per_adult_post = (np.exp(post["beta_num_adults"] + post["beta_adult_era"]) - 1) * 100
    pct_per_minor = (np.exp(post["beta_num_minors"]) - 1) * 100

    pre = pct_per_adult_pre.values.flatten()
    post_ = pct_per_adult_post.values.flatten()
    minor = pct_per_minor.values.flatten()

    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9.2, 4.5))

    all_vals = np.concatenate([pre, post_, minor])
    x_grid = np.linspace(all_vals.min() - 2, all_vals.max() + 2, 600)

    for values, color, label in [
        (pre, BLUE, "adult, pre-1956 (18+ threshold)"),
        (post_, ORANGE, "adult, post-1956 (16+ threshold)"),
    ]:
        kde_y = gaussian_kde(values)(x_grid)
        kde_y = kde_y / kde_y.max()  # normalize to unit peak so curves of differing spread compare cleanly
        ax.fill_between(x_grid, kde_y, color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(x_grid, kde_y, color=color, linewidth=2, label=label, zorder=3)
        mean_val = values.mean()
        ax.axvline(mean_val, color=color, linewidth=1, linestyle="--", zorder=3, ymax=0.87)
        ax.annotate(
            f"{mean_val:+.0f}%",
            xy=(mean_val, 0.9),
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    minor_y = gaussian_kde(minor)(x_grid)
    minor_y = minor_y / minor_y.max()
    ax.plot(x_grid, minor_y, color=MUTED, linewidth=1.5, linestyle=":", label="minor, both eras (reference)", zorder=1)

    ax.set_ylim(0, 1.15)
    ax.set_xlabel("% change in expected num_docs per additional person")
    ax.set_ylabel("posterior density (normalized to peak)")
    ax.set_title(
        "Per-adult effect on dossier size, before vs. after the 1956 threshold change",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    style_axes(ax)
    ax.legend(
        frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    df = load_data(DATA_PATH)
    idata_C = az.from_netcdf(f"{OUT_DIR}/idata_C_adults_minors_era.nc")

    model_C = build_model_C(df)
    with model_C:
        pm.sample_posterior_predictive(idata_C, extend_inferencedata=True, random_seed=42)

    idata_C.to_netcdf(f"{OUT_DIR}/idata_C_adults_minors_era_with_ppc.nc")

    plot_ppc(idata_C, df, f"{OUT_DIR}/posterior_predictive_check.png")
    plot_era_effect(idata_C, f"{OUT_DIR}/era_effect_pre_post_1956.png")

    print(f"Saved plots to {OUT_DIR}/posterior_predictive_check.png and {OUT_DIR}/era_effect_pre_post_1956.png")


if __name__ == "__main__":
    main()
