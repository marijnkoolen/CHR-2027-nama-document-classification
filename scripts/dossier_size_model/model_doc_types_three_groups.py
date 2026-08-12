"""
Per-document-type extension of the corrected three-group dossier-size model
(model_dossier_size_three_groups.py): minor (<16), pre-adult (16-17), adult
(18+), each with their own doc-type-specific coefficient, partially pooled
across the 11 document types (same rationale as model_doc_types.py -- sparse
types borrow strength from the population of doc-type effects).

Fit three times, per user request:
  full : 1952-1965 (all years)
  pre  : 1952-1955 (646 dossiers -- before the pre-adult paperwork change)
  post : 1956-1965 (661 dossiers -- after)

This lets us see, per document type, whether the pre-adult coefficient
found to shrink at the dossier-size level is (a) present consistently
across types or concentrated in a few, and (b) actually shrinks within
each type from the pre- to the post-1956 fit -- a direct, non-parametric
alternative to adding a preadult x era interaction term into one pooled
model, better suited to seeing type-level heterogeneity with this much
data per type.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model_dossier_size_three_groups import DATA_PATH, load_three_group_data
from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42

PERIODS = {
    "full": (1952, 1965),
    "pre1956": (1952, 1955),
    "post1956": (1956, 1965),
}


def load_long_data(path: str, year_min: int, year_max: int) -> pd.DataFrame:
    df = load_three_group_data(path)
    df = df[(df["travel_year"] >= year_min) & (df["travel_year"] <= year_max)].copy()
    long_df = df.melt(
        id_vars=["travel_year", "minor_lt16", "preadult_16_17", "adult_18plus"],
        value_vars=DOC_TYPE_COLS,
        var_name="doc_type",
        value_name="doc_count",
    )
    long_df["doc_count"] = long_df["doc_count"].astype(int)
    return long_df


def build_model(long_df: pd.DataFrame):
    types = DOC_TYPE_COLS
    type_to_idx = {t: i for i, t in enumerate(types)}
    years = np.sort(long_df["travel_year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years)}

    type_idx = long_df["doc_type"].map(type_to_idx).to_numpy()
    year_idx = long_df["travel_year"].map(year_to_idx).to_numpy()
    minor = long_df["minor_lt16"].to_numpy()
    preadult = long_df["preadult_16_17"].to_numpy()
    adult = long_df["adult_18plus"].to_numpy()
    counts = long_df["doc_count"].to_numpy()

    coords = {"type": types, "year": years, "obs": long_df.index}
    with pm.Model(coords=coords) as model:
        type_idx_data = pm.Data("type_idx", type_idx, dims="obs")
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        minor_data = pm.Data("minor_lt16", minor, dims="obs")
        preadult_data = pm.Data("preadult_16_17", preadult, dims="obs")
        adult_data = pm.Data("adult_18plus", adult, dims="obs")

        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=2.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2.0)

        mu_beta_minor = pm.Normal("mu_beta_minor", mu=0.0, sigma=1.0)
        sigma_beta_minor = pm.HalfNormal("sigma_beta_minor", sigma=1.0)
        mu_beta_preadult = pm.Normal("mu_beta_preadult", mu=0.0, sigma=1.0)
        sigma_beta_preadult = pm.HalfNormal("sigma_beta_preadult", sigma=1.0)
        mu_beta_adult = pm.Normal("mu_beta_adult", mu=0.0, sigma=1.0)
        sigma_beta_adult = pm.HalfNormal("sigma_beta_adult", sigma=1.0)

        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)

        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="type")
        alpha_type = pm.Deterministic("alpha_type", mu_alpha + z_alpha * sigma_alpha, dims="type")

        z_beta_minor = pm.Normal("z_beta_minor", 0.0, 1.0, dims="type")
        beta_minor = pm.Deterministic("beta_minor", mu_beta_minor + z_beta_minor * sigma_beta_minor, dims="type")

        z_beta_preadult = pm.Normal("z_beta_preadult", 0.0, 1.0, dims="type")
        beta_preadult = pm.Deterministic(
            "beta_preadult", mu_beta_preadult + z_beta_preadult * sigma_beta_preadult, dims="type"
        )

        z_beta_adult = pm.Normal("z_beta_adult", 0.0, 1.0, dims="type")
        beta_adult = pm.Deterministic("beta_adult", mu_beta_adult + z_beta_adult * sigma_beta_adult, dims="type")

        z_year_type = pm.Normal("z_year_type", 0.0, 1.0, dims=("year", "type"))
        alpha_year_type = pm.Deterministic("alpha_year_type", z_year_type * sigma_year, dims=("year", "type"))

        log_mu = (
            alpha_type[type_idx_data]
            + alpha_year_type[year_idx_data, type_idx_data]
            + beta_minor[type_idx_data] * minor_data
            + beta_preadult[type_idx_data] * preadult_data
            + beta_adult[type_idx_data] * adult_data
        )
        mu = pm.math.exp(log_mu)

        alpha_nb = pm.Gamma("alpha_nb", alpha=2.0, beta=0.1, dims="type")

        pm.NegativeBinomial(
            "doc_count_obs", mu=mu, alpha=alpha_nb[type_idx_data], observed=counts, dims="obs"
        )

    return model


def summarize(idata) -> pd.DataFrame:
    rows = []
    post = idata.posterior
    for t in DOC_TYPE_COLS:
        row = {"doc_type": t}
        for group in ["minor", "preadult", "adult"]:
            draws = post[f"beta_{group}"].sel(type=t).values.flatten()
            pct = (np.exp(draws) - 1) * 100
            row[f"pct_per_{group}_mean"] = pct.mean()
            row[f"pct_per_{group}_hdi_3"] = np.percentile(pct, 3)
            row[f"pct_per_{group}_hdi_97"] = np.percentile(pct, 97)
            row[f"p_{group}_effect_gt_0"] = (draws > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pct_per_preadult_mean", ascending=False).reset_index(drop=True)


def run_period(period_label: str, year_min: int, year_max: int):
    print(f"\n{'=' * 20} period: {period_label} ({year_min}-{year_max}) {'=' * 20}")
    long_df = load_long_data(DATA_PATH, year_min, year_max)
    n_dossiers = long_df.groupby(["travel_year"]).size().sum() // len(DOC_TYPE_COLS)
    print(f"long rows: {len(long_df)} ({n_dossiers} dossiers x {len(DOC_TYPE_COLS)} types)")

    model = build_model(long_df)
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RNG_SEED,
            target_accept=0.92,
        )

    max_rhat = float(az.rhat(idata).max().to_array().max())
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"max r_hat={max_rhat:.3f}, divergences={n_div}")

    table = summarize(idata)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(table[["doc_type", "pct_per_minor_mean", "pct_per_preadult_mean", "pct_per_adult_mean"]])

    idata.to_netcdf(f"{OUT_DIR}/idata_doc_types_3group_{period_label}.nc")
    table.to_csv(f"{OUT_DIR}/doc_type_3group_effects_{period_label}.csv", index=False)
    return idata, table


def main():
    for period_label, (year_min, year_max) in PERIODS.items():
        run_period(period_label, year_min, year_max)


if __name__ == "__main__":
    main()
