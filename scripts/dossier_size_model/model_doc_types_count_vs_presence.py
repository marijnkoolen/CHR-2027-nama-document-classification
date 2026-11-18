"""
For each of the 11 document types, is the number of pre-adults (16-17) in
the unit better modeled as a linear count effect (each additional pre-adult
adds more documents) or as a presence/absence effect (the document type
either gets filed for the unit or not, driven by whether there's >=1
pre-adult, not by how many)? Motivated by the mismatch the user flagged:
the domain-expert account (D.1 always required; D.2/DM.1/NAMA agreement no
longer required for pre-adults after 1956) predicts near-zero pre-adult
counts for D.2/DM.1/NAMA post-1956, which the linear-count model didn't
show -- worth checking whether a presence/absence specification tells a
different, more diagnostic story per type.

Two full-period (1952-1965) models, identical except for how preadult_16_17
enters the linear predictor:
  count model    (same as model_doc_types_three_groups.py, "full" period):
                 beta_preadult[k] * preadult_16_17
  presence model: beta_preadult[k] * I(preadult_16_17 > 0)

minor_lt16 and adult_18plus stay as raw counts in both -- only the
pre-adult predictor's functional form is being tested.

Compared via a PER-TYPE (grouped) pointwise LOO comparison: sum each
model's pointwise elpd_loo within each doc_type's own rows, rather than one
aggregate LOO across all 13070 rows, since the question is "which
specification wins for THIS type," not "which wins overall."
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model_dossier_size_three_groups import DATA_PATH, load_three_group_data
from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42


def load_long_data(path: str) -> pd.DataFrame:
    df = load_three_group_data(path)
    long_df = df.melt(
        id_vars=["travel_year", "minor_lt16", "preadult_16_17", "adult_18plus"],
        value_vars=DOC_TYPE_COLS,
        var_name="doc_type",
        value_name="doc_count",
    )
    long_df["doc_count"] = long_df["doc_count"].astype(int)
    long_df["preadult_present"] = (long_df["preadult_16_17"] > 0).astype(int)
    return long_df.reset_index(drop=True)


def build_model(long_df: pd.DataFrame, preadult_mode: str):
    types = DOC_TYPE_COLS
    type_to_idx = {t: i for i, t in enumerate(types)}
    years = np.sort(long_df["travel_year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years)}

    type_idx = long_df["doc_type"].map(type_to_idx).to_numpy()
    year_idx = long_df["travel_year"].map(year_to_idx).to_numpy()
    minor = long_df["minor_lt16"].to_numpy()
    adult = long_df["adult_18plus"].to_numpy()
    preadult = (long_df["preadult_present"] if preadult_mode == "presence" else long_df["preadult_16_17"]).to_numpy()
    counts = long_df["doc_count"].to_numpy()

    coords = {"type": types, "year": years, "obs": long_df.index}
    with pm.Model(coords=coords) as model:
        type_idx_data = pm.Data("type_idx", type_idx, dims="obs")
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        minor_data = pm.Data("minor_lt16", minor, dims="obs")
        preadult_data = pm.Data("preadult_x", preadult, dims="obs")
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


def fit(long_df, preadult_mode, name):
    model = build_model(long_df, preadult_mode)
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RNG_SEED,
            target_accept=0.92,
            idata_kwargs={"log_likelihood": True},
        )
    max_rhat = float(az.rhat(idata).max().to_array().max())
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"  {name}: max r_hat={max_rhat:.3f}, divergences={n_div}")
    return idata


def grouped_loo_comparison(idata_count, idata_presence, long_df):
    loo_count = az.loo(idata_count, pointwise=True)
    loo_presence = az.loo(idata_presence, pointwise=True)

    elpd_i_count = loo_count.loo_i.values
    elpd_i_presence = loo_presence.loo_i.values

    rows = []
    for t in DOC_TYPE_COLS:
        mask = (long_df["doc_type"] == t).to_numpy()
        elpd_count_sum = elpd_i_count[mask].sum()
        elpd_presence_sum = elpd_i_presence[mask].sum()
        n = mask.sum()
        diff = elpd_presence_sum - elpd_count_sum
        se_diff = np.std(elpd_i_presence[mask] - elpd_i_count[mask]) * np.sqrt(n)
        rows.append({
            "doc_type": t,
            "n_obs": n,
            "elpd_count": elpd_count_sum,
            "elpd_presence": elpd_presence_sum,
            "elpd_diff_presence_minus_count": diff,
            "se_diff": se_diff,
            "winner": "presence" if diff > 0 else "count",
        })
    return pd.DataFrame(rows).sort_values("elpd_diff_presence_minus_count", ascending=False)


def main():
    long_df = load_long_data(DATA_PATH)
    print(f"long rows: {len(long_df)}")

    idata_count = fit(long_df, "count", "preadult_count")
    idata_presence = fit(long_df, "presence", "preadult_presence")

    idata_count.to_netcdf(f"{OUT_DIR}/idata_doc_types_preadult_count_spec.nc")
    idata_presence.to_netcdf(f"{OUT_DIR}/idata_doc_types_preadult_presence_spec.nc")

    cmp_table = grouped_loo_comparison(idata_count, idata_presence, long_df)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== per-type: count vs. presence/absence specification (grouped pointwise LOO) ===")
    print(cmp_table.round(2).to_string(index=False))
    cmp_table.to_csv(f"{OUT_DIR}/doc_type_count_vs_presence_comparison.csv", index=False)


if __name__ == "__main__":
    main()
