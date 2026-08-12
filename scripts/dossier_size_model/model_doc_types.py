"""
Hierarchical Bayesian model of individual document-type counts as a function
of unit composition (num_adults, num_minors), extending the dossier-size
model (model_dossier_size.py) to the 11 document-type columns added to
data/ages_num_docs_pages.tsv.

Data is reshaped to long format (one row per dossier x doc_type) and fit as
a single negative-binomial GLM with partial pooling across document types:

  log(mu) = alpha_type[k] + alpha_year_type[year, k]
            + beta_adult[k] * num_adults + beta_minor[k] * num_minors

alpha_type, beta_adult, beta_minor are drawn from shared population
(hyperprior) distributions across the 11 types, so sparse types (e.g.
"Testimonial medical form", 58% zero) borrow strength from the rest rather
than being estimated in isolation. alpha_year_type is a per-type year
random effect (single shared sigma_year across types) -- some document
types show a clear ramp-up around 1954-1956 (e.g. "Approval notice",
"Judicial and political background check"), so a type-specific year effect
is needed to avoid that historical trend leaking into the adult/minor
slope estimates.

delta[k] = beta_adult[k] - beta_minor[k] is the key comparison quantity:
  delta[k] >> 0  -> document type k is adult-driven
  delta[k] ~ 0   -> document type k scales with total persons, not age
  beta_adult[k] ~ beta_minor[k] ~ 0 -> document type k is roughly constant,
                                        unrelated to unit composition
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model_dossier_size import DATA_PATH, load_data

DOC_TYPE_COLS = [
    "Approval notice",
    "D.1",
    "D.2",
    "DM.1",
    "Judicial and political background check",
    "NAMA agreement",
    "Registration card",
    "Report of selection and medical officers",
    "Testimonial labour (Qualification & Employment Proof)",
    "Testimonial medical form (Medical & Health Documents)",
    "Testimonial medical letter (Medical & Health Documents)",
]

OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42


def load_long_data(path: str) -> pd.DataFrame:
    df = load_data(path)  # dossier-level: adds num_minors, era; drops 2 unknown-age rows
    long_df = df.melt(
        id_vars=["travel_year", "num_adults", "num_minors"],
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
    adults = long_df["num_adults"].to_numpy()
    minors = long_df["num_minors"].to_numpy()
    counts = long_df["doc_count"].to_numpy()

    coords = {"type": types, "year": years, "obs": long_df.index}
    with pm.Model(coords=coords) as model:
        type_idx_data = pm.Data("type_idx", type_idx, dims="obs")
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        adults_data = pm.Data("num_adults", adults, dims="obs")
        minors_data = pm.Data("num_minors", minors, dims="obs")

        # population-level (hyper) priors -- partial pooling across doc types
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=2.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2.0)
        mu_beta_adult = pm.Normal("mu_beta_adult", mu=0.0, sigma=1.0)
        sigma_beta_adult = pm.HalfNormal("sigma_beta_adult", sigma=1.0)
        mu_beta_minor = pm.Normal("mu_beta_minor", mu=0.0, sigma=1.0)
        sigma_beta_minor = pm.HalfNormal("sigma_beta_minor", sigma=1.0)
        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)

        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="type")
        alpha_type = pm.Deterministic("alpha_type", mu_alpha + z_alpha * sigma_alpha, dims="type")

        z_beta_adult = pm.Normal("z_beta_adult", 0.0, 1.0, dims="type")
        beta_adult = pm.Deterministic("beta_adult", mu_beta_adult + z_beta_adult * sigma_beta_adult, dims="type")

        z_beta_minor = pm.Normal("z_beta_minor", 0.0, 1.0, dims="type")
        beta_minor = pm.Deterministic("beta_minor", mu_beta_minor + z_beta_minor * sigma_beta_minor, dims="type")

        pm.Deterministic("delta_adult_minor", beta_adult - beta_minor, dims="type")

        z_year_type = pm.Normal("z_year_type", 0.0, 1.0, dims=("year", "type"))
        alpha_year_type = pm.Deterministic("alpha_year_type", z_year_type * sigma_year, dims=("year", "type"))

        log_mu = (
            alpha_type[type_idx_data]
            + alpha_year_type[year_idx_data, type_idx_data]
            + beta_adult[type_idx_data] * adults_data
            + beta_minor[type_idx_data] * minors_data
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
        b_adult = post["beta_adult"].sel(type=t).values.flatten()
        b_minor = post["beta_minor"].sel(type=t).values.flatten()
        delta = post["delta_adult_minor"].sel(type=t).values.flatten()
        rows.append(
            {
                "doc_type": t,
                "pct_per_adult_mean": (np.exp(b_adult) - 1).mean() * 100,
                "pct_per_adult_hdi_3": np.percentile((np.exp(b_adult) - 1) * 100, 3),
                "pct_per_adult_hdi_97": np.percentile((np.exp(b_adult) - 1) * 100, 97),
                "pct_per_minor_mean": (np.exp(b_minor) - 1).mean() * 100,
                "pct_per_minor_hdi_3": np.percentile((np.exp(b_minor) - 1) * 100, 3),
                "pct_per_minor_hdi_97": np.percentile((np.exp(b_minor) - 1) * 100, 97),
                "delta_mean": delta.mean(),
                "delta_hdi_3": np.percentile(delta, 3),
                "delta_hdi_97": np.percentile(delta, 97),
                "p_delta_gt_0": (delta > 0).mean(),
                "p_adult_effect_gt_0": (b_adult > 0).mean(),
                "p_minor_effect_gt_0": (b_minor > 0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_per_adult_mean", ascending=False).reset_index(drop=True)


def main():
    long_df = load_long_data(DATA_PATH)
    print(f"long-format rows: {len(long_df)} ({long_df['doc_type'].nunique()} types x "
          f"{long_df['travel_year'].nunique()} years x per-dossier)")

    model = build_model(long_df)
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RNG_SEED,
            target_accept=0.92,
        )

    print("\n=== sampling diagnostics ===")
    summary = az.summary(
        idata, var_names=["mu_alpha", "sigma_alpha", "mu_beta_adult", "sigma_beta_adult",
                           "mu_beta_minor", "sigma_beta_minor", "sigma_year"]
    )
    print(summary)
    max_rhat = az.rhat(idata).max()
    print("\nmax r_hat across all parameters:", max_rhat)

    table = summarize(idata)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== per-document-type effects (%% change in expected count per additional person) ===")
    print(table[["doc_type", "pct_per_adult_mean", "pct_per_minor_mean", "delta_mean", "p_delta_gt_0"]])

    idata.to_netcdf(f"{OUT_DIR}/idata_doc_types.nc")
    table.to_csv(f"{OUT_DIR}/doc_type_effects_summary.csv", index=False)
    print(f"\nSaved trace to {OUT_DIR}/idata_doc_types.nc and summary table to "
          f"{OUT_DIR}/doc_type_effects_summary.csv")


if __name__ == "__main__":
    main()
