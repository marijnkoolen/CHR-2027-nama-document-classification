"""
Corrected version of Models B/C from model_dossier_size.py, per domain-expert
input: the legal adult threshold was constant at 18+ throughout 1952-1965 --
it did NOT move to 16+ in 1956. What actually changed in 1956 was a separate
administrative requirement: 16-17 year-olds (never legally "adult") had to
submit their own approval paperwork, and the amount of that paperwork changed
at the 1956 mark. So the earlier "num_adults switches 18+/16+ by era" +
"adult x era interaction" story was a mechanistically wrong (though
numerically similar-looking) account of what's actually a pre-adult-specific
effect.

Three groups, built directly from num_18+ and num_16+ (both present in the
data for every row that isn't fully missing ages):
  minor_lt16       = num_persons - num_16+   (< 16, never files anything)
  preadult_16_17    = num_16+ - num_18+        (16-17, own paperwork; the
                                                 group whose requirement
                                                 changed in 1956)
  adult_18plus      = num_18+                  (18+, constant definition,
                                                 constant requirement)

Models:
  B3: log(mu) = alpha + alpha_year[year] + beta_minor*minor_lt16
                + beta_preadult*preadult_16_17 + beta_adult*adult_18plus
  C3: B3 + beta_preadult_era * preadult_16_17 * era
      (matches the domain-expert account: only the pre-adult requirement
      changed at 1956)
  D3: B3 + era interactions on ALL three groups
      (robustness check -- if C3 fits ~as well as D3, that confirms the
      effect really is isolated to pre-adults specifically, not a broader
      shift also touching adults/minors)
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

DATA_PATH = "data/ages_num_docs_pages.tsv"
OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42


def load_three_group_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["num_18+", "num_16+"]).copy()

    df["travel_year"] = df["travel_year"].astype(int)
    df["adult_18plus"] = df["num_18+"].astype(int)
    df["preadult_16_17"] = (df["num_16+"] - df["num_18+"]).astype(int)
    df["minor_lt16"] = (df["num_persons"] - df["num_16+"]).astype(int)
    df["era"] = (df["travel_year"] >= 1956).astype(int)

    assert (df["preadult_16_17"] >= 0).all()
    assert (df["minor_lt16"] >= 0).all()
    return df


def make_year_index(df: pd.DataFrame):
    years = np.sort(df["travel_year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years)}
    idx = df["travel_year"].map(year_to_idx).to_numpy()
    return idx, years


def build_model(df: pd.DataFrame, era_interaction_groups: list, name: str):
    year_idx, years = make_year_index(df)
    docs = df["num_docs"].to_numpy()
    minor = df["minor_lt16"].to_numpy()
    preadult = df["preadult_16_17"].to_numpy()
    adult = df["adult_18plus"].to_numpy()
    era = df["era"].to_numpy()

    coords = {"year": years, "obs": df.index}
    with pm.Model(coords=coords) as model:
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        era_data = pm.Data("era", era, dims="obs")
        minor_data = pm.Data("minor_lt16", minor, dims="obs")
        preadult_data = pm.Data("preadult_16_17", preadult, dims="obs")
        adult_data = pm.Data("adult_18plus", adult, dims="obs")

        alpha = pm.Normal("alpha", mu=np.log(docs.mean()), sigma=1.5)
        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
        z_year = pm.Normal("z_year", mu=0.0, sigma=1.0, dims="year")
        alpha_year = pm.Deterministic("alpha_year", z_year * sigma_year, dims="year")

        beta_minor = pm.Normal("beta_minor", mu=0.0, sigma=1.0)
        beta_preadult = pm.Normal("beta_preadult", mu=0.0, sigma=1.0)
        beta_adult = pm.Normal("beta_adult", mu=0.0, sigma=1.0)

        log_mu = (
            alpha
            + alpha_year[year_idx_data]
            + beta_minor * minor_data
            + beta_preadult * preadult_data
            + beta_adult * adult_data
        )

        group_data = {"minor": (minor_data, "beta_minor_era"), "preadult": (preadult_data, "beta_preadult_era"),
                      "adult": (adult_data, "beta_adult_era")}
        for group in era_interaction_groups:
            x_data, coef_name = group_data[group]
            beta_era = pm.Normal(coef_name, mu=0.0, sigma=1.0)
            log_mu = log_mu + beta_era * x_data * era_data

        mu = pm.math.exp(log_mu)
        alpha_nb = pm.Gamma("alpha_nb", alpha=2.0, beta=0.1)

        pm.NegativeBinomial("num_docs_obs", mu=mu, alpha=alpha_nb, observed=docs, dims="obs")

    return model


def fit(df, era_interaction_groups, name):
    model = build_model(df, era_interaction_groups, name)
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RNG_SEED,
            target_accept=0.92,
            idata_kwargs={"log_likelihood": True},
        )
    idata.attrs["model_name"] = name
    max_rhat = float(az.rhat(idata).max().to_array().max())
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"  {name}: max r_hat={max_rhat:.3f}, divergences={n_div}")
    return idata


def main():
    df = load_three_group_data(DATA_PATH)
    print(f"n dossiers: {len(df)}")
    print(df[["minor_lt16", "preadult_16_17", "adult_18plus"]].describe())

    idata_B3 = fit(df, [], "B3_three_group")
    idata_C3 = fit(df, ["preadult"], "C3_preadult_era")
    idata_D3 = fit(df, ["minor", "preadult", "adult"], "D3_all_era")

    print("\n=== Model B3 (three groups, no era interaction) ===")
    print(az.summary(idata_B3, var_names=["alpha", "beta_minor", "beta_preadult", "beta_adult",
                                           "sigma_year", "alpha_nb"], hdi_prob=0.94))

    print("\n=== Model C3 (+ preadult x era) ===")
    print(az.summary(idata_C3, var_names=["beta_minor", "beta_preadult", "beta_adult", "beta_preadult_era"],
                      hdi_prob=0.94))

    print("\n=== Model D3 (+ all groups x era) ===")
    print(az.summary(idata_D3, var_names=["beta_minor", "beta_preadult", "beta_adult",
                                           "beta_minor_era", "beta_preadult_era", "beta_adult_era"],
                      hdi_prob=0.94))

    print("\n=== LOO comparison: B3 vs C3 vs D3 ===")
    cmp = az.compare({"B3_no_interaction": idata_B3, "C3_preadult_era": idata_C3, "D3_all_era": idata_D3})
    print(cmp)
    cmp.to_csv(f"{OUT_DIR}/loo_compare_three_group_B3_C3_D3.csv")

    for name, idata in [("B3", idata_B3), ("C3", idata_C3), ("D3", idata_D3)]:
        idata.to_netcdf(f"{OUT_DIR}/idata_three_group_{name}.nc")

    print(f"\nSaved traces and LOO comparison to {OUT_DIR}/")


if __name__ == "__main__":
    main()
