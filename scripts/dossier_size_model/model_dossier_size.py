"""
Bayesian model of dossier size (num_docs) as a function of unit composition
(num_persons vs. num_adults / num_minors), originally built on the premise
that the migration office's minimum adult age changed from 18 to 16 in 1956.

PREMISE SUPERSEDED, MODELS STILL LIVE: domain-expert follow-up established
that premise was wrong -- the legal adult threshold was constant at 18+
throughout; what changed in 1956 was a separate paperwork requirement for
16-17 year-olds ("pre-adults", never legally adult). Model C's original
"adult effect shrinks after 1956" finding (fit on the era-switching data)
was an artifact of that mis-specification, not a real effect -- see
model_dossier_size_three_groups.py (Models B3/C3/D3) for the three-group
(minor/pre-adult/adult) account that actually explains the 1956
discontinuity. That said, A/B/C themselves (refit on the corrected data
below) remain useful as the "how many age groups, and does era matter"
baseline end of the model-comparison grid -- not superseded as
*comparison points*, only the original era-switching-data C result is.

`num_adults`/`num_minors` here originally encoded the (since-corrected)
assumption of an era-dependent adult definition (18+ before 1956, 16+ from
1956) -- ages_num_docs_pages.tsv has since been corrected to a constant
18+ threshold throughout, matching the domain expert's account, so
`num_adults` here now means the same thing as `num_18+` in
model_dossier_size_three_groups.py. B and C predate that fix and were
refit on the corrected data; their specifications (which covariates, era
interaction or not) are unchanged.

Models fit (all negative-binomial GLMs with a log link, hierarchical
partial pooling of a year-level intercept):

  A:     num_docs ~ num_persons
  A_era: num_docs ~ num_persons + persons_x_era          (era = pre/post 1956)
  B:     num_docs ~ num_adults + num_minors               (num_minors = num_persons - num_adults, constant 18+ threshold)
  C:     num_docs ~ num_adults + num_minors + adult_x_era

Together with model_dossier_size_three_groups.py's B3/C3(/D3), these six
models form a 3x2 grid crossed on (1) how many age groups the persons are
split into (none / adult+minor / adult+pre-adult+minor) and (2) whether
there's an era interaction on the youngest adult-adjacent group -- see
compare_all_size_models.py for the combined comparison across all of them.

Model A vs A_era (via LOO) answers: is there any era effect at all on raw
group size, before splitting into adults/minors?
Model A vs B (via LOO) answers: does splitting persons into adults/minors
predict dossier size better than raw group size alone?
Model B vs C (via LOO) answers: did the "cost" (in documents) of an adult
change after the 1956 policy shift?
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

DATA_PATH = "data/ages_num_docs_pages.tsv"
OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["num_adults"]).copy()

    df["travel_year"] = df["travel_year"].astype(int)
    df["num_adults"] = df["num_adults"].astype(int)
    df["num_minors"] = df["num_persons"] - df["num_adults"]
    df["era"] = (df["travel_year"] >= 1956).astype(int)  # 0 = pre-1956 (18+), 1 = post (16+)

    assert (df["num_minors"] >= 0).all(), "num_minors should never be negative"
    return df


def make_year_index(df: pd.DataFrame):
    years = np.sort(df["travel_year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years)}
    idx = df["travel_year"].map(year_to_idx).to_numpy()
    return idx, years


def fit_model(df: pd.DataFrame, predictors: dict, era_interaction_predictor: str | None, name: str,
              era_coef_name: str | None = None):
    """era_interaction_predictor: key into `predictors` to interact with era
    (None for no era interaction). era_coef_name: override for the
    interaction coefficient's name in the trace -- defaults to
    beta_{era_interaction_predictor}_era, but Model C explicitly passes
    "beta_adult_era" (its original, pre-generalization name) since
    build_report.py and plot_dossier_size_results.py both read that exact
    name from Model C's trace; only override when you need to match an
    existing downstream reference like that."""
    year_idx, years = make_year_index(df)
    n_years = len(years)
    docs = df["num_docs"].to_numpy()

    coords = {"year": years, "obs": df.index}
    with pm.Model(coords=coords) as model:
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")

        # Global intercept + non-centered hierarchical year offsets (partial pooling)
        alpha = pm.Normal("alpha", mu=np.log(docs.mean()), sigma=1.5)
        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
        z_year = pm.Normal("z_year", mu=0.0, sigma=1.0, dims="year")
        alpha_year = pm.Deterministic("alpha_year", z_year * sigma_year, dims="year")

        log_mu = alpha + alpha_year[year_idx_data]

        for pname, values in predictors.items():
            x = pm.Data(pname, values, dims="obs")
            beta = pm.Normal(f"beta_{pname}", mu=0.0, sigma=1.0)
            log_mu = log_mu + beta * x

        if era_interaction_predictor is not None:
            era_data = pm.Data("era", df["era"].to_numpy(), dims="obs")
            x_era = predictors[era_interaction_predictor]
            coef_name = era_coef_name or f"beta_{era_interaction_predictor}_era"
            beta_era = pm.Normal(coef_name, mu=0.0, sigma=1.0)
            log_mu = log_mu + beta_era * x_era * era_data

        mu = pm.math.exp(log_mu)
        alpha_nb = pm.Gamma("alpha_nb", alpha=2.0, beta=0.1)  # NB dispersion

        pm.NegativeBinomial("num_docs_obs", mu=mu, alpha=alpha_nb, observed=docs, dims="obs")

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RNG_SEED,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
        )

    idata.attrs["model_name"] = name
    return model, idata


def main():
    df = load_data(DATA_PATH)
    print(f"n dossiers used: {len(df)} (dropped {1309 - len(df)} with unknown ages)")
    print(df[["travel_year", "num_persons", "num_adults", "num_minors", "num_docs"]].describe())

    persons = df["num_persons"].to_numpy()
    adults = df["num_adults"].to_numpy()
    minors = df["num_minors"].to_numpy()

    _, idata_A = fit_model(df, {"num_persons": persons}, era_interaction_predictor=None, name="A_persons")
    _, idata_A_era = fit_model(
        df, {"num_persons": persons}, era_interaction_predictor="num_persons", name="A_era_persons_era"
    )
    _, idata_B = fit_model(
        df, {"num_adults": adults, "num_minors": minors}, era_interaction_predictor=None, name="B_adults_minors"
    )
    _, idata_C = fit_model(
        df, {"num_adults": adults, "num_minors": minors}, era_interaction_predictor="num_adults",
        name="C_adults_minors_era", era_coef_name="beta_adult_era",
    )

    print("\n=== Model A (num_persons) summary ===")
    print(az.summary(idata_A, var_names=["alpha", "beta_num_persons", "sigma_year", "alpha_nb"]))

    print("\n=== Model A_era (num_persons + persons x era) summary ===")
    print(az.summary(
        idata_A_era, var_names=["alpha", "beta_num_persons", "beta_num_persons_era", "sigma_year", "alpha_nb"]
    ))

    print("\n=== Model B (num_adults + num_minors) summary ===")
    print(az.summary(idata_B, var_names=["alpha", "beta_num_adults", "beta_num_minors", "sigma_year", "alpha_nb"]))

    print("\n=== Model C (+ adult x era interaction) summary ===")
    print(
        az.summary(
            idata_C,
            var_names=["alpha", "beta_num_adults", "beta_num_minors", "beta_adult_era", "sigma_year", "alpha_nb"],
        )
    )

    print("\n=== LOO comparison: A vs A_era (does an era effect on raw group size matter at all) ===")
    cmp_a_aera = az.compare({"A_persons": idata_A, "A_era_persons_era": idata_A_era})
    print(cmp_a_aera)

    print("\n=== LOO comparison: A vs B (persons vs adults+minors) ===")
    cmp_ab = az.compare({"A_persons": idata_A, "B_adults_minors": idata_B})
    print(cmp_ab)

    print("\n=== LOO comparison: B vs C (era interaction on adult effect) ===")
    cmp_bc = az.compare({"B_adults_minors": idata_B, "C_adults_minors_era": idata_C})
    print(cmp_bc)

    for name, idata in [
        ("A_persons", idata_A), ("A_era_persons_era", idata_A_era),
        ("B_adults_minors", idata_B), ("C_adults_minors_era", idata_C),
    ]:
        idata.to_netcdf(f"{OUT_DIR}/idata_{name}.nc")

    cmp_a_aera.to_csv(f"{OUT_DIR}/loo_compare_A_vs_A_era.csv")
    cmp_ab.to_csv(f"{OUT_DIR}/loo_compare_A_vs_B.csv")
    cmp_bc.to_csv(f"{OUT_DIR}/loo_compare_B_vs_C.csv")


if __name__ == "__main__":
    main()
