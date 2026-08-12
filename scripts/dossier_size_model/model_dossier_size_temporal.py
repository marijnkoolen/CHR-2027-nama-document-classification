"""
Does dossier size (num_docs) show a structured temporal effect -- net of unit
composition (num_adults, num_minors) -- beyond unstructured year-to-year
noise? And is any such effect in the mean mirrored by a change in variance
(NB dispersion)?

Stage 1 -- mean structure. Four candidate structures for the year effect on
log(mu), all nested around a common residual "noise" term alpha_year[year]
(exchangeable, partially pooled -- same role as in model_dossier_size.py):

  M0 iid        : alpha_year[year] only -- no directional signal, the null
  M1 trend      : + a single linear slope across the whole period
  M2 step       : + a level shift at the 1956 policy change
  M3 step_trend : + step at 1956, with separate linear slopes pre/post

Compared via LOO to let the data choose "no structure" vs. "gradual decline"
vs. "discrete break" vs. "break + within-era trend".

Stage 2 -- variance structure. Take the Stage-1 winner's mean structure and
compare constant / trend / iid dispersion via LOO. Both structured (trend)
and unstructured (iid) dispersion are checked -- a smooth trend can beat
constant dispersion while still losing badly to an unconstrained per-year
estimate (as happened here: trend beat constant by 26 elpd, but iid beat
trend by a further 56.6 elpd, driven partly by one influential 1965 outlier
dossier flagged by the Pareto-k diagnostic) -- so never stop at a two-way
trend-vs-constant comparison.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model_dossier_size import DATA_PATH, load_data

OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42
MEAN_STRUCTURES = ["iid", "trend", "step", "step_trend"]


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_c"] = df["travel_year"] - 1956  # centered on the policy-change year
    return df


def make_year_index(df: pd.DataFrame):
    years = np.sort(df["travel_year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years)}
    idx = df["travel_year"].map(year_to_idx).to_numpy()
    return idx, years


def add_structured_term(structure: str, prefix: str, year_c_data, era_data, sigma_prior: float):
    """Structured (directional) addition to a log-linear predictor. Returns 0 for 'iid'."""
    if structure == "iid":
        return 0.0
    if structure == "trend":
        beta = pm.Normal(f"{prefix}_trend", mu=0.0, sigma=sigma_prior)
        return beta * year_c_data
    if structure == "step":
        beta = pm.Normal(f"{prefix}_step", mu=0.0, sigma=sigma_prior)
        return beta * era_data
    if structure == "step_trend":
        beta_step = pm.Normal(f"{prefix}_step", mu=0.0, sigma=sigma_prior)
        beta_trend_pre = pm.Normal(f"{prefix}_trend_pre", mu=0.0, sigma=sigma_prior)
        beta_trend_post = pm.Normal(f"{prefix}_trend_post", mu=0.0, sigma=sigma_prior)
        return (
            beta_step * era_data
            + beta_trend_pre * year_c_data * (1 - era_data)
            + beta_trend_post * year_c_data * era_data
        )
    raise ValueError(structure)


def build_model(df: pd.DataFrame, mean_structure: str, dispersion_structure: str = "constant"):
    year_idx, years = make_year_index(df)
    docs = df["num_docs"].to_numpy()
    adults = df["num_adults"].to_numpy()
    minors = df["num_minors"].to_numpy()
    year_c = df["year_c"].to_numpy()
    era = df["era"].to_numpy()

    coords = {"year": years, "obs": df.index}
    with pm.Model(coords=coords) as model:
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        year_c_data = pm.Data("year_c", year_c, dims="obs")
        era_data = pm.Data("era", era, dims="obs")
        adults_data = pm.Data("num_adults", adults, dims="obs")
        minors_data = pm.Data("num_minors", minors, dims="obs")

        alpha = pm.Normal("alpha", mu=np.log(docs.mean()), sigma=1.5)
        beta_adult = pm.Normal("beta_adult", mu=0.0, sigma=1.0)
        beta_minor = pm.Normal("beta_minor", mu=0.0, sigma=1.0)

        # residual (undirected) year-to-year noise, always present
        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
        z_year = pm.Normal("z_year", mu=0.0, sigma=1.0, dims="year")
        alpha_year = pm.Deterministic("alpha_year", z_year * sigma_year, dims="year")

        mean_term = add_structured_term(mean_structure, "mean", year_c_data, era_data, sigma_prior=1.0)

        log_mu = (
            alpha
            + alpha_year[year_idx_data]
            + mean_term
            + beta_adult * adults_data
            + beta_minor * minors_data
        )
        mu = pm.math.exp(log_mu)

        if dispersion_structure == "constant":
            alpha_nb = pm.Gamma("alpha_nb", alpha=2.0, beta=0.1)
        elif dispersion_structure == "iid":
            gamma_0 = pm.Normal("gamma_0_disp", mu=2.0, sigma=1.0)
            sigma_year_disp = pm.HalfNormal("sigma_year_disp", sigma=1.0)
            z_year_disp = pm.Normal("z_year_disp", mu=0.0, sigma=1.0, dims="year")
            log_alpha_year_disp = pm.Deterministic(
                "log_alpha_year_disp", gamma_0 + z_year_disp * sigma_year_disp, dims="year"
            )
            alpha_nb = pm.math.exp(log_alpha_year_disp[year_idx_data])
        else:
            gamma_0 = pm.Normal("gamma_0_disp", mu=2.0, sigma=1.0)
            disp_term = add_structured_term(dispersion_structure, "gamma", year_c_data, era_data, sigma_prior=0.5)
            alpha_nb = pm.math.exp(gamma_0 + disp_term)

        pm.NegativeBinomial("num_docs_obs", mu=mu, alpha=alpha_nb, observed=docs, dims="obs")

    return model


def fit(df, mean_structure, dispersion_structure, name):
    model = build_model(df, mean_structure, dispersion_structure)
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
    df = prep_data(load_data(DATA_PATH))
    print(f"n dossiers: {len(df)}, years: {df['travel_year'].min():.0f}-{df['travel_year'].max():.0f}")

    print("\n=== Stage 1: mean structure ===")
    idatas = {}
    for structure in MEAN_STRUCTURES:
        idatas[structure] = fit(df, structure, "constant", f"mean_{structure}")

    cmp_mean = az.compare({f"M_{k}": v for k, v in idatas.items()})
    print("\nLOO comparison (mean structure):")
    print(cmp_mean)
    cmp_mean.to_csv(f"{OUT_DIR}/loo_compare_temporal_mean.csv")

    winner_key = cmp_mean.index[0].replace("M_", "")
    print(f"\nWinning mean structure: {winner_key}")

    for structure in MEAN_STRUCTURES:
        idatas[structure].to_netcdf(f"{OUT_DIR}/idata_temporal_mean_{structure}.nc")

    print(f"\n=== Stage 2: variance structure (on top of '{winner_key}' mean) ===")
    idata_winner_constant = idatas[winner_key]
    idata_disp_trend = fit(df, winner_key, "trend", f"disp_trend_on_{winner_key}")
    idata_disp_iid = fit(df, winner_key, "iid", f"disp_iid_on_{winner_key}")
    idata_disp_trend.to_netcdf(f"{OUT_DIR}/idata_temporal_disp_trend.nc")
    idata_disp_iid.to_netcdf(f"{OUT_DIR}/idata_temporal_disp_iid.nc")

    cmp_disp = az.compare(
        {
            "constant_disp": idata_winner_constant,
            "trend_disp": idata_disp_trend,
            "iid_disp": idata_disp_iid,
        }
    )
    print("\nLOO comparison (dispersion structure: constant vs. trend vs. unconstrained per-year):")
    print(cmp_disp)
    cmp_disp.to_csv(f"{OUT_DIR}/loo_compare_temporal_dispersion.csv")

    n_bad = None
    try:
        loo_trend = az.loo(idata_disp_trend, pointwise=True)
        n_bad = int((loo_trend.pareto_k.values > 0.7).sum())
        print(f"trend_disp: n points with pareto_k > 0.7: {n_bad} / {len(loo_trend.pareto_k)}")
        if n_bad:
            bad_idx = df.index[loo_trend.pareto_k.values > 0.7]
            print(df.loc[bad_idx, ["travel_year", "num_persons", "num_adults", "num_minors", "num_docs"]])
    except Exception as e:
        print("loo pareto-k check failed:", e)

    print("\n=== Coefficient summaries ===")
    for structure in MEAN_STRUCTURES:
        var_names = [v for v in ["mean_trend", "mean_step", "mean_trend_pre", "mean_trend_post"]
                     if v in idatas[structure].posterior]
        if var_names:
            print(f"\n-- {structure} --")
            print(az.summary(idatas[structure], var_names=var_names, hdi_prob=0.94))

    disp_var_names = [v for v in ["gamma_trend", "gamma_step", "gamma_trend_pre", "gamma_trend_post"]
                       if v in idata_disp_trend.posterior]
    if disp_var_names:
        print("\n-- dispersion, trend structure --")
        print(az.summary(idata_disp_trend, var_names=disp_var_names, hdi_prob=0.94))

    with open(f"{OUT_DIR}/temporal_winner.txt", "w") as f:
        f.write(winner_key)
    print(f"\nSaved traces, LOO tables, and winner ('{winner_key}') to {OUT_DIR}/")


if __name__ == "__main__":
    main()
