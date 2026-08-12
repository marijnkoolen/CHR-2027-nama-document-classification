"""
Same temporal-structure question as model_dossier_size_temporal.py, applied
to family/unit size itself (num_persons, num_adults) instead of dossier size
(num_docs). No adult/minor covariates here -- composition is the outcome,
not something to control for.

For each outcome:
  Stage 1 (mean): compare iid / trend / step / step_trend year structures
                  (constant dispersion) via LOO.
  Stage 2 (dispersion): given the Stage-1 winning mean structure, compare
                  constant / trend / iid dispersion via LOO. Both structured
                  and unstructured (iid) are checked -- a smooth trend can
                  win vs. constant while still losing badly to an
                  unconstrained per-year estimate (as happened for num_docs),
                  so never stop at the two-way trend-vs-constant comparison.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model_dossier_size import DATA_PATH, load_data
from model_dossier_size_temporal import add_structured_term, make_year_index, prep_data

OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42
MEAN_STRUCTURES = ["iid", "trend", "step", "step_trend"]
OUTCOMES = ["num_persons", "num_adults"]


def build_model(df: pd.DataFrame, outcome_col: str, mean_structure: str, dispersion_structure: str = "constant"):
    year_idx, years = make_year_index(df)
    y = df[outcome_col].to_numpy().astype(int)
    year_c = df["year_c"].to_numpy()
    era = df["era"].to_numpy()

    coords = {"year": years, "obs": df.index}
    with pm.Model(coords=coords) as model:
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        year_c_data = pm.Data("year_c", year_c, dims="obs")
        era_data = pm.Data("era", era, dims="obs")

        alpha = pm.Normal("alpha", mu=np.log(y.mean()), sigma=1.5)

        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
        z_year = pm.Normal("z_year", mu=0.0, sigma=1.0, dims="year")
        alpha_year = pm.Deterministic("alpha_year", z_year * sigma_year, dims="year")

        mean_term = add_structured_term(mean_structure, "mean", year_c_data, era_data, sigma_prior=1.0)
        log_mu = alpha + alpha_year[year_idx_data] + mean_term
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

        pm.NegativeBinomial(f"{outcome_col}_obs", mu=mu, alpha=alpha_nb, observed=y, dims="obs")

    return model


def fit(df, outcome_col, mean_structure, dispersion_structure, name):
    model = build_model(df, outcome_col, mean_structure, dispersion_structure)
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


def run_for_outcome(df, outcome_col):
    print(f"\n{'=' * 20} outcome: {outcome_col} {'=' * 20}")
    print(f"mean={df[outcome_col].mean():.2f}, sd={df[outcome_col].std():.2f}, "
          f"min={df[outcome_col].min()}, max={df[outcome_col].max()}")

    print("\n-- Stage 1: mean structure --")
    idatas = {}
    for structure in MEAN_STRUCTURES:
        idatas[structure] = fit(df, outcome_col, structure, "constant", f"{outcome_col}_mean_{structure}")

    cmp_mean = az.compare({f"M_{k}": v for k, v in idatas.items()})
    print(cmp_mean)
    cmp_mean.to_csv(f"{OUT_DIR}/loo_compare_{outcome_col}_temporal_mean.csv")

    winner_key = cmp_mean.index[0].replace("M_", "")
    print(f"Winning mean structure: {winner_key}")

    for structure in MEAN_STRUCTURES:
        idatas[structure].to_netcdf(f"{OUT_DIR}/idata_{outcome_col}_temporal_mean_{structure}.nc")

    print(f"\n-- Stage 2: dispersion structure (given '{winner_key}' mean) --")
    idata_trend_disp = fit(df, outcome_col, winner_key, "trend", f"{outcome_col}_disp_trend_on_{winner_key}")
    idata_iid_disp = fit(df, outcome_col, winner_key, "iid", f"{outcome_col}_disp_iid_on_{winner_key}")
    idata_trend_disp.to_netcdf(f"{OUT_DIR}/idata_{outcome_col}_temporal_disp_trend.nc")
    idata_iid_disp.to_netcdf(f"{OUT_DIR}/idata_{outcome_col}_temporal_disp_iid.nc")

    cmp_disp = az.compare({
        "constant_disp": idatas[winner_key],
        "trend_disp": idata_trend_disp,
        "iid_disp": idata_iid_disp,
    })
    print(cmp_disp)
    cmp_disp.to_csv(f"{OUT_DIR}/loo_compare_{outcome_col}_temporal_dispersion.csv")

    print("\n-- coefficient summaries --")
    for structure in MEAN_STRUCTURES:
        var_names = [v for v in ["mean_trend", "mean_step", "mean_trend_pre", "mean_trend_post"]
                     if v in idatas[structure].posterior]
        if var_names:
            print(f"mean structure = {structure}:")
            print(az.summary(idatas[structure], var_names=var_names, hdi_prob=0.94))

    disp_var_names = [v for v in ["gamma_trend", "gamma_step", "gamma_trend_pre", "gamma_trend_post"]
                       if v in idata_trend_disp.posterior]
    if disp_var_names:
        print("dispersion trend:")
        print(az.summary(idata_trend_disp, var_names=disp_var_names, hdi_prob=0.94))

    n_bad = None
    try:
        loo_trend = az.loo(idata_trend_disp, pointwise=True)
        n_bad = int((loo_trend.pareto_k.values > 0.7).sum())
        print(f"trend_disp: n points with pareto_k > 0.7: {n_bad} / {len(loo_trend.pareto_k)}")
    except Exception as e:
        print("loo pareto-k check failed:", e)

    with open(f"{OUT_DIR}/temporal_winner_{outcome_col}.txt", "w") as f:
        f.write(f"mean={winner_key}\ndispersion_winner={cmp_disp.index[0]}\n")

    return {
        "outcome": outcome_col,
        "mean_winner": winner_key,
        "cmp_mean": cmp_mean,
        "cmp_disp": cmp_disp,
    }


def main():
    df = prep_data(load_data(DATA_PATH))
    print(f"n dossiers: {len(df)}, years: {df['travel_year'].min():.0f}-{df['travel_year'].max():.0f}")

    results = [run_for_outcome(df, outcome_col) for outcome_col in OUTCOMES]

    print("\n\n=== overall summary ===")
    for r in results:
        print(f"{r['outcome']}: mean structure winner = {r['mean_winner']}, "
              f"dispersion structure winner = {r['cmp_disp'].index[0]}")


if __name__ == "__main__":
    main()
