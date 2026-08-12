"""
num_adults modeled as Binomial(n=num_persons, p) instead of NB(num_adults).
This is the natural generative model -- an adult count is "successes out of
num_persons trials" -- and correctly handles the under-dispersion (variance
< mean in every single year) that made the NB dispersion analysis in
model_family_size_temporal.py invalid for this outcome.

Stage 0: is plain Binomial enough, or is there real extra-binomial variation
(heterogeneity in the adult-probability p across dossiers beyond binomial
sampling noise)? Tested via LOO, Binomial vs. Beta-Binomial with constant
concentration (kappa). If Beta-Binomial doesn't win, there's no "spread"
question left to ask -- the binomial sampling variance already explains
everything, and Stage 2 is skipped.

Stage 1 (mean): same iid / trend / step / step_trend comparison as before,
now on logit(p), using whichever likelihood Stage 0 selects.

Stage 2 (dispersion, only if Beta-Binomial is needed): constant / trend / iid
structure on log(kappa), given the Stage-1 winning mean structure -- same
two-part check as before (a trend beating constant doesn't mean much if an
unconstrained per-year kappa beats the trend even more).
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


def build_model(df: pd.DataFrame, mean_structure: str, likelihood: str = "binomial",
                dispersion_structure: str = "constant"):
    year_idx, years = make_year_index(df)
    n_trials = df["num_persons"].to_numpy().astype(int)
    k_success = df["num_adults"].to_numpy().astype(int)
    year_c = df["year_c"].to_numpy()
    era = df["era"].to_numpy()

    coords = {"year": years, "obs": df.index}
    with pm.Model(coords=coords) as model:
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs")
        year_c_data = pm.Data("year_c", year_c, dims="obs")
        era_data = pm.Data("era", era, dims="obs")
        n_data = pm.Data("n_trials", n_trials, dims="obs")

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
        z_year = pm.Normal("z_year", mu=0.0, sigma=1.0, dims="year")
        alpha_year = pm.Deterministic("alpha_year", z_year * sigma_year, dims="year")

        mean_term = add_structured_term(mean_structure, "mean", year_c_data, era_data, sigma_prior=1.0)
        logit_p = alpha + alpha_year[year_idx_data] + mean_term
        p = pm.math.invlogit(logit_p)

        if likelihood == "binomial":
            pm.Binomial("num_adults_obs", n=n_data, p=p, observed=k_success, dims="obs")
        else:
            if dispersion_structure == "constant":
                kappa = pm.Gamma("kappa", alpha=2.0, beta=0.1)
            elif dispersion_structure == "iid":
                gamma_0 = pm.Normal("gamma_0_disp", mu=2.0, sigma=1.0)
                sigma_year_disp = pm.HalfNormal("sigma_year_disp", sigma=1.0)
                z_year_disp = pm.Normal("z_year_disp", mu=0.0, sigma=1.0, dims="year")
                log_kappa_year = pm.Deterministic(
                    "log_kappa_year", gamma_0 + z_year_disp * sigma_year_disp, dims="year"
                )
                kappa = pm.math.exp(log_kappa_year[year_idx_data])
            else:
                gamma_0 = pm.Normal("gamma_0_disp", mu=2.0, sigma=1.0)
                disp_term = add_structured_term(dispersion_structure, "gamma", year_c_data, era_data, sigma_prior=0.5)
                kappa = pm.math.exp(gamma_0 + disp_term)

            alpha_bb = p * kappa
            beta_bb = (1 - p) * kappa
            pm.BetaBinomial("num_adults_obs", alpha=alpha_bb, beta=beta_bb, n=n_data, observed=k_success, dims="obs")

    return model


def fit(df, mean_structure, likelihood, dispersion_structure, name):
    model = build_model(df, mean_structure, likelihood, dispersion_structure)
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
    print(f"n dossiers: {len(df)}")

    print("\n=== Stage 0: is there extra-binomial dispersion? ===")
    idata_binom = fit(df, "iid", "binomial", "constant", "adults_binomial_iid")
    idata_betabinom = fit(df, "iid", "beta_binomial", "constant", "adults_betabinom_iid_constant")

    cmp0 = az.compare({"binomial": idata_binom, "beta_binomial": idata_betabinom})
    print(cmp0)
    cmp0.to_csv(f"{OUT_DIR}/loo_compare_adults_binomial_vs_betabinomial.csv")

    use_beta_binomial = cmp0.index[0] == "beta_binomial" and (
        cmp0.loc["beta_binomial", "elpd_loo"] - cmp0.loc["binomial", "elpd_loo"] > cmp0.loc["binomial", "dse"]
    ) if "dse" in cmp0.columns else cmp0.index[0] == "beta_binomial"
    likelihood = "beta_binomial" if use_beta_binomial else "binomial"
    print(f"\nLikelihood selected for the rest of the analysis: {likelihood}")

    print(f"\n=== Stage 1: mean structure (likelihood={likelihood}) ===")
    idatas = {"iid": idata_binom if likelihood == "binomial" else idata_betabinom}
    for structure in MEAN_STRUCTURES:
        if structure == "iid":
            continue
        idatas[structure] = fit(df, structure, likelihood, "constant", f"adults_mean_{structure}_{likelihood}")

    cmp_mean = az.compare({f"M_{k}": v for k, v in idatas.items()})
    print(cmp_mean)
    cmp_mean.to_csv(f"{OUT_DIR}/loo_compare_adults_binom_temporal_mean.csv")
    winner_key = cmp_mean.index[0].replace("M_", "")
    print(f"Winning mean structure: {winner_key}")

    for structure in MEAN_STRUCTURES:
        idatas[structure].to_netcdf(f"{OUT_DIR}/idata_adults_binom_temporal_mean_{structure}.nc")

    print("\n-- coefficient summaries (mean, logit scale) --")
    for structure in MEAN_STRUCTURES:
        var_names = [v for v in ["mean_trend", "mean_step", "mean_trend_pre", "mean_trend_post"]
                     if v in idatas[structure].posterior]
        if var_names:
            print(f"mean structure = {structure}:")
            print(az.summary(idatas[structure], var_names=var_names, hdi_prob=0.94))

    if likelihood != "beta_binomial":
        print("\nNo evidence of extra-binomial dispersion -- plain Binomial fits. "
              "There is no separate 'spread' question beyond the mean structure above: "
              "binomial sampling variance (n * p * (1-p)) already fully explains num_adults' spread.")
        with open(f"{OUT_DIR}/adults_binomial_conclusion.txt", "w") as f:
            f.write(f"likelihood=binomial\nmean_winner={winner_key}\n"
                    f"conclusion=no_extra_binomial_dispersion\n")
        return

    print(f"\n=== Stage 2: dispersion (kappa) structure, given '{winner_key}' mean ===")
    idata_disp_trend = fit(df, winner_key, "beta_binomial", "trend", f"adults_disp_trend_on_{winner_key}")
    idata_disp_iid = fit(df, winner_key, "beta_binomial", "iid", f"adults_disp_iid_on_{winner_key}")
    idata_disp_trend.to_netcdf(f"{OUT_DIR}/idata_adults_binom_temporal_disp_trend.nc")
    idata_disp_iid.to_netcdf(f"{OUT_DIR}/idata_adults_binom_temporal_disp_iid.nc")

    cmp_disp = az.compare({
        "constant_disp": idatas[winner_key],
        "trend_disp": idata_disp_trend,
        "iid_disp": idata_disp_iid,
    })
    print(cmp_disp)
    cmp_disp.to_csv(f"{OUT_DIR}/loo_compare_adults_binom_temporal_dispersion.csv")

    disp_var_names = [v for v in ["gamma_trend", "gamma_step", "gamma_trend_pre", "gamma_trend_post"]
                       if v in idata_disp_trend.posterior]
    if disp_var_names:
        print("dispersion (kappa) trend:")
        print(az.summary(idata_disp_trend, var_names=disp_var_names, hdi_prob=0.94))

    with open(f"{OUT_DIR}/adults_binomial_conclusion.txt", "w") as f:
        f.write(f"likelihood=beta_binomial\nmean_winner={winner_key}\n"
                f"dispersion_winner={cmp_disp.index[0]}\n")


if __name__ == "__main__":
    main()
