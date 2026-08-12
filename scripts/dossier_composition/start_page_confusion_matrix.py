"""
Bayesian confusion matrix for the START-PAGE classifier (same mechanism as
confusion_matrix.py, just 2 classes instead of 11): start_page recall/
precision from the test report (0.98 / 0.95) means predicted_segment_id --
which is a purely deterministic function of predicted_start_page (a new
segment begins exactly when predicted_start_page == "yes") -- is itself
noisy. Every downstream composition analysis currently treats
predicted_segment_id as ground truth, which understates uncertainty for
exactly the same reason document-type predictions needed correcting: some
document boundaries are missed (undercounting instances, merging two real
documents into one) and some are spurious (overcounting instances, splitting
one real document into two).

Model: for each true label i in {no, yes}, the row of predicted-label counts
is Multinomial(n_i, p_i), p_i ~ Dirichlet(kappa * baseline_i) -- identical
structure to confusion_matrix.py, degenerate to a Beta-Binomial at 2 classes
(Dirichlet-multinomial subsumes it, so no separate implementation needed).

Rerun this whenever a new model's test-set evaluation becomes available:
  python3 scripts/dossier_composition/start_page_confusion_matrix.py --test-predictions <path>

On a GPU box (requires requirements-gpu.txt installed):
  python3 scripts/dossier_composition/start_page_confusion_matrix.py --sampler numpyro
"""

import argparse

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from common import DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR

RNG_SEED = 42
DIAGONAL_WEIGHT = 0.7
START_LABELS = ["no", "yes"]


def load_confusion_counts(test_predictions_path: str) -> pd.DataFrame:
    test = pd.read_csv(test_predictions_path, sep="\t")
    counts = pd.crosstab(test["start_page"], test["predicted_start_page"])
    counts = counts.reindex(index=START_LABELS, columns=START_LABELS, fill_value=0)
    return counts


def build_baseline(counts: pd.DataFrame) -> np.ndarray:
    n = len(START_LABELS)
    pred_marginal = counts.to_numpy().sum(axis=0)
    pred_marginal = pred_marginal / pred_marginal.sum()
    return DIAGONAL_WEIGHT * np.eye(n) + (1 - DIAGONAL_WEIGHT) * pred_marginal[None, :]


def build_model(counts: pd.DataFrame, baseline: np.ndarray):
    counts_arr = counts.to_numpy().astype(int)
    row_totals = counts_arr.sum(axis=1)

    coords = {"true_label": START_LABELS, "pred_label": START_LABELS}
    with pm.Model(coords=coords) as model:
        baseline_data = pm.Data("baseline", baseline, dims=("true_label", "pred_label"))
        log_kappa = pm.Normal("log_kappa", mu=np.log(4.0), sigma=0.75)
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))
        p = pm.Dirichlet("p", a=kappa * baseline_data, dims=("true_label", "pred_label"))
        pm.Multinomial(
            "counts", n=row_totals, p=p, observed=counts_arr, dims=("true_label", "pred_label")
        )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-predictions", default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--sampler", choices=["pytensor", "numpyro"], default="pytensor",
                         help="'numpyro' uses PyMC's JAX/numpyro GPU backend -- see requirements-gpu.txt.")
    args = parser.parse_args()

    counts = load_confusion_counts(args.test_predictions)
    print(f"Start-page confusion counts from {args.test_predictions} ({counts.to_numpy().sum()} test pages):")
    print(counts)

    baseline = build_baseline(counts)
    model = build_model(counts, baseline)
    sample_kwargs = dict(draws=1500, tune=2000, chains=4, random_seed=RNG_SEED, target_accept=0.97)
    if args.sampler == "numpyro":
        sample_kwargs["nuts_sampler"] = "numpyro"
        sample_kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}
    else:
        sample_kwargs["max_treedepth"] = 12
    with model:
        idata = pm.sample(**sample_kwargs)

    max_rhat = float(az.rhat(idata).max().to_array().max())
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"\nmax r_hat={max_rhat:.3f}, divergences={n_div}")

    p_mean = idata.posterior["p"].mean(dim=("chain", "draw")).to_pandas()
    print("\nposterior mean confusion matrix (rows=true, cols=predicted):")
    print(p_mean.round(3))
    print("\nposterior mean P(predicted correctly | true label):")
    print(pd.Series(np.diag(p_mean.to_numpy()), index=START_LABELS).round(3))

    idata.to_netcdf(f"{args.out_dir}/idata_start_page_confusion_matrix.nc")
    p_mean.to_csv(f"{args.out_dir}/start_page_confusion_matrix_posterior_mean.csv")
    counts.to_csv(f"{args.out_dir}/start_page_confusion_matrix_test_counts.csv")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
