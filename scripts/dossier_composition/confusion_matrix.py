"""
Bayesian confusion matrix for the document-type classifier, estimated from
the held-out test set (true label + predicted label per page). This is the
foundation the whole composition analysis (occurrence, co-occurrence,
dispersion, order) is built on: classifier reliability is very uneven across
types (e.g. D.1 ~100%/100% precision/recall vs. D.2 ~58% precision), so any
composition statistic computed on raw predicted labels would be
systematically distorted for the weak types.

Model: for each true type i, the row of predicted-type counts is
Multinomial(n_i, p_i), with p_i ~ Dirichlet(kappa * baseline_i) --
hierarchical partial pooling via a single shared concentration kappa across
all 12 rows (11 tracked types + Other), so types with thin test support
(e.g. Approval notice, n=7) borrow strength rather than producing a noisy
empirical confusion row from a handful of examples.

baseline_i is a fixed (not learned) weakly-informative prior mean: mostly
mass on the diagonal (predicted = true), with the remainder spread according
to the test set's overall predicted-type marginal frequency -- a reasonable
default for "probably correct, and if not, confused with something common."

Rerun this whenever a new model's test-set evaluation becomes available:
  python3 scripts/dossier_composition/confusion_matrix.py --test-predictions <path>

On a GPU box (requires requirements-gpu.txt installed -- see that file for
the exact pinned versions and why they matter):
  python3 scripts/dossier_composition/confusion_matrix.py --sampler numpyro
"""

import argparse

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from common import ALL_TYPES, DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR, map_type

RNG_SEED = 42
DIAGONAL_WEIGHT = 0.7  # baseline prior mass on "predicted = true"


def load_confusion_counts(test_predictions_path: str) -> pd.DataFrame:
    test = pd.read_csv(test_predictions_path, sep="\t")
    test["true_mapped"] = test["document_type"].apply(map_type)
    test["pred_mapped"] = test["predicted_document_type"].apply(map_type)
    counts = pd.crosstab(test["true_mapped"], test["pred_mapped"])
    counts = counts.reindex(index=ALL_TYPES, columns=ALL_TYPES, fill_value=0)
    return counts


def build_baseline(counts: pd.DataFrame) -> np.ndarray:
    n = len(ALL_TYPES)
    pred_marginal = counts.to_numpy().sum(axis=0)
    pred_marginal = pred_marginal / pred_marginal.sum()
    baseline = DIAGONAL_WEIGHT * np.eye(n) + (1 - DIAGONAL_WEIGHT) * pred_marginal[None, :]
    return baseline


def build_model(counts: pd.DataFrame, baseline: np.ndarray):
    counts_arr = counts.to_numpy().astype(int)
    row_totals = counts_arr.sum(axis=1)

    coords = {"true_type": ALL_TYPES, "pred_type": ALL_TYPES}
    with pm.Model(coords=coords) as model:
        baseline_data = pm.Data("baseline", baseline, dims=("true_type", "pred_type"))
        # log-scale parameterization: kappa ~ Gamma(4,1) sampled directly showed a mild
        # r_hat excess (1.015) even after heavy tuning -- concentration/scale parameters
        # like this usually sample better on the log scale.
        log_kappa = pm.Normal("log_kappa", mu=np.log(4.0), sigma=0.75)
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))
        p = pm.Dirichlet("p", a=kappa * baseline_data, dims=("true_type", "pred_type"))
        pm.Multinomial(
            "counts", n=row_totals, p=p, observed=counts_arr, dims=("true_type", "pred_type")
        )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-predictions", default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--sampler", choices=["pytensor", "numpyro"], default="pytensor",
                         help="'numpyro' uses PyMC's JAX/numpyro backend, which can use a GPU if jax[cudaXX] "
                              "is installed and visible (see requirements-gpu.txt). 'pytensor' (default) is "
                              "PyMC's normal CPU sampler.")
    args = parser.parse_args()

    counts = load_confusion_counts(args.test_predictions)
    print(f"Confusion counts from {args.test_predictions} ({counts.to_numpy().sum()} test pages):")
    pd.set_option("display.width", 250)
    print(counts)

    row_totals = counts.to_numpy().sum(axis=1)
    print("\nrow (true-type) support:")
    print(pd.Series(row_totals, index=ALL_TYPES))

    baseline = build_baseline(counts)
    model = build_model(counts, baseline)
    sample_kwargs = dict(draws=1500, tune=2000, chains=4, random_seed=RNG_SEED, target_accept=0.97)
    if args.sampler == "numpyro":
        # chain_method (and any other JAX-side NUTS options) must be nested inside
        # nuts_sampler_kwargs, not passed as a top-level pm.sample() kwarg -- NOT
        # tested against a real GPU (none available here); verify this actually
        # runs on the A10 box before trusting it for a real job, and report back
        # if the API doesn't match this PyMC version exactly.
        sample_kwargs["nuts_sampler"] = "numpyro"
        sample_kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}  # 1 GPU, batched chains
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

    diag = np.diag(p_mean.to_numpy())
    print("\nposterior mean P(predicted correctly | true type):")
    print(pd.Series(diag, index=ALL_TYPES).round(3))

    idata.to_netcdf(f"{args.out_dir}/idata_confusion_matrix.nc")
    p_mean.to_csv(f"{args.out_dir}/confusion_matrix_posterior_mean.csv")
    counts.to_csv(f"{args.out_dir}/confusion_matrix_test_counts.csv")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
