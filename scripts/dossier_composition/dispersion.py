"""
Dispersion: for document types that appear more than once in a dossier, do
all instances tend to sit together (contiguous, e.g. filed as one bundle) or
scattered through the sequence (e.g. re-filed at different points, or
disturbed by archival handling)?

Statistic: within a dossier, a type with k instances among n total segments
forms some number of contiguous RUNS -- 1 if all k instances are adjacent,
up to k if none of them are adjacent to another instance of the same type.
Normalized contiguity = (runs - 1) / (k - 1): 0 = fully clustered, 1 =
maximally scattered.

A type with few instances in a long dossier will trivially tend to have
runs close to k (looks "dispersed") regardless of any real pattern, purely
from k and n -- so the raw contiguity score alone isn't enough. Compared
against the EXACT closed-form distribution of the number of runs when k
marked positions are placed uniformly at random among n slots (the
Wald-Wolfowitz runs test):
  E[runs]   = 1 + 2k(n-k)/n
  Var[runs] = 2k(n-k)(2k(n-k) - n) / (n^2 (n-1))
No simulation needed -- this is exact. Each qualifying dossier contributes
(observed_runs - E[runs]) and Var[runs]; summed across all dossiers with
k>=2 for a type and standardized, this gives a single Z-score per type:
credibly negative = more clustered than chance given each dossier's own
length and that type's own instance count; credibly positive = more
scattered than chance.

Both the descriptive contiguity score and the Z-score are computed naively
(raw predicted labels) and corrected (multiple imputation through the
confusion-matrix posterior, same as occurrence.py/co_occurrence.py).

Requires confusion_matrix.py to have been run first.

Rerun with a new model's predictions:
  python3 scripts/dossier_composition/dispersion.py --predictions <path> --test-predictions <path>
"""

import argparse

import numpy as np
import pandas as pd

from common import ALL_TYPES, DEFAULT_PREDICTIONS_PATH, DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR, OTHER
from imputation import (
    load_pages, naive_segments_from_pages, load_confusion_posterior, load_start_page_confusion_posterior,
    sample_segments_for_draw, RNG_SEED as IMPUTATION_RNG_SEED,
)
from occurrence import dossier_type_counts

N_IMPUTATIONS = 200
MIN_QUALIFYING_DOSSIERS = 15  # below this, flag the type's estimate as low-support


def dossier_type_runs(segments: pd.DataFrame, type_idx: np.ndarray, n_dossiers: int, n_types: int) -> np.ndarray:
    """Number of contiguous runs of each type within each dossier. `segments`
    must be pre-sorted by (dossier, order_in_dossier) -- load_segments()
    guarantees this. type_idx: (n_segments,) array of type indices for one
    assignment (naive, or one imputation draw)."""
    dossier_idx = segments["dossier_idx"].to_numpy()
    is_new_dossier = np.empty(len(dossier_idx), dtype=bool)
    is_new_dossier[0] = True
    is_new_dossier[1:] = dossier_idx[1:] != dossier_idx[:-1]
    is_type_change = np.empty(len(type_idx), dtype=bool)
    is_type_change[0] = True
    is_type_change[1:] = type_idx[1:] != type_idx[:-1]
    is_run_start = is_new_dossier | is_type_change

    combined = dossier_idx * n_types + type_idx
    return np.bincount(combined[is_run_start], minlength=n_dossiers * n_types).reshape(n_dossiers, n_types)


def summarize_dispersion(counts: np.ndarray, runs: np.ndarray, dossier_lengths: np.ndarray) -> dict:
    """counts, runs: (n_dossiers, n_types). dossier_lengths: (n_dossiers,).
    Returns per-type dict: z-score (pooled across qualifying dossiers) and
    mean normalized contiguity score, plus n qualifying dossiers."""
    n_types = counts.shape[1]
    k = counts.astype(float)
    n = dossier_lengths[:, None].astype(float)  # broadcast to (n_dossiers, n_types)
    qualifies = k >= 2

    # n=1 is possible under segmentation correction (a dossier can resample down to a
    # single segment, unlike the fixed predicted segmentation where min length was 2) --
    # produces 0/0 here, harmless since `qualifies` masks it out below, but suppress the
    # resulting RuntimeWarning rather than let it print once per draw.
    with np.errstate(invalid="ignore", divide="ignore"):
        e_runs = 1 + 2 * k * (n - k) / n
        var_runs = 2 * k * (n - k) * (2 * k * (n - k) - n) / (n**2 * (n - 1))
    var_runs = np.clip(var_runs, 0, None)  # guard tiny negative floating point noise

    diff = np.where(qualifies, runs - e_runs, 0.0)
    var_masked = np.where(qualifies, var_runs, 0.0)

    denom = np.where(k > 1, k - 1, 1)
    contiguity = np.where(qualifies, (runs - 1) / denom, np.nan)

    z_scores = np.empty(n_types)
    mean_contiguity = np.empty(n_types)
    n_qualifying = qualifies.sum(axis=0)
    for t in range(n_types):
        sum_diff = diff[:, t].sum()
        sum_var = var_masked[:, t].sum()
        z_scores[t] = sum_diff / np.sqrt(sum_var) if sum_var > 0 else np.nan
        vals = contiguity[qualifies[:, t], t]
        mean_contiguity[t] = vals.mean() if len(vals) else np.nan

    return {"z_score": z_scores, "mean_contiguity": mean_contiguity, "n_qualifying": n_qualifying}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--test-predictions", default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-imputations", type=int, default=N_IMPUTATIONS)
    args = parser.parse_args()

    print(f"Loading pages from {args.predictions} ...")
    pages = load_pages(args.predictions)
    dossier_order = pages["pdf_name"].unique()
    n_dossiers = len(dossier_order)
    n_types = len(ALL_TYPES)
    dossier_to_idx = {d: i for i, d in enumerate(dossier_order)}

    segments = naive_segments_from_pages(pages)
    segments["dossier_idx"] = segments["pdf_name"].map(dossier_to_idx)
    print(f"{len(segments)} predicted segments across {n_dossiers} dossiers")

    dossier_lengths = np.zeros(n_dossiers, dtype=int)
    lengths_by_idx = segments.groupby("dossier_idx")["segment_id"].size()
    dossier_lengths[lengths_by_idx.index.to_numpy()] = lengths_by_idx.to_numpy()

    print(f"\nLoading confusion-matrix posteriors from {args.out_dir} ...")
    type_p_draws, type_prior = load_confusion_posterior(
        f"{args.out_dir}/idata_confusion_matrix.nc", args.test_predictions, n_draws=args.n_imputations
    )
    start_p_draws, start_prior = load_start_page_confusion_posterior(
        f"{args.out_dir}/idata_start_page_confusion_matrix.nc", args.test_predictions, n_draws=args.n_imputations
    )

    print(f"Sampling {args.n_imputations} joint (segmentation + type) imputations ...")
    rng = np.random.default_rng(IMPUTATION_RNG_SEED)
    n_imp = type_p_draws.shape[0]
    z_draws = np.empty((n_imp, n_types))
    contiguity_draws = np.empty((n_imp, n_types))
    n_qualifying_draws = np.empty((n_imp, n_types))
    for d in range(n_imp):
        segs_d = sample_segments_for_draw(
            pages, type_p_draws[d], type_prior, start_p_draws[d], start_prior, rng
        )
        segs_d["dossier_idx"] = segs_d["pdf_name"].map(dossier_to_idx)
        # segmentation is resampled this draw, so dossier length (n segments) is too --
        # NOT the same dossier_lengths as the naive/fixed segmentation above
        lengths_d = np.zeros(n_dossiers, dtype=int)
        lengths_by_idx_d = segs_d.groupby("dossier_idx")["segment_id"].size()
        lengths_d[lengths_by_idx_d.index.to_numpy()] = lengths_by_idx_d.to_numpy()

        type_idx_d = segs_d["type_idx"].to_numpy()
        counts_d = dossier_type_counts(segs_d, type_idx_d, n_dossiers, n_types)
        runs_d = dossier_type_runs(segs_d, type_idx_d, n_dossiers, n_types)
        summ = summarize_dispersion(counts_d, runs_d, lengths_d)
        z_draws[d] = summ["z_score"]
        contiguity_draws[d] = summ["mean_contiguity"]
        n_qualifying_draws[d] = summ["n_qualifying"]

    naive_type_idx = segments["type_idx"].to_numpy()
    naive_counts = dossier_type_counts(segments, naive_type_idx, n_dossiers, n_types)
    naive_runs = dossier_type_runs(segments, naive_type_idx, n_dossiers, n_types)
    naive_summ = summarize_dispersion(naive_counts, naive_runs, dossier_lengths)

    rows = []
    for i, t in enumerate(ALL_TYPES):
        if t == OTHER:
            continue
        z_c = z_draws[:, i]
        c_c = contiguity_draws[:, i]
        rows.append({
            "doc_type": t,
            "naive_z_score": naive_summ["z_score"][i],
            "naive_contiguity": naive_summ["mean_contiguity"][i],
            "naive_n_qualifying": naive_summ["n_qualifying"][i],
            "corrected_z_score_mean": np.nanmean(z_c),
            "corrected_z_score_hdi_3": np.nanpercentile(z_c, 3),
            "corrected_z_score_hdi_97": np.nanpercentile(z_c, 97),
            "corrected_contiguity_mean": np.nanmean(c_c),
            "corrected_contiguity_hdi_3": np.nanpercentile(c_c, 3),
            "corrected_contiguity_hdi_97": np.nanpercentile(c_c, 97),
            "corrected_n_qualifying_mean": np.nanmean(n_qualifying_draws[:, i]),
        })

    table = pd.DataFrame(rows)
    # credible: the empirical 94% HDI of the z-score across imputations excludes 0 --
    # i.e. even accounting for classifier uncertainty, the direction (more clustered
    # or more scattered than a random arrangement would produce) is consistent
    table["credible"] = (table["corrected_z_score_hdi_3"] > 0) | (table["corrected_z_score_hdi_97"] < 0)
    table["low_support"] = table["corrected_n_qualifying_mean"] < MIN_QUALIFYING_DOSSIERS
    table = table.sort_values("corrected_z_score_mean")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n=== dispersion: most CLUSTERED (negative z) to most SCATTERED (positive z) ===")
    print(table[["doc_type", "corrected_z_score_mean", "corrected_z_score_hdi_3", "corrected_z_score_hdi_97",
                 "corrected_contiguity_mean", "corrected_n_qualifying_mean", "credible", "low_support"]]
          .round(2).to_string(index=False))

    table.to_csv(f"{args.out_dir}/dispersion_summary.csv", index=False)
    print(f"\nSaved dispersion_summary.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
