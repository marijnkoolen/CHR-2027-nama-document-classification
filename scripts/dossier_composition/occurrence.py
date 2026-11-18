"""
Occurrence analysis: which document types are common/rare, and do they tend
to appear as a single instance or multiple instances per dossier -- computed
both naively (raw predicted labels, as if the classifier were perfect) and
corrected. "Corrected" propagates BOTH sources of classifier uncertainty:
document-type confusion (confusion_matrix.py, uneven per-type recall/
precision) AND segmentation/start_page confusion (start_page_confusion_
matrix.py, missed or spurious document boundaries) -- each imputation draw
resamples start_page per page, rebuilds segment boundaries from that, then
resamples type on the newly-formed segments (see imputation.py's
sample_segments_for_draw). The gap between naive and corrected shows how
much the classifier's unevenness would have distorted a naive reading.

Requires confusion_matrix.py AND start_page_confusion_matrix.py to have been
run first.

Rerun with a new model's predictions:
  python3 scripts/dossier_composition/occurrence.py --predictions <path> --test-predictions <path>
(also rerun confusion_matrix.py / start_page_confusion_matrix.py first with the new model's --test-predictions)
"""

import argparse

import numpy as np
import pandas as pd

from common import ALL_TYPES, DEFAULT_PREDICTIONS_PATH, DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR, OTHER
from imputation import (
    load_pages, naive_segments_from_pages, load_confusion_posterior, load_start_page_confusion_posterior,
    sample_dossier_type_counts_for_draw, RNG_SEED,
)

N_IMPUTATIONS = 200


def dossier_type_counts(segments: pd.DataFrame, type_idx: np.ndarray, n_dossiers: int, n_types: int) -> np.ndarray:
    """type_idx: (n_segments,) array of type indices (naive) for one assignment."""
    dossier_idx = segments["dossier_idx"].to_numpy()
    combined = dossier_idx * n_types + type_idx
    return np.bincount(combined, minlength=n_dossiers * n_types).reshape(n_dossiers, n_types)


def summarize_counts_matrix(counts: np.ndarray) -> dict:
    """counts: (n_dossiers, n_types). Returns per-type dict of scalar summaries
    for a single assignment (naive, or one imputation draw)."""
    present = counts > 0
    return {
        "prevalence": present.mean(axis=0),  # fraction of dossiers with >=1 instance
        "mean_count": counts.mean(axis=0),  # unconditional mean count per dossier
        "mean_count_given_present": np.divide(
            counts.sum(axis=0), present.sum(axis=0), out=np.zeros(counts.shape[1]), where=present.sum(axis=0) > 0
        ),
        "frac_single_given_present": np.divide(
            (counts == 1).sum(axis=0), present.sum(axis=0),
            out=np.zeros(counts.shape[1]), where=present.sum(axis=0) > 0
        ),
    }


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

    # naive: raw predicted type AND raw predicted segmentation, no correction
    segments = naive_segments_from_pages(pages)
    dossier_to_idx = {d: i for i, d in enumerate(dossier_order)}
    segments["dossier_idx"] = segments["pdf_name"].map(dossier_to_idx)
    print(f"{len(segments)} predicted segments across {n_dossiers} dossiers")

    print(f"\nLoading confusion-matrix posteriors from {args.out_dir} ...")
    type_p_draws, type_prior = load_confusion_posterior(
        f"{args.out_dir}/idata_confusion_matrix.nc", args.test_predictions, n_draws=args.n_imputations
    )
    start_p_draws, start_prior = load_start_page_confusion_posterior(
        f"{args.out_dir}/idata_start_page_confusion_matrix.nc", args.test_predictions, n_draws=args.n_imputations
    )
    print(f"prior P(true type) from test set:\n{pd.Series(type_prior, index=ALL_TYPES).round(3)}")

    naive_counts = dossier_type_counts(segments, segments["type_idx"].to_numpy(), n_dossiers, n_types)
    naive_summary = summarize_counts_matrix(naive_counts)

    # corrected: per-draw joint (segmentation + type) resampling, then mean + 94% HDI across draws
    print(f"\nSampling {args.n_imputations} joint (segmentation + type) imputations ...")
    rng = np.random.default_rng(RNG_SEED)
    n_imp = type_p_draws.shape[0]
    corrected_stats = {k: np.empty((n_imp, n_types)) for k in naive_summary}
    for d in range(n_imp):
        counts_d = sample_dossier_type_counts_for_draw(
            pages, type_p_draws[d], type_prior, start_p_draws[d], start_prior, rng, dossier_order
        )
        summ_d = summarize_counts_matrix(counts_d)
        for k, v in summ_d.items():
            corrected_stats[k][d] = v

    rows = []
    for i, t in enumerate(ALL_TYPES):
        row = {"doc_type": t, "naive_prevalence": naive_summary["prevalence"][i],
               "naive_mean_count": naive_summary["mean_count"][i],
               "naive_mean_count_given_present": naive_summary["mean_count_given_present"][i],
               "naive_frac_single_given_present": naive_summary["frac_single_given_present"][i]}
        for k in naive_summary:
            draws_k = corrected_stats[k][:, i]
            row[f"corrected_{k}_mean"] = draws_k.mean()
            row[f"corrected_{k}_hdi_3"] = np.percentile(draws_k, 3)
            row[f"corrected_{k}_hdi_97"] = np.percentile(draws_k, 97)
        rows.append(row)

    table = pd.DataFrame(rows)
    tracked = table[table["doc_type"] != OTHER].sort_values("corrected_prevalence_mean", ascending=False)
    other_row = table[table["doc_type"] == OTHER]
    table = pd.concat([tracked, other_row], ignore_index=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n=== occurrence: naive vs. corrected (94% HDI over imputations) ===")
    print(table[["doc_type", "naive_prevalence", "corrected_prevalence_mean", "corrected_prevalence_hdi_3",
                 "corrected_prevalence_hdi_97", "naive_mean_count_given_present",
                 "corrected_mean_count_given_present_mean"]].round(3).to_string(index=False))

    table.to_csv(f"{args.out_dir}/occurrence_summary.csv", index=False)
    np.save(f"{args.out_dir}/occurrence_naive_counts.npy", naive_counts)
    segments.to_parquet(f"{args.out_dir}/segments.parquet", index=False)
    print(f"\nSaved occurrence_summary.csv, segments.parquet, and count arrays to {args.out_dir}/")


if __name__ == "__main__":
    main()
