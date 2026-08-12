"""
Pairwise co-occurrence: which document types tend to appear together in the
same dossier more/less than chance, computed both naively (raw predicted
labels) and corrected (multiple imputation through the confusion-matrix
posterior, same as occurrence.py).

For each type pair (A, B) and each imputed presence/absence assignment, the
2x2 table (both present / A-only / B-only / neither) has a closed-form
Dirichlet(n11+1, n10+1, n01+1, n00+1) posterior -- no MCMC needed, so no
convergence risk. Pooling draws from that posterior across all imputations
gives one combined posterior per pair that reflects BOTH ordinary
dossier-sampling uncertainty (from the closed-form Dirichlet) AND classifier
uncertainty (from varying which imputation draw the table came from). This
is much cheaper than refitting a joint multivariate model per imputation,
while still being fully Bayesian.

Reported measure: log odds ratio log((n11*n00)/(n10*n01)) -- positive means
the pair co-occurs more than chance, negative means less.

IMPORTANT CONFOUND: dossiers vary a lot in overall "richness" (how many of
the 11 types they contain at all -- from 1 to 11, mean 9, SD 2.5). A dossier
that happens to be more complete will show elevated presence for EVERY type
at once, which inflates the raw pairwise log-odds-ratio for ALL pairs
uniformly (confirmed empirically: with the raw/unstratified measure, all 45
pairs come out positive and "credible" -- not a plausible substantive
result, a classic co-occurrence-analysis artifact). To control for it, every
pair also gets a Mantel-Haenszel stratified log-odds-ratio: dossiers are
split into 4 strata by their richness *excluding* the pair's own two types
(rank-based, so exactly balanced), the 2x2 table is computed within each
stratum, and the strata are pooled via the standard Mantel-Haenszel
estimator (with Robins-Breslow-Greenland variance) -- a standard
epidemiological technique for exactly this kind of confound. The MH version
is what answers "do A and B specifically go together," net of one dossier
simply being more complete than another; the raw version is kept alongside
it for transparency about how much of the raw signal was confound.

Requires confusion_matrix.py to have been run first.

Rerun with a new model's predictions:
  python3 scripts/dossier_composition/co_occurrence.py --predictions <path> --test-predictions <path>
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from common import ALL_TYPES, DEFAULT_PREDICTIONS_PATH, DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR, OTHER
from imputation import (
    load_pages, naive_segments_from_pages, load_confusion_posterior, load_start_page_confusion_posterior,
    sample_dossier_type_counts_for_draw, RNG_SEED as IMPUTATION_RNG_SEED,
)
from occurrence import dossier_type_counts

N_IMPUTATIONS = 200
INNER_DRAWS = 20  # Dirichlet posterior draws per imputation per pair
MH_STRATA = 4  # richness strata for the Mantel-Haenszel control
RNG_SEED = 42


def pair_log_odds_draws(presence_a: np.ndarray, presence_b: np.ndarray, rng: np.random.Generator, n_draws: int):
    n11 = int((presence_a & presence_b).sum())
    n10 = int((presence_a & ~presence_b).sum())
    n01 = int((~presence_a & presence_b).sum())
    n00 = int((~presence_a & ~presence_b).sum())
    draws = rng.dirichlet([n11 + 1, n10 + 1, n01 + 1, n00 + 1], size=n_draws)  # (n_draws, 4)
    p11, p10, p01, p00 = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3]
    log_odds = np.log(p11) + np.log(p00) - np.log(p10) - np.log(p01)
    lift = p11 / ((p11 + p10) * (p11 + p01))
    return log_odds, lift


def compute_pairwise(presence_matrices: list, pairs: list, type_idx_of: dict, n_draws_per: int, rng_seed: int):
    """presence_matrices: list of (n_dossiers, n_types) boolean arrays, one per
    imputation (or a single-element list for the naive/uncorrected case)."""
    rng = np.random.default_rng(rng_seed)
    results = {}
    for a, b in pairs:
        ia, ib = type_idx_of[a], type_idx_of[b]
        all_log_odds, all_lift = [], []
        for presence in presence_matrices:
            lo, lift = pair_log_odds_draws(presence[:, ia], presence[:, ib], rng, n_draws_per)
            all_log_odds.append(lo)
            all_lift.append(lift)
        results[(a, b)] = (np.concatenate(all_log_odds), np.concatenate(all_lift))
    return results


def mantel_haenszel_log_odds(presence_a: np.ndarray, presence_b: np.ndarray, strata: np.ndarray, n_strata: int):
    """Stratified log odds ratio (Mantel-Haenszel) + its Robins-Breslow-Greenland
    variance, controlling for whatever `strata` encodes (here: dossier richness
    excluding A and B)."""
    sum_r = sum_s = sum_pr = sum_ps_qr = sum_qs = 0.0
    for k in range(n_strata):
        mask = strata == k
        n = mask.sum()
        if n == 0:
            continue
        a, b = presence_a[mask], presence_b[mask]
        n11 = float((a & b).sum())
        n10 = float((a & ~b).sum())
        n01 = float((~a & b).sum())
        n00 = float((~a & ~b).sum())
        if n11 + n10 + n01 + n00 == 0:
            continue
        r_k = n11 * n00 / n
        s_k = n10 * n01 / n
        p_k = (n11 + n00) / n
        q_k = (n10 + n01) / n
        sum_r += r_k
        sum_s += s_k
        sum_pr += p_k * r_k
        sum_ps_qr += p_k * s_k + q_k * r_k
        sum_qs += q_k * s_k

    if sum_r <= 0 or sum_s <= 0:
        return np.nan, np.nan
    log_or = np.log(sum_r / sum_s)
    var_log_or = sum_pr / (2 * sum_r**2) + sum_ps_qr / (2 * sum_r * sum_s) + sum_qs / (2 * sum_s**2)
    return log_or, var_log_or


def richness_strata(presence: np.ndarray, ia: int, ib: int, n_strata: int) -> np.ndarray:
    """Rank-based (exactly balanced) strata by dossier richness excluding types
    ia and ib, so a pair's own two types can't mechanically determine their
    own stratification."""
    richness_other = presence.sum(axis=1).astype(int) - presence[:, ia].astype(int) - presence[:, ib].astype(int)
    order = np.argsort(richness_other, kind="stable")
    strata = np.empty(len(richness_other), dtype=int)
    for k, idxs in enumerate(np.array_split(order, n_strata)):
        strata[idxs] = k
    return strata


def compute_pairwise_mh(presence_matrices: list, pairs: list, type_idx_of: dict, n_draws_per: int,
                          n_strata: int, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    results = {}
    for a, b in pairs:
        ia, ib = type_idx_of[a], type_idx_of[b]
        draws = []
        for presence in presence_matrices:
            strata = richness_strata(presence, ia, ib, n_strata)
            log_or, var_log_or = mantel_haenszel_log_odds(presence[:, ia], presence[:, ib], strata, n_strata)
            if np.isnan(log_or) or var_log_or <= 0:
                continue
            draws.append(rng.normal(log_or, np.sqrt(var_log_or), size=n_draws_per))
        results[(a, b)] = np.concatenate(draws) if draws else np.array([np.nan])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--test-predictions", default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-imputations", type=int, default=N_IMPUTATIONS)
    parser.add_argument("--inner-draws", type=int, default=INNER_DRAWS)
    args = parser.parse_args()

    print(f"Loading pages from {args.predictions} ...")
    pages = load_pages(args.predictions)
    dossier_order = pages["pdf_name"].unique()
    n_dossiers = len(dossier_order)
    n_types = len(ALL_TYPES)

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

    print(f"Sampling {args.n_imputations} joint (segmentation + type) imputations ...")
    rng = np.random.default_rng(IMPUTATION_RNG_SEED)
    corrected_presence = []
    for d in range(type_p_draws.shape[0]):
        counts_d = sample_dossier_type_counts_for_draw(
            pages, type_p_draws[d], type_prior, start_p_draws[d], start_prior, rng, dossier_order
        )
        corrected_presence.append(counts_d > 0)

    naive_counts = dossier_type_counts(segments, segments["type_idx"].to_numpy(), n_dossiers, n_types)
    naive_presence = [naive_counts > 0]

    type_idx_of = {t: i for i, t in enumerate(ALL_TYPES)}
    tracked = [t for t in ALL_TYPES if t != OTHER]
    pairs = list(itertools.combinations(tracked, 2))
    print(f"\nComputing pairwise co-occurrence for {len(pairs)} type pairs ...")

    corrected_results = compute_pairwise(corrected_presence, pairs, type_idx_of, args.inner_draws, RNG_SEED)
    naive_results = compute_pairwise(naive_presence, pairs, type_idx_of, args.inner_draws * args.n_imputations, RNG_SEED + 1)

    print("Computing richness-stratified (Mantel-Haenszel) co-occurrence ...")
    mh_corrected_results = compute_pairwise_mh(
        corrected_presence, pairs, type_idx_of, args.inner_draws, MH_STRATA, RNG_SEED + 2
    )
    mh_naive_results = compute_pairwise_mh(
        naive_presence, pairs, type_idx_of, args.inner_draws * args.n_imputations, MH_STRATA, RNG_SEED + 3
    )

    rows = []
    for a, b in pairs:
        lo_c, lift_c = corrected_results[(a, b)]
        lo_n, lift_n = naive_results[(a, b)]
        mh_c = mh_corrected_results[(a, b)]
        mh_n = mh_naive_results[(a, b)]
        rows.append({
            "type_a": a, "type_b": b,
            "naive_log_odds_mean": lo_n.mean(),
            "corrected_log_odds_mean": lo_c.mean(),
            "corrected_log_odds_hdi_3": np.percentile(lo_c, 3),
            "corrected_log_odds_hdi_97": np.percentile(lo_c, 97),
            "corrected_p_positive": (lo_c > 0).mean(),
            "corrected_lift_mean": lift_c.mean(),
            "corrected_lift_hdi_3": np.percentile(lift_c, 3),
            "corrected_lift_hdi_97": np.percentile(lift_c, 97),
            "naive_mh_log_odds_mean": np.nanmean(mh_n),
            "corrected_mh_log_odds_mean": np.nanmean(mh_c),
            "corrected_mh_log_odds_hdi_3": np.nanpercentile(mh_c, 3),
            "corrected_mh_log_odds_hdi_97": np.nanpercentile(mh_c, 97),
            "corrected_mh_p_positive": np.nanmean(mh_c > 0),
        })

    table = pd.DataFrame(rows)
    table["credible"] = (table["corrected_p_positive"] > 0.975) | (table["corrected_p_positive"] < 0.025)
    table["mh_credible"] = (table["corrected_mh_p_positive"] > 0.975) | (table["corrected_mh_p_positive"] < 0.025)
    table = table.sort_values("corrected_mh_log_odds_mean", ascending=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n=== raw (unstratified): all pairs, for comparison ===")
    print(f"all positive: {(table['corrected_log_odds_mean'] > 0).all()} -- "
          f"this is the richness confound, see MH-controlled results below")

    print("\n=== richness-controlled (Mantel-Haenszel): strongest positive associations ===")
    print(table.head(10)[["type_a", "type_b", "corrected_mh_log_odds_mean", "corrected_mh_log_odds_hdi_3",
                            "corrected_mh_log_odds_hdi_97", "mh_credible"]].round(3).to_string(index=False))
    print("\n=== richness-controlled (Mantel-Haenszel): strongest negative associations ===")
    print(table.tail(10)[["type_a", "type_b", "corrected_mh_log_odds_mean", "corrected_mh_log_odds_hdi_3",
                            "corrected_mh_log_odds_hdi_97", "mh_credible"]].round(3).to_string(index=False))
    print(f"\n{table['mh_credible'].sum()} / {len(table)} pairs credible after richness control "
          f"(vs. {table['credible'].sum()} / {len(table)} without it)")

    table.to_csv(f"{args.out_dir}/co_occurrence_summary.csv", index=False)
    print(f"\nSaved co_occurrence_summary.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
