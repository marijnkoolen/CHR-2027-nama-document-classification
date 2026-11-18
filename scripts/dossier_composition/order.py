"""
Order: (A) does each document type have a fairly fixed typical position in
the dossier sequence, or is it spread throughout? (B) for pairs of types
that co-occur, is there a consistent typical order (e.g. D.1 usually before
D.2)?

Part A -- marginal position: pool each type's instances' normalized position
(order_in_dossier / (dossier length - 1), so 0 = first segment, 1 = last)
across the corpus. Mean position (with SE-based interval) shows where a type
typically sits; SD of position shows how consistent that is (low SD = fixed
spot, high SD = spread throughout).

Part B -- pairwise order via a Bradley-Terry model: for every dossier where
both A and B occur, record whether A's first occurrence precedes B's. This
gives, per pair, a win/loss count -- fit as a Bradley-Terry paired-comparison
model (equivalent to a Binomial GLM on a +1/-1 contrast design, one type
fixed as reference for identifiability) to get one latent "typical sequence
rank" theta per type: more negative = tends to come earlier, more positive =
tends to come later. This is a single unified model instead of 45 separate
pairwise tests, and directly gives P(A before B) for any pair via
sigmoid(theta_B - theta_A).

Both parts run per imputation draw (multiple imputation through the
confusion-matrix posterior, same as occurrence/co_occurrence/dispersion),
using a Normal approximation (mean + SE from statsmodels' GLM fit, or from
the sample mean/SD) pooled across draws -- same "cheap closed-form-ish
approximation per draw, pool across draws" pattern as co_occurrence.py's
Mantel-Haenszel step, rather than refitting a full MCMC model 200 times.

The Bradley-Terry sign convention is validated against synthetic data with
known true ranks before trusting it on real data -- run
`python3 scripts/dossier_composition/order.py --self-test` to repeat that
check.

Requires confusion_matrix.py to have been run first.

Rerun with a new model's predictions:
  python3 scripts/dossier_composition/order.py --predictions <path> --test-predictions <path>
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import ALL_TYPES, DEFAULT_PREDICTIONS_PATH, DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR, OTHER, TRACKED_TYPES
from imputation import (
    load_pages, naive_segments_from_pages, load_confusion_posterior, load_start_page_confusion_posterior,
    sample_segments_for_draw, RNG_SEED as IMPUTATION_RNG_SEED,
)

N_IMPUTATIONS = 200
INNER_DRAWS = 20
RNG_SEED = 42
REF_TYPE_IDX = 0  # reference type (theta fixed at 0) within TRACKED_TYPES, for identifiability


def first_occurrence_matrix(segments: pd.DataFrame, type_idx: np.ndarray, n_dossiers: int, n_types: int) -> np.ndarray:
    """(n_dossiers, n_types) array of each type's first occurrence order within
    its dossier (np.inf if absent)."""
    dossier_idx = segments["dossier_idx"].to_numpy()
    order = segments["order_in_dossier"].to_numpy()
    combined = dossier_idx * n_types + type_idx
    flat = np.full(n_dossiers * n_types, np.inf)
    np.minimum.at(flat, combined, order)
    return flat.reshape(n_dossiers, n_types)


def bradley_terry_fit(win_counts: dict, tracked_types: list, ref_idx: int):
    """win_counts: {(i, j): (n_i_first, n_j_first)} for i<j indices into
    tracked_types. Returns (theta, cov) for the non-reference types, in
    tracked_types order with the reference dropped."""
    n_types = len(tracked_types)
    free_idx = [i for i in range(n_types) if i != ref_idx]
    col_of = {t: c for c, t in enumerate(free_idx)}

    rows, endog = [], []
    for (i, j), (n_i_first, n_j_first) in win_counts.items():
        if n_i_first + n_j_first == 0:
            continue
        x = np.zeros(len(free_idx))
        if i != ref_idx:
            x[col_of[i]] = -1.0
        if j != ref_idx:
            x[col_of[j]] = 1.0
        rows.append(x)
        endog.append([n_i_first, n_j_first])

    exog = np.array(rows)
    endog = np.array(endog)
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()
    return result.params, result.cov_params(), free_idx


def full_theta(theta_free: np.ndarray, free_idx: list, n_types: int, ref_idx: int) -> np.ndarray:
    theta = np.zeros(n_types)
    theta[free_idx] = theta_free
    theta[ref_idx] = 0.0
    return theta


def self_test():
    """Synthetic check: 4 types, known true theta, verify recovered theta
    ranks and roughly matches (within noise) before trusting real data."""
    rng = np.random.default_rng(0)
    types = ["A", "B", "C", "D"]
    true_theta = np.array([0.0, 2.0, -1.5, 0.5])  # A=ref, C most early, B most late
    n_per_pair = 500

    win_counts = {}
    for i in range(4):
        for j in range(i + 1, 4):
            p_i_first = 1 / (1 + np.exp(true_theta[i] - true_theta[j]))  # sigmoid(theta_j - theta_i)
            n_i_first = rng.binomial(n_per_pair, p_i_first)
            win_counts[(i, j)] = (n_i_first, n_per_pair - n_i_first)

    theta_free, cov, free_idx = bradley_terry_fit(win_counts, types, ref_idx=0)
    theta = full_theta(theta_free, free_idx, 4, ref_idx=0)

    print("true theta:", true_theta)
    print("fitted theta:", theta.round(3))
    order_true = np.argsort(true_theta)
    order_fit = np.argsort(theta)
    assert list(order_true) == list(order_fit), "Bradley-Terry recovered ranking does not match truth!"
    max_err = np.max(np.abs(theta - true_theta))
    print(f"max abs error: {max_err:.3f}")
    assert max_err < 0.3, "Bradley-Terry recovered theta too far from truth!"
    print("self-test passed: ranking and magnitude recovered correctly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--test-predictions", default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-imputations", type=int, default=N_IMPUTATIONS)
    parser.add_argument("--inner-draws", type=int, default=INNER_DRAWS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    print(f"Loading pages from {args.predictions} ...")
    pages = load_pages(args.predictions)
    dossier_order = pages["pdf_name"].unique()
    n_dossiers = len(dossier_order)
    n_types = len(ALL_TYPES)
    dossier_to_idx = {d: i for i, d in enumerate(dossier_order)}

    segments = naive_segments_from_pages(pages)
    segments["dossier_idx"] = segments["pdf_name"].map(dossier_to_idx)
    print(f"{len(segments)} predicted segments across {n_dossiers} dossiers")

    print(f"\nLoading confusion-matrix posteriors from {args.out_dir} ...")
    type_p_draws, type_prior = load_confusion_posterior(
        f"{args.out_dir}/idata_confusion_matrix.nc", args.test_predictions, n_draws=args.n_imputations
    )
    start_p_draws, start_prior = load_start_page_confusion_posterior(
        f"{args.out_dir}/idata_start_page_confusion_matrix.nc", args.test_predictions, n_draws=args.n_imputations
    )

    tracked_all_idx = [ALL_TYPES.index(t) for t in TRACKED_TYPES]
    tracked_pos_in_all = {ALL_TYPES.index(t): k for k, t in enumerate(TRACKED_TYPES)}
    n_tracked = len(TRACKED_TYPES)

    def run_one(segs: pd.DataFrame):
        type_idx_arr = segs["type_idx"].to_numpy()
        segs_n_dossiers = n_dossiers  # segs may resample segmentation but never adds/drops dossiers

        # Part A: marginal position, pooled instances of tracked types
        pos_stats = {}
        norm_pos = segs["norm_position"].to_numpy()
        for t_all, t_local in tracked_pos_in_all.items():
            vals = norm_pos[type_idx_arr == t_all]
            n = len(vals)
            if n < 2:
                pos_stats[t_local] = (np.nan, np.nan, np.nan, np.nan, n)
                continue
            mean_p, sd_p = vals.mean(), vals.std(ddof=1)
            se_mean = sd_p / np.sqrt(n)
            se_sd = sd_p / np.sqrt(2 * (n - 1))
            pos_stats[t_local] = (mean_p, se_mean, sd_p, se_sd, n)

        # Part B: pairwise first-occurrence order -> Bradley-Terry
        first_pos = first_occurrence_matrix(segs, type_idx_arr, segs_n_dossiers, n_types)
        win_counts = {}
        for a in range(n_tracked):
            for b in range(a + 1, n_tracked):
                fa, fb = first_pos[:, tracked_all_idx[a]], first_pos[:, tracked_all_idx[b]]
                valid = np.isfinite(fa) & np.isfinite(fb)
                n_a_first = int((fa[valid] < fb[valid]).sum())
                n_b_first = int(valid.sum() - n_a_first)
                win_counts[(a, b)] = (n_a_first, n_b_first)

        theta_free, cov, free_idx = bradley_terry_fit(win_counts, TRACKED_TYPES, REF_TYPE_IDX)
        return pos_stats, theta_free, cov, free_idx, win_counts

    print("\nRunning position + Bradley-Terry per imputation draw ...")
    rng = np.random.default_rng(IMPUTATION_RNG_SEED)
    pos_mean_draws = {t: [] for t in range(n_tracked)}
    pos_sd_draws = {t: [] for t in range(n_tracked)}
    theta_draws = []

    pos_stats_n, theta_free_n, cov_n, free_idx_n, win_counts_n = run_one(segments)
    theta_full_naive = full_theta(theta_free_n, free_idx_n, n_tracked, REF_TYPE_IDX)
    theta_full_naive -= theta_full_naive.mean()

    print(f"Sampling {args.n_imputations} joint (segmentation + type) imputations ...")
    for d in range(type_p_draws.shape[0]):
        segs_d = sample_segments_for_draw(
            pages, type_p_draws[d], type_prior, start_p_draws[d], start_prior, rng
        )
        segs_d["dossier_idx"] = segs_d["pdf_name"].map(dossier_to_idx)
        pos_stats, theta_free, cov, free_idx, _ = run_one(segs_d)
        for t in range(n_tracked):
            mean_p, se_mean, sd_p, se_sd, n = pos_stats[t]
            if np.isnan(mean_p):
                continue
            pos_mean_draws[t].append(rng.normal(mean_p, se_mean, size=args.inner_draws))
            pos_sd_draws[t].append(rng.normal(sd_p, se_sd, size=args.inner_draws))

        theta_sample = rng.multivariate_normal(theta_free, cov, size=args.inner_draws)
        theta_full = np.array([full_theta(ts, free_idx, n_tracked, REF_TYPE_IDX) for ts in theta_sample])
        theta_full -= theta_full.mean(axis=1, keepdims=True)  # recenter for interpretability
        theta_draws.append(theta_full)

    theta_all = np.concatenate(theta_draws, axis=0)  # (n_imp * inner_draws, n_tracked)

    rows = []
    for t, type_name in enumerate(TRACKED_TYPES):
        mean_draws = np.concatenate(pos_mean_draws[t]) if pos_mean_draws[t] else np.array([np.nan])
        sd_draws = np.concatenate(pos_sd_draws[t]) if pos_sd_draws[t] else np.array([np.nan])
        th = theta_all[:, t]
        rows.append({
            "doc_type": type_name,
            "naive_mean_position": pos_stats_n[t][0],
            "naive_sd_position": pos_stats_n[t][2],
            "corrected_mean_position_mean": np.nanmean(mean_draws),
            "corrected_mean_position_hdi_3": np.nanpercentile(mean_draws, 3),
            "corrected_mean_position_hdi_97": np.nanpercentile(mean_draws, 97),
            "corrected_sd_position_mean": np.nanmean(sd_draws),
            "corrected_sd_position_hdi_3": np.nanpercentile(sd_draws, 3),
            "corrected_sd_position_hdi_97": np.nanpercentile(sd_draws, 97),
            "naive_rank_theta": theta_full_naive[t],
            "corrected_rank_theta_mean": th.mean(),
            "corrected_rank_theta_hdi_3": np.percentile(th, 3),
            "corrected_rank_theta_hdi_97": np.percentile(th, 97),
        })

    table = pd.DataFrame(rows).sort_values("corrected_rank_theta_mean")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n=== typical sequence order (Bradley-Terry rank, most-early to most-late) ===")
    print(table[["doc_type", "corrected_rank_theta_mean", "corrected_rank_theta_hdi_3",
                 "corrected_rank_theta_hdi_97", "corrected_mean_position_mean"]].round(3).to_string(index=False))

    table.to_csv(f"{args.out_dir}/order_summary.csv", index=False)

    # also save pairwise P(A before B) implied by the pooled theta draws, for specific-pair queries
    pair_rows = []
    for a in range(n_tracked):
        for b in range(a + 1, n_tracked):
            p_a_first = (1 / (1 + np.exp(theta_all[:, a] - theta_all[:, b])))
            pair_rows.append({
                "type_a": TRACKED_TYPES[a], "type_b": TRACKED_TYPES[b],
                "p_a_before_b_mean": p_a_first.mean(),
                "p_a_before_b_hdi_3": np.percentile(p_a_first, 3),
                "p_a_before_b_hdi_97": np.percentile(p_a_first, 97),
            })
    pd.DataFrame(pair_rows).to_csv(f"{args.out_dir}/order_pairwise.csv", index=False)

    print(f"\nSaved order_summary.csv and order_pairwise.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
