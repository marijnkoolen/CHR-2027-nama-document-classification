"""
Multiple-imputation counterpart to doc_type_period_diff.py /
doc_type_period_diff_uncertainty.py: same pre-1956 vs. post-1956 pre-adult-
effect difference, computed from the full MI posteriors (20 imputations x
4000 MCMC draws each, pooled per period) rather than the raw or point-
corrected fits.

--mode pool (doc_type_three_groups_uncertainty.py) only saves each period's
POOLED SUMMARY (mean/HDI) per type, not the underlying pooled draws, so it
alone isn't enough to compute a period difference -- this script re-derives
the pooled draws itself by loading all idata_mi_joint_{period}_{k:03d}.nc
files directly (same as run_pool does internally) and keeps them, rather
than just their marginal summary.

Same independent-draws-pairing logic as the other two period-diff scripts:
pre-1956 and post-1956 are pooled as two independent large samples (80,000
draws each: 20 imputations x 4000 MCMC draws), then paired index-wise via
random sampling to get Monte Carlo draws from their difference -- valid
since the two period fits (and, within a period, the 20 imputations) are
independent.
"""

import numpy as np
import pandas as pd
import arviz as az

from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model/uncertainty"
RNG_SEED = 42


def pooled_preadult_pct(period_label: str, n_imputations: int) -> dict:
    """{doc_type: pooled pct-per-preadult draws array} across all found
    imputations for one period."""
    pooled = {t: [] for t in DOC_TYPE_COLS}
    found = 0
    for k in range(n_imputations):
        path = f"{OUT_DIR}/idata_mi_joint_{period_label}_{k:03d}.nc"
        try:
            idata = az.from_netcdf(path)
        except FileNotFoundError:
            continue
        found += 1
        for t in DOC_TYPE_COLS:
            draws = idata.posterior["beta_preadult"].sel(type=t).values.flatten()
            pooled[t].append((np.exp(draws) - 1) * 100)
    print(f"  {period_label}: {found} imputations found")
    return {t: np.concatenate(v) for t, v in pooled.items()}


def main():
    pre = pooled_preadult_pct("pre1956", 20)
    post = pooled_preadult_pct("post1956", 20)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for t in DOC_TYPE_COLS:
        pre_draws, post_draws = pre[t], post[t]
        n = min(len(pre_draws), len(post_draws))
        pre_s = rng.choice(pre_draws, n, replace=False)
        post_s = rng.choice(post_draws, n, replace=False)
        diff = post_s - pre_s
        rows.append({
            "doc_type": t,
            "preadult_pct_pre1956": pre_s.mean(),
            "preadult_pct_post1956": post_s.mean(),
            "diff_mean": diff.mean(),
            "diff_hdi_3": np.percentile(diff, 3),
            "diff_hdi_97": np.percentile(diff, 97),
            "p_post_lt_pre": (post_s < pre_s).mean(),
        })

    table = pd.DataFrame(rows).sort_values("p_post_lt_pre", ascending=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))

    out_path = f"{OUT_DIR}/mi_joint_period_diff.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
