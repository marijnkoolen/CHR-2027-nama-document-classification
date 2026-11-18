"""
Point-corrected counterpart to doc_type_period_diff.py: same pre-1956 vs.
post-1956 pre-adult-effect difference, computed from the classifier-
uncertainty-corrected period fits (doc_type_three_groups_uncertainty.py
--mode point) instead of the raw ones, so the "which types actually
shifted" story can be checked against corrected counts, not just raw ones.

Same independent-draws-pairing logic as doc_type_period_diff.py (the two
period models are independent fits, so pairing draws index-wise and
subtracting is a valid Monte Carlo sample from the difference distribution).
"""

import numpy as np
import pandas as pd
import arviz as az

from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model/uncertainty"
RNG_SEED = 42


def main():
    idata_pre = az.from_netcdf(f"{OUT_DIR}/idata_point_pre1956.nc")
    idata_post = az.from_netcdf(f"{OUT_DIR}/idata_point_post1956.nc")

    rows = []
    for t in DOC_TYPE_COLS:
        pre_draws = idata_pre.posterior["beta_preadult"].sel(type=t).values.flatten()
        post_draws = idata_post.posterior["beta_preadult"].sel(type=t).values.flatten()
        rng = np.random.default_rng(RNG_SEED)
        n = min(len(pre_draws), len(post_draws))
        pre_s = rng.choice(pre_draws, n, replace=False)
        post_s = rng.choice(post_draws, n, replace=False)

        pre_pct = (np.exp(pre_s) - 1) * 100
        post_pct = (np.exp(post_s) - 1) * 100
        diff_pct = post_pct - pre_pct

        rows.append({
            "doc_type": t,
            "preadult_pct_pre1956": pre_pct.mean(),
            "preadult_pct_post1956": post_pct.mean(),
            "diff_mean": diff_pct.mean(),
            "diff_hdi_3": np.percentile(diff_pct, 3),
            "diff_hdi_97": np.percentile(diff_pct, 97),
            "p_post_lt_pre": (post_s < pre_s).mean(),
        })

    table = pd.DataFrame(rows).sort_values("p_post_lt_pre", ascending=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))

    table.to_csv(f"{OUT_DIR}/point_period_diff.csv", index=False)
    print(f"\nSaved {OUT_DIR}/point_period_diff.csv")


if __name__ == "__main__":
    main()
