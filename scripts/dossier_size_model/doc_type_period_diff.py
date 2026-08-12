"""
Per-document-type pre-1956 vs. post-1956 difference in the pre-adult (16-17)
effect, from the two independent period fits in model_doc_types_three_groups.py
(idata_doc_types_3group_pre1956.nc, idata_doc_types_3group_post1956.nc).

Since the two period models are independent fits, pairing draws index-wise
and subtracting gives valid Monte Carlo samples from the difference
distribution (the two posteriors are independent random variables, so this
is just how you compute the distribution of their difference) -- this is
what identified which specific document types (D.1, Testimonial medical
form, Report of selection, Approval notice) drive the aggregate pre-adult
shrinkage found in the three-group dossier-size model, vs. which types show
no credible change.

Originally run as an ad hoc analysis snippet; formalized here as its own
script so it's a reproducible pipeline step rather than a one-off.
"""

import numpy as np
import pandas as pd
import arviz as az

from model_doc_types import DOC_TYPE_COLS

OUT_DIR = "data/dossier_size_model"
RNG_SEED = 42


def main():
    idata_pre = az.from_netcdf(f"{OUT_DIR}/idata_doc_types_3group_pre1956.nc")
    idata_post = az.from_netcdf(f"{OUT_DIR}/idata_doc_types_3group_post1956.nc")

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

    table.to_csv(f"{OUT_DIR}/doc_type_3group_period_diff.csv", index=False)
    print(f"\nSaved doc_type_3group_period_diff.csv to {OUT_DIR}/")


if __name__ == "__main__":
    main()
