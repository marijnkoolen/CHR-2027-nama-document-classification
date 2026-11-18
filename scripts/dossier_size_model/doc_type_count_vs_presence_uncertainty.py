"""
Classifier-uncertainty-corrected version of model_doc_types_count_vs_presence.py:
same count-vs-presence specification comparison (does each document type's
pre-adult predictor act as a linear count or a presence/absence effect),
full period only, but on corrected document-type counts instead of the raw
TSV counts -- see doc_type_uncertainty_common.py's docstring for why and how.

Two modes, same pattern as doc_type_three_groups_uncertainty.py (see that
script's docstring for the full GPU setup instructions -- identical here,
just a different model):

  --mode point   Fast, CPU-only, bias-corrected point estimate via the
                 document-type confusion matrix's posterior mean; segmentation
                 held fixed at predicted boundaries (not corrected in this mode).
  --mode mi      Rigorous multiple imputation correcting BOTH document-type
                 AND segmentation uncertainty jointly, GPU-only at practical
                 imputation counts. Output filenames carry a `joint` tag
                 (cvp_mi_joint_*, not cvp_mi_*) so they can't collide with or
                 be silently mixed with checkpoints from the earlier
                 type-only version of this pipeline -- see doc_type_
                 uncertainty_common.py's docstring for why that matters:
                     CUDA_VISIBLE_DEVICES=0 python3 scripts/dossier_size_model/doc_type_count_vs_presence_uncertainty.py \\
                         --mode mi --n-imputations 20 --shard 0/2 --gpu &
                     CUDA_VISIBLE_DEVICES=1 python3 scripts/dossier_size_model/doc_type_count_vs_presence_uncertainty.py \\
                         --mode mi --n-imputations 20 --shard 1/2 --gpu &
                     wait
                 Then pool:
                     python3 scripts/dossier_size_model/doc_type_count_vs_presence_uncertainty.py --mode pool --n-imputations 20

Note the LOO-based winner comparison is a per-imputation-draw quantity (each
imputation gets its own count/presence winner per type); pooling reports how
often each type's winner was "presence" across imputations, rather than
averaging elpd differences directly (elpd differences aren't on a scale
that's meaningful to average across independently-imputed datasets, since
each imputation is a different underlying count dataset -- summarizing which
specification wins, and how often, is the honest way to combine).
"""

import argparse
from collections import Counter
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from doc_type_uncertainty_common import (
    RNG_SEED,
    UNCERTAINTY_OUT_DIR,
    gpu_sample_kwargs,
    hard_imputed_counts_joint,
    join_pages_to_size_model,
    load_start_page_posterior,
    load_test_prior,
    point_correction_matrix,
    select_posterior_draws,
    soft_expected_counts,
    to_long_format,
)
from imputation import naive_segments_from_pages  # sys.path set up by doc_type_uncertainty_common's import, above
from model_doc_types import DOC_TYPE_COLS
from model_doc_types_count_vs_presence import build_model, grouped_loo_comparison

OUT = Path(UNCERTAINTY_OUT_DIR)


def add_preadult_present(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["preadult_present"] = (long_df["preadult_16_17"] > 0).astype(int)
    return long_df


def fit_one(long_df: pd.DataFrame, preadult_mode: str, use_gpu: bool, seed: int):
    model = build_model(long_df, preadult_mode)
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=4, random_seed=seed, target_accept=0.92,
            idata_kwargs={"log_likelihood": True},
            **gpu_sample_kwargs(use_gpu),
        )
    return idata


def run_point(pages: pd.DataFrame, dossier_meta: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    segments = naive_segments_from_pages(pages)  # segmentation NOT corrected in point mode
    correction = point_correction_matrix()
    expected = soft_expected_counts(segments, correction)
    long_df = add_preadult_present(to_long_format(dossier_meta, expected, round_counts=False))
    print(f"point-corrected: {len(long_df) // len(DOC_TYPE_COLS)} dossiers")

    idata_count = fit_one(long_df, "count", use_gpu=False, seed=RNG_SEED)
    idata_presence = fit_one(long_df, "presence", use_gpu=False, seed=RNG_SEED)

    cmp_table = grouped_loo_comparison(idata_count, idata_presence, long_df)
    pd.set_option("display.width", 250)
    print(cmp_table.round(2).to_string(index=False))

    idata_count.to_netcdf(OUT / "idata_cvp_point_count.nc")
    idata_presence.to_netcdf(OUT / "idata_cvp_point_presence.nc")
    cmp_table.to_csv(OUT / "cvp_point.csv", index=False)
    print(f"\nSaved point-corrected results to {OUT}/")


def run_mi(pages: pd.DataFrame, dossier_meta: pd.DataFrame, n_imputations: int, shard: str, use_gpu: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    shard_idx, n_shards = (int(x) for x in shard.split("/"))
    my_indices = list(range(shard_idx, n_imputations, n_shards))
    print(f"shard {shard_idx}/{n_shards}: {len(my_indices)} imputations -> {my_indices}")

    type_p_draws = select_posterior_draws(n_imputations)
    type_prior = load_test_prior()
    start_p_draws, start_prior = load_start_page_posterior(n_imputations)

    for k in my_indices:
        out_path = OUT / f"cvp_mi_joint_{k:03d}.csv"
        if out_path.exists():
            print(f"skip (already done): {out_path}")
            continue

        rng = np.random.default_rng(RNG_SEED + 2000 + k)
        counts = hard_imputed_counts_joint(pages, type_p_draws[k], type_prior, start_p_draws[k], start_prior, rng)
        long_df = add_preadult_present(to_long_format(dossier_meta, counts, round_counts=False))

        print(f"\n=== mi imputation {k} (joint segmentation+type correction) ===")
        idata_count = fit_one(long_df, "count", use_gpu=use_gpu, seed=RNG_SEED + k)
        idata_presence = fit_one(long_df, "presence", use_gpu=use_gpu, seed=RNG_SEED + k + 500)
        for name, idata in [("count", idata_count), ("presence", idata_presence)]:
            max_rhat = float(az.rhat(idata).max().to_array().max())
            n_div = int(idata.sample_stats["diverging"].sum())
            print(f"  {name}: max r_hat={max_rhat:.3f}, divergences={n_div}")

        cmp_table = grouped_loo_comparison(idata_count, idata_presence, long_df)
        cmp_table["imputation"] = k
        cmp_table.to_csv(out_path, index=False)

    print(f"\nShard {shard_idx}/{n_shards} done.")


def run_pool(n_imputations: int):
    tables = []
    found = 0
    for k in range(n_imputations):
        path = OUT / f"cvp_mi_joint_{k:03d}.csv"
        if not path.exists():
            continue
        found += 1
        tables.append(pd.read_csv(path))

    if found < n_imputations:
        print(f"WARNING: only {found}/{n_imputations} imputations found -- pooling what's available, "
              f"rerun the missing shard(s) before treating this as final.")
    if found == 0:
        print("no imputations found")
        return

    all_rows = pd.concat(tables, ignore_index=True)
    rows = []
    for t in DOC_TYPE_COLS:
        sub = all_rows[all_rows["doc_type"] == t]
        winner_counts = Counter(sub["winner"])
        rows.append({
            "doc_type": t,
            "n_imputations_pooled": found,
            "elpd_diff_mean": sub["elpd_diff_presence_minus_count"].mean(),
            "elpd_diff_sd_across_imputations": sub["elpd_diff_presence_minus_count"].std(),
            "frac_presence_wins": winner_counts.get("presence", 0) / found,
        })
    table = pd.DataFrame(rows).sort_values("frac_presence_wins", ascending=False)
    pd.set_option("display.width", 250)
    print(f"\n=== pooled count-vs-presence, {found}/{n_imputations} imputations ===")
    print(table.round(3).to_string(index=False))
    table.to_csv(OUT / "cvp_mi_joint_pooled.csv", index=False)
    print(f"\nSaved pooled results to {OUT}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["point", "mi", "pool"], required=True)
    parser.add_argument("--n-imputations", type=int, default=20)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.mode == "pool":
        run_pool(args.n_imputations)
        return

    print("Joining pages to the size-model 1307-dossier subset ...")
    pages, dossier_meta = join_pages_to_size_model()
    print(f"{len(pages)} pages, {dossier_meta['pdf_name'].nunique()} dossiers")

    if args.mode == "point":
        run_point(pages, dossier_meta)
    else:
        run_mi(pages, dossier_meta, args.n_imputations, args.shard, args.gpu)


if __name__ == "__main__":
    main()
