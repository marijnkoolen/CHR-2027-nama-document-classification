"""
Classifier-uncertainty-corrected version of model_doc_types_three_groups.py:
same model (minor/pre-adult/adult per document type, partially pooled,
year x type effects), fit per period (full/pre1956/post1956), but on
corrected document-type counts instead of the raw TSV counts -- see
doc_type_uncertainty_common.py's docstring for why and how.

Two modes:

  --mode point   Fast (~same cost as the original script, ~25 min total for
                 all 3 periods), CPU-only. Deconvolves counts via the
                 document-type confusion matrix's posterior MEAN; holds
                 segmentation fixed at its predicted boundaries (see
                 doc_type_uncertainty_common.py's docstring for why). Bias-
                 corrected point estimate; credible intervals still
                 understate true uncertainty (doesn't propagate the
                 classifier's own uncertainty, and doesn't correct
                 segmentation at all). Run this locally first for a quick
                 read on whether the correction moves anything.

  --mode mi      Rigorous multiple imputation, correcting BOTH document-type
                 AND segmentation uncertainty jointly: draws hard imputed
                 counts per posterior draw of BOTH confusion matrices,
                 refits per imputation, pools posteriors. Not practical on
                 CPU at useful imputation counts (~25 min PER imputation PER
                 period => 20 imputations x 3 periods ~ 25 hours on CPU).

                 GPU setup (on the A10 box, not runnable here):
                     pip install numpyro "jax[cuda12]"
                 Verify JAX sees the GPU:
                     python3 -c "import jax; print(jax.devices())"

                 Split the 20 (or more) imputations across both A10s as two
                 independent processes, each pinned to one GPU, each getting
                 half the imputation indices (same --shard pattern as
                 scripts/ocr/benchmark_vllm.py):

                     CUDA_VISIBLE_DEVICES=0 python3 scripts/dossier_size_model/doc_type_three_groups_uncertainty.py \\
                         --mode mi --n-imputations 20 --shard 0/2 --gpu &
                     CUDA_VISIBLE_DEVICES=1 python3 scripts/dossier_size_model/doc_type_three_groups_uncertainty.py \\
                         --mode mi --n-imputations 20 --shard 1/2 --gpu &
                     wait

                 Each imputation's result is saved immediately (checkpointed)
                 as it completes, so a crashed/killed run can be resumed by
                 rerunning the same command -- already-saved imputations are
                 skipped. Once both shards finish, pool them:

                     python3 scripts/dossier_size_model/doc_type_three_groups_uncertainty.py --mode pool --n-imputations 20
"""

import argparse
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
from model_doc_types_three_groups import PERIODS, build_model, summarize

OUT = Path(UNCERTAINTY_OUT_DIR)


def fit_one(long_df: pd.DataFrame, use_gpu: bool, seed: int):
    model = build_model(long_df)
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=4, random_seed=seed, target_accept=0.92,
            **gpu_sample_kwargs(use_gpu),
        )
    return idata


def run_point(pages: pd.DataFrame, dossier_meta: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    segments = naive_segments_from_pages(pages)  # segmentation NOT corrected in point mode
    correction = point_correction_matrix()
    expected = soft_expected_counts(segments, correction)

    for period_label, (year_min, year_max) in PERIODS.items():
        meta_p = dossier_meta[(dossier_meta["travel_year"] >= year_min) & (dossier_meta["travel_year"] <= year_max)]
        long_df = to_long_format(meta_p, expected, round_counts=False)
        n_dossiers = len(long_df) // len(DOC_TYPE_COLS)
        print(f"\n=== point-corrected: {period_label} ({year_min}-{year_max}), {n_dossiers} dossiers ===")

        idata = fit_one(long_df, use_gpu=False, seed=RNG_SEED)
        max_rhat = float(az.rhat(idata).max().to_array().max())
        n_div = int(idata.sample_stats["diverging"].sum())
        print(f"max r_hat={max_rhat:.3f}, divergences={n_div}")

        table = summarize(idata)
        print(table[["doc_type", "pct_per_minor_mean", "pct_per_preadult_mean", "pct_per_adult_mean"]].round(1))

        idata.to_netcdf(OUT / f"idata_point_{period_label}.nc")
        table.to_csv(OUT / f"point_{period_label}.csv", index=False)

    print(f"\nSaved point-corrected results to {OUT}/")


def run_mi(pages: pd.DataFrame, dossier_meta: pd.DataFrame, n_imputations: int, shard: str, use_gpu: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    shard_idx, n_shards = (int(x) for x in shard.split("/"))
    my_indices = list(range(shard_idx, n_imputations, n_shards))
    print(f"shard {shard_idx}/{n_shards}: {len(my_indices)} imputations -> {my_indices}")

    type_p_draws = select_posterior_draws(n_imputations)
    type_prior = load_test_prior()
    start_p_draws, start_prior = load_start_page_posterior(n_imputations)
    # each (imputation, period) fit is independent -- as in the original
    # model_doc_types_three_groups.py, the 3 periods are separate analyses,
    # not required to agree on any single dossier's imputed label. A fixed
    # per-period offset (not Python's hash(), which is randomized per
    # process and would break reproducibility across shard runs) keeps each
    # (k, period) combination's random stream distinct and deterministic.
    period_offset = {label: i * 10_000 for i, label in enumerate(PERIODS)}

    for k in my_indices:
        for period_label, (year_min, year_max) in PERIODS.items():
            out_path = OUT / f"idata_mi_joint_{period_label}_{k:03d}.nc"
            if out_path.exists():
                print(f"skip (already done): {out_path}")
                continue

            period_rng = np.random.default_rng(RNG_SEED + 1000 + k + period_offset[period_label])
            meta_p = dossier_meta[(dossier_meta["travel_year"] >= year_min) & (dossier_meta["travel_year"] <= year_max)]
            pages_p = pages[pages["pdf_name"].isin(meta_p["pdf_name"])]

            counts = hard_imputed_counts_joint(
                pages_p, type_p_draws[k], type_prior, start_p_draws[k], start_prior, period_rng
            )
            long_df = to_long_format(meta_p, counts, round_counts=False)

            print(f"\n=== mi imputation {k}, period {period_label} (joint segmentation+type correction) ===")
            idata = fit_one(long_df, use_gpu=use_gpu, seed=RNG_SEED + k)
            max_rhat = float(az.rhat(idata).max().to_array().max())
            n_div = int(idata.sample_stats["diverging"].sum())
            print(f"max r_hat={max_rhat:.3f}, divergences={n_div}")

            idata.to_netcdf(out_path)
    print(f"\nShard {shard_idx}/{n_shards} done.")


def run_pool(n_imputations: int):
    for period_label in PERIODS:
        tables = []
        found = 0
        for k in range(n_imputations):
            path = OUT / f"idata_mi_joint_{period_label}_{k:03d}.nc"
            if not path.exists():
                continue
            found += 1
            idata = az.from_netcdf(path)
            for t in DOC_TYPE_COLS:
                for group in ["minor", "preadult", "adult"]:
                    draws = idata.posterior[f"beta_{group}"].sel(type=t).values.flatten()
                    pct = (np.exp(draws) - 1) * 100
                    tables.append({"doc_type": t, "group": group, "pct": pct})

        if found < n_imputations:
            print(f"WARNING: only {found}/{n_imputations} imputations found for period={period_label}, "
                  f"pooling what's available -- rerun the missing shard(s) before treating this as final.")
        if found == 0:
            print(f"period={period_label}: no imputations found, skipping")
            continue

        rows = []
        for t in DOC_TYPE_COLS:
            row = {"doc_type": t, "n_imputations_pooled": found}
            for group in ["minor", "preadult", "adult"]:
                pooled = np.concatenate([r["pct"] for r in tables if r["doc_type"] == t and r["group"] == group])
                row[f"pct_per_{group}_mean"] = pooled.mean()
                row[f"pct_per_{group}_hdi_3"] = np.percentile(pooled, 3)
                row[f"pct_per_{group}_hdi_97"] = np.percentile(pooled, 97)
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("pct_per_preadult_mean", ascending=False)
        pd.set_option("display.width", 250)
        print(f"\n=== pooled mi, period={period_label} ({found}/{n_imputations} imputations) ===")
        print(table.round(1).to_string(index=False))
        table.to_csv(OUT / f"mi_joint_pooled_{period_label}.csv", index=False)

    print(f"\nSaved pooled results to {OUT}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["point", "mi", "pool"], required=True)
    parser.add_argument("--n-imputations", type=int, default=20)
    parser.add_argument("--shard", default="0/1", help="e.g. 0/2 for the first of 2 shards")
    parser.add_argument("--gpu", action="store_true", help="use numpyro/JAX on GPU (requires --mode mi)")
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
