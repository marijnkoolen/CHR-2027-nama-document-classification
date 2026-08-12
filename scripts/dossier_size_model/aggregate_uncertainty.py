"""
Classifier-uncertainty-corrected version of the AGGREGATE (total num_docs)
dossier-size models: model_dossier_size_three_groups.py's B3/C3/D3 (the
current three-group account, section 6 of the report) and
model_dossier_size_temporal.py's temporal mean/dispersion structure search
(section 3).

Unlike the per-document-TYPE analyses (doc_type_*_uncertainty.py), a
dossier's total document count is insensitive to document-type confusion --
mislabeling a page's predicted type doesn't change whether that page STARTS
a new document -- so only the start_page (segmentation) confusion matrix
matters here, not the document-type one.

--mode point (the only mode implemented): fast, CPU-only. Replaces each
page's hard predicted_start_page label with its posterior-mean-corrected
P(true start='yes') (point_correction_start_page), sums per dossier to get
an EXPECTED (fractional) num_docs, and refits B3/C3/D3 and the temporal
mean+dispersion models on that corrected outcome via a NegativeBinomial
likelihood (which PyMC accepts for non-integer observed values -- the same
approximation already used throughout doc_type_*_uncertainty.py). Same
tradeoff as those scripts: corrects the average segmentation bias, but
understates the classifier's own posterior uncertainty.

A full multiple-imputation (--mode mi) version would resample start_page
per page per posterior draw and refit per draw, exactly like
doc_type_three_groups_uncertainty.py's --mode mi -- not implemented here
since point mode already answers the qualitative question (does the
corrected story change), and mi mode would need the same GPU treatment as
those scripts if it's ever wanted at scale.

Reuses model_dossier_size_three_groups.fit and model_dossier_size_temporal's
fit/prep_data/MEAN_STRUCTURES directly (same priors, same model structure)
rather than reimplementing them -- only the `num_docs` column each one
reads is swapped for the corrected value.

Run:
  python3 scripts/dossier_size_model/aggregate_uncertainty.py
"""

import argparse
from pathlib import Path

import arviz as az
import pandas as pd

from doc_type_uncertainty_common import (
    UNCERTAINTY_OUT_DIR,
    expected_num_docs_point,
    join_pages_to_aggregate_data,
    point_correction_start_page,
)
from model_dossier_size_temporal import MEAN_STRUCTURES, fit as fit_temporal, prep_data
from model_dossier_size_three_groups import fit as fit_three_group

OUT = Path(UNCERTAINTY_OUT_DIR)


def corrected_dossier_meta() -> pd.DataFrame:
    """One row per dossier (the same 1307-dossier subset used throughout),
    with `num_docs` replaced by its segmentation-corrected expected value
    and the raw value kept alongside as `num_docs_raw` for comparison."""
    pages, dossier_meta = join_pages_to_aggregate_data()
    correction_yes = point_correction_start_page()
    expected = expected_num_docs_point(pages, correction_yes)

    dossier_meta = dossier_meta.set_index("pdf_name")
    dossier_meta["num_docs_raw"] = dossier_meta["num_docs"]
    dossier_meta["num_docs"] = expected.reindex(dossier_meta.index)
    assert dossier_meta["num_docs"].notna().all(), "some dossiers missing an expected num_docs -- join issue"
    return dossier_meta.reset_index(drop=True)


def run_three_group(df: pd.DataFrame):
    print(f"\n=== corrected three-group models (n={len(df)}) ===")
    idata_B3 = fit_three_group(df, [], "B3_three_group_corrected")
    idata_C3 = fit_three_group(df, ["preadult"], "C3_preadult_era_corrected")
    idata_D3 = fit_three_group(df, ["minor", "preadult", "adult"], "D3_all_era_corrected")

    cmp = az.compare({
        "B3_no_interaction": idata_B3, "C3_preadult_era": idata_C3, "D3_all_era": idata_D3,
    })
    print(cmp)
    cmp.to_csv(OUT / "loo_compare_three_group_B3_C3_D3_corrected.csv")

    for name, idata in [("B3", idata_B3), ("C3", idata_C3), ("D3", idata_D3)]:
        idata.to_netcdf(OUT / f"idata_three_group_{name}_corrected.nc")
    print(f"Saved corrected three-group traces to {OUT}/")


def run_temporal(df: pd.DataFrame):
    df = prep_data(df)
    print(f"\n=== corrected temporal models (n={len(df)}) ===")
    idatas = {}
    for structure in MEAN_STRUCTURES:
        idatas[structure] = fit_temporal(df, structure, "constant", f"mean_{structure}_corrected")

    cmp_mean = az.compare({f"M_{k}": v for k, v in idatas.items()})
    print(cmp_mean)
    cmp_mean.to_csv(OUT / "loo_compare_temporal_mean_corrected.csv")
    winner_key = cmp_mean.index[0].replace("M_", "")
    print(f"Winning mean structure (corrected): {winner_key}")

    for structure in MEAN_STRUCTURES:
        idatas[structure].to_netcdf(OUT / f"idata_temporal_mean_{structure}_corrected.nc")

    idata_disp_trend = fit_temporal(df, winner_key, "trend", f"disp_trend_on_{winner_key}_corrected")
    idata_disp_iid = fit_temporal(df, winner_key, "iid", f"disp_iid_on_{winner_key}_corrected")
    idata_disp_trend.to_netcdf(OUT / "idata_temporal_disp_trend_corrected.nc")
    idata_disp_iid.to_netcdf(OUT / "idata_temporal_disp_iid_corrected.nc")

    cmp_disp = az.compare({
        "constant_disp": idatas[winner_key], "trend_disp": idata_disp_trend, "iid_disp": idata_disp_iid,
    })
    print(cmp_disp)
    cmp_disp.to_csv(OUT / "loo_compare_temporal_dispersion_corrected.csv")

    with open(OUT / "temporal_winner_corrected.txt", "w") as f:
        f.write(winner_key)
    print(f"Saved corrected temporal traces to {OUT}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["three_group", "temporal", "all"], default="all")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("Computing point-corrected expected num_docs (segmentation-only correction) ...")
    df = corrected_dossier_meta()
    print(df[["num_docs_raw", "num_docs"]].describe())
    diff = df["num_docs"] - df["num_docs_raw"]
    print(f"mean correction: {diff.mean():+.3f} docs/dossier, "
          f"{(diff.abs() > 0.5).sum()}/{len(df)} dossiers shift by >0.5 docs")
    df.to_csv(OUT / "aggregate_num_docs_corrected.csv", index=False)

    if args.stage in ("three_group", "all"):
        run_three_group(df)
    if args.stage in ("temporal", "all"):
        run_temporal(df)


if __name__ == "__main__":
    main()
