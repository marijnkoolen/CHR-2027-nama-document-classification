"""
LOO-CV comparison across all seven num_docs models built in this analysis,
organized as a 3x2 grid crossed on (1) how many age groups the persons in
a dossier are split into, and (2) whether there's an era interaction --
plus D3 as an extra cell in the three-group row (two era flavors there:
pre-adult-only interaction, C3; interaction on all three groups, D3).

                          no era interaction   with era interaction
  (1) num_persons          A                    A_era
  (2) adult(18+)/minor     B                    C
  (3) adult/pre-adult/minor B3                   C3   (+ D3, all-groups era)

All seven use the SAME constant 18+ adult threshold now -- A/B/C
(model_dossier_size.py) originally used an era-switching (18+ pre-1956,
16+ post-1956) `num_adults` column that was later found to be a data-
generation bug, not a real definitional change; ages_num_docs_pages.tsv
has since been corrected and A/B/C refit on it, so `num_adults` here now
means the same thing as `num_18+` in model_dossier_size_three_groups.py's
B3/C3/D3. The grid therefore cleanly isolates "how many groups" and "is
there an era effect" as the only two things varying across cells -- no
longer confounded by two different age-threshold conventions, which was a
real problem with this same comparison before the data fix.

Does not refit anything -- loads the seven idata traces each upstream
script already saved and compares them directly.

Valid because all seven share the same outcome (num_docs), the same
likelihood family (negative binomial), and the same 1307-dossier subset in
the same row order -- checked below before comparing rather than assumed,
since a silent misalignment would produce a plausible-looking but wrong
comparison.

Saves three tables:
  loo_compare_grid.csv       the clean 3x2 grid (6 models: A, A_era, B, C,
                              B3, C3) -- the headline comparison.
  loo_compare_grid_with_D3.csv  the same 6 plus D3 (7 models) -- for
                              arguing about D3 specifically (e.g. "is
                              interacting era with ALL three groups, not
                              just pre-adult, ever worth it") against the
                              full model set, not just against B3/C3.
  loo_compare_B3_vs_C3.csv   direct pairwise B3 vs. C3 -- NOT derivable
                              from either grid table above, since both
                              measure every model's distance from whichever
                              model ranks best in THAT table, not from each
                              other. Answers whether C3's era-interaction
                              term earns a decisive predictive-accuracy
                              edge over the simpler B3 (it doesn't, as of
                              the last run: elpd_diff well under 1 dse) --
                              a different question from whether the era-
                              interaction coefficient itself is credible
                              (see the report's C3 writeup, based on the
                              coefficient's own posterior, not this LOO
                              comparison).

Rerun after refitting any of the seven upstream models:
  python3 scripts/dossier_size_model/compare_all_size_models.py
"""

import arviz as az
import numpy as np
import pandas as pd

OUT_DIR = "data/dossier_size_model"

MODEL_PATHS = {
    "A_persons": f"{OUT_DIR}/idata_A_persons.nc",
    "A_era_persons_era": f"{OUT_DIR}/idata_A_era_persons_era.nc",
    "B_adults_minors": f"{OUT_DIR}/idata_B_adults_minors.nc",
    "C_adults_minors_era": f"{OUT_DIR}/idata_C_adults_minors_era.nc",
    "B3_three_group": f"{OUT_DIR}/idata_three_group_B3.nc",
    "C3_preadult_era": f"{OUT_DIR}/idata_three_group_C3.nc",
    "D3_all_era": f"{OUT_DIR}/idata_three_group_D3.nc",
}

# the clean 3x2 grid -- D3 is deliberately left out of this one (see module docstring)
GRID_MODEL_NAMES = [
    "A_persons", "A_era_persons_era",
    "B_adults_minors", "C_adults_minors_era",
    "B3_three_group", "C3_preadult_era",
]

GRID_LAYOUT = {
    "(1) num_persons": (("A", "A_persons"), ("A_era", "A_era_persons_era")),
    "(2) adult(18+)/minor": (("B", "B_adults_minors"), ("C", "C_adults_minors_era")),
    "(3) adult/pre-adult/minor": (("B3", "B3_three_group"), ("C3", "C3_preadult_era")),
}


def load_and_verify() -> dict:
    idatas = {name: az.from_netcdf(path) for name, path in MODEL_PATHS.items()}

    reference_name = next(iter(idatas))
    reference = idatas[reference_name]
    reference_var = list(reference.observed_data.data_vars)[0]
    reference_obs = reference.observed_data[reference_var].values

    for name, idata in idatas.items():
        assert "log_likelihood" in idata.groups(), (
            f"{name}: no log_likelihood saved in this trace -- cannot compute LOO, refit with "
            f"idata_kwargs={{'log_likelihood': True}} (or PyMC's default, if already on)"
        )
        obs_var = list(idata.observed_data.data_vars)[0]
        obs = idata.observed_data[obs_var].values
        assert len(obs) == len(reference_obs), (
            f"{name}: {len(obs)} observations vs. {reference_name}'s {len(reference_obs)} -- "
            f"not the same dossier subset, not directly comparable"
        )
        assert np.array_equal(obs, reference_obs), (
            f"{name}: observed {obs_var} values differ from {reference_name}'s {reference_var} -- "
            f"rows aren't aligned (or the outcome itself differs), not directly comparable"
        )

    return idatas


def print_pareto_k(idatas: dict):
    print("\nPareto-k diagnostics (PSIS-LOO is unreliable if these are bad, regardless of elpd_diff):")
    for name, idata in idatas.items():
        loo = az.loo(idata, pointwise=True)
        max_k = float(loo.pareto_k.values.max())
        n_bad = int((loo.pareto_k.values > 0.7).sum())
        n_ok_high = int(((loo.pareto_k.values > 0.5) & (loo.pareto_k.values <= 0.7)).sum())
        print(f"  {name:22s} max k={max_k:.3f}  n(k>0.7)={n_bad}  n(0.5<k<=0.7)={n_ok_high}")


def print_grid_pivot(cmp: pd.DataFrame, title: str):
    """Convenience view of elpd_loo shaped like the 3x2 (or 3x2+1) grid --
    for eyeballing only; the flat az.compare() table (with rank/dse) is the
    statistically complete source, this is just easier to read at a glance."""
    print(f"\n{title} (elpd_loo, higher = better fit):")
    for row_label, cells in GRID_LAYOUT.items():
        vals = "  ".join(
            f"{label:>6s}={cmp.loc[name, 'elpd_loo']:.1f}" for label, name in cells if name in cmp.index
        )
        print(f"  {row_label:28s} {vals}")


def main():
    idatas = load_and_verify()
    n_obs = len(next(iter(idatas.values())).observed_data.to_array().values.flatten())
    print(f"All {len(idatas)} models verified on {n_obs} shared observations (same num_docs, same row order).")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)

    grid_idatas = {name: idatas[name] for name in GRID_MODEL_NAMES}
    cmp_grid = az.compare(grid_idatas)
    print("\n=== LOO comparison: the 3x2 grid (6 models, D3 excluded) ===")
    print(cmp_grid[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "warning"]])
    print_grid_pivot(cmp_grid, "Grid view")
    cmp_grid.to_csv(f"{OUT_DIR}/loo_compare_grid.csv")
    print(f"\nSaved {OUT_DIR}/loo_compare_grid.csv")

    cmp_full = az.compare(idatas)
    print("\n=== LOO comparison: grid + D3 (7 models) ===")
    print(cmp_full[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "warning"]])
    cmp_full.to_csv(f"{OUT_DIR}/loo_compare_grid_with_D3.csv")
    print(f"Saved {OUT_DIR}/loo_compare_grid_with_D3.csv")

    print_pareto_k(idatas)

    # Direct pairwise B3 vs. C3 -- see module docstring for why this isn't
    # derivable from either table above.
    print("\n=== LOO comparison: B3 vs. C3 directly (not both-vs-grid-best) ===")
    cmp_b3_c3 = az.compare({"B3_three_group": idatas["B3_three_group"], "C3_preadult_era": idatas["C3_preadult_era"]})
    print(cmp_b3_c3[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "warning"]])
    cmp_b3_c3.to_csv(f"{OUT_DIR}/loo_compare_B3_vs_C3.csv")
    print(f"Saved {OUT_DIR}/loo_compare_B3_vs_C3.csv")


if __name__ == "__main__":
    main()
