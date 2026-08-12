"""
Generates the styled HTML report (data/dossier_size_model/report.html)
pulling together the full dossier-size / family-composition investigation:
does unit composition predict dossier size, is there a temporal trend, and
the domain-expert correction that overturned the initial "adult effect
shrinks after 1956" finding in favor of a pre-adult-specific one.

Full tables are built from the summary CSVs, so rerunning after new data
regenerates them correctly. Coefficient-level numbers not saved to CSV are
read directly from the saved posterior traces (.nc files) via arviz. The
narrative prose is a fixed template with inline numbers pulled from the same
sources (so cited figures stay in sync), but the qualitative interpretation
is written, not derived -- re-read it after a rerun, since new data could
change which finding is the interesting one.

Rerun: python3 scripts/dossier_size_model/build_report.py
"""

import base64
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

OUT = Path("data/dossier_size_model")
UNC = OUT / "uncertainty"


def b64_img(filename: str) -> str:
    data = (OUT / filename).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def fmt_pct(x, d=1, sign=False):
    s = f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%"
    return s


def fmt(x, d=3):
    return f"{x:.{d}f}"


def hdi_pct(lo, hi, d=1):
    return f"[{fmt_pct(lo, d)}, {fmt_pct(hi, d)}]"


def hdi(lo, hi, d=3):
    return f"[{fmt(lo, d)}, {fmt(hi, d)}]"


def table_html(headers: list, rows: list, classes: str = "") -> str:
    thead = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs.append(f"<tr>{tds}</tr>")
    return (
        f'<div class="table-wrap"><table class="{classes}"><thead><tr>{thead}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table></div>'
    )


def pct_summary(idata, varname, extra=None):
    """Posterior mean + 94% HDI of exp(beta)-1 as a percent, optionally beta + extra (another draws array to add)."""
    draws = idata.posterior[varname].values.flatten()
    if extra is not None:
        draws = draws + idata.posterior[extra].values.flatten()
    pct = (np.exp(draws) - 1) * 100
    return pct.mean(), np.percentile(pct, 3), np.percentile(pct, 97)


def loo_table(path) -> pd.DataFrame:
    df = pd.read_csv(OUT / path, index_col=0)
    return df[["rank", "elpd_loo", "elpd_diff", "dse", "warning"]]


def loo_html(df: pd.DataFrame) -> str:
    rows = []
    for name, r in df.iterrows():
        rows.append([
            name, int(r["rank"]), fmt(r["elpd_loo"], 1), fmt(r["elpd_diff"], 2), fmt(r["dse"], 2),
            "yes" if r["warning"] else "",
        ])
    return table_html(["Model", "Rank", "elpd_loo", "elpd_diff", "dse", "Pareto-k warning"], rows, "data-table narrow")


def doc_type_2group_table(df: pd.DataFrame) -> str:
    df = df.sort_values("pct_per_adult_mean", ascending=False)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["doc_type"],
            f'<span class="mono">{fmt_pct(r["pct_per_adult_mean"])}</span> '
            f'<span class="hdi">{hdi_pct(r["pct_per_adult_hdi_3"], r["pct_per_adult_hdi_97"])}</span>',
            f'<span class="mono">{fmt_pct(r["pct_per_minor_mean"])}</span> '
            f'<span class="hdi">{hdi_pct(r["pct_per_minor_hdi_3"], r["pct_per_minor_hdi_97"])}</span>',
        ])
    return table_html(["Document type", "Per adult (94% HDI)", "Per minor (94% HDI)"], rows, "data-table")


def doc_type_3group_table(df: pd.DataFrame, sort_col="pct_per_preadult_mean") -> str:
    df = df.sort_values(sort_col, ascending=False)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["doc_type"],
            f'<span class="mono">{fmt_pct(r["pct_per_minor_mean"])}</span> '
            f'<span class="hdi">{hdi_pct(r["pct_per_minor_hdi_3"], r["pct_per_minor_hdi_97"])}</span>',
            f'<span class="mono">{fmt_pct(r["pct_per_preadult_mean"])}</span> '
            f'<span class="hdi">{hdi_pct(r["pct_per_preadult_hdi_3"], r["pct_per_preadult_hdi_97"])}</span>',
            f'<span class="mono">{fmt_pct(r["pct_per_adult_mean"])}</span> '
            f'<span class="hdi">{hdi_pct(r["pct_per_adult_hdi_3"], r["pct_per_adult_hdi_97"])}</span>',
        ])
    return table_html(["Document type", "Per minor <16 (94% HDI)", "Per pre-adult 16-17 (94% HDI)",
                        "Per adult 18+ (94% HDI)"], rows, "data-table wide")


def period_diff_table(df: pd.DataFrame) -> str:
    df = df.sort_values("p_post_lt_pre", ascending=False)
    rows = []
    for _, r in df.iterrows():
        credible = r["p_post_lt_pre"] >= 0.95
        dot = '<span class="dot dot-yes" title="P(shrinks) >= 95%"></span>' if credible else \
              '<span class="dot dot-no" title="not credible"></span>'
        rows.append([
            r["doc_type"],
            f'<span class="mono">{fmt_pct(r["preadult_pct_pre1956"])}</span>',
            f'<span class="mono">{fmt_pct(r["preadult_pct_post1956"])}</span>',
            f'<span class="mono">{fmt_pct(r["diff_mean"], sign=True)}</span> '
            f'<span class="hdi">{hdi_pct(r["diff_hdi_3"], r["diff_hdi_97"])}</span>',
            f'<span class="mono">{fmt_pct(r["p_post_lt_pre"] * 100, 1)}</span>',
            dot,
        ])
    return table_html(["Document type", "Pre-1956", "Post-1956", "Difference (94% HDI)",
                        "P(shrinks)", "Credible"], rows, "data-table wide")


def count_vs_presence_table(df: pd.DataFrame) -> str:
    df = df.sort_values("elpd_diff_presence_minus_count", ascending=False)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["doc_type"], int(r["n_obs"]), fmt(r["elpd_count"], 2), fmt(r["elpd_presence"], 2),
            fmt(r["elpd_diff_presence_minus_count"], 2), fmt(r["se_diff"], 2), r["winner"],
        ])
    return table_html(["Document type", "n", "elpd (count)", "elpd (presence)", "elpd diff (presence-count)",
                        "SE of diff", "Winner"], rows, "data-table wide")


def main():
    occ_2g = pd.read_csv(OUT / "doc_type_effects_summary.csv")
    cvp = pd.read_csv(OUT / "doc_type_count_vs_presence_comparison.csv")
    doc3_full = pd.read_csv(OUT / "doc_type_3group_effects_full.csv")
    doc3_pre = pd.read_csv(OUT / "doc_type_3group_effects_pre1956.csv")
    doc3_post = pd.read_csv(OUT / "doc_type_3group_effects_post1956.csv")
    period_diff = pd.read_csv(OUT / "doc_type_3group_period_diff.csv")
    n_doc_types = len(doc3_full)

    loo_ab = loo_table("loo_compare_A_vs_B.csv")
    loo_bc = loo_table("loo_compare_B_vs_C.csv")
    loo_temp_mean = loo_table("loo_compare_temporal_mean.csv")
    loo_temp_disp = loo_table("loo_compare_temporal_dispersion.csv")
    loo_np_mean = loo_table("loo_compare_num_persons_temporal_mean.csv")
    loo_np_disp = loo_table("loo_compare_num_persons_temporal_dispersion.csv")
    loo_na_mean = loo_table("loo_compare_num_adults_temporal_mean.csv")
    loo_na_disp = loo_table("loo_compare_num_adults_temporal_dispersion.csv")
    loo_bb = loo_table("loo_compare_adults_binomial_vs_betabinomial.csv")
    loo_binom_mean = loo_table("loo_compare_adults_binom_temporal_mean.csv")
    loo_3group = loo_table("loo_compare_three_group_B3_C3_D3.csv")
    loo_b3_c3 = loo_table("loo_compare_B3_vs_C3.csv")
    loo_grid = loo_table("loo_compare_grid.csv")
    loo_grid_d3 = loo_table("loo_compare_grid_with_D3.csv")
    loo_a_aera = loo_table("loo_compare_A_vs_A_era.csv")

    idata_A_era = az.from_netcdf(OUT / "idata_A_era_persons_era.nc")
    a_era_coef = pct_summary(idata_A_era, "beta_num_persons_era")
    a_era_credible = not (a_era_coef[1] < 0 < a_era_coef[2])

    idata_B = az.from_netcdf(OUT / "idata_B_adults_minors.nc")
    idata_C = az.from_netcdf(OUT / "idata_C_adults_minors_era.nc")
    idata_temp_trend = az.from_netcdf(OUT / "idata_temporal_mean_trend.nc")
    idata_temp_disp_trend = az.from_netcdf(OUT / "idata_temporal_disp_trend.nc")
    idata_np_trend = az.from_netcdf(OUT / "idata_num_persons_temporal_mean_trend.nc")
    idata_na_step_trend = az.from_netcdf(OUT / "idata_num_adults_temporal_mean_step_trend.nc")
    idata_C3 = az.from_netcdf(OUT / "idata_three_group_C3.nc")

    b_adult = pct_summary(idata_B, "beta_num_adults")
    b_minor = pct_summary(idata_B, "beta_num_minors")
    c_adult = pct_summary(idata_C, "beta_num_adults")
    c_era = idata_C.posterior["beta_adult_era"].values.flatten()
    c_era_mean, c_era_lo, c_era_hi = c_era.mean(), np.percentile(c_era, 3), np.percentile(c_era, 97)

    trend_pct = pct_summary(idata_temp_trend, "mean_trend")
    trend_13yr = np.exp(idata_temp_trend.posterior["mean_trend"].values.flatten() * 13)
    trend_13yr_stats = (trend_13yr.mean(), np.percentile(trend_13yr, 3), np.percentile(trend_13yr, 97))
    gamma_trend = idata_temp_disp_trend.posterior["gamma_trend"].values.flatten()
    gamma_trend_stats = (gamma_trend.mean(), np.percentile(gamma_trend, 3), np.percentile(gamma_trend, 97))

    np_trend = idata_np_trend.posterior["mean_trend"].values.flatten()
    np_trend_pct = ((np.exp(np_trend) - 1) * 100)
    np_trend_stats = (np_trend_pct.mean(), np.percentile(np_trend_pct, 3), np.percentile(np_trend_pct, 97))

    na_step = idata_na_step_trend.posterior["mean_step"].values.flatten()
    na_trend_pre = idata_na_step_trend.posterior["mean_trend_pre"].values.flatten()
    na_trend_post = idata_na_step_trend.posterior["mean_trend_post"].values.flatten()

    adult_c3 = pct_summary(idata_C3, "beta_adult")
    minor_c3 = pct_summary(idata_C3, "beta_minor")
    preadult_pre_c3 = pct_summary(idata_C3, "beta_preadult")
    preadult_post_c3 = pct_summary(idata_C3, "beta_preadult", extra="beta_preadult_era")
    era_draws_c3 = idata_C3.posterior["beta_preadult_era"].values.flatten()
    p_era_neg = (era_draws_c3 < 0).mean()

    images = {name: b64_img(f"{name}.png") for name in [
        "posterior_predictive_check", "era_effect_pre_post_1956", "doc_type_adult_minor_effects",
        "dossier_size_temporal_trend", "family_size_temporal_trend", "adults_binomial_p_per_year",
        "three_group_era_effect_corrected", "doc_type_3group_effects_by_period",
    ]}

    n_credible_diff = int((period_diff["p_post_lt_pre"] >= 0.95).sum())

    d1_pre_row = doc3_pre.loc[doc3_pre["doc_type"] == "D.1"].iloc[0]
    d1_post_row = doc3_post.loc[doc3_post["doc_type"] == "D.1"].iloc[0]
    d1_pre_str = f"{fmt_pct(d1_pre_row['pct_per_preadult_mean'])} {hdi_pct(d1_pre_row['pct_per_preadult_hdi_3'], d1_pre_row['pct_per_preadult_hdi_97'])}"
    d1_post_str = f"{fmt_pct(d1_post_row['pct_per_preadult_mean'])} {hdi_pct(d1_post_row['pct_per_preadult_hdi_3'], d1_post_row['pct_per_preadult_hdi_97'])}"
    d1_p_shrink_raw = fmt_pct(period_diff.loc[period_diff["doc_type"] == "D.1", "p_post_lt_pre"].item() * 100, 1)

    # --- Segmentation-uncertainty correction for the AGGREGATE num_docs models
    # (§3 temporal, §6 three-group) -- see aggregate_uncertainty.py. num_docs
    # totals only depend on segment BOUNDARIES, not document TYPE, so only the
    # start_page confusion matrix is relevant here (unlike §7, which also needs
    # the document-type confusion matrix).
    agg_corrected = pd.read_csv(UNC / "aggregate_num_docs_corrected.csv")
    agg_diff = agg_corrected["num_docs"] - agg_corrected["num_docs_raw"]
    agg_n_shift = int((agg_diff.abs() > 0.5).sum())
    agg_mean_shift = agg_diff.mean()

    idata_trend_corrected = az.from_netcdf(UNC / "idata_temporal_mean_trend_corrected.nc")
    trend_pct_corrected = pct_summary(idata_trend_corrected, "mean_trend")
    trend_13yr_corrected = np.exp(idata_trend_corrected.posterior["mean_trend"].values.flatten() * 13)
    trend_13yr_corrected_stats = (
        trend_13yr_corrected.mean(), np.percentile(trend_13yr_corrected, 3), np.percentile(trend_13yr_corrected, 97)
    )
    loo_temp_mean_corrected = loo_table("uncertainty/loo_compare_temporal_mean_corrected.csv")
    with open(UNC / "temporal_winner_corrected.txt") as f:
        temporal_winner_corrected = f.read().strip()

    idata_C3_corrected = az.from_netcdf(UNC / "idata_three_group_C3_corrected.nc")
    adult_c3_corrected = pct_summary(idata_C3_corrected, "beta_adult")
    minor_c3_corrected = pct_summary(idata_C3_corrected, "beta_minor")
    preadult_pre_c3_corrected = pct_summary(idata_C3_corrected, "beta_preadult")
    preadult_post_c3_corrected = pct_summary(idata_C3_corrected, "beta_preadult", extra="beta_preadult_era")
    era_draws_c3_corrected = idata_C3_corrected.posterior["beta_preadult_era"].values.flatten()
    p_era_neg_corrected = (era_draws_c3_corrected < 0).mean()
    loo_3group_corrected = loo_table("uncertainty/loo_compare_three_group_B3_C3_D3_corrected.csv")

    # --- Per-document-type joint (document-type + segmentation) point-mode
    # correction for §7 -- doc_type_three_groups_uncertainty.py --mode point,
    # already run; doc_type_period_diff_uncertainty.py's paired difference.
    point_full = pd.read_csv(UNC / "point_full.csv")
    point_pre = pd.read_csv(UNC / "point_pre1956.csv")
    point_post = pd.read_csv(UNC / "point_post1956.csv")
    point_period_diff = pd.read_csv(UNC / "point_period_diff.csv")
    cvp_point = pd.read_csv(UNC / "cvp_point.csv")
    n_credible_diff_point = int((point_period_diff["p_post_lt_pre"] >= 0.95).sum())
    n_credible_grow_point = int((point_period_diff["p_post_lt_pre"] <= 0.05).sum())
    shrink_types_point = point_period_diff.loc[point_period_diff["p_post_lt_pre"] >= 0.95, "doc_type"].tolist()
    grow_types_point = point_period_diff.loc[point_period_diff["p_post_lt_pre"] <= 0.05, "doc_type"].tolist()
    shrink_types_point_str = " and ".join(shrink_types_point) if shrink_types_point else "none"
    grow_types_point_str = " and ".join(grow_types_point) if grow_types_point else "none"

    raw_shrink_types = set(period_diff.loc[period_diff["p_post_lt_pre"] >= 0.95, "doc_type"])
    raw_shrink_types_str = " and ".join(sorted(raw_shrink_types)) if raw_shrink_types else "no type"
    kept_types = sorted(raw_shrink_types & set(shrink_types_point))
    dropped_types = sorted(raw_shrink_types - set(shrink_types_point))
    new_types = sorted(set(shrink_types_point) - raw_shrink_types)
    kept_types_str = " and ".join(kept_types) if kept_types else "none of them"
    dropped_types_str = " and ".join(dropped_types) if dropped_types else "none"

    # sentence branches on how the raw and point-corrected credible-shrink sets
    # relate -- written out rather than forced into one template, since which
    # branch holds is itself a substantive finding that can change with the data
    # (e.g. it used to be a 4-type raw set narrowing to 2; after fixing a stale
    # confusion-matrix bug the raw set turned out to be 1 type all along).
    if raw_shrink_types == set(shrink_types_point) and raw_shrink_types:
        shrink_narrative = (
            f"<strong>{kept_types_str}</strong> is the only type whose pre-adult effect credibly shrinks "
            f"after 1956, and stays the only one once jointly corrected for document-type and segmentation "
            f"uncertainty &mdash; the finding does not depend on trusting raw predicted counts."
        )
    elif not raw_shrink_types and not shrink_types_point:
        shrink_narrative = "No type shows a credible pre-adult shrinkage in either the raw or the corrected fit."
    elif not shrink_types_point:
        shrink_narrative = (
            f"Of the {len(raw_shrink_types)} type{'s' if len(raw_shrink_types) != 1 else ''} credible in the "
            f"raw fit ({raw_shrink_types_str}), <strong>none survive</strong> joint document-type and "
            f"segmentation correction &mdash; the raw-fit shrinkage isn't robust to classifier uncertainty "
            f"(see &sect;10)."
        )
    else:
        shrink_narrative = (
            f"Of the {len(raw_shrink_types)} type{'s' if len(raw_shrink_types) != 1 else ''} credible in the "
            f"raw fit ({raw_shrink_types_str}), <strong>{kept_types_str}</strong> survive{'s' if len(kept_types) == 1 else ''} "
            f"correction; <strong>{dropped_types_str}</strong> do{'es' if len(dropped_types) == 1 else ''} not "
            f"(see &sect;10)."
        )
        if new_types:
            shrink_narrative += (
                f" <strong>{' and '.join(new_types)}</strong> newly becomes credible once corrected, not "
                f"credible in the raw fit."
            )

    # short version of the same finding for the §8 synthesis bullet
    if kept_types:
        shrink_synthesis_phrase = f"{kept_types_str}'s credible pre-adult shrinkage holds up under correction too"
    else:
        shrink_synthesis_phrase = "the per-document-type pre-adult shrinkage seen in the raw fit does not survive correction"

    html = f"""<title>Dossier Size &amp; Family Composition Analysis</title>
<style>
:root {{
  --paper: #eef0ec;
  --surface: #f8f9f6;
  --surface-2: #ffffff;
  --ink: #1b1e1a;
  --ink-secondary: #52564e;
  --ink-muted: #868a7f;
  --rule: #d6dad0;
  --accent: #2464a8;
  --accent-soft: #dce8f2;
  --caution: #a8501f;
  --caution-soft: #f3e3d8;
  --retract: #8a2f2f;
  --retract-soft: #f3dcdc;
  --good: #2e7d4f;
  --font-serif: Charter, "Iowan Old Style", "Palatino Linotype", "Georgia", serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --paper: #14161a; --surface: #1b1e17; --surface-2: #20231c;
    --ink: #eef0ec; --ink-secondary: #b9bcae; --ink-muted: #7d8177; --rule: #33362f;
    --accent: #74a9de; --accent-soft: #223349;
    --caution: #d98a52; --caution-soft: #3a2a1e;
    --retract: #d97a7a; --retract-soft: #3a2222;
    --good: #5cb385;
    color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #14161a; --surface: #1b1e17; --surface-2: #20231c;
  --ink: #eef0ec; --ink-secondary: #b9bcae; --ink-muted: #7d8177; --rule: #33362f;
  --accent: #74a9de; --accent-soft: #223349;
  --caution: #d98a52; --caution-soft: #3a2a1e;
  --retract: #d97a7a; --retract-soft: #3a2222;
  --good: #5cb385;
  color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--paper); color: var(--ink); font-family: var(--font-serif); font-size: 18px; line-height: 1.6; margin: 0; padding: 0 24px 96px; }}
.page {{ max-width: 800px; margin: 0 auto; }}
header.masthead {{ max-width: 800px; margin: 0 auto; padding: 64px 0 28px; }}
.eyebrow {{ font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); margin: 0 0 14px; }}
h1 {{ font-size: 38px; line-height: 1.15; margin: 0 0 10px; text-wrap: balance; font-weight: 600; }}
.subtitle {{ color: var(--ink-secondary); font-size: 19px; max-width: 65ch; margin: 0 0 28px; }}
.status-banner {{ border: 1px solid var(--rule); background: var(--surface); border-left: 3px solid var(--accent); padding: 14px 18px; font-size: 15px; color: var(--ink-secondary); border-radius: 2px; }}
.status-banner code {{ font-family: var(--font-mono); font-size: 13px; background: var(--surface-2); padding: 1px 5px; border-radius: 3px; }}
section {{ max-width: 800px; margin: 0 auto; padding: 40px 0; border-top: 1px solid var(--rule); }}
section.wide-section {{ max-width: 1040px; }}
section.wide-section > .prose {{ max-width: 800px; margin: 0 auto; }}
.section-head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 22px; }}
.section-num {{ font-family: var(--font-mono); font-size: 14px; color: var(--accent); white-space: nowrap; }}
h2 {{ font-size: 27px; margin: 0; font-weight: 600; text-wrap: balance; }}
h3 {{ font-size: 18px; font-weight: 600; margin: 26px 0 10px; }}
p {{ margin: 0 0 16px; }}
.lede {{ font-size: 19px; color: var(--ink-secondary); }}
a {{ color: var(--accent); }}
strong {{ font-weight: 600; }}
.callout {{ background: var(--caution-soft); border-left: 3px solid var(--caution); padding: 14px 18px; border-radius: 2px; font-size: 16px; margin: 20px 0; }}
.callout .callout-label {{ font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--caution); display: block; margin-bottom: 6px; }}
.retracted {{ background: var(--retract-soft); border-left: 3px solid var(--retract); padding: 14px 18px; border-radius: 2px; font-size: 16px; margin: 20px 0; }}
.retracted .callout-label {{ font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--retract); display: block; margin-bottom: 6px; }}
figure {{ margin: 24px 0; }}
figure img {{ width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 3px; background: var(--surface-2); }}
figcaption {{ font-family: var(--font-mono); font-size: 12.5px; color: var(--ink-muted); margin-top: 8px; text-align: center; }}
.mono {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.hdi {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--ink-muted); font-size: 0.88em; }}
details {{ margin: 12px 0 8px; }}
details > summary {{ font-family: var(--font-mono); font-size: 13px; letter-spacing: 0.03em; color: var(--accent); cursor: pointer; list-style: none; padding: 8px 0; }}
details > summary::-webkit-details-marker {{ display: none; }}
details > summary::before {{ content: "\\25B8 "; }}
details[open] > summary::before {{ content: "\\25BE "; }}
.table-wrap {{ overflow-x: auto; margin: 10px 0 4px; border: 1px solid var(--rule); border-radius: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; background: var(--surface-2); }}
table.narrow {{ max-width: 620px; }}
th, td {{ text-align: left; padding: 8px 13px; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th {{ font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-muted); background: var(--surface); position: sticky; top: 0; }}
tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--accent-soft); }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; }}
.dot-yes {{ background: var(--good); }}
.dot-no {{ background: var(--ink-muted); opacity: 0.4; }}
.synthesis-list {{ padding: 0; margin: 0; list-style: none; }}
.synthesis-list li {{ padding: 16px 0; border-bottom: 1px solid var(--rule); }}
.synthesis-list li:last-child {{ border-bottom: none; }}
.synthesis-list .tag {{ font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent); display: block; margin-bottom: 6px; }}
.refs {{ font-size: 15px; }}
.refs ol {{ padding-left: 22px; }}
.refs li {{ margin-bottom: 12px; color: var(--ink-secondary); }}
.cite {{ font-family: var(--font-mono); font-size: 0.72em; color: var(--accent); vertical-align: super; text-decoration: none; margin-left: 1px; }}
.timeline {{ font-family: var(--font-mono); font-size: 13px; color: var(--ink-muted); border-left: 2px solid var(--rule); padding-left: 14px; margin: 20px 0; }}
.timeline div {{ margin-bottom: 8px; }}
.timeline strong {{ color: var(--ink); }}
footer {{ max-width: 800px; margin: 40px auto 0; padding-top: 28px; border-top: 1px solid var(--rule); font-size: 13.5px; color: var(--ink-muted); font-family: var(--font-mono); }}
footer code {{ display: block; background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 12px 14px; margin: 10px 0; overflow-x: auto; white-space: pre; color: var(--ink); }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>

<header class="masthead">
  <p class="eyebrow">Migration Dossiers &middot; Size &amp; Composition Analysis</p>
  <h1>What predicts dossier size, and a correction along the way</h1>
  <p class="subtitle">A family of Bayesian models across 1,307 dossiers (1952&ndash;1965) tracing whether unit composition and time predict dossier size, plus a classifier-uncertainty recheck &mdash; including an initial finding that a later domain-expert conversation overturned, kept visible deliberately.</p>
  <div class="status-banner">
    This report documents the full investigative arc as it happened, not just the final answer &mdash; &sect;1&ndash;2 are <strong>superseded</strong> by &sect;6&ndash;7 (and, on predictive grounds rather than a data bug, by &sect;8's grid). &sect;3, &sect;6, and &sect;7 additionally carry a classifier-uncertainty check (point estimate, see &sect;10): the aggregate findings hold up essentially unchanged, the per-document-type story in &sect;7 narrows. Regenerate end to end with <code>make -C scripts/dossier_size_model all</code> (~1.5&ndash;2 hours; individual targets in <code>make help</code>); the uncertainty check is a separate step, <code>python3 scripts/dossier_size_model/aggregate_uncertainty.py</code>.
  </div>
  <div class="timeline">
    <div><strong>&sect;1&ndash;2</strong> originally: num_adults defined by an era-switching 18+/16+ threshold &rarr; "adult effect shrinks after 1956" (data bug, since fixed; refit here on corrected data)</div>
    <div><strong>&sect;3&ndash;5</strong> temporal trend and family-composition side investigations</div>
    <div><strong>&sect;6</strong> domain expert: the threshold never moved &mdash; a separate pre-adult (16-17) paperwork rule changed instead &rarr; corrected model</div>
    <div><strong>&sect;7</strong> which specific document types drove it &mdash; and what that reveals about selection-officer discretion</div>
    <div><strong>&sect;8</strong> all seven num_docs models on one footing &mdash; how many age groups, and does era matter</div>
  </div>
</header>

<section>
  <div class="section-head"><span class="section-num">&sect;0</span><h2>Method, throughout</h2></div>
  <p>Every model here is a negative binomial GLM<a href="#ref-1" class="cite">[1]</a> with a log link, fit with NUTS<a href="#ref-2" class="cite">[2]</a> via PyMC<a href="#ref-3" class="cite">[3]</a>, using hierarchical partial pooling<a href="#ref-4" class="cite">[4]</a> for year-level (and, where noted, document-type-level) effects so sparse years or types borrow strength rather than being estimated in isolation. Structural questions (which predictor form, whether a temporal trend is real, which likelihood is appropriate) are answered by comparing models via LOO cross-validation<a href="#ref-5" class="cite">[5]</a> rather than by fitting one model and trusting it.</p>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;1</span><h2>Does composition predict dossier size?</h2></div>
    <div class="retracted">
      <span class="callout-label">Superseded by &sect;6</span>
      This section's "adult"/"minor" covariates originally used an era-switching (18+ pre-1956, 16+ post-1956) definition -- later found to be a data-generation bug, not a real policy change. <code>ages_num_docs_pages.tsv</code> has since been corrected to a constant 18+ threshold throughout, and the models below are refit on that corrected data. Still superseded by &sect;6's three-group model, but the reason has changed: it's no longer a data bug, it's that two groups is simply a coarser specification than three, and predicts decisively worse (see &sect;8's model-comparison grid). Kept as the "(2) adult/minor" row of that grid, not as a historical reproduction of a since-fixed bug.
    </div>
    <p class="lede">Three models: A (num_docs ~ num_persons), B (~ num_adults + num_minors), C (B + adult&times;era interaction).</p>
    <p><strong>B beats A decisively</strong> (elpd diff {fmt(loo_ab.loc['A_persons','elpd_diff'],1)}, dse {fmt(loo_ab.loc['A_persons','dse'],1)}) &mdash; splitting persons into adults/minors predicts dossier size far better than raw group size: <strong>{fmt_pct(b_adult[0])} {hdi_pct(b_adult[1],b_adult[2])}</strong> more documents per adult vs. <strong>{fmt_pct(b_minor[0])} {hdi_pct(b_minor[1],b_minor[2])}</strong> per minor.</p>
    <p><strong>C does not beat B</strong> (elpd diff {fmt(loo_bc.loc['B_adults_minors','elpd_diff'],1)}, dse {fmt(loo_bc.loc['B_adults_minors','dse'],1)} &mdash; not decisive): the adult&times;era interaction is null, {fmt_pct(c_era_mean)} {hdi_pct(c_era_lo,c_era_hi)}, no era effect on adults at all. This matches &sect;6's finding that the legal adult threshold never moved. Worth being precise about what changed: the <em>original</em> version of this model, fit on the era-switching data before it was corrected, showed a spuriously credible-looking shrink -- refitting on clean data doesn't just make that finding statistically indistinguishable from noise, it makes the effect disappear entirely, coefficient and LOO comparison both agreeing there's nothing here.</p>
    <figure>
      <img src="{images['posterior_predictive_check']}" alt="Posterior predictive check for the dossier-size model" />
      <figcaption>posterior predictive check (density overlay and predicted-vs-observed)</figcaption>
    </figure>
    <figure>
      <img src="{images['era_effect_pre_post_1956']}" alt="No credible pre/post-1956 adult effect shift, on corrected data" />
      <figcaption>corrected data: no adult-effect shift at 1956 (contrast with &sect;6's pre-adult-specific shift)</figcaption>
    </figure>
    <details><summary>Model A vs. B (LOO)</summary>{loo_html(loo_ab)}</details>
    <details><summary>Model B vs. C (LOO)</summary>{loo_html(loo_bc)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;2</span><h2>Per-document-type breakdown (two-group)</h2></div>
    <div class="retracted"><span class="callout-label">Superseded by &sect;7</span>Same corrected constant-threshold "adult"/"minor" covariates as &sect;1, same reason for being superseded: a coarser two-group specification than &sect;7's three-group, per-period version, not a data-bug artifact.</div>
    <p>Partially-pooled hierarchical model across the {n_doc_types} tracked document types, adult and minor coefficients per type. D.2, NAMA agreement, and Testimonial labour show the largest adult effects (36&ndash;60% more documents per adult); minor effects are near zero for every type.</p>
    <figure>
      <img src="{images['doc_type_adult_minor_effects']}" alt="Per-document-type adult vs minor effects, two-group model" />
      <figcaption>per-type adult/minor effects (94% HDI)</figcaption>
    </figure>
    <details><summary>Full table ({n_doc_types} types)</summary>{doc_type_2group_table(occ_2g)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;3</span><h2>Is dossier size changing over time?</h2></div>
    <p class="lede">Four candidate mean structures (iid / trend / step at 1956 / step+trend) and three dispersion structures (constant / trend / unconstrained per-year), net of composition.</p>
    <p><strong>Mean</strong>: all four structures are statistically indistinguishable by LOO (largest elpd diff {fmt(loo_temp_mean.iloc[-1]["elpd_diff"],2)}, well under 1 dse) &mdash; no decisive evidence a directional trend beats plain year-to-year noise. The trend coefficient itself is nominally credible: <strong>{fmt_pct(trend_pct[0])} {hdi_pct(trend_pct[1],trend_pct[2])} per year</strong>, compounding to <strong>{fmt(trend_13yr_stats[0],2)}&times;</strong> {hdi(trend_13yr_stats[1],trend_13yr_stats[2],2)} over 1952&rarr;1965 &mdash; dossiers got slightly <em>larger</em> over time, the opposite of what a naive reading of &sect;1 might suggest.</p>
    <p><strong>Dispersion</strong>: decisive, and not what a first pass suggested. A smooth trend beats constant dispersion (elpd diff {fmt(loo_temp_disp.loc['trend_disp','elpd_diff'],1)}), but an <strong>unconstrained per-year estimate beats the trend by a further {fmt(loo_temp_disp.loc['iid_disp','elpd_diff']-loo_temp_disp.loc['trend_disp','elpd_diff'],1)} elpd</strong> &mdash; dispersion genuinely varies by year but isn't a smooth trend; one influential 1965 dossier (7 persons, 78 documents) was flagged by the Pareto-k diagnostic as disproportionately driving the naive trend-shaped read.</p>
    <div class="callout">
      <span class="callout-label">Segmentation-uncertainty check (point estimate)</span>
      num_docs here is the raw predicted segment count, which assumes the start_page classifier's boundaries are ground truth. Refit with each dossier's num_docs replaced by its segmentation-corrected expected value (point_correction_start_page, same posterior-mean approximation as &sect;7's point-mode correction &mdash; {agg_n_shift}/1307 dossiers shift by more than half a document, {agg_mean_shift:+.2f} documents/dossier on average): the trend barely moves, <strong>{fmt_pct(trend_pct_corrected[0])} {hdi_pct(trend_pct_corrected[1],trend_pct_corrected[2])} per year</strong> (vs. {fmt_pct(trend_pct[0])} {hdi_pct(trend_pct[1],trend_pct[2])} raw), compounding to {fmt(trend_13yr_corrected_stats[0],2)}&times; {hdi(trend_13yr_corrected_stats[1],trend_13yr_corrected_stats[2],2)} over 1952&rarr;1965. The mean-structure LOO ranking is unchanged (<strong>{temporal_winner_corrected}</strong> still ranks first, largest elpd diff {fmt(loo_temp_mean_corrected.iloc[-1]["elpd_diff"],2)}, same "statistically indistinguishable" verdict as raw). This section's temporal story is not an artifact of segmentation noise.
    </div>
    <figure>
      <img src="{images['dossier_size_temporal_trend']}" alt="Dossier size over time: fitted mean trend and per-year spread" />
      <figcaption>mean trend (left) and per-year spread, no smooth trend imposed (right)</figcaption>
    </figure>
    <details><summary>Mean structure (LOO)</summary>{loo_html(loo_temp_mean)}</details>
    <details><summary>Dispersion structure (LOO)</summary>{loo_html(loo_temp_disp)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;4</span><h2>Is family composition itself changing over time?</h2></div>
    <p class="lede">Same iid/trend/step/step_trend &times; constant/trend/iid-dispersion comparison, now with num_persons and num_adults as the outcome.</p>
    <p><strong>num_persons</strong>: mean structures indistinguishable (largest elpd diff {fmt(loo_np_mean.iloc[-1]['elpd_diff'],2)}); trend coefficient borderline, {fmt_pct(np_trend_stats[0])} {hdi_pct(np_trend_stats[1],np_trend_stats[2])} per year &mdash; a weak, inconclusive hint of decline. Dispersion: iid beats constant ({fmt(loo_np_disp.loc['constant_disp','elpd_diff'],1)} elpd) and beats trend too &mdash; real but unstructured year-to-year heterogeneity, same pattern as &sect;3.</p>
    <p><strong>num_adults</strong>: no credible mean structure at all &mdash; not even the nominal LOO "winner" (step_trend) has an individually credible coefficient (step {fmt_pct(na_step.mean())}, pre-trend {fmt_pct(na_trend_pre.mean())}, post-trend {fmt_pct(na_trend_post.mean())}, all HDIs crossing zero). Family units have been remarkably stable in adult count across the whole period.</p>
    <figure>
      <img src="{images['family_size_temporal_trend']}" alt="num_persons and num_adults over time" />
      <figcaption>num_persons (top) and num_adults (bottom), mean and spread, per year</figcaption>
    </figure>
    <details><summary>num_persons mean structure (LOO)</summary>{loo_html(loo_np_mean)}</details>
    <details><summary>num_persons dispersion structure (LOO)</summary>{loo_html(loo_np_disp)}</details>
    <details><summary>num_adults mean structure (LOO)</summary>{loo_html(loo_na_mean)}</details>
    <details><summary>num_adults dispersion structure (LOO)</summary>{loo_html(loo_na_disp)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;5</span><h2>num_adults spread: NB breaks, Binomial resolves it</h2></div>
    <p>num_adults is under-dispersed relative to Poisson in <em>every single year</em> (variance/mean ratio 0.10&ndash;0.67) &mdash; negative binomial can only add variance beyond Poisson, so it structurally cannot fit this, and an NB dispersion analysis on num_adults produced a flat, uninformative fit disconnected from the raw data.</p>
    <p>Reframing as <strong>num_adults ~ Binomial(num_persons, p)</strong> &mdash; the natural model, since an adult count is "successes out of num_persons trials" &mdash; resolves it cleanly: <strong>plain Binomial beats Beta-Binomial by {fmt(loo_bb.loc['beta_binomial','elpd_diff'],1)} elpd</strong>, meaning no evidence of extra-binomial dispersion. Once correctly conditioned on unit size, ordinary binomial sampling variance fully explains num_adults' spread &mdash; there is no separate "spread" question left to ask. Mean structure (logit p over time): again indistinguishable by LOO, trend nominally wins with a barely-credible coefficient.</p>
    <figure>
      <img src="{images['adults_binomial_p_per_year']}" alt="Fitted vs observed adult proportion per year, Binomial model" />
      <figcaption>fitted adult probability per year vs. observed proportion</figcaption>
    </figure>
    <details><summary>Binomial vs. Beta-Binomial (LOO)</summary>{loo_html(loo_bb)}</details>
    <details><summary>Mean structure, given Binomial (LOO)</summary>{loo_html(loo_binom_mean)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;6</span><h2>The correction: three real groups, not two</h2></div>
    <div class="callout">
      <span class="callout-label">Domain-expert input</span>
      The legal adult threshold was constant at <strong>18+</strong> throughout 1952&ndash;1965 &mdash; it never moved to 16+. What changed in 1956 was a separate administrative requirement: 16-17 year-olds (never legally "adult") had to submit their own approval paperwork, and the amount of that paperwork changed. &sect;1&ndash;2's "adult effect shrinks" finding was a mechanistically wrong (if numerically similar-looking) account of what's actually a pre-adult-specific effect.
    </div>
    <p>Three groups built directly from num_18+ and num_16+: minor (&lt;16), pre-adult (16-17, own paperwork whose requirement changed in 1956), adult (18+, constant definition and constant requirement). Three models: B3 (no era interaction), C3 (+ pre-adult&times;era, the domain-expert account), D3 (+ era interaction on all three groups, a robustness check).</p>
    <p><strong>C3's era-interaction coefficient is credible, matching the domain expert's account.</strong> Adult effect is <strong>stable across 1956</strong>: {fmt_pct(adult_c3[0])} {hdi_pct(adult_c3[1],adult_c3[2])}, with D3's adult&times;era term not credible (HDI includes 0). The pre-adult effect drops credibly: <strong>{fmt_pct(preadult_pre_c3[0])} {hdi_pct(preadult_pre_c3[1],preadult_pre_c3[2])} pre-1956 &rarr; {fmt_pct(preadult_post_c3[0])} {hdi_pct(preadult_post_c3[1],preadult_post_c3[2])} post-1956</strong> (P(decrease) = {fmt_pct(p_era_neg*100,1)}). Minor effect negligible throughout ({fmt_pct(minor_c3[0])} {hdi_pct(minor_c3[1],minor_c3[2])}).</p>
    <p>Neither direction of added complexity earns a decisive predictive edge over C3 by LOO, though. D3 edges out C3 by only {fmt(loo_3group.loc['C3_preadult_era','elpd_diff'],1)} elpd (dse {fmt(loo_3group.loc['C3_preadult_era','dse'],1)}) &mdash; not decisive, so the extra era terms on minor and adult aren't earning their keep. In the other direction, C3 itself edges out the simpler B3 (no era term at all) by only {fmt(loo_b3_c3.loc['B3_three_group','elpd_diff'],1)} elpd (dse {fmt(loo_b3_c3.loc['B3_three_group','dse'],1)}) &mdash; also not decisive. So the case for the pre-adult&times;era term specifically rests on its own credible interval above (P(decrease) = {fmt_pct(p_era_neg*100,1)}), not on a predictive-accuracy win over B3: with ~1,300 count-noisy dossiers, LOO comparison is a conservative test, and a coefficient can be credible on its own terms without moving overall predictive fit enough to register as decisive. Both readings point the same direction (the era-specific pre-adult drop is real), they just rest on different kinds of evidence, worth keeping distinct rather than treating LOO as having settled it.</p>
    <div class="callout">
      <span class="callout-label">Segmentation-uncertainty check (point estimate)</span>
      Refit C3 with each dossier's num_docs segmentation-corrected (same point-mode correction as &sect;3): adult <strong>{fmt_pct(adult_c3_corrected[0])} {hdi_pct(adult_c3_corrected[1],adult_c3_corrected[2])}</strong>, pre-adult <strong>{fmt_pct(preadult_pre_c3_corrected[0])} {hdi_pct(preadult_pre_c3_corrected[1],preadult_pre_c3_corrected[2])} &rarr; {fmt_pct(preadult_post_c3_corrected[0])} {hdi_pct(preadult_post_c3_corrected[1],preadult_post_c3_corrected[2])}</strong> (P(decrease) = {fmt_pct(p_era_neg_corrected*100,1)}), minor <strong>{fmt_pct(minor_c3_corrected[0])} {hdi_pct(minor_c3_corrected[1],minor_c3_corrected[2])}</strong> &mdash; all three within a point or two of the raw estimates above, and D3 vs. C3 stays similarly non-decisive ({fmt(loo_3group_corrected.loc['C3_preadult_era','elpd_diff'],1)} elpd, dse {fmt(loo_3group_corrected.loc['C3_preadult_era','dse'],1)}). This section's headline finding is not an artifact of segmentation noise either.
    </div>
    <figure>
      <img src="{images['three_group_era_effect_corrected']}" alt="Corrected: adult effect stable, pre-adult effect shrinks at 1956" />
      <figcaption>corrected: adult stable across 1956, pre-adult effect shrinks (three-group model C3)</figcaption>
    </figure>
    <details><summary>B3 vs. C3 vs. D3 (LOO)</summary>{loo_html(loo_3group)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;7</span><h2>Which document types actually drove it &mdash; and what that reveals about officer discretion</h2></div>
    <p>Same three-group model, partially pooled across the {n_doc_types} document types, fit separately for the full period and each era.</p>
    <p>The aggregate pre-adult shrinkage from &sect;6 is concentrated in <strong>{n_credible_diff} of {n_doc_types} types</strong>: {raw_shrink_types_str} show{'s' if n_credible_diff == 1 else ''} P(post-1956 &lt; pre-1956) &ge; 95% (see the paired-difference table below). The other {n_doc_types - n_credible_diff} show no credible change either direction &mdash; the 1956 reform reads as targeted at specific forms, not a blanket reduction.</p>
    <div class="callout">
      <span class="callout-label">Domain-expert input, revised</span>
      An earlier draft of this report flagged a mismatch here against the domain expert's account and speculated it might be a swapped or inconsistent D.1/D.2 label mapping. That's been checked and ruled out &mdash; the label mapping is correct, D.1 and D.2 are <strong>not</strong> swapped. The domain expert's original account (pre-adults filed only D.1 after 1956, with D.2/DM.1/NAMA no longer required) was itself the part that needed revising. The corrected account: pre-adults were <em>not</em> required to submit D.1 &mdash; selection officers instead required D.2, DM.1, and NAMA agreement from them. This reflects real discretion selection officers had over what to demand from young applicants: the same officers are known, from other case files, to have treated some 14&ndash;15 year-olds (below even the pre-adult threshold) as employable and required them to complete the same paperwork as older applicants. Read against this corrected account, the data match precisely rather than conflicting: <strong>D.1 shrinks hardest</strong> (P={d1_p_shrink_raw}) &mdash; the 1956 reform dropped it specifically for pre-adults, its per-pre-adult effect falling from a credible {d1_pre_str} pre-1956 to a non-credible {d1_post_str} post-1956 (see the paired-difference table below) &mdash; while <strong>D.2, DM.1, and NAMA stay flat or rise</strong>, consistent with officers continuing to require them throughout. (This is the raw-count reading; see below for how it holds up once corrected for classifier uncertainty.)
    </div>
    <figure>
      <img src="{images['doc_type_3group_effects_by_period']}" alt="Per-document-type minor/pre-adult/adult effects, full period, pre-1956, post-1956" />
      <figcaption>per-type effects, full period vs. pre-1956 vs. post-1956</figcaption>
    </figure>
    <details><summary>Full period ({n_doc_types} types)</summary>{doc_type_3group_table(doc3_full)}</details>
    <details><summary>Pre-1956 ({n_doc_types} types)</summary>{doc_type_3group_table(doc3_pre)}</details>
    <details><summary>Post-1956 ({n_doc_types} types)</summary>{doc_type_3group_table(doc3_post)}</details>
    <details><summary>Pre/post-1956 paired difference, pre-adult effect ({n_credible_diff}/{n_doc_types} credible)</summary>{period_diff_table(period_diff)}</details>
    <h3>Classifier-uncertainty-corrected version (point estimate)</h3>
    <p>Everything above uses raw predicted counts, which assume both classifiers (document-type and start_page) are ground truth. Refit with each dossier's per-type counts jointly corrected for document-type confusion and segmentation uncertainty (doc_type_three_groups_uncertainty.py <code>--mode point</code>, deconvolving via each confusion matrix's posterior mean &mdash; corrects the average bias, but not a full multiple-imputation treatment of the classifiers' own posterior uncertainty; see that script's docstring for the rigorous <code>--mode mi</code> GPU path).</p>
    <p>{shrink_narrative} Separately, <strong>{grow_types_point_str}</strong> newly shows a <em>credible increase</em> (P(grows) = {fmt_pct((1-point_period_diff.loc[point_period_diff['doc_type']==grow_types_point[0],'p_post_lt_pre'].item())*100,1)}) once corrected &mdash; not credible in the raw fit &mdash; consistent with the revised domain-expert account that officers leaned on D.2 (among others) for pre-adults, and did so increasingly after 1956.</p>
    <details><summary>Full period, corrected ({n_doc_types} types)</summary>{doc_type_3group_table(point_full)}</details>
    <details><summary>Pre-1956, corrected ({n_doc_types} types)</summary>{doc_type_3group_table(point_pre)}</details>
    <details><summary>Post-1956, corrected ({n_doc_types} types)</summary>{doc_type_3group_table(point_post)}</details>
    <details><summary>Pre/post-1956 paired difference, corrected ({n_credible_diff_point}/{n_doc_types} credible shrink, {n_credible_grow_point}/{n_doc_types} credible grow)</summary>{period_diff_table(point_period_diff)}</details>
    <h3>Count vs. presence specification check</h3>
    <p>For each type, is the pre-adult predictor better modeled as a linear count (each extra pre-adult adds more documents) or presence/absence (the form is filed once, or not, regardless of count)? Every difference is small relative to its standard error (largest: D.2, favoring count by {fmt(cvp.loc[cvp['doc_type']=='D.2','elpd_diff_presence_minus_count'].item(),1)} elpd, se {fmt(cvp.loc[cvp['doc_type']=='D.2','se_diff'].item(),1)}, ~1.6 SE) &mdash; not decisive for any type, and doesn't explain the mismatch above.</p>
    <details><summary>Full table ({n_doc_types} types)</summary>{count_vs_presence_table(cvp)}</details>
    <p>Rerun on the classifier-uncertainty-corrected counts: still not decisive for any type (largest: {cvp_point.loc[cvp_point['elpd_diff_presence_minus_count'].abs().idxmax(),'doc_type']}, {fmt(cvp_point['elpd_diff_presence_minus_count'].abs().max()/cvp_point.loc[cvp_point['elpd_diff_presence_minus_count'].abs().idxmax(),'se_diff'],1)} SE) &mdash; the correction doesn't surface a count-vs-presence distinction the raw fit missed.</p>
    <details><summary>Full table, corrected ({n_doc_types} types)</summary>{count_vs_presence_table(cvp_point)}</details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;8</span><h2>The full grid: how many groups, and does era matter?</h2></div>
    <p class="lede">&sect;1's A/B/C and &sect;6's B3/C3(/D3) answer two separate questions -- how many age groups the persons in a dossier are split into, and whether there's an era interaction -- but were never compared side by side on one footing until now. All seven models share the same outcome (num_docs), same likelihood, same 1307 dossiers, and (since the era-switching data bug was fixed) the same underlying 18+ adult threshold, so they're directly LOO-comparable:</p>
    <div class="table-wrap"><table class="data-table narrow"><thead><tr><th></th><th>no era interaction</th><th>with era interaction</th></tr></thead><tbody>
      <tr><td>(1) num_persons</td><td class="mono">A: {fmt(loo_grid.loc['A_persons','elpd_loo'],1)}</td><td class="mono">A_era: {fmt(loo_grid.loc['A_era_persons_era','elpd_loo'],1)}</td></tr>
      <tr><td>(2) adult(18+)/minor</td><td class="mono">B: {fmt(loo_grid.loc['B_adults_minors','elpd_loo'],1)}</td><td class="mono">C: {fmt(loo_grid.loc['C_adults_minors_era','elpd_loo'],1)}</td></tr>
      <tr><td>(3) adult/pre-adult/minor</td><td class="mono">B3: {fmt(loo_grid.loc['B3_three_group','elpd_loo'],1)}</td><td class="mono">C3: {fmt(loo_grid.loc['C3_preadult_era','elpd_loo'],1)}</td></tr>
    </tbody></table></div>
    <p class="lede" style="margin-top:14px">(elpd_loo, higher = better out-of-sample fit; see the full table below for rank/dse.)</p>
    <p><strong>Number of groups is what matters.</strong> Going from 1 to 2 groups (A/A_era &rarr; B/C) gains roughly 80 elpd &mdash; the same decisive jump &sect;1 already reports for A vs. B. Going from 2 to 3 groups (B/C &rarr; B3/C3) gains a further ~26 elpd, similarly decisive (B trails the grid's best model (C3) by {fmt(loo_grid.loc['B_adults_minors','elpd_diff'],1)} elpd; B3 trails by only {fmt(loo_grid.loc['B3_three_group','elpd_diff'],1)}). Each step of splitting persons into a finer age breakdown earns its keep; nothing about this is close.</p>
    <p><strong>Era interaction, on its own, essentially never does.</strong> Within every row, the era-interaction column is statistically indistinguishable from its no-era neighbor: A vs. A_era (&sect;1-style comparison, elpd diff {fmt(loo_a_aera.loc['A_persons','elpd_diff'],1)}, dse {fmt(loo_a_aera.loc['A_persons','dse'],1)}), B vs. C (&sect;1: elpd diff {fmt(loo_bc.loc['B_adults_minors','elpd_diff'],1)}, dse {fmt(loo_bc.loc['B_adults_minors','dse'],1)}), B3 vs. C3 (&sect;6: elpd diff {fmt(loo_b3_c3.loc['B3_three_group','elpd_diff'],1)}, dse {fmt(loo_b3_c3.loc['B3_three_group','dse'],1)}). The one exception is a narrow one: A_era's persons&times;era coefficient is itself {'credible' if a_era_credible else 'not credible'} ({fmt_pct(a_era_coef[0])} {hdi_pct(a_era_coef[1],a_era_coef[2])}) at the coarsest, unsplit level -- but that signal disappears entirely once persons is split into adults/minors (&sect;1's B vs. C, adult&times;era null). Read together with &sect;6's pre-adult&times;era finding, the pattern is consistent: a blanket "does era matter" question gets a no at every level of grouping, but a narrow, theoretically-motivated question ("does era matter specifically for pre-adults") gets a credible yes at the coefficient level, even though it likewise doesn't win decisively on LOO (&sect;6). Coarse era interactions and the one substantively-motivated one behave differently, and only the grid makes that contrast visible.</p>
    <p>D3 (era interaction on all three groups) is included in the second table below but deliberately left out of the grid above -- it isn't a second "with era" cell for row (3), it's a broader robustness check on whether era matters for <em>any</em> group, not just pre-adults (see &sect;6). It doesn't decisively beat C3 either ({fmt(loo_3group.loc['C3_preadult_era','elpd_diff'],1)} elpd, dse {fmt(loo_3group.loc['C3_preadult_era','dse'],1)}), for the same reason: blanket era interactions don't earn their keep, only the targeted pre-adult one does at the coefficient level.</p>
    <details><summary>The grid, six models (LOO)</summary>{loo_html(loo_grid)}</details>
    <details><summary>Grid + D3, seven models (LOO)</summary>{loo_html(loo_grid_d3)}</details>
  </div>
</section>

<section>
  <div class="section-head"><span class="section-num">&sect;9</span><h2>Putting it together</h2></div>
  <ul class="synthesis-list">
    <li><span class="tag">The corrected headline</span>Dossier size scales with adult count (constant ~24% more documents per adult, both eras) and, specifically, with pre-adults (16-17) whose own paperwork requirement dropped credibly after 1956 (~30% &rarr; ~17% per pre-adult). Minors contribute almost nothing. This is a mechanistically correct account; the earlier "adult effect shrinks" finding (&sect;1) was an artifact of a mis-specified era-switching adult definition -- one now fixed at the data level, after which even the two-group model agrees there's no adult-era effect at all (&sect;1, &sect;8).</li>
    <li><span class="tag">Composition is stable, paperwork isn't</span>Neither num_persons nor num_adults shows a credible temporal trend (&sect;4) &mdash; the changing document burden (&sect;3's modest but credible trend, and &sect;6's 1956 pre-adult shift) reflects administrative practice changing, not the population of migrating families changing.</li>
    <li><span class="tag">The recurring lesson: check the null model</span>Three separate points in this investigation flipped on closer inspection: dispersion "trending" turned out to be unstructured noise dominated by one outlier (&sect;3); num_adults' "spread problem" turned out to be a wrong-likelihood problem, resolved by switching to Binomial (&sect;5); and the "adult effect shrinks" finding turned out to be a definitional artifact, confirmed twice over -- first by the three-group correction (&sect;6), then again when the underlying data bug itself was fixed and even the original two-group model stopped showing it (&sect;1, &sect;8). In each case the fix was to compare against a more general or more correctly-specified alternative via LOO (or, for &sect;1, simply against corrected data) rather than trust the first model that ran.</li>
    <li><span class="tag">Officer discretion, not a labeling error</span>&sect;7's document-type pattern (D.1 drops out for pre-adults after 1956; D.2/DM.1/NAMA stay flat or rise) was initially flagged as a possible D.1/D.2 label-mapping mismatch against the domain expert's account. Checked and ruled out. The domain expert's account has since been revised: pre-adults were never required to submit D.1 &mdash; selection officers instead required D.2, DM.1, and NAMA agreement, reflecting real discretion officers had over what to demand from young applicants (also seen in some 14&ndash;15 year-olds, below the pre-adult threshold, being treated as employable and required to file the same forms).</li>
    <li><span class="tag">Classifier uncertainty checked, not just assumed away</span>Every model above treats predicted document counts as ground truth. Rechecked against document-type and segmentation classifier uncertainty (point estimate; &sect;3, &sect;6, &sect;7): the aggregate findings (temporal trend, three-group adult/pre-adult effects) barely move, but {shrink_synthesis_phrase}. D.2 newly shows a credible <em>increase</em> for pre-adults once corrected, not credible in the raw fit &mdash; reinforcing the revised officer-discretion account above.</li>
  </ul>
</section>

<section>
  <div class="section-head"><span class="section-num">&sect;10</span><h2>Limitations</h2></div>
  <p>&sect;1&ndash;2 are refit on the same corrected data as everything else and are no longer historical reproductions of a bug -- but they're still superseded, now simply as the coarser "(2) adult/minor" row of &sect;8's grid, decisively beaten by the three-group specification in &sect;6&ndash;7 on predictive grounds.</p>
  <p>Several LOO comparisons throughout are close (elpd differences within 1&ndash;2 dse), particularly the mean-structure comparisons in &sect;3&ndash;4, every era-interaction comparison in &sect;8's grid (A vs. A_era, B vs. C, B3 vs. C3, and D3 vs. C3 in &sect;6) &mdash; treat "no decisive winner" as a real answer (the data don't support extra structure) rather than a failure to find one. The one era-related exception is C3's pre-adult&times;era <em>coefficient</em>, which is credible even though C3 doesn't decisively out-predict B3 on LOO (&sect;6, &sect;8) -- a coefficient-level finding, not a model-comparison one, and worth not conflating with the LOO verdicts around it.</p>
  <p>Sample size drops sharply in 1961&ndash;1965 (4&ndash;17 dossiers/year vs. 55&ndash;293 in earlier years), and the pre/post-1956 split in &sect;7 necessarily halves the already-modest per-document-type data &mdash; corrected intervals are correspondingly wide for some types.</p>
  <p>&sect;7's document-type story rests on the domain expert's revised account (officer discretion over pre-adult paperwork, not a fixed D.1-only requirement) rather than a written policy document &mdash; plausible and consistent with the data, but still an oral-history reconstruction of 1950s&ndash;60s office practice.</p>
  <p>The classifier-uncertainty corrections in &sect;3, &sect;6, and &sect;7 are all <strong>point estimates</strong> (deconvolving via each confusion matrix's posterior mean): they correct the average bias from segmentation and document-type misclassification, but still understate the classifiers' own posterior uncertainty, and &sect;3/&sect;6's segmentation-only correction doesn't touch the document-type confusion that &sect;7 alone is exposed to. A full multiple-imputation treatment (resampling both confusion matrices per posterior draw and refitting per imputation, the same GPU-scale approach already used in the dossier-composition analysis) would give honest intervals; not run here for the aggregate models, see aggregate_uncertainty.py's docstring.</p>
</section>

<section class="refs">
  <div class="section-head"><span class="section-num">&sect;11</span><h2>References</h2></div>
  <ol>
    <li id="ref-1">Hilbe, J. M. (2011). <em>Negative Binomial Regression</em> (2nd ed.). Cambridge University Press. &mdash; the count-regression family used for every dossier-size and document-type model.</li>
    <li id="ref-2">Hoffman, M. D., &amp; Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. <em>Journal of Machine Learning Research</em>, 15(1), 1593&ndash;1623.</li>
    <li id="ref-3">Abril-Pla, O., et al. (2023). PyMC: a modern, and comprehensive probabilistic programming framework in Python. <em>PeerJ Computer Science</em>, 9, e1516.</li>
    <li id="ref-4">Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., &amp; Rubin, D. B. (2013). <em>Bayesian Data Analysis</em> (3rd ed.). CRC Press. &mdash; hierarchical partial pooling, used for year-level and document-type-level effects throughout.</li>
    <li id="ref-5">Vehtari, A., Gelman, A., &amp; Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. <em>Statistics and Computing</em>, 27(5), 1413&ndash;1432. &mdash; LOO-CV and the Pareto-k diagnostic, used for essentially every model-comparison decision in this report.</li>
    <li id="ref-6">McCullagh, P., &amp; Nelder, J. A. (1989). <em>Generalized Linear Models</em> (2nd ed.). Chapman &amp; Hall. &mdash; the binomial GLM used in &sect;5.</li>
  </ol>
</section>

<footer>
  <p>Rerun end to end (~1.5&ndash;2 hours):</p>
  <code>make -C scripts/dossier_size_model all</code>
  <p>Or an individual stage, e.g.:</p>
  <code>make -C scripts/dossier_size_model three_groups_plot</code>
  <p>Code: scripts/dossier_size_model/ &middot; Data: data/dossier_size_model/</p>
</footer>
"""

    out_path = OUT / "report.html"
    out_path.write_text(html)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
