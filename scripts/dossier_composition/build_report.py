"""
Generates the styled HTML report (data/dossier_composition/report.html) from
the four analyses' summary CSVs -- full pairwise/per-type tables are built
entirely from the data, so rerunning after new predictions regenerates them
correctly. The narrative prose is a fixed template with inline numbers
pulled from the same CSVs (so cited figures stay in sync), but the
*qualitative interpretation* is written, not derived -- re-read it after a
rerun rather than trusting it blindly, since a new model could change which
finding is the interesting one.

Rerun: python3 scripts/dossier_composition/build_report.py
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_TEST_PREDICTIONS_PATH, OUT_DIR, TRACKED_TYPES

OUT = Path(OUT_DIR)


def run_info(test_predictions_path: str) -> dict:
    """Derive a human-readable run/combo label + headline test-set metrics
    from the test-predictions path, e.g.
    runs/per_task/<run_name>/<combo_name>/predictions.tsv -- reads the
    metrics.json sitting alongside predictions.tsv rather than duplicating
    numbers that are already computed by the eval pipeline.

    Deliberately uses the macro, END-TO-END variants (doc_macro_f1_e2e,
    doc_accuracy_e2e, start_macro_f1) -- metrics.json used to also carry a
    non-e2e "doc_macro_f1"/"doc_accuracy" pair and per-class positive-class-
    only scores (per_class_metrics_document_type.tsv), which overstated
    performance by scoring document-type accuracy against the TRUE
    segmentation rather than the model's own (error-prone) predicted
    segmentation. Those keys have since been removed from metrics.json
    entirely; only report keys that still exist, so a future eval-pipeline
    regression that reintroduces a non-e2e/positive-class-only metric fails
    loudly here (KeyError) instead of silently being picked up again."""
    combo_dir = Path(test_predictions_path).resolve().parent
    metrics = json.loads((combo_dir / "metrics.json").read_text())
    return {
        "run_name": combo_dir.parent.name,
        "combo_name": combo_dir.name,
        "doc_macro_f1": metrics["doc_macro_f1_e2e"],
        "doc_accuracy": metrics["doc_accuracy_e2e"],
        "start_f1": metrics["start_macro_f1"],
    }


def b64_img(filename: str) -> str:
    data = (OUT / filename).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def fmt_pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def fmt(x, d=2):
    return f"{x:.{d}f}"


def hdi(lo, hi, d=2, pct=False):
    if pct:
        return f"[{fmt_pct(lo, 1)}, {fmt_pct(hi, 1)}]"
    return f"[{fmt(lo, d)}, {fmt(hi, d)}]"


def load():
    return {
        "confusion": pd.read_csv(OUT / "confusion_matrix_posterior_mean.csv", index_col=0),
        "test_counts": pd.read_csv(OUT / "confusion_matrix_test_counts.csv", index_col=0),
        "occ": pd.read_csv(OUT / "occurrence_summary.csv"),
        "co": pd.read_csv(OUT / "co_occurrence_summary.csv"),
        "disp": pd.read_csv(OUT / "dispersion_summary.csv"),
        "order": pd.read_csv(OUT / "order_summary.csv"),
        "order_pw": pd.read_csv(OUT / "order_pairwise.csv"),
    }


def precision_series(confusion: pd.DataFrame, test_counts: pd.DataFrame) -> pd.Series:
    """P(true=i | predicted=i) per type, i.e. the column-normalized counterpart
    to `confusion`'s rows (which are P(predicted=j | true=i), i.e. recall) --
    needed because occurrence-prevalence corrections are driven by precision
    problems (over-prediction), not recall problems (under-prediction), and
    the two can diverge sharply per type."""
    prior = test_counts.sum(axis=1)
    prior = prior / prior.sum()
    unnorm = confusion.values * prior.reindex(confusion.index).values[:, None]
    precision = unnorm / unnorm.sum(axis=0, keepdims=True)
    return pd.Series(np.diag(precision), index=confusion.index)


def g(df, key_col, key_val, col):
    return df.loc[df[key_col] == key_val, col].item()


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


def build_recall_table(confusion: pd.DataFrame) -> str:
    diag = pd.Series(np.diag(confusion.values), index=confusion.index)
    diag = diag.drop("Other").sort_values(ascending=False)
    rows = [[t, f'<span class="mono">{fmt_pct(v, 1)}</span>'] for t, v in diag.items()]
    return table_html(["Document type", "Recall"], rows, classes="data-table narrow")


def build_occurrence_table(occ: pd.DataFrame) -> str:
    occ = occ[occ["doc_type"] != "Other"].sort_values("corrected_prevalence_mean", ascending=False)
    rows = []
    for _, r in occ.iterrows():
        rows.append([
            r["doc_type"],
            f'<span class="mono">{fmt_pct(r["naive_prevalence"])}</span>',
            f'<span class="mono">{fmt_pct(r["corrected_prevalence_mean"])}</span> '
            f'<span class="hdi">{hdi(r["corrected_prevalence_hdi_3"], r["corrected_prevalence_hdi_97"], pct=True)}</span>',
            f'<span class="mono">{fmt(r["naive_mean_count_given_present"])}</span>',
            f'<span class="mono">{fmt(r["corrected_mean_count_given_present_mean"])}</span> '
            f'<span class="hdi">{hdi(r["corrected_mean_count_given_present_hdi_3"], r["corrected_mean_count_given_present_hdi_97"])}</span>',
        ])
    return table_html(
        ["Document type", "Prevalence (naive)", "Prevalence (corrected, 94% HDI)",
         "Multiplicity (naive)", "Multiplicity (corrected, 94% HDI)"],
        rows, classes="data-table"
    )


def build_co_occurrence_table(co: pd.DataFrame) -> str:
    co = co.sort_values("corrected_mh_log_odds_mean", ascending=False)
    rows = []
    for _, r in co.iterrows():
        credible = '<span class="dot dot-yes" title="94% HDI excludes 0"></span>' if r["mh_credible"] else \
                   '<span class="dot dot-no" title="94% HDI includes 0"></span>'
        rows.append([
            r["type_a"], r["type_b"],
            f'<span class="mono">{fmt(r["corrected_log_odds_mean"])}</span>',
            f'<span class="mono">{fmt(r["corrected_mh_log_odds_mean"])}</span> '
            f'<span class="hdi">{hdi(r["corrected_mh_log_odds_hdi_3"], r["corrected_mh_log_odds_hdi_97"])}</span>',
            credible,
        ])
    return table_html(
        ["Type A", "Type B", "Log-odds (raw)", "Log-odds (richness-controlled, 94% HDI)", "Credible"],
        rows, classes="data-table wide"
    )


def build_dispersion_table(disp: pd.DataFrame) -> str:
    disp = disp.sort_values("corrected_z_score_mean")
    rows = []
    for _, r in disp.iterrows():
        rows.append([
            r["doc_type"],
            f'<span class="mono">{fmt(r["corrected_z_score_mean"], 1)}</span> '
            f'<span class="hdi">{hdi(r["corrected_z_score_hdi_3"], r["corrected_z_score_hdi_97"], d=1)}</span>',
            f'<span class="mono">{fmt(r["corrected_contiguity_mean"])}</span> '
            f'<span class="hdi">{hdi(r["corrected_contiguity_hdi_3"], r["corrected_contiguity_hdi_97"])}</span>',
            f'<span class="mono">{int(round(r["corrected_n_qualifying_mean"]))}</span>',
        ])
    return table_html(
        ["Document type", "Z-score vs. random-placement null (94% HDI)", "Contiguity score (94% HDI)",
         "Qualifying dossiers (mean)"],
        rows, classes="data-table"
    )


def build_order_table(order: pd.DataFrame) -> str:
    order = order.sort_values("corrected_rank_theta_mean")
    rows = []
    for _, r in order.iterrows():
        rows.append([
            r["doc_type"],
            f'<span class="mono">{fmt(r["corrected_rank_theta_mean"])}</span> '
            f'<span class="hdi">{hdi(r["corrected_rank_theta_hdi_3"], r["corrected_rank_theta_hdi_97"])}</span>',
            f'<span class="mono">{fmt(r["corrected_mean_position_mean"])}</span> '
            f'<span class="hdi">{hdi(r["corrected_mean_position_hdi_3"], r["corrected_mean_position_hdi_97"])}</span>',
        ])
    return table_html(
        ["Document type", "Bradley-Terry rank (94% HDI)", "Mean normalized position (94% HDI)"],
        rows, classes="data-table"
    )


def build_order_pairwise_table(order_pw: pd.DataFrame) -> str:
    order_pw = order_pw.sort_values("p_a_before_b_mean", ascending=False)
    rows = []
    for _, r in order_pw.iterrows():
        credible = (r["p_a_before_b_hdi_3"] > 0.5) or (r["p_a_before_b_hdi_97"] < 0.5)
        dot = '<span class="dot dot-yes" title="94% HDI excludes 0.5"></span>' if credible else \
              '<span class="dot dot-no" title="94% HDI includes 0.5"></span>'
        rows.append([
            r["type_a"], r["type_b"],
            f'<span class="mono">{fmt_pct(r["p_a_before_b_mean"])}</span> '
            f'<span class="hdi">{hdi(r["p_a_before_b_hdi_3"], r["p_a_before_b_hdi_97"], pct=True)}</span>',
            dot,
        ])
    return table_html(
        ["Type A", "Type B", "P(A before B), 94% HDI", "Credible"],
        rows, classes="data-table wide"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-predictions", default=DEFAULT_TEST_PREDICTIONS_PATH)
    args = parser.parse_args()

    info = run_info(args.test_predictions)
    run_label = f"{info['run_name']} / {info['combo_name']}"
    run_doc_f1 = fmt(info["doc_macro_f1"], 2)
    run_doc_acc = fmt_pct(info["doc_accuracy"], 1)
    run_start_f1 = fmt(info["start_f1"], 2)

    d = load()
    confusion, occ, co, disp, order, order_pw = (
        d["confusion"], d["occ"], d["co"], d["disp"], d["order"], d["order_pw"]
    )

    n_tracked = len(TRACKED_TYPES)
    n_all_types = n_tracked + 1  # + Other

    n_credible_co = int(co["mh_credible"].sum())
    n_pairs_co = len(co)
    n_raw_positive_co = int((co["naive_log_odds_mean"] > 0).sum())
    n_credible_order = int(((order_pw["p_a_before_b_hdi_3"] > 0.5) | (order_pw["p_a_before_b_hdi_97"] < 0.5)).sum())
    n_pairs_order = len(order_pw)

    # richness (naive/predicted-label segmentation): how many of the n_all_types
    # buckets a dossier contains at all -- the confound the MH stratification in
    # co_occurrence.py controls for. Computed here (not saved by occurrence.py
    # itself) since it's only needed for this one descriptive callout.
    segments_for_richness = pd.read_parquet(OUT / "segments.parquet")
    richness = segments_for_richness.groupby("pdf_name")["type_mapped"].nunique()
    richness_min, richness_max, richness_mean = int(richness.min()), int(richness.max()), richness.mean()

    images = {
        "occurrence": b64_img("occurrence_naive_vs_corrected.png"),
        "co_occurrence": b64_img("co_occurrence_heatmap.png"),
        "dispersion": b64_img("dispersion_naive_vs_corrected.png"),
        "order": b64_img("order_naive_vs_corrected.png"),
    }

    recall = pd.Series(np.diag(confusion.values), index=confusion.index)
    precision = precision_series(confusion, d["test_counts"])
    recall_t = recall.drop("Other")
    precision_t = precision.drop("Other")
    best_recall_type, best_recall = recall_t.idxmax(), fmt_pct(recall_t.max(), 1)
    worst_recall_type, worst_recall = recall_t.idxmin(), fmt_pct(recall_t.min(), 1)
    worst_prec_type, worst_prec = precision_t.idxmin(), fmt_pct(precision_t.min(), 1)
    second_prec_type = precision_t.drop(worst_prec_type).idxmin()
    second_prec = fmt_pct(precision_t.drop(worst_prec_type).min(), 1)
    n_test_pages = int(d["test_counts"].to_numpy().sum())
    approval_test_n = int(d["test_counts"].loc["Approval notice"].sum())
    testmed_test_n = int(d["test_counts"].loc["Testimonial medical form (Medical & Health Documents)"].sum())

    d1_prev_naive = fmt_pct(g(occ, "doc_type", "D.1", "naive_prevalence"))
    d1_prev_c = fmt_pct(g(occ, "doc_type", "D.1", "corrected_prevalence_mean"))
    d1_prev_lo = fmt_pct(g(occ, "doc_type", "D.1", "corrected_prevalence_hdi_3"))
    d1_prev_hi = fmt_pct(g(occ, "doc_type", "D.1", "corrected_prevalence_hdi_97"))

    d2_prev_naive = fmt_pct(g(occ, "doc_type", "D.2", "naive_prevalence"))
    d2_prev_c = fmt_pct(g(occ, "doc_type", "D.2", "corrected_prevalence_mean"))
    d2_prev_lo = fmt_pct(g(occ, "doc_type", "D.2", "corrected_prevalence_hdi_3"))
    d2_prev_hi = fmt_pct(g(occ, "doc_type", "D.2", "corrected_prevalence_hdi_97"))

    ap_prev_naive = fmt_pct(g(occ, "doc_type", worst_prec_type, "naive_prevalence"))
    ap_prev_c = fmt_pct(g(occ, "doc_type", worst_prec_type, "corrected_prevalence_mean"))
    ap_prev_lo = fmt_pct(g(occ, "doc_type", worst_prec_type, "corrected_prevalence_hdi_3"))
    ap_prev_hi = fmt_pct(g(occ, "doc_type", worst_prec_type, "corrected_prevalence_hdi_97"))

    d1d2_row = co[(co["type_a"] == "D.1") & (co["type_b"] == "D.2")].iloc[0]
    d1d2_lo = fmt(d1d2_row["corrected_mh_log_odds_mean"])
    d1d2_lo_lo = fmt(d1d2_row["corrected_mh_log_odds_hdi_3"])
    d1d2_lo_hi = fmt(d1d2_row["corrected_mh_log_odds_hdi_97"])

    confound_row = co[
        ((co["type_a"] == "D.2") & (co["type_b"] == "Judicial and political background check"))
        | ((co["type_b"] == "D.2") & (co["type_a"] == "Judicial and political background check"))
    ].iloc[0]
    confound_before = fmt(confound_row["corrected_log_odds_mean"])
    confound_after = fmt(confound_row["corrected_mh_log_odds_mean"])

    # D.1's own weakest/strongest co-occurrence partners, and which types are
    # weakest overall (averaged across all their pairs) -- both picked
    # dynamically since which types end up weakest is a substantive finding
    # that can flip between classifiers, not a fixed cast of characters.
    d1_pairs = co[(co["type_a"] == "D.1") | (co["type_b"] == "D.1")]
    d1_n_partners = len(d1_pairs)
    d1_partner_lo = fmt(d1_pairs["corrected_mh_log_odds_mean"].min())
    d1_partner_hi = fmt(d1_pairs["corrected_mh_log_odds_mean"].max())

    avg_mh_by_type = {}
    for t in TRACKED_TYPES:
        sub = co[(co["type_a"] == t) | (co["type_b"] == t)]
        avg_mh_by_type[t] = sub["corrected_mh_log_odds_mean"].mean()
    weakest_co_types = sorted(avg_mh_by_type, key=avg_mh_by_type.get)[:3]
    weakest_co_types_str = ", ".join(weakest_co_types[:-1]) + f", and {weakest_co_types[-1]}"

    # which type is most/least rigidly clustered shifts with the classifier (a
    # weak-recall type can look artificially "tight" naively, then spread out
    # once correction recovers instances the naive read had missed) -- so
    # these are picked dynamically from corrected_contiguity_mean rather than
    # hardcoded to D.1, which is not always the extreme case.
    disp_t = disp[disp["doc_type"] != "Other"]
    mc_row = disp_t.loc[disp_t["corrected_contiguity_mean"].idxmin()]
    lc_row = disp_t.loc[disp_t["corrected_contiguity_mean"].idxmax()]
    mc_type, mc_contig = mc_row["doc_type"], fmt(mc_row["corrected_contiguity_mean"])
    mc_contig_lo, mc_contig_hi = fmt(mc_row["corrected_contiguity_hdi_3"]), fmt(mc_row["corrected_contiguity_hdi_97"])
    mc_z = fmt(mc_row["corrected_z_score_mean"], 1)
    lc_type, lc_contig = lc_row["doc_type"], fmt(lc_row["corrected_contiguity_mean"])

    disp_t = disp_t.assign(contig_gap=(disp_t["corrected_contiguity_mean"] - disp_t["naive_contiguity"]).abs())
    disp_t = disp_t.sort_values("contig_gap", ascending=False)
    g1_row, g2_row = disp_t.iloc[0], disp_t.iloc[1]
    g1_type = g1_row["doc_type"]
    g1_naive, g1_c = fmt(g1_row["naive_contiguity"]), fmt(g1_row["corrected_contiguity_mean"])
    g2_type = g2_row["doc_type"]
    g2_naive, g2_c = fmt(g2_row["naive_contiguity"]), fmt(g2_row["corrected_contiguity_mean"])

    d1_pos = fmt(g(order, "doc_type", "D.1", "corrected_mean_position_mean"))
    d2_pos = fmt(g(order, "doc_type", "D.2", "corrected_mean_position_mean"))
    d1d2_p = order_pw[(order_pw["type_a"] == "D.1") & (order_pw["type_b"] == "D.2")].iloc[0]
    d1d2_p_mean = fmt_pct(d1d2_p["p_a_before_b_mean"])
    d1d2_p_lo = fmt_pct(d1d2_p["p_a_before_b_hdi_3"])
    d1d2_p_hi = fmt_pct(d1d2_p["p_a_before_b_hdi_97"])

    approval_pos = fmt(g(order, "doc_type", "Approval notice", "corrected_mean_position_mean"))

    # which types fall "in the middle" (and their sub-ordering) is itself a
    # substantive finding that can shift between classifiers, e.g. a type
    # whose recall used to be poor can look artificially early/late naively
    # -- so this is read off the Bradley-Terry ranking dynamically rather
    # than hardcoded to whatever ordering an earlier model produced.
    middle_types = [t for t in TRACKED_TYPES if t not in ("D.1", "D.2", "Approval notice")]
    order_t = order[order["doc_type"] != "Other"]
    middle_ranked = order_t[order_t["doc_type"].isin(middle_types)].sort_values("corrected_rank_theta_mean")
    middle_order_str = ", ".join(middle_ranked["doc_type"].tolist()[:-1]) + f", and {middle_ranked['doc_type'].iloc[-1]}"
    widest_row = order_t.loc[
        (order_t["corrected_rank_theta_hdi_97"] - order_t["corrected_rank_theta_hdi_3"]).idxmax()
    ]
    widest_type = widest_row["doc_type"]

    html = f"""<title>Dossier Composition Analysis</title>
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
  --good: #2e7d4f;
  --font-serif: Charter, "Iowan Old Style", "Palatino Linotype", "Georgia", serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --paper: #14161a;
    --surface: #1b1e17;
    --surface-2: #20231c;
    --ink: #eef0ec;
    --ink-secondary: #b9bcae;
    --ink-muted: #7d8177;
    --rule: #33362f;
    --accent: #74a9de;
    --accent-soft: #223349;
    --caution: #d98a52;
    --caution-soft: #3a2a1e;
    --good: #5cb385;
    color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #14161a;
  --surface: #1b1e17;
  --surface-2: #20231c;
  --ink: #eef0ec;
  --ink-secondary: #b9bcae;
  --ink-muted: #7d8177;
  --rule: #33362f;
  --accent: #74a9de;
  --accent-soft: #223349;
  --caution: #d98a52;
  --caution-soft: #3a2a1e;
  --good: #5cb385;
  color-scheme: dark;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-serif);
  font-size: 18px;
  line-height: 1.6;
  margin: 0;
  padding: 0 24px 96px;
}}
.page {{
  max-width: 800px;
  margin: 0 auto;
}}
.wide {{
  max-width: 1040px;
  margin: 0 auto;
}}
header.masthead {{
  max-width: 800px;
  margin: 0 auto;
  padding: 64px 0 28px;
}}
.eyebrow {{
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 14px;
}}
h1 {{
  font-size: 40px;
  line-height: 1.15;
  margin: 0 0 10px;
  text-wrap: balance;
  font-weight: 600;
}}
.subtitle {{
  color: var(--ink-secondary);
  font-size: 19px;
  max-width: 65ch;
  margin: 0 0 28px;
}}
.status-banner {{
  border: 1px solid var(--rule);
  background: var(--surface);
  border-left: 3px solid var(--accent);
  padding: 14px 18px;
  font-size: 15px;
  color: var(--ink-secondary);
  border-radius: 2px;
}}
.status-banner code {{
  font-family: var(--font-mono);
  font-size: 13px;
  background: var(--surface-2);
  padding: 1px 5px;
  border-radius: 3px;
}}
section {{
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 0;
  border-top: 1px solid var(--rule);
}}
section.wide-section {{ max-width: 1040px; }}
section.wide-section > .prose {{ max-width: 800px; margin: 0 auto; }}
.section-head {{
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 22px;
}}
.section-num {{
  font-family: var(--font-mono);
  font-size: 14px;
  color: var(--accent);
  white-space: nowrap;
}}
h2 {{
  font-size: 28px;
  margin: 0;
  font-weight: 600;
  text-wrap: balance;
}}
h3 {{
  font-size: 19px;
  font-weight: 600;
  margin: 28px 0 10px;
}}
p {{ margin: 0 0 16px; }}
.lede {{ font-size: 19px; color: var(--ink-secondary); }}
a {{ color: var(--accent); }}
strong {{ font-weight: 600; }}
.callout {{
  background: var(--caution-soft);
  border-left: 3px solid var(--caution);
  padding: 14px 18px;
  border-radius: 2px;
  font-size: 16px;
  margin: 20px 0;
}}
.callout .callout-label {{
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--caution);
  display: block;
  margin-bottom: 6px;
}}
figure {{ margin: 24px 0; }}
figure img {{
  width: 100%;
  height: auto;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: var(--surface-2);
}}
figcaption {{
  font-family: var(--font-mono);
  font-size: 12.5px;
  color: var(--ink-muted);
  margin-top: 8px;
  text-align: center;
}}
.mono {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.hdi {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--ink-muted); font-size: 0.88em; }}
.table-toggle {{
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.03em;
  color: var(--accent);
  background: none;
  border: 1px solid var(--accent);
  border-radius: 3px;
  padding: 7px 14px;
  cursor: pointer;
  margin: 8px 0 4px;
}}
.table-toggle:hover {{ background: var(--accent-soft); }}
details {{ margin: 12px 0 8px; }}
details > summary {{
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.03em;
  color: var(--accent);
  cursor: pointer;
  list-style: none;
  padding: 8px 0;
}}
details > summary::-webkit-details-marker {{ display: none; }}
details > summary::before {{ content: "▸ "; }}
details[open] > summary::before {{ content: "▾ "; }}
.table-wrap {{ overflow-x: auto; margin: 10px 0 4px; border: 1px solid var(--rule); border-radius: 3px; }}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 14.5px;
  background: var(--surface-2);
}}
table.narrow {{ max-width: 420px; }}
th, td {{
  text-align: left;
  padding: 9px 14px;
  border-bottom: 1px solid var(--rule);
  white-space: nowrap;
}}
th {{
  font-family: var(--font-mono);
  font-size: 11.5px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ink-muted);
  background: var(--surface);
  position: sticky;
  top: 0;
}}
tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--accent-soft); }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; }}
.dot-yes {{ background: var(--good); }}
.dot-no {{ background: var(--ink-muted); opacity: 0.4; }}
.synthesis-list {{ padding: 0; margin: 0; list-style: none; }}
.synthesis-list li {{
  padding: 16px 0;
  border-bottom: 1px solid var(--rule);
}}
.synthesis-list li:last-child {{ border-bottom: none; }}
.synthesis-list .tag {{
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  display: block;
  margin-bottom: 6px;
}}
.refs {{ font-size: 15px; }}
.refs ol {{ padding-left: 22px; }}
.refs li {{ margin-bottom: 12px; color: var(--ink-secondary); }}
.refs li a {{ color: var(--ink-secondary); }}
.cite {{
  font-family: var(--font-mono);
  font-size: 0.72em;
  color: var(--accent);
  vertical-align: super;
  text-decoration: none;
  margin-left: 1px;
}}
footer {{
  max-width: 800px;
  margin: 40px auto 0;
  padding-top: 28px;
  border-top: 1px solid var(--rule);
  font-size: 13.5px;
  color: var(--ink-muted);
  font-family: var(--font-mono);
}}
footer code {{
  display: block;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 12px 14px;
  margin: 10px 0;
  overflow-x: auto;
  white-space: pre;
  color: var(--ink);
}}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>

<header class="masthead">
  <p class="eyebrow">Migration Dossiers &middot; Composition Analysis</p>
  <h1>What a dossier typically contains, and how sure we can be</h1>
  <p class="subtitle">Occurrence, co-occurrence, dispersion, and order of the {n_tracked} tracked document types across 3,343 dossiers &mdash; corrected for a document-type classifier whose reliability varies sharply by type.</p>
  <div class="status-banner">
    This run uses the <code>{run_label}</code> pipeline (document-type macro F1 {run_doc_f1}, accuracy {run_doc_acc}, both end-to-end against the model's own predicted segmentation; start-page macro F1 {run_start_f1}). If a better-performing model's predictions become available, every table and figure below regenerates with no code changes:
    <code>make -C scripts/dossier_composition all PREDICTIONS=... TEST_PREDICTIONS=...</code>
    Treat this as the current best answer, not the final one.
  </div>
</header>

<section>
  <div class="section-head"><span class="section-num">&sect;0</span><h2>Why correction, not just counting</h2></div>
  <p>This classifier's recall is fairly even across types &mdash; from {best_recall_type} ({best_recall}) down to a still-solid {worst_recall_type} ({worst_recall}). The larger remaining unevenness is on <em>precision</em>: {worst_prec_type} ({worst_prec}) and D.2 ({second_prec}) are the two types most often over-predicted, which inflates their naive prevalence (see &sect;1). Naive counts on predicted labels are still systematically off for these types, just less so, and less broadly, than with a weaker classifier.</p>
  <p>Every analysis below is built on the same foundation: a Bayesian confusion matrix<a href="#ref-1" class="cite">[1]</a> (hierarchical Dirichlet-multinomial, partially pooled across types<a href="#ref-2" class="cite">[2]</a>, fit with NUTS<a href="#ref-3" class="cite">[3]</a> via PyMC<a href="#ref-4" class="cite">[4]</a>) estimated from a {n_test_pages}-page held-out test set, used to multiply-impute<a href="#ref-5" class="cite">[5]</a> each predicted document instance's plausible true type &mdash; 200 imputations, each also resampling the confusion matrix's own posterior. Every statistic is reported both <strong>naive</strong> (raw predicted labels) and <strong>corrected</strong> (pooled across imputations), so the gap between them shows exactly how much the classifier's unevenness would have distorted a naive reading.</p>
  <h3>Recall by document type</h3>
  {build_recall_table(confusion)}
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;1</span><h2>Occurrence</h2></div>
    <p class="lede">Which types are common, which are rare, and do they tend to appear once or several times?</p>
    <p><strong>D.1 is near-universal</strong> ({d1_prev_c} of dossiers, {d1_prev_lo}&ndash;{d1_prev_hi}) &mdash; the closest thing to a mandatory document in this set. Most other tracked types sit in the 55&ndash;92% prevalence range.</p>
    <p>The naive-vs-corrected gap now mostly shows the classifier's <em>precision</em> unevenness, not recall (&sect;0). <strong>{worst_prec_type}</strong> drops from {ap_prev_naive} naive prevalence to {ap_prev_c} corrected ({ap_prev_lo}&ndash;{ap_prev_hi}) &mdash; the largest correction in the analysis, in the direction its precision problem predicts: much of what the classifier calls &ldquo;{worst_prec_type}&rdquo; is actually misclassified &ldquo;Other&rdquo; pages. <strong>D.2</strong> shows the same pattern, more mildly ({d2_prev_naive} &rarr; {d2_prev_c}, {d2_prev_lo}&ndash;{d2_prev_hi}). With recall now solid across the board (&sect;0), no type shows the opposite pattern (recall recovering hidden instances) by more than a percentage point or two &mdash; every other type's naive and corrected prevalence agree closely.</p>
    <figure>
      <img src="{images['occurrence']}" alt="Occurrence: naive vs. corrected prevalence and multiplicity per document type" />
      <figcaption>prevalence and typical multiplicity, naive vs. classifier-corrected</figcaption>
    </figure>
    <details>
      <summary>Full occurrence table ({n_tracked} tracked types)</summary>
      {build_occurrence_table(occ)}
    </details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;2</span><h2>Co-occurrence</h2></div>
    <p class="lede">Which document types tend to appear together, and which avoid each other?</p>
    <div class="callout">
      <span class="callout-label">Confound caught before trusting this</span>
      The raw pairwise association came back with {n_raw_positive_co} of {n_pairs_co} pairs positive &mdash; not plausible on its face. Cause: dossiers vary hugely in overall &ldquo;richness&rdquo; ({richness_min}&ndash;{richness_max} of the {n_all_types} possible types present, mean {fmt(richness_mean, 1)}), so a more-complete dossier shows elevated presence for every type at once, inflating every pair uniformly. Controlled with a Mantel-Haenszel<a href="#ref-6" class="cite">[6]</a> richness-stratified estimator (4 strata by how many <em>other</em> types a dossier contains, variance via Robins&ndash;Breslow&ndash;Greenland<a href="#ref-7" class="cite">[7]</a>).
    </div>
    <p>After control, every pair is <em>still</em> positive &mdash; the confound explained part of the raw signal (weaker pairs shrink substantially, e.g. D.2&ndash;Judicial background check: {confound_before} &rarr; {confound_after} log-odds), but nothing flipped negative. Read with &sect;1, this is a coherent picture: the {n_tracked} tracked types function as parts of one largely-standard &ldquo;complete case&rdquo; packet rather than having substitutable pairs. <strong>D.1&ndash;D.2</strong> specifically: log-odds {d1d2_lo} ({d1d2_lo_lo}&ndash;{d1d2_lo_hi}), among the strongest pairs in the corpus.</p>
    <p><strong>D.1 is the strongest co-occurring partner with everything</strong> (log-odds {d1_partner_lo}&ndash;{d1_partner_hi} across its {d1_n_partners} partners). Averaged across all their pairs, the weakest-but-still-positive associations cluster around {weakest_co_types_str}. With this classifier's much more even reliability (&sect;0), that clustering is no longer just a residual classifier artifact for all three: {worst_prec_type}'s and D.2's weak precision (&sect;0) still explain part of it, but Judicial background check's recall and precision are both solid now &mdash; its weak co-occurrence looks like a genuine substantive signal rather than a correction gap.</p>
    <figure>
      <img src="{images['co_occurrence']}" alt="Pairwise co-occurrence heatmap, Mantel-Haenszel richness-controlled log-odds ratio" />
      <figcaption>pairwise co-occurrence, richness-controlled log-odds ratio</figcaption>
    </figure>
    <details>
      <summary>Full pairwise table (all {n_pairs_co} pairs, {n_credible_co} credible at 94% HDI)</summary>
      {build_co_occurrence_table(co)}
    </details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;3</span><h2>Dispersion</h2></div>
    <p class="lede">When a type occurs more than once, are the instances filed together or scattered through the dossier?</p>
    <p>Every type is dramatically more contiguous than random placement would produce &mdash; z-scores in the hundreds against the exact closed-form runs-test null<a href="#ref-8" class="cite">[8]</a>, for all {n_tracked} types (validated against a hand-worked synthetic example before trusting output this large). Same-type documents are essentially never scattered arbitrarily &mdash; whether that reflects original filing practice, archival practice, or both.</p>
    <p><strong>{mc_type} is the most rigidly clustered</strong> (contiguity {mc_contig}, {mc_contig_lo}&ndash;{mc_contig_hi}, z={mc_z}, on a 0=always-one-block to 1=maximally-scattered scale), while {lc_type} sits at the loose end ({lc_contig}) &mdash; still far more clustered than chance, just less rigidly so. Which type anchors each end is itself sensitive to the classifier: a type whose weak instances get merged into &ldquo;Other&rdquo; naively looks artificially tight until correction spreads it back out, so re-check this pairing whenever the underlying model changes.</p>
    <p><strong>Where correction matters most</strong>: not simply the weakest-recall types (recall is fairly even now, &sect;0) but the types whose predicted instances get relocated the most once segmentation and type are jointly corrected. {g1_type}'s contiguity moves {g1_naive} &rarr; {g1_c}, and {g2_type} {g2_naive} &rarr; {g2_c} &mdash; instances recovered or reassigned sit in different positions than the naively-detected cluster, revealing genuinely more scattered true patterns for these two. The clearest case in this analysis of the naive answer being substantively wrong, not just imprecise.</p>
    <figure>
      <img src="{images['dispersion']}" alt="Dispersion: z-score vs random-placement null, and descriptive contiguity score" />
      <figcaption>contiguity vs. random-placement null, naive vs. classifier-corrected</figcaption>
    </figure>
    <details>
      <summary>Full dispersion table ({n_tracked} tracked types)</summary>
      {build_dispersion_table(disp)}
    </details>
  </div>
</section>

<section class="wide-section">
  <div class="prose">
    <div class="section-head"><span class="section-num">&sect;4</span><h2>Order</h2></div>
    <p class="lede">Does a type sit in a fairly fixed position in the sequence? Is there a typical order between pairs?</p>
    <p><strong>D.1 anchors the start</strong> (mean position {d1_pos}) and <strong>D.2 anchors the end</strong> (mean position {d2_pos}) &mdash; fit with a Bradley-Terry paired-comparison model<a href="#ref-9" class="cite">[9]</a> on which type's first occurrence comes earlier, for every dossier where a pair co-occurs. This directly confirms the example that motivated this whole line of analysis: <strong>P(D.1 before D.2) = {d1d2_p_mean}, 94% HDI {d1d2_p_lo}&ndash;{d1d2_p_hi}</strong>.</p>
    <p>Between the anchors, the remaining types rank (earliest to latest, mostly overlapping intervals, no sharp distinctions among neighbors): {middle_order_str}. <strong>{widest_type}</strong> has the widest uncertainty band of any type: its position is genuinely inconsistent across dossiers, not just poorly estimated.</p>
    <div class="callout">
      <span class="callout-label">Worth checking with domain expertise</span>
      Approval notice ranks among the <em>earliest</em> documents (mean position {approval_pos}), surprising if it represents final case approval. Could be an intermediate eligibility approval rather than final sign-off, or an archival-filing convention rather than original processing order &mdash; not resolvable from this data alone.
    </div>
    <p style="font-size:14.5px;color:var(--ink-muted)">Before trusting the Bradley-Terry fit on real data, its sign convention was checked against synthetic data with a known true ranking &mdash; it caught a bug in the synthetic-data generator on the first attempt (fixed, then reconfirmed). Rerunnable with <code class="mono">python3 scripts/dossier_composition/order.py --self-test</code>, also wired as a Makefile prerequisite of the <code class="mono">order</code> target.</p>
    <figure>
      <img src="{images['order']}" alt="Typical sequence order (Bradley-Terry rank) and absolute position within the dossier" />
      <figcaption>typical sequence order and absolute position, naive vs. classifier-corrected</figcaption>
    </figure>
    <details>
      <summary>Full position table ({n_tracked} tracked types)</summary>
      {build_order_table(order)}
    </details>
    <details>
      <summary>Full pairwise order table (all {n_pairs_order} pairs, {n_credible_order} credible at 94% HDI)</summary>
      {build_order_pairwise_table(order_pw)}
    </details>
  </div>
</section>

<section>
  <div class="section-head"><span class="section-num">&sect;5</span><h2>Putting the four pieces together</h2></div>
  <ul class="synthesis-list">
    <li><span class="tag">The anchor</span>D.1 is present in nearly every dossier, comes first, is the most tightly clustered type, and is the strongest co-occurring partner with everything else &mdash; consistent with it being a foundational intake form.</li>
    <li><span class="tag">The closer</span>D.2 is reliably last when present and positively associated with everything, but more weakly than D.1. Its occurrence rate still needs a real correction ({d2_prev_naive} &rarr; {d2_prev_c}) &mdash; second only to {worst_prec_type} ({ap_prev_naive} &rarr; {ap_prev_c}), the type that now needs the single biggest correction. Interpret both types' numbers with more caution than the rest.</li>
    <li><span class="tag">A cross-cutting diagnostic</span>{worst_prec_type} and D.2 &mdash; the two weakest-precision types (&sect;0) &mdash; keep showing up as needing the biggest naive-vs-corrected corrections in occurrence and co-occurrence specifically; Testimonial labour and DM.1 show the same pattern in dispersion instead. No single type is unreliable across every analysis at once with this classifier &mdash; the earlier, weaker model produced a more uniform "usual suspects" list than this one does.</li>
    <li><span class="tag">One packet, not competing forms</span>The {n_tracked} tracked types read as components of one largely-standard &ldquo;complete case&rdquo; packet, with a fairly stable typical order (D.1 &rarr; middle cluster &rarr; D.2), rather than a set of interchangeable or competing forms &mdash; no pair showed evidence of substituting for another, even after confound control.</li>
  </ul>
</section>

<section>
  <div class="section-head"><span class="section-num">&sect;6</span><h2>Limitations</h2></div>
  <p>The confusion matrix is estimated from only {n_test_pages} test pages; some types have thin support even after merging (Approval notice test n={approval_test_n}, Testimonial medical n={testmed_test_n}) &mdash; hierarchical pooling mitigates but doesn't eliminate the resulting uncertainty, visible as wider corrected intervals for those types throughout.</p>
  <p>&ldquo;Other&rdquo; (everything outside the {n_tracked} tracked types) is a deliberately heterogeneous residual bucket, excluded from the co-occurrence, dispersion, and order rankings &mdash; it still acts correctly as an &ldquo;interrupter&rdquo; in dispersion/order calculations, just isn't reported on as its own type.</p>
  <p>Several approximations trade statistical purity for tractability: Mantel-Haenszel stratification approximately controls the richness confound rather than eliminating it; position statistics pool instances across dossiers as if independent; pairwise order uses a Normal approximation to the Bradley-Terry likelihood rather than full MCMC per imputation. All are standard, well-understood approximations, not ad hoc shortcuts.</p>
  <p>This analysis describes patterns in the corrected <em>digitized and predicted</em> record. It cannot on its own distinguish original bureaucratic filing order from disturbance introduced during archival handling or digitization &mdash; the dispersion and order results are evidence about the surviving record, not directly about 1950s&ndash;60s office practice, though the two are presumably related.</p>
</section>

<section class="refs">
  <div class="section-head"><span class="section-num">&sect;7</span><h2>References</h2></div>
  <ol>
    <li id="ref-1">Dawid, A. P., &amp; Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. <em>Journal of the Royal Statistical Society: Series C</em>, 28(1), 20&ndash;28. &mdash; estimating a confusion/error-rate matrix from noisy classifier or rater labels, the basis for the correction used throughout.</li>
    <li id="ref-2">Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., &amp; Rubin, D. B. (2013). <em>Bayesian Data Analysis</em> (3rd ed.). CRC Press. &mdash; hierarchical/partial-pooling modeling generally, used for the confusion matrix and throughout the wider project.</li>
    <li id="ref-3">Hoffman, M. D., &amp; Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. <em>Journal of Machine Learning Research</em>, 15(1), 1593&ndash;1623. &mdash; the MCMC sampler used to fit the confusion matrix.</li>
    <li id="ref-4">Salvatier, J., Wiecki, T. V., &amp; Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. <em>PeerJ Computer Science</em>, 2, e55.</li>
    <li id="ref-5">Rubin, D. B. (1987). <em>Multiple Imputation for Nonresponse in Surveys</em>. John Wiley &amp; Sons. &mdash; the multiple-imputation framework used to propagate classifier uncertainty into every downstream statistic.</li>
    <li id="ref-6">Mantel, N., &amp; Haenszel, W. (1959). Statistical aspects of the analysis of data from retrospective studies of disease. <em>Journal of the National Cancer Institute</em>, 22(4), 719&ndash;748. &mdash; the stratified estimator used to control the dossier-richness confound in &sect;2.</li>
    <li id="ref-7">Robins, J., Breslow, N., &amp; Greenland, S. (1986). Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models. <em>Biometrics</em>, 42(2), 311&ndash;323. &mdash; the variance estimator used alongside the Mantel-Haenszel odds ratio.</li>
    <li id="ref-8">Wald, A., &amp; Wolfowitz, J. (1940). On a test whether two samples are from the same population. <em>Annals of Mathematical Statistics</em>, 11(2), 147&ndash;162. &mdash; the runs-test null distribution used in &sect;3 to judge dispersion against random placement.</li>
    <li id="ref-9">Bradley, R. A., &amp; Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. <em>Biometrika</em>, 39(3/4), 324&ndash;345. &mdash; the paired-comparison ranking model used in &sect;4 to derive a typical sequence order.</li>
  </ol>
</section>

<footer>
  <p>Rerun against new predictions:</p>
  <code>make -C scripts/dossier_composition all \\
    PREDICTIONS=data/predictions-new-model.tsv \\
    TEST_PREDICTIONS=runs/new_model/test_predictions.tsv</code>
  <p>Code: scripts/dossier_composition/ &middot; Data: data/dossier_composition/</p>
</footer>
"""

    out_path = OUT / "report.html"
    out_path.write_text(html)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
