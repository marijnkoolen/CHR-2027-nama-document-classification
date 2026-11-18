"""
Tests whether the top-performing config (per metric) is *significantly*
better than each of the others, on the same held-out test PDFs, correcting
for the multiple comparisons this entails.

Method:
    Paired bootstrap test (Berg-Kirkpatrick, Burkett & Klein, EMNLP 2012;
    see also Dror, Baumer & Reichart's ACL 2018 survey "The Hitchhiker's
    Guide to Testing Statistical Significance in NLP"). PDFs, not pages, are
    the resampling unit: pages within one PDF share a lot of what determines
    correctness (the same true document, the same segmentation decisions,
    often the same failure mode), so resampling pages directly would treat
    correlated pages as independent evidence and understate the true
    uncertainty - the same reasoning already applied to --metrics pq's
    per-stream averaging in evaluate_segmentation.py.

    For each metric and each comparison config C (against that metric's own
    top-scoring config T): draw a bootstrap sample of PDFs with replacement
    (same draw applied to both T and C - "paired", so correlation between
    the two configs' performance on the same real PDFs is preserved, which
    increases power over resampling them independently), recompute the
    metric for both on that resampled data, and record whether T's observed
    advantage held up. The one-sided p-value is the fraction of bootstrap
    resamples where it didn't (T <= C) - the standard paired-bootstrap
    formula from Berg-Kirkpatrick et al. Each resampled occurrence of a PDF
    is relabeled to its own pseudo-id first, so a PDF drawn twice becomes
    two independent streams rather than one double-length one (segment
    construction groups by PDF id, so leaving duplicates under the same id
    would silently merge them into a single corrupted stream).

Multiple comparisons: for a metric with N configs, there are N-1
comparisons against the top. Bonferroni (dividing alpha by N-1) would be
needlessly conservative for this many, mostly-correlated comparisons drawn
from only 9 test PDFs - it would likely wash out real differences. Instead,
p-values are Benjamini-Hochberg FDR-corrected *within each metric's own
family* of N-1 comparisons (not pooled across metrics - the metrics measure
different things and answer different questions, so treating all of them as
one family would be an arbitrary scope choice, not a principled one).

Usage:
    python scripts/classification/significance_test.py \\
        --run-dirs runs/from_embeddings/* \\
        --n-bootstrap 1000 --seed 0 \\
        --out runs/from_embeddings/significance_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_segmentation import (
    attachment_scores, macro_f1_report, majority_label_from_segments,
    panoptic_quality, segments_from_id_col, segments_from_start_col, start_page_report,
)

PDF_COL, PAGE_COL = "pdf_name", "page_num"

PRED_LABEL_COL = {
    "document_type": "predicted_document_type",
    "layout_type": "predicted_layout_type",
    "functional_category": "predicted_functional_category",
}


def attachment_metric(df: pd.DataFrame, label_col: str, key: str) -> float:
    true_segments = segments_from_start_col(df, PDF_COL, PAGE_COL, "start_page")
    pred_segments = segments_from_id_col(df, PDF_COL, PAGE_COL, "predicted_segment_id")
    true_labels = majority_label_from_segments(df, PDF_COL, PAGE_COL, label_col, true_segments)
    pred_labels = majority_label_from_segments(df, PDF_COL, PAGE_COL, PRED_LABEL_COL[label_col], pred_segments)
    return attachment_scores(true_segments, pred_segments, true_labels, pred_labels)[key]


def pq_metric(df: pd.DataFrame) -> float:
    true_segments = segments_from_start_col(df, PDF_COL, PAGE_COL, "start_page")
    pred_segments = segments_from_id_col(df, PDF_COL, PAGE_COL, "predicted_segment_id")
    return panoptic_quality(true_segments, pred_segments)["pq"]


def build_metrics(classes: dict, label_col: str) -> dict:
    """{metric_name: df -> float}. F1 metrics always average over the full
    trained vocabulary (classes.json), not whatever happens to appear in a
    given bootstrap resample - same reasoning as evaluate_segmentation.py's
    macro_f1_report, and necessary here for the metric to mean the same
    thing across every resample."""
    metrics = {}
    for target, true_col, pred_col, class_key in [
        ("document_type", "document_type", "predicted_document_type", "document_type"),
        ("layout_type", "layout_type", "predicted_layout_type", "layout_type"),
        ("functional_category", "functional_category", "predicted_functional_category", "functional_category"),
    ]:
        cls = classes[class_key]
        metrics[f"{target}_macro_f1"] = (
            lambda df, tc=true_col, pc=pred_col, c=cls: macro_f1_report(df, tc, pc, c)["macro_f1"]
        )
        metrics[f"{target}_weighted_f1"] = (
            lambda df, tc=true_col, pc=pred_col, c=cls: macro_f1_report(df, tc, pc, c)["weighted_f1"]
        )
    metrics["start_page_f1"] = lambda df: start_page_report(df, "start_page", "predicted_start_page")["f1"]
    metrics["pq"] = pq_metric
    metrics["uas"] = lambda df, lc=label_col: attachment_metric(df, lc, "uas")
    metrics["las"] = lambda df, lc=label_col: attachment_metric(df, lc, "las")
    return metrics


def split_by_pdf(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {pdf_id: sub for pdf_id, sub in df.groupby(PDF_COL, sort=False)}


def build_resample(pdf_dict: dict[str, pd.DataFrame], chosen: np.ndarray) -> pd.DataFrame:
    """Concatenates the drawn PDFs, relabeling pdf_name per-occurrence so a
    PDF drawn more than once becomes multiple independent pseudo-streams -
    see module docstring."""
    parts = []
    for i, pdf_id in enumerate(chosen):
        part = pdf_dict[pdf_id].copy()
        part[PDF_COL] = f"{pdf_id}__boot{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Returns BH-adjusted q-values in the same order as `p_values`."""
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    q_ranked = ranked * m / (np.arange(m) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]  # enforce monotonicity
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty(m)
    q[order] = q_ranked
    return q.tolist()


def paired_bootstrap_family(
    metric_fn, top_dict: dict, other_dicts: dict[str, dict], pdf_ids: list[str],
    n_bootstrap: int, rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    """Compares one metric's top config against every other config in one
    pass, sharing the top config's resample across all of them per bootstrap
    iteration (each `other` still needs its own resample - only the shared
    top side is redundant to rebuild once per comparison). Returns
    {other_name: (top_value, other_value, one_sided_p)} - p is the fraction
    of paired bootstrap resamples in which the top config's advantage over
    `other` disappeared or reversed (Berg-Kirkpatrick et al. 2012)."""
    top_value = metric_fn(pd.concat(top_dict.values(), ignore_index=True))
    other_values = {name: metric_fn(pd.concat(d.values(), ignore_index=True)) for name, d in other_dicts.items()}
    obs_diffs = {name: top_value - v for name, v in other_values.items()}

    n_reversed = {name: 0 for name in other_dicts}
    for _ in range(n_bootstrap):
        chosen = rng.choice(pdf_ids, size=len(pdf_ids), replace=True)
        top_value_b = metric_fn(build_resample(top_dict, chosen))
        for name, d in other_dicts.items():
            diff_b = top_value_b - metric_fn(build_resample(d, chosen))
            if obs_diffs[name] >= 0:
                n_reversed[name] += diff_b <= 0
            else:
                n_reversed[name] += diff_b >= 0

    return {name: (top_value, other_values[name], n_reversed[name] / n_bootstrap) for name in other_dicts}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                         help="each must contain test_predictions.tsv (see predict_from_embeddings.py) and "
                              "classes.json")
    parser.add_argument("--label-col", choices=["document_type", "layout_type", "functional_category"],
                         default="document_type", help="which task --metrics uas/las scores")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                         help="500 gives p-value resolution of 0.002 and keeps a full 12-config x 10-metric run "
                              "to roughly 10-15 minutes; raise for a final report (min achievable p-value is "
                              "1/n_bootstrap).")
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    configs: dict[str, dict[str, pd.DataFrame]] = {}
    classes = None
    for run_dir in args.run_dirs:
        pred_path = run_dir / "test_predictions.tsv"
        if not pred_path.exists():
            print(f"skipping {run_dir} (no test_predictions.tsv - run predict_from_embeddings.py first)")
            continue
        name = run_dir.name
        df = pd.read_csv(pred_path, sep="\t")
        configs[name] = split_by_pdf(df)
        these_classes = json.loads((run_dir / "classes.json").read_text())
        if classes is None:
            classes = these_classes
        elif classes != these_classes:
            raise SystemExit(
                f"{run_dir}'s classes.json doesn't match the others' - these configs weren't trained on the "
                f"same label vocabulary, so their metrics (especially macro-f1) aren't directly comparable."
            )

    if len(configs) < 2:
        raise SystemExit("need at least 2 configs with test_predictions.tsv to compare")

    pdf_id_sets = {name: frozenset(pdfs.keys()) for name, pdfs in configs.items()}
    first_set = next(iter(pdf_id_sets.values()))
    mismatched = {name: s for name, s in pdf_id_sets.items() if s != first_set}
    if mismatched:
        raise SystemExit(f"configs don't share the same test PDFs - not a valid paired comparison: {list(mismatched)}")
    pdf_ids = sorted(first_set)
    print(f"{len(configs)} configs, {len(pdf_ids)} shared test PDFs, {args.n_bootstrap} bootstrap resamples")

    metrics = build_metrics(classes, args.label_col)

    report_lines = [f"{len(configs)} configs compared on {len(pdf_ids)} shared test PDFs "
                     f"(paired bootstrap, n={args.n_bootstrap}, seed={args.seed}); "
                     f"p-values are one-sided (H0: top is not better) and BH-FDR-corrected "
                     f"within each metric (alpha={args.fdr_alpha})."]
    tsv_rows = []

    for metric_name, metric_fn in metrics.items():
        observed = {name: metric_fn(pd.concat(pdfs.values(), ignore_index=True)) for name, pdfs in configs.items()}
        top_name = max(observed, key=observed.get)
        others = [n for n in configs if n != top_name]

        print(f"\n--- {metric_name} (top: {top_name} = {observed[top_name]:.3f}) ---")
        report_lines.append(f"\n--- {metric_name} (top: {top_name} = {observed[top_name]:.3f}) ---")

        results = paired_bootstrap_family(
            metric_fn, configs[top_name], {n: configs[n] for n in others}, pdf_ids, args.n_bootstrap, rng
        )
        rows = [(other_name, *results[other_name]) for other_name in others]
        p_values = [r[3] for r in rows]

        q_values = benjamini_hochberg(p_values)
        for (other_name, top_value, other_value, p), q in sorted(zip(rows, q_values), key=lambda r: r[1]):
            sig = "*" if q < args.fdr_alpha else " "
            line = (f"  {sig} {other_name:<32} {top_name}={top_value:.3f}  vs  {other_value:.3f}  "
                    f"diff={top_value - other_value:+.3f}  p={p:.4f}  q={q:.4f}")
            print(line)
            report_lines.append(line)
            tsv_rows.append({
                "metric": metric_name, "top_config": top_name, "other_config": other_name,
                "top_value": top_value, "other_value": other_value, "diff": top_value - other_value,
                "p_value": p, "q_value": q, "significant": q < args.fdr_alpha,
            })

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(report_lines) + "\n")
        tsv_path = args.out.with_suffix(".tsv")
        pd.DataFrame(tsv_rows).to_csv(tsv_path, sep="\t", index=False)
        print(f"\nWrote {args.out} and {tsv_path}")


if __name__ == "__main__":
    main()
