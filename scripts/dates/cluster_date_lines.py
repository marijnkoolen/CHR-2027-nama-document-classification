"""
cluster_date_lines.py
----------------------
Groups the date-bearing lines extract_dates.py found (data/dates/*.tsv or
*.jsonl) by document type, then clusters them by their *non-date* text, so
that the recurring formulaic phrases of a pre-printed form ("Date : ...",
"Sighted on ...", "X-Ray Film: Dated ...") surface as clusters instead of
being split apart by the many different actual dates on each line.

Pipeline:
  1. mask_date_spans   - replace every date find_dates matched in a line
                          with a placeholder, so lines that only differ by
                          their date collapse to the same text.
  2. cluster_masked_texts - within each predicted_document_type, cluster the
                          *distinct* masked strings by edit-distance
                          similarity (rapidfuzz), using average-linkage
                          agglomerative clustering so a chain of borderline
                          pairs can't drag two truly different phrases into
                          one cluster (see module docstring note below on
                          the similarity threshold).
  3. summarize_clusters - one row per cluster: representative phrase(s) (the
                          most frequent exact variant(s) - simpler and more
                          transparent than synthesizing a consensus string),
                          member count, and a few example source lines for
                          provenance/review.

Output is meant for human review, not as ground truth: cluster labels don't
say what a field *means* (e.g. that "Sighted on <DATE>" is the medical
inspection date) - only that its lines are textually similar. Turning a
reviewed cluster's phrase into a fuzzy search over the full page corpus
(to catch OCR-mangled dates find_dates missed, or slightly different form
wording) is a deliberate follow-up step, not done here.

NOTE ON THE SIMILARITY THRESHOLD: short masked lines make edit-distance
similarity an imperfect proxy for "same field". Concretely, in this
corpus "Sighted on <DATE>" vs "Signed on <DATE>" (different fields) scores
90.9, *higher* than "Date: <DATE>" vs "Datum: <DATE>" (same field, en/nl)
at 88.0 - no single threshold gets both right. The default (90) favours
precision: it won't falsely merge distinct fields, at the cost of leaving
some genuine near-duplicates (like Date:/Datum:) as separate, adjacent
clusters for a reviewer to merge by eye.

Run:
    python scripts/dates/cluster_date_lines.py [--tsv data/dates/safe_dates.tsv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from markdown_text import strip_inline_markdown
from extract_dates import find_dates

# Deliberately shares no substring with any real field-label word (in
# particular not "date" itself - classify_by_formula.py registers "DATE"
# as a real term for some document types, and a placeholder containing
# "date" would fuzzy-match that term against every single masked line,
# since every masked line contains the placeholder by construction).
PLACEHOLDER = "MASKEDVALUE"


def mask_date_spans(line_text: str, placeholder: str = PLACEHOLDER) -> str | None:
    """Strips markdown the same way extract_dates.py does before matching,
    then replaces every date find_dates finds with *placeholder*. Spans are
    replaced right-to-left so an earlier replacement can't shift a later
    span's offsets. Returns None if the (re-derived) text has no date at
    all - re-derives rather than trusting a stored span_start/span_end,
    since those are offsets into the markdown-stripped text while a TSV's
    line_text column is the pre-strip original (see module docstring)."""
    if not line_text:
        return None
    stripped = strip_inline_markdown(line_text)
    hits = find_dates(stripped)
    if not hits:
        return None
    masked = stripped
    for hit in sorted(hits, key=lambda h: h["span_start"], reverse=True):
        masked = masked[: hit["span_start"]] + placeholder + masked[hit["span_end"] :]
    return " ".join(masked.split())


def load_masked_lines(tsv_path: Path, placeholder: str = PLACEHOLDER) -> pd.DataFrame:
    """Reads the extract_dates.py TSV and collapses it to one row per
    distinct (predicted_document_type, masked_text): count (how many source
    rows/lines produced this masked text - a line with N date hits still
    counts once, since mask_date_spans masks all of them together) and a
    handful of example source lines for provenance."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df["masked_text"] = df["line_text"].apply(lambda t: mask_date_spans(t, placeholder))
    unmatched = df["masked_text"].isna().sum()
    if unmatched:
        print(f"[WARN] {unmatched} rows had no date on re-derivation and were dropped", file=sys.stderr)
    df = df.dropna(subset=["masked_text"])

    # one row per physical line: a line with 2+ date hits appears as 2+ TSV
    # rows sharing the same line_text, and therefore the same masked_text -
    # drop_duplicates on (document type, line_text) counts it once.
    lines = df.drop_duplicates(subset=["predicted_document_type", "dossier", "page_num", "line_id", "line_text"])

    grouped = (
        lines.groupby(["predicted_document_type", "masked_text"])
        .agg(
            count=("line_text", "size"),
            examples=("line_text", lambda s: list(dict.fromkeys(s))[:5]),
        )
        .reset_index()
    )
    grouped["has_context"] = grouped["masked_text"] != placeholder
    return grouped


def cluster_masked_texts(
    masked_texts: list[str],
    similarity_threshold: float,
) -> list[int]:
    """Clusters *masked_texts* (assumed already de-duplicated) by edit-
    distance similarity, returning a cluster id per input string.
    Average-linkage agglomerative clustering: a candidate merge needs the
    *average* similarity across all cross-cluster pairs to clear the
    threshold, not just one close pair - deliberately more conservative
    than single-linkage/DBSCAN-with-min_samples=1, which chains through
    borderline pairs and can merge phrases that are individually similar
    only in a transitive, not pairwise, sense."""
    n = len(masked_texts)
    if n == 1:
        return [0]
    similarity = rf_process.cdist(masked_texts, masked_texts, scorer=fuzz.ratio)
    distance = 100.0 - similarity
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=100.0 - similarity_threshold,
        n_clusters=None,
    )
    return clustering.fit_predict(distance).tolist()


def summarize_clusters(grouped: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    """Runs cluster_masked_texts per predicted_document_type and returns one
    row per cluster: representative phrase (the most frequent exact masked
    variant - simpler and more transparent than synthesizing a consensus
    string via alignment), how many distinct variants and total lines it
    covers, and example source lines."""
    rows: list[dict] = []
    for doc_type, group in grouped.groupby("predicted_document_type"):
        group = group.reset_index(drop=True)
        cluster_ids = cluster_masked_texts(group["masked_text"].tolist(), similarity_threshold)
        group = group.assign(cluster_id=cluster_ids)

        for cluster_id, cluster in group.groupby("cluster_id"):
            cluster = cluster.sort_values("count", ascending=False)
            examples = [ex for exs in cluster["examples"] for ex in exs][:5]
            rows.append(
                {
                    "predicted_document_type": doc_type,
                    "cluster_id": int(cluster_id),
                    "representative_phrase": cluster.iloc[0]["masked_text"],
                    "other_phrases": [p for p in cluster["masked_text"].tolist()[1:5]],
                    "n_variants": len(cluster),
                    "n_lines": int(cluster["count"].sum()),
                    "has_context": bool(cluster.iloc[0]["has_context"]),
                    "examples": examples,
                }
            )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["predicted_document_type", "n_lines"], ascending=[True, False]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=Path("data/dates/safe_dates.tsv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/dates/date_line_clusters.tsv"),
        help="Where to write the per-cluster summary.",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=90.0,
        help="rapidfuzz.fuzz.ratio (0-100) below which two masked lines are "
             "never merged into the same cluster. See the module docstring "
             "for why 90 is the default trade-off.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading and masking dates in {args.tsv} ...")
    grouped = load_masked_lines(args.tsv)
    print(f"{len(grouped)} distinct (document type, masked text) pairs "
          f"across {grouped['predicted_document_type'].nunique()} document types")

    print(f"Clustering at similarity >= {args.similarity_threshold} ...")
    summary = summarize_clusters(grouped, args.similarity_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(summary)} clusters to {args.output}")

    for doc_type, group in summary.groupby("predicted_document_type"):
        print(f"\n=== {doc_type} ({len(group)} clusters) ===")
        for _, row in group.head(10).iterrows():
            print(f"  [{row['n_lines']:>5} lines, {row['n_variants']:>3} variants] "
                  f"{row['representative_phrase']!r}")


if __name__ == "__main__":
    main()
