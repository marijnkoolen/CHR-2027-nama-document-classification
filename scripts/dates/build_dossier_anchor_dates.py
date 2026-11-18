"""
build_dossier_anchor_dates.py
------------------------------
Links classify_by_formula.py's per-document-type formula matching back to
individual dossiers, producing a dossier-level table of procedural (not
biographical) anchor dates: application-side signing dates (D.1/D.2/DM.1's
own 'Date'/'Datum' field), approval-side dates (Approval notice, NAMA
agreement), and X-ray dates (wherever 'Film Dated'/'X-ray Report' matched) -
one row per (dossier, matched line), so multiple copies of the same form in
one dossier (re-submissions, corrections) are preserved rather than
collapsed.

Deliberately excludes every 'biographical' and 'none'-role match (Date of
Birth, Date of Marriage, Migration File No., etc.) - this script's whole
purpose is procedural timeline reconstruction, not biographical data, and
dropping 'biographical' rows means no date-of-birth information ever enters
the output, by construction rather than by redaction.

The output carries no raw line text - only dossier id, page number, document
type, the matched term/role, and the normalized date components - so it's
safe to hand off for further analysis without re-exposing OCR'd prose.

Run (needs the full per-line dates TSV, e.g. data/dates/_all_dates.tsv -
NOT one of the files handed to the analysis assistant):
    python scripts/dates/build_dossier_anchor_dates.py \\
        --tsv data/dates/_all_dates.tsv \\
        --output data/dates/dossier_anchor_dates.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_by_formula import TERMS_PATH, build_searcher, classify_line, load_terms
from cluster_date_lines import mask_date_spans

KEPT_ROLES = {"processing-step", "reference"}

OUTPUT_COLUMNS = [
    "dossier",
    "page_num",
    "line_id",
    "predicted_document_type",
    "assigned_term",
    "assigned_role",
    "similarity",
    "day_norm",
    "month_norm",
    "year_norm",
    "date_valid",
    "swapped",
]


def classify_document_type(df: pd.DataFrame, document_type: str, terms: list[dict]) -> pd.DataFrame:
    """Classifies every distinct masked_text for one document_type once,
    then maps the result back onto every row of that type - avoids running
    the fuzzy searcher once per raw line when the same masked_text recurs
    thousands of times."""
    searcher = build_searcher(document_type, terms)
    distinct_texts = df["masked_text"].dropna().unique()
    lookup = {}
    for text in distinct_texts:
        match = classify_line(searcher, text)
        lookup[text] = (match.term, match.role, match.similarity) if match else (None, None, None)

    assigned = df["masked_text"].map(lookup)
    df = df.copy()
    df["assigned_term"] = assigned.map(lambda t: t[0] if isinstance(t, tuple) else None)
    df["assigned_role"] = assigned.map(lambda t: t[1] if isinstance(t, tuple) else None)
    df["similarity"] = assigned.map(lambda t: t[2] if isinstance(t, tuple) else None)
    return df


def build_dossier_anchor_dates(all_dates: pd.DataFrame, terms: list[dict]) -> pd.DataFrame:
    all_dates = all_dates.copy()
    all_dates["masked_text"] = all_dates["line_text"].apply(mask_date_spans)

    classified_parts = []
    for document_type, group in all_dates.groupby("predicted_document_type"):
        classified_parts.append(classify_document_type(group, document_type, terms))
    classified = pd.concat(classified_parts, ignore_index=True)

    kept = classified[classified["assigned_role"].isin(KEPT_ROLES)]
    return kept[OUTPUT_COLUMNS].sort_values(["dossier", "predicted_document_type", "page_num"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=Path("data/dates/_all_dates.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/dates/dossier_anchor_dates.tsv"))
    parser.add_argument("--terms", type=Path, default=TERMS_PATH)
    args = parser.parse_args()

    all_dates = pd.read_csv(args.tsv, sep="\t", dtype=str)
    terms = load_terms(args.terms)

    result = build_dossier_anchor_dates(all_dates, terms)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result)} anchor-date rows for {result['dossier'].nunique()} distinct dossiers to {args.output}")

    print()
    print(result.groupby(["predicted_document_type", "assigned_term"])["dossier"].nunique().sort_values(ascending=False))


if __name__ == "__main__":
    main()
