"""
find_formula_variants.py
--------------------------
Runs a fuzzy_search FuzzyPhraseSearcher directly over the per-page markdown
OCR text (not the already-recognised dates in dossier_anchor_dates.tsv) to
find where the REGISTERED FORMULA LABELS THEMSELVES
(docs/project_knowledge-date_context_formulas.json) appear with small
orthographic/OCR deviations - e.g. "Dbte:" for "Date:". This is a
complementary, "reverse" search to the existing date-extraction pipeline:
extract_dates.py finds dates first (by pattern) and classify_by_formula.py
then labels the text around them, so a badly OCR'd date that
extract_dates.py's date-pattern regexes never recognise in the first place
never enters that pipeline at all, however cleanly its accompanying label
read. Searching for the labels directly, independent of whether a date was
ever recognised nearby, surfaces those otherwise-invisible cases.

One FuzzyPhraseSearcher per (predicted_document_type, page_num_in_document)
pair - not just per document type - each built from a REDUCED PhraseModel
containing only the glossary terms with empirical evidence of appearing at
that specific within-document page position (derived from
data/dates/dossier_anchor_dates.tsv, joined against the page-stream
segmentation in the predictions file). A term with no page-position
evidence anywhere for its document type is included on every page of that
type instead of being silently excluded - there's nothing to restrict it
by yet, and excluding it outright risks losing genuine matches purely
because that page position happens to be under-represented in the anchor
data so far.

"page_num_in_document" is NOT the same as the dossier-wide "page_num" - a
dossier's pages are shared across many different document instances back
to back, so "page 16" means something different in every dossier; a
document's OWN 1st/2nd/... page, derived from predicted_segment_id (a
per-dossier counter that increments at every predicted document boundary),
is what's actually comparable across dossiers - the D.1 form's own
back/signature page, for instance, is consistently its page 2, whatever
the dossier-wide page_num happens to be.

Each page's PhraseModel keeps the same distractor relationships
classify_by_formula.build_searcher() uses (every eligible term registered
as every other eligible term's distractor, filter_distractors enabled on
both the PhraseModel and the FuzzyPhraseSearcher) - this still lets a
strong match to one term suppress a weaker match to an orthographically
similar one within the SAME reduced, page-scoped term set.

Output: one row per fuzzy match found, never per line (a line with two
matches produces two rows) - match_phrase/context_formula (the registered
glossary term matched - identical today, kept as two separate columns
since a future glossary with several phrase variants per term could make
them diverge), match_string (the actual OCR'd substring), similarity,
line_num, line_text, dossier, page_num (dossier-wide), document_type,
page_num_in_document.

This script was written without read (or even directory-listing) access to
data/markdown-per-page/, so it is UNTESTED against real OCR content -
verify on a small slice before a full run:
    python scripts/dates/find_formula_variants.py --dossier 1155385
    python scripts/dates/find_formula_variants.py --limit 200

Full run (114,987 pages as of the predictions file this was written
against - this will take a while):
    python scripts/dates/find_formula_variants.py \\
        --predictions data/predictions/predictions-top_combo_latefusion-114k-fixed.tsv \\
        --anchor-dates data/dates/dossier_anchor_dates.tsv \\
        --markdown-root data/markdown-per-page \\
        --output data/dates/formula_variant_matches.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from fuzzy_search import FuzzyPhraseSearcher

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from classify_by_formula import DEFAULT_CONFIG, TERMS_PATH, _entries_for_document_type, load_terms
from extract_dates import read_markdown_lines
from markdown_text import strip_inline_markdown, strip_markdown_fencing

DEFAULT_PREDICTIONS_PATH = Path("data/predictions/predictions-top_combo_latefusion-114k-fixed.tsv")
DEFAULT_MARKDOWN_ROOT = Path("data/markdown-per-page")


def load_page_positions(predictions_path: Path) -> pd.DataFrame:
    """One row per (dossier, page_num): predicted_document_type and
    page_num_in_document - this document instance's own 1st/2nd/... page,
    derived from predicted_segment_id, NOT the dossier-wide page_num (see
    module docstring for why those are different things)."""
    df = pd.read_csv(
        predictions_path, sep="\t",
        usecols=["dossier", "page_num", "predicted_document_type", "predicted_segment_id"],
    )
    df["dossier_id"] = df["dossier"].str.rsplit("-", n=1).str[-1].astype(int)
    df = df.sort_values(["dossier", "page_num"])
    df["page_num_in_document"] = df.groupby(["dossier", "predicted_segment_id"]).cumcount() + 1
    return df


def load_term_page_positions(anchor_dates_path: Path, page_positions: pd.DataFrame) -> dict[tuple[str, str], set[int]]:
    """(predicted_document_type, assigned_term) -> the set of
    page_num_in_document values that term has actually been found at, per
    the already-classified anchor dates - this is the empirical evidence
    the page-scoping in eligible_entries() is built from."""
    anchor = pd.read_csv(anchor_dates_path, sep="\t")
    anchor = anchor.dropna(subset=["assigned_term"])
    merged = anchor.merge(
        page_positions[["dossier_id", "page_num", "page_num_in_document"]],
        left_on=["dossier", "page_num"], right_on=["dossier_id", "page_num"], how="inner",
    )
    positions: dict[tuple[str, str], set[int]] = {}
    for (doc_type, term), group in merged.groupby(["predicted_document_type", "assigned_term"]):
        positions[(doc_type, term)] = set(group["page_num_in_document"].unique())
    return positions


def eligible_entries(
    document_type: str, page_num_in_document: int, terms: list[dict],
    term_page_positions: dict[tuple[str, str], set[int]],
) -> list[dict]:
    """Glossary entries applicable to *document_type* AND, where we have
    empirical evidence at all for that (document_type, term), only if
    *page_num_in_document* is one of the positions it's actually been seen
    at. A term with NO evidence anywhere for this document_type is kept
    unrestricted - safer than guessing it doesn't belong on this page."""
    eligible = []
    for e in _entries_for_document_type(document_type, terms):
        positions = term_page_positions.get((document_type, e["term"]))
        if positions is None or page_num_in_document in positions:
            eligible.append(e)
    return eligible


def build_page_searcher(entries: list[dict]) -> FuzzyPhraseSearcher | None:
    if not entries:
        return None
    names = [e["term"] for e in entries]
    phrase_model = [
        {"phrase": e["term"], "distractors": [n for n in names if n != e["term"]], "label": e.get("role")}
        for e in entries
    ]
    return FuzzyPhraseSearcher(config=DEFAULT_CONFIG, phrase_model=phrase_model)


def drop_nested_matches(matches: list) -> list:
    """Drops a match whose span sits entirely inside another, STRICTLY
    LARGER match's span in the same result set - a short generic term
    (e.g. "Datum") fuzzy-matching a substring of an unrelated, longer
    compound match (e.g. "Date of Birth") is a coincidental artifact, not
    a second real finding, and filter_distractors doesn't catch this
    itself since the two are different registered phrases, not competing
    for the identical span (the same underlying issue
    classify_by_formula.classify_line's span-length tie-break addresses
    for single-best-match classification - there's no masked-date
    placeholder to gate on here, since this script searches raw,
    not-yet-date-extracted text, so the fix here is the more general
    "drop strictly-nested matches" rule instead). Two matches tied on the
    exact same span are both kept - a genuine either/or ambiguity worth a
    human's attention, unlike strict nesting."""
    kept = []
    for m in matches:
        nested_in_larger = any(
            other is not m and other.offset <= m.offset and m.end <= other.end
            and (other.end - other.offset) > (m.end - m.offset)
            for other in matches
        )
        if not nested_in_larger:
            kept.append(m)
    return kept


def iter_page_matches(md_path: Path, searcher: FuzzyPhraseSearcher):
    """Yields (line_num, line_text, match) for every fuzzy match on every
    non-blank line of *md_path* - line numbering and markdown-stripping
    match extract_dates.process_markdown_page exactly, so line_num here is
    directly comparable to dossier_anchor_dates.tsv's line_id."""
    lines = read_markdown_lines(md_path)
    content_lines = strip_markdown_fencing(lines)
    for line_num, line in enumerate(content_lines):
        plain_text = line.strip()
        if not plain_text:
            continue
        stripped_text = strip_inline_markdown(plain_text)
        if not stripped_text:
            continue
        for match in drop_nested_matches(searcher.find_matches(stripped_text)):
            yield line_num, stripped_text, match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--anchor-dates", type=Path, default=Path("data/dates/dossier_anchor_dates.tsv"))
    parser.add_argument("--terms", type=Path, default=TERMS_PATH)
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--output", type=Path, default=Path("data/dates/formula_variant_matches.tsv"))
    parser.add_argument(
        "--dossier", type=int, default=None,
        help="Restrict to one dossier (bare numeric id, e.g. 1155385) - for testing on a small slice first.",
    )
    parser.add_argument("--document-types", type=str, default=None, help="Comma-separated predicted_document_type filter.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many pages - for testing.")
    args = parser.parse_args()

    terms = load_terms(args.terms)
    page_positions = load_page_positions(args.predictions)
    term_page_positions = load_term_page_positions(args.anchor_dates, page_positions)

    rows = page_positions
    if args.dossier is not None:
        rows = rows[rows["dossier_id"] == args.dossier]
    if args.document_types:
        wanted = {d.strip() for d in args.document_types.split(",")}
        rows = rows[rows["predicted_document_type"].isin(wanted)]
    if args.limit:
        rows = rows.head(args.limit)

    searcher_cache: dict[tuple[str, int], FuzzyPhraseSearcher | None] = {}
    n_pages = 0
    n_matches = 0
    n_missing_files = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.writer(out_fh, delimiter="\t")
        writer.writerow([
            "dossier", "page_num", "document_type", "page_num_in_document",
            "line_num", "match_phrase", "context_formula", "match_string",
            "similarity", "line_text",
        ])

        for row in rows.itertuples(index=False):
            n_pages += 1
            key = (row.predicted_document_type, row.page_num_in_document)
            if key not in searcher_cache:
                entries = eligible_entries(row.predicted_document_type, row.page_num_in_document, terms, term_page_positions)
                searcher_cache[key] = build_page_searcher(entries)
            searcher = searcher_cache[key]
            if searcher is None:
                continue

            md_path = args.markdown_root / row.dossier / f"{row.dossier}_page_{row.page_num:04d}.markdown.md"
            if not md_path.exists():
                n_missing_files += 1
                continue

            for line_num, line_text, match in iter_page_matches(md_path, searcher):
                writer.writerow([
                    row.dossier_id, row.page_num, row.predicted_document_type, row.page_num_in_document,
                    line_num, match.phrase.phrase_string, match.phrase.phrase_string,
                    line_text[match.offset:match.end], round(match.levenshtein_similarity, 4), line_text,
                ])
                n_matches += 1

            if n_pages % 2000 == 0:
                print(f"...{n_pages} pages processed, {n_matches} matches so far, {n_missing_files} missing files", file=sys.stderr)

    print(f"Done: {n_pages} pages processed, {n_matches} matches written to {args.output}, {n_missing_files} markdown files not found.")


if __name__ == "__main__":
    main()
