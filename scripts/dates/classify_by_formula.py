"""
classify_by_formula.py
------------------------
Disambiguates masked date lines between semantically distinct but
orthographically similar formulaic field labels - the cases
cluster_date_lines.py deliberately keeps apart (its similarity threshold
favours precision, see its module docstring), but whose many minor
variants still need sorting into the right field. The motivating example:
"Sighted on <DATE>" and "Signed on <DATE>" are two different fields (90.9
similarity - higher than some same-field pairs cluster_date_lines.py *does*
merge), yet each has dozens of its own OCR-noisy/formatting variants that
should still be matched to the correct one of the two, not left as
singleton clusters.

The confirmed set of distinct fields per document type lives in
docs/project_knowledge-date_context_formulas.json (term, meaning, role,
document_type - the project's human-curated glossary of what each date
context actually means), not here - this module only turns that list into
a fuzzy_search PhraseModel where every term is registered as every other
applicable term's distractor, and its role ("processing-step" /
"biographical" / "reference" / "annotation" - what kind of date context
this is, not just which field) as the phrase's label, then classifies
text against it.

Uses fuzzy_search (the Huygens library, PyPI package "fuzzy-search") rather
than character-edit-distance fuzzy matching (fuzzysearch, tried first):
its word/ngram-based matching correctly handles cases where OCR or a form
variant inserts extra tokens between a field label and the date
placeholder ("Sighted on = B <DATE>"), which a pure Levenshtein-budget
substring search misses - validated on the real Sighted/Signed variants
before switching.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from fuzzy_search import FuzzyPhraseSearcher

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_date_lines import PLACEHOLDER

TERMS_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "project_knowledge-date_context_formulas.json"

DEFAULT_CONFIG = {
    "ignorecase": True,
    "filter_distractors": True,
    "levenshtein_threshold": 0.6,
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
}


class FormulaMatch(NamedTuple):
    term: str
    role: str | None
    similarity: float


def load_terms(terms_path: Path = TERMS_PATH) -> list[dict]:
    return json.loads(terms_path.read_text(encoding="utf-8"))


def _entries_for_document_type(document_type: str, terms: list[dict]) -> list[dict]:
    """Entries that apply to *document_type*: those explicitly scoped to
    it, plus every universally-scoped entry (document_type: [])."""
    return [
        t for t in terms
        if not t["document_type"] or document_type in t["document_type"]
    ]


def terms_for_document_type(document_type: str, terms: list[dict] | None = None) -> list[str]:
    """Term names that apply to *document_type* - see _entries_for_document_type."""
    terms = terms if terms is not None else load_terms()
    return [e["term"] for e in _entries_for_document_type(document_type, terms)]


def build_searcher(
    document_type: str,
    terms: list[dict] | None = None,
    config: dict | None = None,
) -> FuzzyPhraseSearcher:
    """Builds a FuzzyPhraseSearcher for *document_type* where every
    applicable term is registered as every other applicable term's
    distractor, so find_matches only returns a term when it's a closer
    match than every orthographically similar alternative
    (filter_distractors). Each term's role ("processing-step" /
    "biographical" / "reference" / "annotation") is registered as the
    phrase's label, so a match carries not just which field it is but what
    kind of date context that field represents."""
    terms = terms if terms is not None else load_terms()
    entries = _entries_for_document_type(document_type, terms)
    names = [e["term"] for e in entries]
    phrase_model = [
        {
            "phrase": e["term"],
            "distractors": [other for other in names if other != e["term"]],
            "label": e.get("role"),
        }
        for e in entries
    ]
    return FuzzyPhraseSearcher(config=config or DEFAULT_CONFIG, phrase_model=phrase_model)


def classify_line(searcher: FuzzyPhraseSearcher, text: str) -> FormulaMatch | None:
    """Returns the best-matching term for *text* (highest
    levenshtein_similarity among the matches find_matches returns, ties
    broken by the longer/more specific match span), or None if nothing
    matched within the searcher's thresholds.

    A short registered term that's a prefix of a longer one ("Date" vs.
    "Date of Birth") scores an identical 1.0 exact-substring similarity on
    any line containing the longer phrase - filter_distractors doesn't
    resolve this because both really are valid matches, just at different
    spans. Breaking ties by span length (matched-character count) prefers
    the more specific/informative phrase instead of falling back to
    whichever match happens to come first in find_matches's return order
    (effectively arbitrary - it tracked phrase registration order, which
    silently favoured "Date" over "Date of Birth" simply because "Date"
    was registered earlier in the terms file).

    Also requires at least one match to start before some occurrence of
    the masked-date placeholder in *text* - every field label in the
    glossary precedes the date it labels ("Date: <DATE>", "Sighted on
    <DATE>", "your letter of <DATE>"), never follows it. A generic term
    like "Date" can otherwise fuzzy-match the literal word inside an
    unrelated idiom sitting AFTER the date it has nothing to do with
    ("<DATE> - to date", "<DATE> up to the present date" - both meaning
    "ongoing to the present", not a dated field at all). This is a gate,
    not a hard filter on individual matches: if at least one before-the-
    placeholder match exists, every match (before or after) still competes
    normally on similarity - an after-placeholder match only gets
    disqualified when it's the ONLY evidence for this line, so it can't
    force a weak before-placeholder match to win over a strong
    after-placeholder one it doesn't actually conflict with (e.g. "Date"
    matching a trailing "(date)" annotation stays the best answer on a
    line that also weakly, coincidentally matches an unrelated
    abbreviation earlier in the same line)."""
    matches = searcher.find_matches(text)
    if not matches:
        return None
    placeholder_positions = [m.start() for m in re.finditer(re.escape(PLACEHOLDER), text)]
    if placeholder_positions and not any(m.offset < pos for m in matches for pos in placeholder_positions):
        return None
    best = max(matches, key=lambda m: (m.levenshtein_similarity, m.end - m.offset))
    return FormulaMatch(best.phrase.phrase_string, best.phrase.label, best.levenshtein_similarity)


def classify_variants(
    masked_texts: list[str],
    document_type: str,
    terms: list[dict] | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Runs classify_line over a batch of masked lines for one document
    type (e.g. the distinct masked_text values
    cluster_date_lines.load_masked_lines produces), returning one row per
    input with the assigned term (None if nothing matched)."""
    searcher = build_searcher(document_type, terms, config)
    rows = []
    for text in masked_texts:
        match = classify_line(searcher, text)
        rows.append(
            {
                "masked_text": text,
                "assigned_term": match.term if match else None,
                "assigned_role": match.role if match else None,
                "similarity": match.similarity if match else None,
            }
        )
    return rows


def main() -> None:
    from cluster_date_lines import load_masked_lines

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=Path("data/dates/safe_dates.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/dates/classified_dates.tsv"))
    parser.add_argument("--terms", type=Path, default=TERMS_PATH)
    args = parser.parse_args()

    grouped = load_masked_lines(args.tsv)
    terms = load_terms(args.terms)

    all_rows = []
    for document_type, group in grouped.groupby("predicted_document_type"):
        rows = classify_variants(group["masked_text"].tolist(), document_type, terms)
        for row, count in zip(rows, group["count"]):
            row["predicted_document_type"] = document_type
            row["n_lines"] = count
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result)} classified (distinct masked) lines to {args.output}")

    print()
    summary = result.groupby(["predicted_document_type", "assigned_role"], dropna=False)["n_lines"].sum()
    print(summary)


if __name__ == "__main__":
    main()
