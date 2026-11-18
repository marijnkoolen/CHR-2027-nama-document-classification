"""
mine_candidate_formulas.py
----------------------------
Mines candidate date-context formulas from cluster_date_lines.py's cluster
summary, for human review before promoting them into
docs/project_knowledge-date_context_formulas.json.

Splits each document type's clusters into two populations by word count:
  - form-field phrases (<= --word-count-cutoff words): the pre-printed
    form's own field labels ("Date :", "Sighted on") - short, structured.
  - running-text phrases (longer): narrative sentences that happen to
    contain a date. Most of these are NOT processing-step dates (see
    date_context_formulas.json's "role" field and its module docstring in
    classify_by_formula.py) - they're biographical facts, contractual
    clauses, or regulatory/treaty references. A few genuinely are (e.g.
    the background check's "...disclose at this date" closing sentence).
    Read the actual matched lines before promoting a running-text
    candidate, not just the n-gram frequency - a frequent n-gram here is
    NOT reliable evidence of being a processing step (the "Yellow Fever"
    regulation clause was the highest-frequency Approval notice
    running-text bigram and is not a processing-step date at all).

For each population, mines the most frequent word n-grams (1-4 words,
weighted by each cluster's line count - cluster_date_lines.py already
collapsed exact/near-duplicate variants, so this weight approximates how
many actual source lines each n-gram represents). The placeholder itself
is kept as a token (see tokenize_with_placeholder), so an n-gram can show
the anchor context on EITHER side of a masked date - "sighted on
MASKEDVALUE" as well as "MASKEDVALUE (date)" - not just what precedes it.
Unigrams are filtered against a small English/Dutch stopword list (a bare
stopword would otherwise dominate every document type's top-N the way a
stopword pair doesn't). Drops any n-gram that's already a substring of a
currently-registered term for that document type (placeholder stripped
first), so the output surfaces what's NOT yet catalogued rather than
re-surfacing "sighted on MASKEDVALUE" once "Sighted on" is a confirmed
formula.

Run (after cluster_date_lines.py has produced its cluster summary):
    python scripts/dates/mine_candidate_formulas.py \\
        --clusters data/dates/date_line_clusters.tsv \\
        --output data/dates/candidate_formulas.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_by_formula import TERMS_PATH, load_terms

WORD_COUNT_CUTOFF = 10
# must match cluster_date_lines.PLACEHOLDER - not imported directly to
# keep this script usable against any TSV produced with an older/newer
# placeholder, as long as --placeholder is passed to match it.
PLACEHOLDER = "MASKEDVALUE"


def tokenize_with_placeholder(phrase: str, placeholder: str = PLACEHOLDER) -> list[str]:
    """Tokenizes *phrase* keeping the placeholder as its own token, in its
    original position and case, rather than stripping it out. This lets an
    n-gram show the anchor context on BOTH sides of a masked date
    ("sighted on MASKEDVALUE", "MASKEDVALUE (date)"), while still
    correctly stopping unrelated words on either side from being treated
    as adjacent - the placeholder token itself sits between them, so
    "born MASKEDVALUE at Amsterdam" can never produce a bare ("born",
    "at") bigram (an earlier version of this script did exactly that by
    discarding the placeholder before tokenizing)."""
    pattern = re.escape(placeholder) + r"|[A-Za-zÀ-ÿ]+"
    return re.findall(pattern, phrase)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def already_covered(
    ngram_tokens: tuple[str, ...],
    registered_terms: list[str],
    placeholder: str = PLACEHOLDER,
) -> bool:
    """True if *ngram_tokens*, with any placeholder token removed, appears
    as a contiguous run within any already-registered term's own tokens
    (case-insensitive) - i.e. this n-gram isn't telling us anything new.
    Stripping the placeholder first means "sighted on MASKEDVALUE" is
    recognised as covered once "Sighted on" is registered, not kept
    around as a redundant near-duplicate. Registered term strings never
    contain the placeholder, so tokenizing them needs no substitution -
    just word-extraction (an empty-string placeholder here would make
    str.replace("", " ") insert a space between every character)."""
    real_tokens = [t for t in ngram_tokens if t != placeholder]
    if not real_tokens:
        return False
    ngram_str = " ".join(t.lower() for t in real_tokens)
    for term in registered_terms:
        term_str = " ".join(t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ]+", term))
        if ngram_str in term_str:
            return True
    return False


# Common English/Dutch function words, excluded from unigram mining only -
# bigrams/trigrams are naturally selective enough that stopword pairs
# ("of the") just sink to the bottom of the ranking, but a bare stopword
# unigram would otherwise dominate every document type's top-N.
_UNIGRAM_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "been",
    "were", "also", "your", "you", "are", "was", "not", "but", "all",
    "any", "who", "which", "will", "shall", "would", "should", "must",
    "can", "may", "has", "had", "his", "her", "him", "she", "they",
    "them", "their", "our", "its", "been", "being",
    "de", "het", "een", "van", "voor", "met", "aan", "bij", "als",
    "dat", "die", "deze", "dit", "wordt", "worden", "zijn", "niet",
    "geen", "ook", "maar", "en", "of", "te", "in", "op",
}


def mine_ngrams_for_group(
    clusters: pd.DataFrame,
    document_type: str,
    running_text: bool,
    word_count_cutoff: int,
    registered_terms: list[str],
    top_n: int,
    placeholder: str,
) -> pd.DataFrame:
    g = clusters[
        (clusters["predicted_document_type"] == document_type) & (clusters["has_context"])
    ].copy()
    g["word_count"] = g["representative_phrase"].str.split().apply(len)
    g = g[g["word_count"] > word_count_cutoff] if running_text else g[g["word_count"] <= word_count_cutoff]

    def normalize(token: str) -> str:
        # keep the placeholder visually distinct (and un-lowercased) in
        # ngram output, so a candidate like "sighted on MASKEDVALUE"
        # immediately shows where the date sits relative to the anchor text.
        return token if token == placeholder else token.lower()

    counters: dict[int, Counter] = {n: Counter() for n in (1, 2, 3, 4)}
    for _, row in g.iterrows():
        tokens = tokenize_with_placeholder(row["representative_phrase"], placeholder)
        for size, counter in counters.items():
            for ng in ngrams(tokens, size):
                if size == 1:
                    token = ng[0]
                    if token == placeholder:
                        continue  # a bare masked date isn't an informative "formula" on its own
                    normalized_token = token.lower()
                    if len(normalized_token) < 4 or normalized_token in _UNIGRAM_STOPWORDS:
                        continue
                    counter[(normalized_token,)] += row["n_lines"]
                elif any(t != placeholder for t in ng):
                    # skip n-grams made entirely of placeholder tokens
                    # ("MASKEDVALUE MASKEDVALUE", from adjacent dates with
                    # nothing between them) - no anchor text, no signal
                    counter[tuple(normalize(t) for t in ng)] += row["n_lines"]

    rows = []
    for size, counts in sorted(counters.items()):
        for ngram, count in counts.most_common(top_n):
            if already_covered(ngram, registered_terms, placeholder):
                continue
            rows.append(
                {
                    "predicted_document_type": document_type,
                    "context": "running-text" if running_text else "form-field",
                    "ngram_size": size,
                    "ngram": " ".join(ngram),
                    "weighted_count": count,
                }
            )
    return pd.DataFrame(rows)


def mine_candidate_formulas(
    clusters: pd.DataFrame,
    terms: list[dict],
    word_count_cutoff: int = WORD_COUNT_CUTOFF,
    top_n: int = 25,
    placeholder: str = PLACEHOLDER,
) -> pd.DataFrame:
    all_rows = []
    for document_type in sorted(clusters["predicted_document_type"].unique()):
        registered = [
            t["term"] for t in terms
            if not t["document_type"] or document_type in t["document_type"]
        ]
        for running_text in (False, True):
            all_rows.append(
                mine_ngrams_for_group(
                    clusters, document_type, running_text, word_count_cutoff,
                    registered, top_n, placeholder,
                )
            )
    result = pd.concat(all_rows, ignore_index=True)
    return result.sort_values(
        ["predicted_document_type", "context", "weighted_count"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=Path("data/dates/date_line_clusters.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/dates/candidate_formulas.tsv"))
    parser.add_argument(
        "--word-count-cutoff", type=int, default=WORD_COUNT_CUTOFF,
        help="Cluster phrases at or under this many words are treated as form-field "
             "labels; longer ones as running text.",
    )
    parser.add_argument(
        "--top-n", type=int, default=25,
        help="How many top n-grams (per size, 1-4 words) to keep per document type/context.",
    )
    parser.add_argument("--terms", type=Path, default=TERMS_PATH)
    parser.add_argument("--placeholder", default=PLACEHOLDER)
    args = parser.parse_args()

    clusters = pd.read_csv(args.clusters, sep="\t")
    terms = load_terms(args.terms)

    result = mine_candidate_formulas(
        clusters, terms, args.word_count_cutoff, args.top_n, args.placeholder,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result)} candidate n-grams to {args.output}")

    for (doc_type, context), group in result.groupby(["predicted_document_type", "context"]):
        print(f"\n=== {doc_type} / {context} ({len(group)} candidates) ===")
        for _, row in group.head(10).iterrows():
            print(f"  [{row['weighted_count']:>5}] {row['ngram']}")


if __name__ == "__main__":
    main()
