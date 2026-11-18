"""
extract_dates.py
----------------
Extract every text line that contains a recognisable date from the PageXML
files that correspond to pages listed in annotations.

Results are written to:
  data/dates/<dossier-id>.jsonl  – one JSON object per date-bearing line
  data/dates/_all_dates.jsonl    – combined file across all annotated dossiers

Run from the workspace root:
    python scripts/extract_dates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from markdown_text import strip_inline_markdown, strip_markdown_fencing
from normalize_dates import (
    _EN_MONTHS_FULL,
    _EN_MONTHS_ABBR,
    _NL_MONTHS_FULL,
    _NL_MONTHS_ABBR,
    normalize_hit,
)


# --------------------------------------------------------------------------- #
# Building blocks
#
# The markdown is OCR output, so every "space" between date components may in
# fact be a comma, an abbreviation dot, a dash, several spaces, a non-breaking
# space, or nothing at all. The blocks below are therefore deliberately
# permissive about *separators* while staying strict about the *components*
# (a month name, 1-2 digit day, 2-4 digit year).
# --------------------------------------------------------------------------- #

# straight and curly apostrophes / primes, as in "'60" or "’60"
_APOS = r"['‘’ʼ`´]"

# horizontal whitespace only (never a newline - lines are matched one by one)
_SP = r"[ \t ]"

# separators inside numeric dates: hyphen variants, slash, dot
_NUM_SEP = r"[-‐‑–—/.]"

# the "glue" between date components: any run of spaces and/or punctuation,
# possibly empty (OCR often drops the space: "13 Juni1933")
_GAP = rf"(?:{_SP}|[,;:.·‐‑–—-])*"

# English and Dutch ordinal suffixes: 1st, 2nd, 3rd, 27th, 21ste, 27sten, 14e
_ORD = r"(?:sten|ste|nd|rd|th|st|den|de|e)\.?"

# _EN_MONTHS_*/_NL_MONTHS_* (full names and abbreviations) live in
# normalize_dates.py, which needs the same vocabulary to resolve a
# captured month string back to a month number - imported above.


def _alternation(*name_lists: list[str]) -> str:
    """Longest alternative first, so "September" wins over "Sep" and "Sept"
    over "Sep" without relying on backtracking."""
    names = sorted({n for lst in name_lists for n in lst}, key=len, reverse=True)
    return "(?:" + "|".join(names) + ")"


_MONTH_FULL = _alternation(_EN_MONTHS_FULL, _NL_MONTHS_FULL)
_MONTH_ABBR = _alternation(_EN_MONTHS_ABBR, _NL_MONTHS_ABBR)

# Only an *abbreviation* may carry a trailing dot. A dot after a full month
# name is a sentence period, not an abbreviation dot - letting it be absorbed
# here is what would turn "...in September. 1960 was a hard year" into a date.
# The dot is matched but kept out of the captured group, so the month field is
# always the bare name ("Sept"). find_dates merges the two groups.
#
# The trailing (?![A-Za-z]) stops a longer name from being backtracked into a
# shorter one to satisfy the rest of a pattern - without it "1960 Sept 27"
# matches as month="Sep", stranding the "t".
_MONTH = rf"(?:(?P<month>{_MONTH_FULL})|(?P<month_abbr>{_MONTH_ABBR})\.?)(?![A-Za-z])"

# day, optionally with an ordinal suffix. (?<!\d) / (?!\d) stop the day from
# being carved out of a longer number such as a year or a file number.
_DAY = rf"(?<!\d)(?P<day>\d{{1,2}})(?!\d)(?:{_ORD})?"

# year: 4 digits, or 2 digits optionally introduced by an apostrophe ("'60")
_YEAR = rf"(?:{_APOS}{_SP}*)?(?P<year>\d{{2,4}})(?!\d)"

# Day-less patterns (month+year, year+month) carry much weaker evidence that
# they are a date at all, so they get two extra restrictions:
#
#   _GAP_TIGHT   no dots and no dashes in the glue, so a sentence boundary
#                ("...in 1960. September was quiet") cannot join a year to a
#                month, and a range ("1960-1965") cannot look like one.
#   _YEAR_BARE   a bare 2-digit number next to a month name is far more often
#                a quantity than a year, so require either an apostrophe or a
#                plausible 4-digit year - which also keeps reference numbers
#                ("No. 1234 Mar") from being read as dates. Widen the range
#                here if the corpus ever reaches outside 1500-2099.
_GAP_TIGHT = rf"(?:{_SP}|[,;:])*"
_YEAR_BARE = (
    rf"(?:{_APOS}{_SP}*(?P<year_short>\d{{2}})|(?P<year>1[5-9]\d\d|20\d\d))(?!\d)"
)

# numeric separator run, tolerating spaces around it ("13 - 7 - '56")
_NUM_GAP = rf"{_SP}*{_NUM_SEP}+{_SP}*"

# Set to False to drop the noisiest pattern (day + month with no year at all).
INCLUDE_DAY_MONTH_WITHOUT_YEAR = True

# Still uncaptured on purpose:
#   - month names split by OCR ("Januar y") - see parse_dates.resolve_month
#   - misspelled month names ("Janaury")   - see parse_dates.resolve_month
#   - bare years ("in 1960"), far too noisy to extract as dates
DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "15 February 1909" / "29 JULI 1956" / "27th Sept., '60" / "27-Sept-60"
    # / "13 Juni1933" / "21ste Juli 1956"
    (
        "day_monthname_year",
        re.compile(rf"{_DAY}{_GAP}{_MONTH}{_GAP}{_YEAR}", re.IGNORECASE),
    ),
    # "February 15, 1909" / "February 15th, '09" / "Sept. 27 1960"
    (
        "monthname_day_year",
        re.compile(
            rf"\b{_MONTH}{_GAP}(?P<day>\d{{1,2}})(?!\d)(?:{_ORD})?{_GAP}{_YEAR}",
            re.IGNORECASE,
        ),
    ),
    # DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY / "13-7-'56" / "13 - 7 - 1956"
    (
        "numeric_dmy",
        re.compile(
            rf"(?<!\d)(?P<day>\d{{1,2}}){_NUM_GAP}(?P<month>\d{{1,2}}){_NUM_GAP}{_YEAR}"
        ),
    ),
    # YYYY-MM-DD (ISO)
    (
        "numeric_ymd",
        re.compile(
            rf"(?<!\d)(?P<year>\d{{4}}){_NUM_GAP}(?P<month>\d{{1,2}}){_NUM_GAP}(?P<day>\d{{1,2}})(?!\d)"
        ),
    ),
    # "February 1909" / "Sept. '60" / "Sept., 1960"  (no day)
    (
        "monthname_year",
        re.compile(rf"\b{_MONTH}{_GAP_TIGHT}{_YEAR_BARE}", re.IGNORECASE),
    ),
    # "1960 Sept." / "1909 February" / "'60 Sept"  (no day, year first)
    (
        "year_monthname",
        re.compile(
            rf"(?<!\d){_YEAR_BARE}{_GAP_TIGHT}{_MONTH}(?!{_GAP_TIGHT}\d)",
            re.IGNORECASE,
        ),
    ),
]

if INCLUDE_DAY_MONTH_WITHOUT_YEAR:
    # "27th Sept." with no year in sight. The look-ahead keeps it from firing
    # where one of the year-bearing patterns above already applies.
    DATE_PATTERNS.append(
        (
            "day_monthname",
            re.compile(rf"{_DAY}{_GAP}{_MONTH}(?!{_GAP}\d)", re.IGNORECASE),
        )
    )


def find_dates(text: str) -> list[Dict[str, str|int]]:
    """Return a list of date-match dicts found in *text*.

    Patterns are tried most-specific first; any match that overlaps an
    already-accepted match is dropped, so "27th Sept., '60" is reported once
    as a full date rather than also as a bare day+month.

    Each hit carries two labels for filtering downstream: "pattern" names the
    exact shape that matched, and "components" says which fields are present -
    "dmy" for a complete date, "my" for a day-less month+year (either order),
    "dm" for a day+month with no year. Selecting only complete dates is
    therefore `rec["components"] == "dmy"`, independent of pattern names.
    """
    hits: list[Dict[str, str|int]] = []
    taken: list[tuple[int, int]] = []

    for pattern_name, pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            # skip if it overlaps an earlier (higher-priority) match at all
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            gd = m.groupdict()
            day = gd.get("day")
            month = gd.get("month") or gd.get("month_abbr")
            year = gd.get("year") or gd.get("year_short")
            hits.append(
                {
                    "pattern": pattern_name,
                    "components": (
                        ("d" if day else "")
                        + ("m" if month else "")
                        + ("y" if year else "")
                    ),
                    "date_string": m.group(0),
                    "day": day,
                    "month": month,
                    "year": year,
                    "span_start": span[0],
                    "span_end": span[1],
                }
            )

    hits.sort(key=lambda h: h["span_start"])
    return hits


def coords_to_dict(coords) -> Dict[str, int]:
    return {
        "x": coords.x,
        "y": coords.y,
        "w": coords.w,
        "h": coords.h,
        "left": coords.left,
        "top": coords.top,
        "right": coords.right,
        "bottom": coords.bottom,
        "points": coords.points,
    }


def load_annotations(csv_path: Path) -> dict[str, dict[str, dict]]:
    """Load all_annotations.csv.

    Returns a nested dict:  dossier_id -> page_number_str -> annotation_row
    The dossier key has the .pdf suffix stripped.
    """
    index: dict[str, dict[str, dict]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dossier = row["dossier"].removesuffix(".pdf")
            page = row["page_number"]
            index.setdefault(dossier, {})[page] = row
    return index


def process_pagexml_page(
    xml_path: Path,
    dossier_id: str,
    page_name: str,
    earliest_year: int,
    latest_year: int,
) -> list[Dict[str, str|int]]:

    # imported lazily so the date patterns can be imported/tested without
    # the pagexml dependency installed
    from pagexml.parser import parse_pagexml_file

    try:
        scan = parse_pagexml_file(str(xml_path))
    except Exception as exc:
        print(f"  [WARN] could not parse {xml_path}: {exc}", file=sys.stderr)
        return []

    ann_meta: dict = {}

    records: list[dict] = []
    for line in scan.get_lines():
        text = line.text
        if not text:
            continue

        date_hits = find_dates(text)
        if not date_hits:
            continue

        coords_dict = coords_to_dict(line.coords) if line.coords else None

        for hit in date_hits:
            records.append(
                {
                    "dossier": dossier_id,
                    "page": page_name,
                    "line_id": line.id,
                    "line_text": text,
                    "coords": coords_dict,
                    **ann_meta,
                    **normalize_hit(hit, earliest_year, latest_year),
                }
            )

    return records


def parse_md_filename(md_path: Path) -> tuple[int, int]:
    """Return (dossier_id, page_number) parsed from a per-page markdown filename."""
    part1, part2 = md_path.name.split('_page_')
    return int(part1.split('-')[-1]), int(part2[:4])


def read_markdown_lines(md_path: Path):
    with open(md_path, 'rt') as fh:
        return fh.readlines()


def write_dossier_dates(dossier_id: int, dossier_records: list[Dict[str, str|int|None]],
                        output_root: Path):
    # write per-dossier JSONL
    out_path = output_root / f"dates-{dossier_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in dossier_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_markdown_page(
    md_path: Path,
    dossier_id: int,
    page_num: int,
    earliest_year: int,
    latest_year: int,
) -> list[Dict[str, str|int|None]]:

    records: list[Dict[str, str|int|None]] = []

    lines = read_markdown_lines(md_path)
    content_lines = strip_markdown_fencing(lines)
    
    for li, line in enumerate(content_lines):
        # Strip markdown markup (e.g. the "*"/"**" around an emphasised
        # date) before matching - the DATE_PATTERNS regexes above expect
        # plain prose, not markdown syntax. Uses strip_inline_markdown, not
        # strip_markdown: the latter renders full CommonMark block grammar,
        # which reads a leading "<digits>. " as an ordered-list marker and
        # discards it - silently eating the day off dates like
        # "14. 3.1928" -> "3.1928". strip_inline_markdown only removes
        # matched emphasis/code delimiters, so it can't do that, and never
        # introduces line breaks that weren't already there.
        plain_text = line.strip()
        if not plain_text:
            continue

        stripped_text = strip_inline_markdown(plain_text)
        if not stripped_text:
            continue

        plain_date_hits = find_dates(plain_text)
        stripped_date_hits = find_dates(stripped_text)
        if len(plain_date_hits) > len(stripped_date_hits):
            print(f"plain_text: {plain_text}\n\t{plain_date_hits}")
            print(f"stripped_text: {stripped_text}\n\t{stripped_date_hits}")
        date_hits = stripped_date_hits
        if not date_hits:
            continue

        for hit in date_hits:
            records.append(
                {
                    "dossier": dossier_id,
                    "page": page_num,
                    "line_id": li,
                    "line_text": stripped_text,
                    "coords": None,
                    **normalize_hit(hit, earliest_year, latest_year),
                }
            )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--creation-period-end", type=int, default=1965,
        help="Last year the forms in this dataset were filled out. Sets the "
             "highest year normalize_dates.normalize_hit will accept "
             "(creation-period-end + 1), e.g. 1965 for the 1950-1965 dataset "
             "or 1990 for the 1966-1990 one.",
    )
    parser.add_argument(
        "--earliest-year", type=int, default=1870,
        help="Earliest year normalize_hit will accept for a birth year "
             "(these can reach into the 19th century).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    earliest_year = args.earliest_year
    latest_year = args.creation_period_end + 1

    root = Path(__file__).resolve().parent.parent.parent
    print(f"root: {root}")
    text_type = 'markdown'
    if text_type == 'pagexml':
        input_root = root / "data/pagexml-per-page"
    elif text_type == 'markdown':
        input_root = root / "data/markdown-per-page"
    else:
        raise ValueError(f"Invalid 'text_type '{text_type}', must be one of ['pagexml', 'markdown'].")
    output_root = root / "data" / "dates"
    output_root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, str|int|None]] = []

    print(f"input root: {input_root} - exists: {input_root.exists()}")
    # Sort by (dossier, page): the loop below streams one dossier at a time and
    # writes its JSONL when the dossier id changes, so a dossier's pages must be
    # contiguous. Raw glob order is filesystem order, which can interleave
    # dossiers - the second run of a dossier would then overwrite the JSONL
    # written by the first.
    md_files = sorted(input_root.glob('**/*.markdown.md'), key=parse_md_filename)
    print(f"number of md files: {len(md_files)}")

    if len(md_files) == 0:
        print(f"Nothing to do. Exiting\n")
        sys.exit(0)

    dossier_records: list[dict[str, str|int|None]] = []
    prev_dossier = None

    per_doc_count = 0
    for mi, md_file in enumerate(md_files):
        curr_dossier, page_num = parse_md_filename(md_file)

        # page_annotations = annotations[dossier_id]
        if curr_dossier != prev_dossier:
            if prev_dossier is not None:
                write_dossier_dates(prev_dossier, dossier_records, output_root)
            per_doc_count += len(dossier_records)
            all_records.extend(dossier_records)
            dossier_records: list[dict[str, str|int|None]] = []

        hits = process_markdown_page(md_file, curr_dossier, page_num, earliest_year, latest_year)
        dossier_records.extend(hits)
        prev_dossier = curr_dossier


        if (mi+1) % 1000 == 0:
            print(f"{mi+1} of {len(md_files)} processed, {per_doc_count} records (all: {len(all_records)})")

    write_dossier_dates(prev_dossier, dossier_records, output_root)
    all_records.extend(dossier_records)
    per_doc_count += len(dossier_records)

    # write combined JSONL
    combined_path = output_root / "_all_dates.jsonl"
    with combined_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{mi+1} of {len(md_files)} processed, {per_doc_count} records (all: {len(all_records)})")
    print(f"\nDone. {len(all_records)} total date hits across all dossiers.")


if __name__ == "__main__":
    main()
