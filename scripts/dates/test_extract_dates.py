"""
test_extract_dates.py
---------------------
Regression tests for the date patterns in extract_dates.find_dates.

The corpus is OCR output of mostly English and some Dutch typescript, so the
cases below are organised around the ways OCR mangles a date: punctuation where
a space should be, abbreviation dots, apostrophe years, ordinals, and missing
spaces. Each case is a string shape actually worth extracting - when you widen
a pattern, add the shape here first.

The NEGATIVE cases matter as much as the positive ones: the patterns are
deliberately permissive about separators, and the failure mode of a permissive
date regex is not "no match", it is "a plausible-looking wrong match" carved
out of a file number or a quantity.

Run standalone (no dependencies):
    python scripts/dates/test_extract_dates.py

Or under pytest:
    pytest scripts/dates/test_extract_dates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_dates import find_dates


# (text, (day, month, year)) - at least one hit must have exactly these fields.
# `None` for a field means the pattern is not expected to capture it.
POSITIVE_CASES: list[tuple[str, tuple[str | None, str | None, str | None]]] = [
    # --- the shapes that motivated this pattern set -----------------------
    ("27th Sept., '60", ("27", "Sept", "60")),
    ("27th Sep 60", ("27", "Sep", "60")),
    ("27 Sept. 1960", ("27", "Sept", "1960")),
    ("Sept. '60", (None, "Sept", "60")),
    ("Sept., 1960", (None, "Sept", "1960")),

    # --- ordinal suffixes, English and Dutch ------------------------------
    ("1st October 1953", ("1", "October", "1953")),
    ("2nd Jan. '47", ("2", "Jan", "47")),
    ("3rd Aug 1949", ("3", "Aug", "1949")),
    ("21ste Juli 1956", ("21", "Juli", "1956")),
    ("27sten Sept. 1960", ("27", "Sept", "1960")),
    ("14e Mei 1940", ("14", "Mei", "1940")),

    # --- punctuation standing in for the separator ------------------------
    ("15, February, 1909", ("15", "February", "1909")),
    ("15 February,1909", ("15", "February", "1909")),
    ("February 15, 1909", ("15", "February", "1909")),
    ("February 15th, '09", ("15", "February", "09")),
    ("27-Sept-60", ("27", "Sept", "60")),
    ("27 . Sept . 1960", ("27", "Sept", "1960")),
    ("29 JULI 1956", ("29", "JULI", "1956")),

    # --- OCR noise: dropped spaces, curly apostrophe, non-breaking space ---
    ("13 Juni1933", ("13", "Juni", "1933")),
    ("27th Sept.,’60", ("27", "Sept", "60")),
    ("27 Sept. 1960", ("27", "Sept", "1960")),

    # --- month abbreviations. "Oct" in particular was absent from both the
    # English and Dutch lists, so English `5 Oct 1947` never matched.
    ("5 Oct 1947", ("5", "Oct", "1947")),
    ("5 Okt. 1947", ("5", "Okt", "1947")),
    ("5 Febr. 1947", ("5", "Febr", "1947")),
    ("5 Mrt. '47", ("5", "Mrt", "47")),

    # --- numeric ----------------------------------------------------------
    ("13-7-'56", ("13", "7", "56")),
    ("13 - 7 - 1956", ("13", "7", "1956")),
    ("13.7.56", ("13", "7", "56")),
    ("13/07/1956", ("13", "07", "1956")),
    ("1956-07-13", ("13", "07", "1956")),

    # --- day + month with no year in the line -----------------------------
    ("on 27th Sept. the ship", ("27", "Sept", None)),

    # --- in running text --------------------------------------------------
    ("No. 15 September 1960", ("15", "September", "1960")),

    # --- day-less month+year, both orders (pattern "monthname_year" /
    # "year_monthname", components "my") -----------------------------------
    ("February 1909", (None, "February", "1909")),
    ("Sept. 1960", (None, "Sept", "1960")),
    ("in 1960 Sept the", (None, "Sept", "1960")),
    ("1909 February", (None, "February", "1909")),
    ("1960, Sept.", (None, "Sept", "1960")),
    ("'60 Sept", (None, "Sept", "60")),
    ("1960Sept", (None, "Sept", "1960")),
]


# Text that must produce no hits at all.
NEGATIVE_CASES: list[str] = [
    "no dates at all here",
    "page 1927 of the file",
    "12 34 56",
    "tel. 1960 1234",

    # --- day-less month+year is deliberately permissive, so these check the
    # guards that keep it from firing on non-dates -------------------------
    # a sentence boundary must not glue a year to a month, in either order
    "he arrived in September. 1960 was a hard year",
    "transferred in 1960. September was quiet",
    # a full month name never takes an abbreviation dot, so the dot above is
    # a sentence period; an abbreviation still does (see POSITIVE_CASES)
    "Augustus. 1953 brought no change",
    # reference numbers are not years
    "No. 1234 Mar",
    "file 0912 Sept",
    # a bare 2-digit number next to a month is a quantity, not a year
    "Sept. 60 men were transferred",
    # backtracking must not shorten "Sept" to "Sep" to make a pattern fit
    "1960 Sept 27",
]


def test_positive_cases() -> None:
    for text, expected in POSITIVE_CASES:
        hits = find_dates(text)
        got = [(h["day"], h["month"], h["year"]) for h in hits]
        assert expected in got, f"{text!r}: expected {expected}, got {got}"


def test_negative_cases() -> None:
    for text in NEGATIVE_CASES:
        hits = find_dates(text)
        assert not hits, f"{text!r}: expected no hits, got {hits}"


def test_components_label_separates_partial_from_complete_dates() -> None:
    """The whole point of the day-less patterns: they must be trivial to filter
    out downstream without knowing the pattern names."""
    cases = {
        "27th Sept., '60": ("day_monthname_year", "dmy"),
        "13/07/1956": ("numeric_dmy", "dmy"),
        "1956-07-13": ("numeric_ymd", "dmy"),
        "February 15, 1909": ("monthname_day_year", "dmy"),
        "Sept., 1960": ("monthname_year", "my"),
        "1960 Sept": ("year_monthname", "my"),
        "on 27th Sept. the ship": ("day_monthname", "dm"),
    }
    for text, (pattern, components) in cases.items():
        hits = find_dates(text)
        assert len(hits) == 1, f"{text!r}: expected 1 hit, got {hits}"
        assert (hits[0]["pattern"], hits[0]["components"]) == (pattern, components), \
            f"{text!r}: got {hits[0]['pattern']}/{hits[0]['components']}"


def test_only_abbreviations_take_a_trailing_dot() -> None:
    """A dot after a full month name is a sentence period. Absorbing it would
    let "...in September. 1960 was a hard year" become a date."""
    assert find_dates("Sept. 1960")[0]["year"] == "1960"
    assert find_dates("September. 1960") == []


def test_backtracking_does_not_truncate_a_month_name() -> None:
    """Without a trailing letter guard, "1960 Sept 27" matches as month="Sep",
    stranding the "t", because the shorter alternative makes the pattern fit."""
    assert find_dates("1960 Sept 27") == []


def test_day_is_not_carved_out_of_a_year() -> None:
    """Regression: the separator between components may be punctuation and may
    be empty, which lets a day-shaped match start inside a 4-digit year.
    `Sept., 1960` once matched as day=19, year=60."""
    hits = find_dates("Sept., 1960")
    assert [(h["day"], h["month"], h["year"]) for h in hits] == [(None, "Sept", "1960")]


def test_two_dates_on_one_line_are_reported_separately() -> None:
    hits = find_dates("born 3 January 1945; died 27th Sept., '60")
    assert [h["date_string"] for h in hits] == ["3 January 1945", "27th Sept., '60"]


def test_hits_do_not_overlap_and_are_ordered_by_position() -> None:
    """A full date must be reported once, not also as its own day+month prefix."""
    hits = find_dates("filed 27th Sept., '60 and again on 1st October 1953")
    spans = [(h["span_start"], h["span_end"]) for h in hits]
    assert spans == sorted(spans)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end <= next_start, f"overlapping spans: {spans}"


def _main() -> int:
    failures = 0

    for text, expected in POSITIVE_CASES:
        hits = find_dates(text)
        got = [(h["day"], h["month"], h["year"]) for h in hits]
        ok = expected in got
        failures += not ok
        detail = [(h["pattern"], h["date_string"]) for h in hits]
        print(f"{'ok  ' if ok else 'FAIL'} {text!r:34s} -> {detail}")

    for text in NEGATIVE_CASES:
        hits = find_dates(text)
        ok = not hits
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {text!r:34s} -> (expected no hits) "
              f"{[(h['pattern'], h['date_string']) for h in hits]}")

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")

    total = len(POSITIVE_CASES) + len(NEGATIVE_CASES)
    print(f"\n{total - failures} of {total} cases pass"
          f"{'' if not failures else f' ({failures} failure(s))'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
