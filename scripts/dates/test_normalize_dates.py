"""
test_normalize_dates.py
------------------------
Regression tests for normalize_dates.py's resolve_month/resolve_year/
resolve_day/try_swap_day_year/normalize_hit.

Run standalone (no dependencies):
    python scripts/dates/test_normalize_dates.py

Or under pytest:
    pytest scripts/dates/test_normalize_dates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize_dates import (
    normalize_hit,
    resolve_day,
    resolve_month,
    resolve_year,
    try_swap_day_year,
)

# The two datasets this project actually normalises: forms filled out
# 1950-1965, and a later batch filled out 1966-1990. latest_year is always
# creation-period-end + 1; earliest_year is the same 19th-century floor for
# both (see resolve_year's docstring for why: it's a birth-year floor, not
# tied to when the form itself was filled out).
PERIOD_1950_1965 = dict(earliest_year=1870, latest_year=1966)
PERIOD_1966_1990 = dict(earliest_year=1870, latest_year=1991)


def test_resolve_month_names_english_and_dutch() -> None:
    cases = {
        "September": 9, "sept": 9, "SEPT": 9, "Sep": 9,
        "Okt": 10, "Oktober": 10, "Maart": 3, "Mrt": 3, "Juli": 7, "Mei": 5,
        "February": 2, "Febr": 2,
    }
    for raw, expected in cases.items():
        assert resolve_month(raw) == expected, f"{raw!r}: expected {expected}"


def test_resolve_month_numeric() -> None:
    assert resolve_month("11") == 11
    assert resolve_month("01") == 1
    assert resolve_month("0") is None
    assert resolve_month("13") is None


def test_resolve_month_unresolvable_or_missing() -> None:
    assert resolve_month(None) is None
    assert resolve_month("") is None
    assert resolve_month("Blergust") is None


def test_resolve_year_two_digit_completion_period_1950_1965() -> None:
    # the exact examples the 1950-1965 dataset was configured against:
    # earliest_year=1870 keeps '64 from completing to the out-of-range 1864.
    assert resolve_year("71", **PERIOD_1950_1965) == 1871
    assert resolve_year("64", **PERIOD_1950_1965) == 1964


def test_resolve_year_ambiguous_completion_prefers_earliest_century() -> None:
    # 1966-1990 dataset (latest_year=1991): both 1888 and 1988 are in range,
    # and these are predominantly birth years, so the 19th-century reading
    # wins.
    assert resolve_year("88", **PERIOD_1966_1990) == 1888


def test_resolve_year_four_digit_range_check() -> None:
    assert resolve_year("1955", **PERIOD_1950_1965) == 1955
    assert resolve_year("1869", **PERIOD_1950_1965) is None  # below earliest
    assert resolve_year("1967", **PERIOD_1950_1965) is None  # above latest


def test_resolve_year_two_digit_out_of_range_resolves_to_none() -> None:
    # only century 19 overlaps [1950, 1966], and 1905 falls outside it -
    # no century produces an in-range year, so this is an OCR misread digit.
    assert resolve_year("05", earliest_year=1950, latest_year=1966) is None


def test_resolve_year_missing_or_malformed() -> None:
    assert resolve_year(None, **PERIOD_1950_1965) is None
    assert resolve_year("", **PERIOD_1950_1965) is None
    assert resolve_year("12a", **PERIOD_1950_1965) is None
    assert resolve_year("123", **PERIOD_1950_1965) is None  # not a 2- or 4-digit shape


def test_resolve_day_calendar_validity() -> None:
    assert resolve_day("15", month=6, year=1960) == 15
    assert resolve_day("30", month=2, year=1960) is None  # no Feb 30
    assert resolve_day("31", month=4, year=1960) is None  # no Apr 31
    assert resolve_day("29", month=2, year=1960) == 29    # 1960 is a leap year
    assert resolve_day("29", month=2, year=1961) is None  # 1961 is not


def test_resolve_day_without_known_month_or_year() -> None:
    assert resolve_day("29", month=None, year=None) == 29
    assert resolve_day("32", month=None, year=None) is None
    # Feb 29 must not be rejected just because the year didn't resolve.
    assert resolve_day("29", month=2, year=None) == 29


def test_resolve_day_missing_or_malformed() -> None:
    assert resolve_day(None, month=1, year=1960) is None
    assert resolve_day("0", month=1, year=1960) is None


def test_swap_day_year_resolves_the_shuffled_example() -> None:
    # "81 November 17" is a year-month-day date, but the day-first
    # DATE_PATTERNS shape reads it as day="81" (impossible), month=11,
    # year="17".
    swap = try_swap_day_year("81", 11, "17", **PERIOD_1966_1990)
    assert swap == (17, 1881)


def test_swap_day_year_none_when_swap_does_not_fix_it() -> None:
    # day=32 is impossible either way round (32 is not a valid day, and as
    # a year-completion candidate "32" doesn't turn the original year "60"
    # into a valid day either).
    assert try_swap_day_year("32", 6, "60", **PERIOD_1950_1965) is None


def test_swap_day_year_missing_inputs() -> None:
    assert try_swap_day_year(None, 11, "17", **PERIOD_1966_1990) is None
    assert try_swap_day_year("81", None, "17", **PERIOD_1966_1990) is None
    assert try_swap_day_year("81", 11, None, **PERIOD_1966_1990) is None


def test_normalize_hit_valid_full_date() -> None:
    hit = {"day": "27", "month": "Sept", "year": "1960"}
    norm = normalize_hit(hit, **PERIOD_1950_1965)
    assert norm["day_norm"] == 27
    assert norm["month_norm"] == 9
    assert norm["year_norm"] == 1960
    assert norm["date_valid"] is True
    assert norm["swapped"] is False
    # original raw fields must survive untouched
    assert norm["day"] == "27" and norm["month"] == "Sept" and norm["year"] == "1960"


def test_normalize_hit_day_month_year_shuffled() -> None:
    hit = {"day": "81", "month": "November", "year": "17"}
    norm = normalize_hit(hit, **PERIOD_1966_1990)
    assert (norm["day_norm"], norm["month_norm"], norm["year_norm"]) == (17, 11, 1881)
    assert norm["date_valid"] is True
    assert norm["swapped"] is True


def test_normalize_hit_impossible_month_and_day() -> None:
    hit = {"day": "30", "month": "13", "year": "1950"}
    norm = normalize_hit(hit, **PERIOD_1950_1965)
    assert norm["month_norm"] is None
    assert norm["date_valid"] is False


def test_normalize_hit_out_of_range_year_becomes_none_but_hit_is_kept() -> None:
    """Every extracted date is retained - a normalisation failure clears the
    normalised field, it never drops the record."""
    hit = {"day": "5", "month": "Mar", "year": "2075"}
    norm = normalize_hit(hit, **PERIOD_1950_1965)
    assert norm["year_norm"] is None
    assert norm["date_valid"] is False
    assert norm["day"] == "5" and norm["year"] == "2075"  # raw fields untouched


def test_normalize_hit_partial_date_not_penalised_for_absent_components() -> None:
    """A day-less "my" hit (find_dates' monthname_year/year_monthname
    patterns) has no "day" key at all - that must not count against
    date_valid."""
    hit = {"day": None, "month": "September", "year": "1960"}
    norm = normalize_hit(hit, **PERIOD_1950_1965)
    assert norm["day_norm"] is None
    assert norm["date_valid"] is True


def _main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")

    total = len([n for n in globals() if n.startswith("test_")])
    print(f"\n{total - failures} of {total} cases pass"
          f"{'' if not failures else f' ({failures} failure(s))'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
