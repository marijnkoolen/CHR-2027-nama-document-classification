"""
normalize_dates.py
-------------------
Normalises and verifies the raw day/month/year strings captured by
extract_dates.find_dates: resolves month names (English & Dutch, full and
abbreviated) or numeric months to a month number, completes 2-digit years
against a configurable [earliest_year, latest_year] range and rejects years
outside it, and validates the day against the calendar (leap years included
where the year is known).

Every extracted hit is retained - normalisation only adds fields
(day_norm/month_norm/year_norm/date_valid/swapped), it never drops a
record. A component that can't be resolved/verified (impossible month
number, out-of-range year, day that doesn't exist in that month) is set to
None in the normalised field; the raw day/month/year strings from
find_dates are left untouched for inspection.

The month-name lists live here rather than in extract_dates.py (which
imports them back to build its DATE_PATTERNS regexes) so both modules
share one vocabulary without a circular import.
"""

from __future__ import annotations

import calendar
from typing import Dict

_EN_MONTHS_FULL = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_EN_MONTHS_ABBR = [
    "Jan", "Feb", "Febr", "Mar", "Apr", "Jun", "Jul", "Aug",
    "Sep", "Sept", "Oct", "Octo", "Nov", "Dec",
]

_NL_MONTHS_FULL = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December",
]
_NL_MONTHS_ABBR = [
    "Jan", "Feb", "Febr", "Mrt", "Mar", "Apr", "Jun", "Jul", "Aug",
    "Sep", "Sept", "Okt", "Nov", "Dec",
]

# name (lowercase) -> month number, for the full-name lists: both are
# ordered January..December so a positional zip is safe.
_FULL_MONTH_NUM: dict[str, int] = {}
for _names in (_EN_MONTHS_FULL, _NL_MONTHS_FULL):
    for _i, _name in enumerate(_names, start=1):
        _FULL_MONTH_NUM[_name.lower()] = _i

# Abbreviations are NOT 1:1 positional with the full-name lists (several
# months have more than one abbreviation, e.g. "Sep"/"Sept", "Oct"/"Octo",
# and the two abbreviation lists above are 14 entries long for 12 months) -
# hand-mapped instead of derived by index.
_ABBR_MONTH_NUM: dict[str, int] = {
    "jan": 1, "feb": 2, "febr": 2, "mar": 3, "mrt": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "octo": 10, "okt": 10, "nov": 11, "dec": 12,
}
assert set(_ABBR_MONTH_NUM) >= {n.lower() for n in _EN_MONTHS_ABBR + _NL_MONTHS_ABBR}, (
    "extract_dates.py's month abbreviations changed - update _ABBR_MONTH_NUM"
)

MONTH_NAME_TO_NUM: dict[str, int] = {**_FULL_MONTH_NUM, **_ABBR_MONTH_NUM}


def resolve_month(raw_month: str | None) -> int | None:
    """day_monthname_year/monthname_day_year hits capture a month *name*
    (English or Dutch, full or abbreviated); numeric_dmy/numeric_ymd hits
    capture a plain 1-2 digit month number instead - this handles both.
    Returns None for a number outside 1-12, or (defensively) a name
    find_dates' DATE_PATTERNS could not actually have captured."""
    if not raw_month:
        return None
    token = raw_month.strip()
    if token.isdigit():
        month = int(token)
        return month if 1 <= month <= 12 else None
    return MONTH_NAME_TO_NUM.get(token.lower())


def resolve_year(raw_year: str | None, earliest_year: int, latest_year: int) -> int | None:
    """Completes a 2-digit year to 4 digits by picking the century that
    lands the result inside [earliest_year, latest_year]; a bare 4-digit
    year is just range-checked. If more than one century fits (the range
    spans over 100 years), the earliest one wins - these are
    predominantly birth years and the corpus skews toward older
    applicants, so "'88" resolves to 1888 rather than 1988 when both are
    in range. Returns None if no century fits (almost certainly an OCR
    misread digit) or *raw_year* isn't a plain 2- or 4-digit number."""
    if not raw_year:
        return None
    token = raw_year.strip()
    if not token.isdigit():
        return None
    if len(token) == 4:
        year = int(token)
        return year if earliest_year <= year <= latest_year else None
    if len(token) != 2:
        return None
    digits = int(token)
    for century in range(earliest_year // 100, latest_year // 100 + 1):
        candidate = century * 100 + digits
        if earliest_year <= candidate <= latest_year:
            return candidate
    return None


def resolve_day(raw_day: str | None, month: int | None, year: int | None) -> int | None:
    """Validates the day against the calendar when the month is known
    (falling back to a generic 1-31 bound when it isn't), returning None
    if the day can't be a real day of that month - the "impossible date"
    signal try_swap_day_year acts on."""
    if not raw_day:
        return None
    token = raw_day.strip()
    if not token.isdigit():
        return None
    day = int(token)
    if day < 1:
        return None
    if month is None:
        return day if day <= 31 else None
    # calendar.monthrange needs a real year for the Feb-29 case; fall back
    # to a leap year when *year* itself isn't known/valid, so a genuine
    # Feb 29 isn't rejected just because its year couldn't be resolved.
    max_day = calendar.monthrange(year if year else 1980, month)[1]
    return day if day <= max_day else None


def try_swap_day_year(
    raw_day: str | None,
    month: int | None,
    raw_year: str | None,
    earliest_year: int,
    latest_year: int,
) -> tuple[int, int] | None:
    """For a date whose day is impossible for its month, checks whether
    the source actually had year and day the other way around from what
    the {DAY} {MONTH} {YEAR} (or {MONTH} {DAY} {YEAR}) shape assumes - e.g.
    "81 November 17" is a year-month-day date typed as if day-first, so
    find_dates reads day="81" (impossible), month=11, year="17". Trying
    "17" as the day and completing "81" as the year gives 1981-11-17, a
    valid date. Returns the (day, year) pair to use if swapping fixes it,
    else None - never proposes a swap for a date that already resolved,
    or that no swap can fix."""
    if not raw_day or not raw_year or month is None:
        return None
    swapped_year = resolve_year(raw_day, earliest_year, latest_year)
    swapped_day = resolve_day(raw_year, month, swapped_year)
    if swapped_year is not None and swapped_day is not None:
        return swapped_day, swapped_year
    return None


def normalize_hit(
    hit: Dict[str, str | int | None],
    earliest_year: int,
    latest_year: int,
) -> Dict[str, str | int | bool | None]:
    """Returns a copy of a find_dates hit with normalised fields added:
    day_norm, month_norm, year_norm (None where that component couldn't be
    resolved/verified), date_valid (True iff every component actually
    present in the hit resolved), and swapped (True iff date_valid was
    only reached by trying day/year the other way around - see
    try_swap_day_year). Components absent from the hit to begin with
    (e.g. a "my" month+year hit has no day) don't count against
    date_valid. Never drops or mutates the original hit fields."""
    raw_day = hit.get("day")
    raw_month = hit.get("month")
    raw_year = hit.get("year")

    month_norm = resolve_month(raw_month)
    year_norm = resolve_year(raw_year, earliest_year, latest_year)
    day_norm = resolve_day(raw_day, month_norm, year_norm)

    swapped = False
    if raw_day and day_norm is None:
        swap = try_swap_day_year(raw_day, month_norm, raw_year, earliest_year, latest_year)
        if swap is not None:
            day_norm, year_norm = swap
            swapped = True

    date_valid = (
        (not raw_day or day_norm is not None)
        and (not raw_month or month_norm is not None)
        and (not raw_year or year_norm is not None)
    )

    return {
        **hit,
        "day_norm": day_norm,
        "month_norm": month_norm,
        "year_norm": year_norm,
        "date_valid": date_valid,
        "swapped": swapped,
    }
