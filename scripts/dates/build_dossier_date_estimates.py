"""
build_dossier_date_estimates.py
---------------------------------
Turns dossier_anchor_dates.tsv (per-line procedural anchor dates, built by
build_dossier_anchor_dates.py) into one row per dossier: a robust "signing
bundle" date estimate (D.1 + DM.1 + NAMA agreement - confirmed empirically
to be the same or near-same real-world event, see project discussion) and
an Approval notice date estimate, each with an explicit digit-correction
trail rather than a silently "fixed" value.

Digit-error reconciliation, per group (bundle / approval notice)
independently: within one dossier, the same real signing event sometimes
gets OCR'd with a single wrong day, month, or year digit on one of its
several source lines (multiple pages/copies of the same form, or several
form types signed together). Two distinct dates that agree on exactly 2 of
their 3 (day, month, year) components are treated as the same real event
with one field misread - EXCEPT when day is the only differing component,
since two genuinely different documents signed a few days apart within the
same dossier is an expected, common pattern (see project discussion: D.1/
DM.1/NAMA median gaps of 1-14 days), not a recognition error. Correcting
those away would erase real signal, not just noise.

The "primary" (most-voted) distinct date per group anchors the estimate;
other distinct dates are either reconciled into it (2-of-3 match, not
day-only) - recorded in *_correction_trace, contributing their vote to
*_n_rows_reconciled - or left as "unexplained" residual dates, which is
itself the signal for a possible multi-round procedure.

Also reconciles the Approval notice's validity-deadline fields ("Only
Valid Until"/"Uitsluitend Geldig Tot") as a third group, flags any
date_estimate outside the dossier series' known 1951-1966 coverage, and -
when a per-dossier travel_year is available (data/dates/travel_year.tsv) -
uses it as a light tie-breaking nudge (TRAVEL_YEAR_TIEBREAK_BONUS, worth
less than one raw vote) in picking the primary: it can only swing a
genuine tie among weakly-supported candidates, never override a cluster
where the dossier's own documents already agree - travel_year is manually
typed and can itself carry errors, so same-dossier agreement always wins
over it. Every case where travel_year did change the outcome is traced in
*_correction_trace exactly like a digit correction, not applied silently.

Deliberately keeps every raw distinct date and its provenance out of the
final table (only counts and ISO date strings) - this table is meant to be
safe to work with directly, not just an intermediate.

Run:
    python scripts/dates/build_dossier_date_estimates.py \\
        --anchor-dates data/dates/dossier_anchor_dates.tsv \\
        --output data/dates/dossier_date_estimates.tsv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SERIES_YEAR_RANGE = (1951, 1966)  # dossier series covers 1951-1965 applications; +1 year slack for
                                   # a late-1965 application whose approval validity runs into 1966

VALIDITY_DEADLINE_TERMS = {
    ("Approval notice", "Only Valid Until"), ("Approval notice", "Uitsluitend Geldig Tot"),
}

XRAY_TERMS = {
    ("DM.1", "Film Dated"), ("DM.1", "X-ray Report"),
    ("Report of selection and medical officers", "Film Dated"),
    ("Report of selection and medical officers", "MED"),
}

BUNDLE_TERMS = {
    ("D.1", "Date"), ("D.1", "Datum"),
    ("DM.1", "Date"),
    ("NAMA agreement", "Date"), ("NAMA agreement", "Datum"),
    ("NAMA agreement", "Handtekening"), ("NAMA agreement", "Date Datum"),
}
APPROVAL_NOTICE_TERMS = {
    ("Approval notice", "Date"), ("Approval notice", "Date of Issue"), ("Approval notice", "Date of Approval"),
}


def load_dates(anchor_dates_path: Path) -> pd.DataFrame:
    df = pd.read_csv(anchor_dates_path, sep="\t")
    df = df[df[["day_norm", "month_norm", "year_norm"]].notna().all(axis=1) & df["date_valid"]].copy()
    for c in ("year_norm", "month_norm", "day_norm"):
        df[c] = df[c].astype(int)

    def make_date(row):
        try:
            return pd.Timestamp(year=row["year_norm"], month=row["month_norm"], day=row["day_norm"])
        except (ValueError, OverflowError):
            return pd.NaT

    df["date"] = df.apply(make_date, axis=1)
    return df.dropna(subset=["date"])


def differing_component(a: pd.Timestamp, b: pd.Timestamp) -> str | None:
    """Returns which single component (day/month/year) differs between
    two dates that agree on the other two, or None if they differ on 0, 2,
    or 3 components (not a 2-of-3 match)."""
    diffs = []
    if a.day != b.day:
        diffs.append("day")
    if a.month != b.month:
        diffs.append("month")
    if a.year != b.year:
        diffs.append("year")
    return diffs[0] if len(diffs) == 1 else None


TRAVEL_YEAR_TIEBREAK_BONUS = 0.5  # < 1 raw vote: only ever swings a tie among weakly-supported
                                   # candidates, never overrides a genuine multi-source cluster
CROSS_ANCHOR_TIEBREAK_BONUS = 0.5  # same rationale, for the same-dossier cross-anchor check below
CROSS_ANCHOR_WINDOW_DAYS = 400  # bundle -> approval-side gap is 1-13 months in the observed data
                                 # (see project discussion); generous on purpose since it's only
                                 # ever a tie-breaking nudge, never an override


def reconcile_group(
    dates: pd.Series, doc_types: pd.Series, terms: pd.Series, pages: pd.Series,
    scatter_threshold_days: int = 31, travel_year: float | None = None,
    cross_anchors: list[tuple[str, pd.Timestamp, str]] | None = None,
) -> dict:
    """Reconciles one dossier's dates for one group (bundle, approval
    notice, validity deadline, or x-ray). Returns a dict of summary fields
    - see module docstring for the digit-correction rule.

    Separately from digit correction, a non-primary date only counts as
    genuinely "scattered" (vs. just normal multi-document variation) if
    it's more than *scatter_threshold_days* from the primary - a handful
    of days apart between different documents in the same signing episode
    is the expected pattern (see project discussion: D.1/DM.1/NAMA median
    gaps of 1-14 days), not something to flag.

    *cross_anchors*, when given, is a list of (label, anchor_date,
    direction) - each the dossier's OTHER, independently reconciled date
    group (e.g. the approval-side estimate, when reconciling the bundle),
    direction being 'before'/'after' (bundle precedes approval-side) or
    'near' (x-ray dates are roughly concurrent with the bundle, not
    reliably before or after it). Each satisfied anchor adds its own
    tie-breaking nudge (they stack: a candidate backed by two independent
    anchors outweighs one backed by only one) if the candidate falls
    within CROSS_ANCHOR_WINDOW_DAYS on the expected side. Needs no
    external metadata, so unlike travel_year this generalizes directly to
    a series with no such metadata available."""
    if len(dates) == 0:
        return {
            "n_rows": 0, "n_distinct_dates": 0, "date_estimate": None,
            "confidence": "none", "n_rows_reconciled": 0, "correction_trace": "",
            "scattered_dates": "",
        }

    votes = dates.value_counts()
    if len(votes) == 1:
        primary = votes.index[0]
        return {
            "n_rows": len(dates), "n_distinct_dates": 1, "date_estimate": primary.date().isoformat(),
            "confidence": "single", "n_rows_reconciled": 0, "correction_trace": "",
            "scattered_dates": "",
        }

    # Pick the primary by local support rather than raw vote count alone:
    # when most distinct dates are one-off singletons (common - many rows
    # are each some family member's own single page), raw-vote-count
    # picks whichever singleton happens to be listed first - which can be
    # an isolated outlier while two OTHER singletons that are a few days
    # apart from each other (and so clearly the same real signing episode)
    # get passed over. Support = a date's own votes plus the votes of
    # every other distinct date that's within scatter_threshold_days of it
    # or a 2-of-3 digit-match to it - i.e. "how much evidence backs this
    # date, including near neighbors", not just "how many rows literally
    # share this exact value".
    distinct = list(votes.index)

    def raw_support(d):
        total = int(votes[d])
        for other in distinct:
            if other == d:
                continue
            comp = differing_component(other, d)
            if comp in ("month", "year") or abs((other - d).days) <= scatter_threshold_days:
                total += int(votes[other])
        return total

    def anchor_satisfied(d, anchor_date, direction):
        if direction == "before":
            gap = (anchor_date - d).days
        elif direction == "after":
            gap = (d - anchor_date).days
        else:  # "near": roughly concurrent, either side
            gap = abs((d - anchor_date).days)
        return 0 <= gap <= CROSS_ANCHOR_WINDOW_DAYS

    def bonus(d):
        total = 0.0
        if travel_year is not None and abs(d.year - int(travel_year)) <= 1:
            total += TRAVEL_YEAR_TIEBREAK_BONUS
        for _, anchor_date, direction in cross_anchors or []:
            if anchor_satisfied(d, anchor_date, direction):
                total += CROSS_ANCHOR_TIEBREAK_BONUS
        return total

    def support(d):
        return raw_support(d) + bonus(d)

    in_range = lambda d: SERIES_YEAR_RANGE[0] <= d.year <= SERIES_YEAR_RANGE[1]  # noqa: E731
    primary = max(distinct, key=lambda d: (support(d), int(votes[d]), in_range(d), -d.toordinal()))

    corrections = []
    if travel_year is not None or cross_anchors:
        # Was the external/cross-dossier evidence actually needed to reach
        # this primary, or would the same date have won on the dossier's
        # own internal evidence alone? Traced explicitly rather than
        # silently folded in - a bonus-backed pick rests on weaker
        # evidence (travel_year is manually typed and can carry its own
        # errors; a cross-anchor is itself just another reconciled
        # estimate, not ground truth) than a same-dossier document
        # directly agreeing.
        primary_without_bonus = max(
            distinct, key=lambda d: (raw_support(d), int(votes[d]), in_range(d), -d.toordinal())
        )
        if primary != primary_without_bonus:
            reasons = []
            if travel_year is not None and abs(primary.year - int(travel_year)) <= 1:
                reasons.append(f"travel_year={int(travel_year)}")
            for label, anchor_date, direction in cross_anchors or []:
                if anchor_satisfied(primary, anchor_date, direction):
                    reasons.append(f"{label}={anchor_date.date().isoformat()}({direction})")
            corrections.append(
                f"primary set via {' + '.join(reasons)}: "
                f"{primary_without_bonus.date().isoformat()} -> {primary.date().isoformat()}"
            )

    reconciled_votes = int(votes[primary])
    scattered = []
    for other_date in distinct:
        if other_date == primary:
            continue
        n = int(votes[other_date])
        comp = differing_component(other_date, primary)
        if comp in ("month", "year"):
            reconciled_votes += int(n)
            example_idx = dates[dates == other_date].index[0]
            corrections.append(
                f"{doc_types[example_idx]}/{terms[example_idx]} p{pages[example_idx]}: "
                f"{other_date.date().isoformat()} -> {primary.date().isoformat()} ({comp} corrected, n={n})"
            )
        elif abs((other_date - primary).days) > scatter_threshold_days:
            scattered.append(f"{other_date.date().isoformat()} (n={n})")
        # else: a handful of days apart (including a lone differing "day"
        # component) - normal variation, not corrected and not flagged.

    confidence = "consistent" if not scattered else "scattered"
    return {
        "n_rows": len(dates),
        "n_distinct_dates": len(votes),
        "date_estimate": primary.date().isoformat(),
        "confidence": confidence,
        "n_rows_reconciled": reconciled_votes - int(votes[primary]),
        "correction_trace": "; ".join(corrections),
        "scattered_dates": ", ".join(scattered),
    }


def year_in_series_range(iso_date: str | None) -> bool | None:
    if not iso_date:
        return None
    year = int(iso_date[:4])
    return SERIES_YEAR_RANGE[0] <= year <= SERIES_YEAR_RANGE[1]


def years_from_dates_field(field: str) -> set[int]:
    """Pulls the year out of every ISO date in a *_scattered_dates /
    *_date_estimate-shaped string, e.g. '1928-10-16 (n=1), 1947-09-04
    (n=1)' -> {1928, 1947}. Used to check whether a travel year matches
    ANY residual scattered candidate, not just the chosen estimate."""
    return {int(y) for y in re.findall(r"(\d{4})-\d{2}-\d{2}", field or "")}


def travel_year_conflict_note(
    confidence: str, date_estimate: str | None, scattered_dates: str, travel_year: float | None,
) -> str:
    """A short hint for which side (our extracted date vs. the manually-
    typed travel year) is more likely wrong, per the project's own
    heuristic: internal agreement among our dates, contradicted by travel
    year, points at travel year being the error; conversely if our dates
    are internally scattered but ONE of the discarded candidates matches
    travel year, that candidate may be the right one after all."""
    if travel_year is None or not date_estimate:
        return ""
    travel_year = int(travel_year)
    estimate_year = int(date_estimate[:4])
    if confidence in ("single", "consistent"):
        return "" if abs(estimate_year - travel_year) <= 1 else "travel_year_likely_wrong"
    if confidence == "scattered":
        if abs(estimate_year - travel_year) <= 1:
            return ""
        other_years = years_from_dates_field(scattered_dates)
        return "travel_year_matches_a_scattered_candidate" if travel_year in other_years or (travel_year - 1) in other_years or (travel_year + 1) in other_years else "unclear"
    return ""


def build_dossier_date_estimates(
    anchor_dates: pd.DataFrame, travel_years: pd.DataFrame | None = None, scatter_threshold_days: int = 31,
) -> pd.DataFrame:
    anchor_dates = anchor_dates.copy()
    anchor_dates["pair"] = list(zip(anchor_dates["predicted_document_type"], anchor_dates["assigned_term"]))
    bundle = anchor_dates[anchor_dates["pair"].isin(BUNDLE_TERMS)]
    notice = anchor_dates[anchor_dates["pair"].isin(APPROVAL_NOTICE_TERMS)]
    validity = anchor_dates[anchor_dates["pair"].isin(VALIDITY_DEADLINE_TERMS)]
    xray = anchor_dates[anchor_dates["pair"].isin(XRAY_TERMS)]

    travel_year_by_dossier = {}
    if travel_years is not None:
        tv = travel_years.dropna(subset=["travel_year"])
        travel_year_by_dossier = dict(zip(tv["identifier"], tv["travel_year"]))

    all_dossiers = sorted(set(anchor_dates["dossier"]))
    rows = []
    for dossier in all_dossiers:
        b = bundle[bundle["dossier"] == dossier]
        n = notice[notice["dossier"] == dossier]
        v = validity[validity["dossier"] == dossier]
        x = xray[xray["dossier"] == dossier]
        travel_year = travel_year_by_dossier.get(dossier)
        b_args = (b["date"], b["predicted_document_type"], b["assigned_term"], b["page_num"])
        n_args = (n["date"], n["predicted_document_type"], n["assigned_term"], n["page_num"])
        v_args = (v["date"], v["predicted_document_type"], v["assigned_term"], v["page_num"])
        x_args = (x["date"], x["predicted_document_type"], x["assigned_term"], x["page_num"])

        # Pass 1: each group reconciled independently (plus travel_year,
        # which needs no other group). Needed before cross-anchoring since
        # bundle, approval-side, and x-ray each want another group as an
        # anchor.
        b_summary = reconcile_group(*b_args, scatter_threshold_days, travel_year)
        n_summary = reconcile_group(*n_args, scatter_threshold_days, travel_year)
        v_summary = reconcile_group(*v_args, scatter_threshold_days, travel_year)
        x_summary = reconcile_group(*x_args, scatter_threshold_days, travel_year)

        # Pass 2: re-reconcile bundle using whichever approval-side
        # estimate is available (validity deadline preferred - more
        # commonly present) AND the x-ray estimate (x-ray dates should be
        # roughly concurrent with the bundle, per project discussion - a
        # form of ID has to still be medically valid at signing time), then
        # re-reconcile the approval-side groups using the (now
        # cross-anchored) bundle estimate in turn. One round each way, not
        # iterated further.
        bundle_cross_anchors = []
        approval_side_estimate = v_summary["date_estimate"] or n_summary["date_estimate"]
        if approval_side_estimate:
            bundle_cross_anchors.append(("approval_side", pd.Timestamp(approval_side_estimate), "before"))
        if x_summary["date_estimate"]:
            bundle_cross_anchors.append(("xray", pd.Timestamp(x_summary["date_estimate"]), "near"))
        if bundle_cross_anchors:
            b_summary = reconcile_group(*b_args, scatter_threshold_days, travel_year, bundle_cross_anchors)

        if b_summary["date_estimate"]:
            bundle_anchor = [("bundle", pd.Timestamp(b_summary["date_estimate"]), "after")]
            n_summary = reconcile_group(*n_args, scatter_threshold_days, travel_year, bundle_anchor)
            v_summary = reconcile_group(*v_args, scatter_threshold_days, travel_year, bundle_anchor)
            x_summary = reconcile_group(
                *x_args, scatter_threshold_days, travel_year,
                [("bundle", pd.Timestamp(b_summary["date_estimate"]), "near")],
            )

        row = {"dossier": dossier}
        row.update({f"bundle_{k}": v for k, v in b_summary.items()})
        row.update({f"approval_notice_{k}": v for k, v in n_summary.items()})
        row.update({f"validity_deadline_{k}": v for k, v in v_summary.items()})
        row.update({f"xray_{k}": v for k, v in x_summary.items()})

        for group in ("bundle", "approval_notice", "validity_deadline", "xray"):
            row[f"{group}_year_in_series_range"] = year_in_series_range(row[f"{group}_date_estimate"])
            # A 'scattered' dossier can still have a trustworthy point
            # estimate if travel_year or the cross-anchor picked it out
            # from otherwise-tied candidates - this makes that visible
            # without grepping correction_trace for 'cross_anchor|travel_year'.
            trace = row[f"{group}_correction_trace"] or ""
            row[f"{group}_estimate_backed_by_cross_check"] = "primary set via" in trace

        if row["bundle_date_estimate"] and row["approval_notice_date_estimate"]:
            gap = (pd.Timestamp(row["approval_notice_date_estimate"]) - pd.Timestamp(row["bundle_date_estimate"])).days
            row["approval_notice_gap_days"] = gap
            # Empirically (see project discussion) the Approval notice is
            # dated ~1-2.5 months after the D.1/DM.1/NAMA bundle in the
            # large majority of dossiers, essentially never before it by
            # more than a few weeks. A single-source Approval notice date
            # has no internal cross-check (confidence='single' only means
            # "no conflicting Approval notice date", not "verified"), so
            # this external check against the bundle catches wild misreads
            # that internal reconciliation alone can't.
            row["approval_notice_plausible_gap"] = -31 <= gap <= 366
        else:
            row["approval_notice_gap_days"] = None
            row["approval_notice_plausible_gap"] = None

        row["travel_year"] = travel_year
        for group in ("bundle", "approval_notice", "validity_deadline", "xray"):
            estimate = row[f"{group}_date_estimate"]
            if travel_year is not None and estimate:
                gap_years = int(estimate[:4]) - int(travel_year)
                row[f"{group}_travel_year_gap"] = gap_years
                row[f"{group}_travel_year_flag"] = abs(gap_years) > 1
            else:
                row[f"{group}_travel_year_gap"] = None
                row[f"{group}_travel_year_flag"] = None
            row[f"{group}_travel_year_conflict_note"] = travel_year_conflict_note(
                row[f"{group}_confidence"], estimate, row[f"{group}_scattered_dates"], travel_year,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dates", type=Path, default=Path("data/dates/dossier_anchor_dates.tsv"))
    parser.add_argument("--travel-years", type=Path, default=Path("data/dates/travel_year.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/dates/dossier_date_estimates.tsv"))
    parser.add_argument(
        "--scatter-threshold-days", type=int, default=31,
        help="A non-primary date within this many days of the primary (and not resolved as a digit "
             "correction) counts as normal multi-document variation, not 'scattered'.",
    )
    args = parser.parse_args()

    anchor_dates = load_dates(args.anchor_dates)
    travel_years = pd.read_csv(args.travel_years, sep="\t") if args.travel_years.exists() else None
    result = build_dossier_date_estimates(anchor_dates, travel_years, args.scatter_threshold_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result)} dossier rows to {args.output}")
    print()
    for group in ("bundle", "approval_notice", "validity_deadline", "xray"):
        print(f"{group}_confidence:")
        print(result[f"{group}_confidence"].value_counts())
        print(f"  digit corrections applied: {(result[f'{group}_n_rows_reconciled'] > 0).sum()}")
        print(f"  estimate backed by cross-check: {(result[f'{group}_estimate_backed_by_cross_check']).sum()}")
        out_of_range = result[f"{group}_year_in_series_range"] == False  # noqa: E712 - explicit vs. None
        print(f"  year_estimate outside {SERIES_YEAR_RANGE}: {out_of_range.sum()}")
        print()

    has_travel_year = result["travel_year"].notna()
    print(f"dossiers with a travel_year available: {has_travel_year.sum()}")
    for group in ("bundle", "approval_notice", "validity_deadline", "xray"):
        flagged = result[f"{group}_travel_year_flag"] == True  # noqa: E712
        print(f"  {group} vs travel_year, flagged (>1yr gap): {flagged.sum()}")
        note_counts = result.loc[has_travel_year, f"{group}_travel_year_conflict_note"].value_counts()
        note_counts = note_counts[note_counts.index != ""]
        if len(note_counts):
            print(f"    conflict notes: {note_counts.to_dict()}")


if __name__ == "__main__":
    main()
