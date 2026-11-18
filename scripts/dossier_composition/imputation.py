"""
Shared library for turning the full-corpus page-level predictions into
document-instance ("segment") sequences per dossier, and for multiply
imputing each segment's "true" document type using the confusion-matrix
posterior from confusion_matrix.py.

Not a standalone analysis -- imported by occurrence.py, co-occurrence.py,
dispersion.py, order.py, and (via dossier_size_model's
doc_type_uncertainty_common.py) the doc-type dossier-size analyses, so the
imputation logic and its uncertainty propagation stay identical everywhere.

Usage pattern for a downstream script (TYPE correction only -- segmentation
held fixed at its predicted boundaries):

    from imputation import load_segments, load_confusion_posterior, sample_true_types

    segments = load_segments(predictions_path)
    p_draws, prior = load_confusion_posterior(confusion_idata_path, test_predictions_path)
    imputed = sample_true_types(segments, p_draws, prior, n_imputations=200)
    # imputed: int array (n_imputations, n_segments), values index into ALL_TYPES

SEGMENTATION correction (start_page uncertainty, on top of type correction):
predicted_segment_id is a deterministic function of predicted_start_page (a
new segment begins exactly when predicted_start_page == "yes" -- verified
directly against the predictions file), and start_page recall/precision are
0.98/0.95 per the test report -- not perfect, so predicted_segment_id is
itself noisy: missed boundaries merge two real documents into one segment,
spurious boundaries split one real document into two. See
start_page_confusion_matrix.py for the same Bayesian-confusion-matrix
treatment applied to the 2-class start_page problem. Two entry points,
depending on whether the caller needs segment-level ordering (dispersion.py,
order.py) or just per-dossier counts (occurrence.py, co_occurrence.py, and
dossier_size_model's doc-type analyses):

    from imputation import (
        load_pages, load_confusion_posterior, load_start_page_confusion_posterior,
        sample_segments_for_draw, sample_dossier_type_counts_for_draw,
    )

    pages = load_pages(predictions_path)
    type_p_draws, type_prior = load_confusion_posterior(type_idata_path, type_test_path, n_draws=20)
    start_p_draws, start_prior = load_start_page_confusion_posterior(start_idata_path, type_test_path, n_draws=20)
    rng = np.random.default_rng(42)
    for d in range(20):
        segs_d = sample_segments_for_draw(
            pages, type_p_draws[d], type_prior, start_p_draws[d], start_prior, rng
        )  # segment-level DataFrame, this draw's own boundaries + types

IMPORTANT precondition for both `pages` and the `segments`-shaped output of
sample_segments_for_draw: functions in this module use plain numpy
row-position logic (not pandas index-label alignment) internally precisely
because callers often pass a FILTERED subset (e.g. dossier_size_model joins
to a 1307-dossier subset, or filters further to one era) -- filtering with a
boolean mask preserves row order but NOT a contiguous 0..n-1 index, and an
earlier version of reconstruct_segments_from_start silently produced NaN
segment ids for ~60% of dossiers in exactly that situation (pandas Series
`+` aligns by index label, not position) before self_test() below caught it
on synthetic data with a deliberately non-contiguous index. Passing a
plain-numpy-safe `pages` (any row order/index is fine, contents matter, not
labels) is the actual contract; run `python3
scripts/dossier_composition/imputation.py` to repeat that check.
"""

import arviz as az
import numpy as np
import pandas as pd

from common import ALL_TYPES, map_type

RNG_SEED = 42
START_LABELS = ["no", "yes"]


def load_pages(predictions_path: str) -> pd.DataFrame:
    """Page-level data (one row per page), sorted by (pdf_name, page_num),
    with the document-type mapping applied but NOT yet grouped into
    segments -- the building block both the fixed-segmentation path
    (load_segments) and the segmentation-correction path
    (sample_segments_for_draw) start from.

    Normalizes a `dossier` column to `pdf_name` if that's what the file
    uses (some pipeline runs name it one way, some the other) -- everything
    downstream assumes `pdf_name`.
    """
    df = pd.read_csv(predictions_path, sep="\t")
    if "dossier" in df.columns and "pdf_name" not in df.columns:
        df = df.rename(columns={"dossier": "pdf_name"})
    df["type_mapped"] = df["predicted_document_type"].apply(map_type)
    df = df.sort_values(["pdf_name", "page_num"]).reset_index(drop=True)
    return df


def pages_to_segments(pages: pd.DataFrame, segment_id_col: str) -> pd.DataFrame:
    """Aggregate page-level rows into one row per segment_id_col value:
    dossier, position in the dossier's sequence, majority-vote mapped type
    (ties broken by summed confidence), mean confidence. Shared by
    load_segments (fixed, predicted_segment_id) and the reconstructed-
    segmentation path (reconstruct_segments_from_start's output column).

    Fully vectorized (two plain groupbys + a merge) rather than a Python-
    level groupby().apply() with a per-group nested groupby -- the original
    apply-based version took ~51s per call on the full corpus (~81k
    segments), which is fine called once but not called 200x per
    imputation-heavy analysis script; this version is milliseconds.
    Cross-checked against the apply-based version for identical output
    before replacing it (see self_test() at the bottom of this module).
    """
    conf_sum = (
        pages.groupby([segment_id_col, "type_mapped"], sort=False)["document_type_confidence"]
        .sum()
        .reset_index()
    )
    top_idx = conf_sum.groupby(segment_id_col, sort=False)["document_type_confidence"].idxmax()
    majority = conf_sum.loc[top_idx, [segment_id_col, "type_mapped"]].rename(
        columns={"type_mapped": "_majority_type"}
    )

    merged = pages.merge(majority, on=segment_id_col, how="left")
    is_majority = merged["type_mapped"] == merged["_majority_type"]

    segments = merged.groupby(segment_id_col, sort=False).agg(
        pdf_name=("pdf_name", "first"),
        n_pages=("page_num", "size"),
        first_page_num=("page_num", "min"),
    )
    segments["type_mapped"] = majority.set_index(segment_id_col)["_majority_type"]
    segments["mean_confidence"] = (
        merged.loc[is_majority].groupby(segment_id_col, sort=False)["document_type_confidence"].mean()
    )
    segments = segments.reset_index().rename(columns={segment_id_col: "segment_id"})

    segments = segments.sort_values(["pdf_name", "first_page_num"]).reset_index(drop=True)
    segments["order_in_dossier"] = segments.groupby("pdf_name").cumcount()
    segments["n_segments_in_dossier"] = segments.groupby("pdf_name")["segment_id"].transform("size")
    segments["norm_position"] = segments["order_in_dossier"] / (segments["n_segments_in_dossier"] - 1).clip(lower=1)

    segments["type_idx"] = segments["type_mapped"].map({t: i for i, t in enumerate(ALL_TYPES)})
    return segments


def naive_segments_from_pages(pages: pd.DataFrame) -> pd.DataFrame:
    """One row per predicted document instance, using the PREDICTED
    start_page sequence as-is (no start_page correction -- see
    sample_segments_for_draw for that). predicted_document_type is usually
    but not always constant within a segment (~4% of segments mix types
    across pages) -- resolved by majority vote in pages_to_segments.

    Segment boundaries are DERIVED from predicted_start_page via
    reconstruct_segments_from_start, rather than trusting the file's own
    predicted_segment_id column directly -- that column's uniqueness scope
    isn't guaranteed across prediction pipelines (one run's segment ids were
    a globally-unique string per dossier; another's were a small integer
    reused starting from 1 in every dossier, which would silently merge
    unrelated dossiers' segments together under a direct groupby). Deriving
    it ourselves from predicted_start_page gives the same boundaries either
    way (verified: a new segment begins exactly when predicted_start_page
    == "yes", full stop) and is robust regardless of what convention the
    source file happens to use.

    Use this (not pages_to_segments(pages, "predicted_segment_id") directly)
    anywhere a "naive"/uncorrected baseline segmentation is needed from an
    already-loaded `pages` DataFrame; load_segments below is the same thing
    from a file path.
    """
    start_bool = (pages["predicted_start_page"] == "yes").to_numpy()
    derived_segment_id = reconstruct_segments_from_start(pages, start_bool)
    return pages_to_segments(pages.assign(_derived_seg=derived_segment_id), "_derived_seg")


def load_segments(predictions_path: str) -> pd.DataFrame:
    """naive_segments_from_pages, loading `pages` from a file path first."""
    return naive_segments_from_pages(load_pages(predictions_path))


def reconstruct_segments_from_start(pages: pd.DataFrame, start_bool: np.ndarray) -> pd.Series:
    """pages: sorted by (pdf_name, page_num) -- must match load_pages' order.
    start_bool: same-length boolean array, one entry per page-row, True =
    this page begins a new document instance. The first page of every
    dossier is forced to True regardless of start_bool (a dossier can't
    have zero documents), matching how the original predicted_segment_id
    is constructed (verified: it always starts "yes" on a dossier's first
    page). Returns a Series of new segment-id strings, same length/order as
    `pages`, suitable as `segment_id_col` data for pages_to_segments.

    Vectorized via a running-count trick: a plain cumulative sum of
    start_bool gives a running segment counter that only *resets* at
    dossier boundaries if we subtract, at every row, the counter's value at
    that row's dossier's first page.

    Everything below works in plain numpy, not pandas Series arithmetic --
    `pages["pdf_name"] + pd.Series(...)` would align by pandas INDEX LABEL,
    not row position, and silently produces NaN wherever `pages` has a
    non-default index (e.g. any filtered subset, which every caller outside
    this module's own tests actually passes) -- caught via
    test_reconstruct_segments_from_start_with_filtered_index in this
    module's self_test(), after it surfaced as ~60% of dossiers vanishing
    from a real downstream count table.
    """
    pdf_name = pages["pdf_name"].to_numpy()
    is_first_page_of_dossier = np.empty(len(pdf_name), dtype=bool)
    is_first_page_of_dossier[0] = True
    is_first_page_of_dossier[1:] = pdf_name[1:] != pdf_name[:-1]
    start_bool = start_bool | is_first_page_of_dossier

    global_cum = np.cumsum(start_bool)
    dossier_start_offset = pd.Series(np.where(is_first_page_of_dossier, global_cum, np.nan)).ffill().to_numpy()
    seg_local_idx = (global_cum - dossier_start_offset).astype(int)

    seg_id = np.char.add(np.char.add(pdf_name.astype(str), "::rseg"), seg_local_idx.astype(str))
    return pd.Series(seg_id, index=pages.index)


def load_start_page_confusion_posterior(confusion_idata_path: str, test_predictions_path: str,
                                          n_draws: int = None):
    """Same shape/contract as load_confusion_posterior, for the 2-class
    start_page confusion matrix (start_page_confusion_matrix.py)."""
    idata = az.from_netcdf(confusion_idata_path)
    p = idata.posterior["p"].stack(sample=("chain", "draw")).transpose("sample", "true_label", "pred_label")
    p_draws = p.values
    if n_draws is not None and n_draws < p_draws.shape[0]:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(p_draws.shape[0], size=n_draws, replace=False)
        p_draws = p_draws[idx]

    test = pd.read_csv(test_predictions_path, sep="\t")
    counts = test["start_page"].value_counts().reindex(START_LABELS, fill_value=0).to_numpy()
    prior_true_start = (counts + 1.0) / (counts + 1.0).sum()

    return p_draws, prior_true_start


def _sample_categorical(pred_idx: np.ndarray, p_draw: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """P(true=i | predicted=j) ~ p_draw[i, j] * prior[i], sampled for every
    row. Shared inner loop for both the type and start-page corrections."""
    w = p_draw[:, pred_idx].T * prior[None, :]
    w = w / w.sum(axis=1, keepdims=True)
    cum_w = np.cumsum(w, axis=1)
    u = rng.random(len(pred_idx))
    return (u[:, None] < cum_w).argmax(axis=1)


def load_confusion_posterior(confusion_idata_path: str, test_predictions_path: str, n_draws: int = None):
    """Returns (p_draws, prior_true_type):
    p_draws: array (n_draws, n_types, n_types), posterior draws of P(pred|true).
    prior_true_type: array (n_types,), Dirichlet-smoothed empirical P(true type)
      from the test set's true-label marginal -- the base rate used to invert
      P(pred|true) into P(true|pred) via Bayes' rule.
    """
    idata = az.from_netcdf(confusion_idata_path)
    p = idata.posterior["p"].stack(sample=("chain", "draw")).transpose("sample", "true_type", "pred_type")
    p_draws = p.values
    if n_draws is not None and n_draws < p_draws.shape[0]:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(p_draws.shape[0], size=n_draws, replace=False)
        p_draws = p_draws[idx]

    test = pd.read_csv(test_predictions_path, sep="\t")
    test["true_mapped"] = test["document_type"].apply(map_type)
    counts = test["true_mapped"].value_counts().reindex(ALL_TYPES, fill_value=0).to_numpy()
    prior_true_type = (counts + 1.0) / (counts + 1.0).sum()  # +1 Dirichlet smoothing

    return p_draws, prior_true_type


def sample_true_types(segments: pd.DataFrame, p_draws: np.ndarray, prior_true_type: np.ndarray) -> np.ndarray:
    """For each posterior draw d of the confusion matrix, sample one plausible
    true type per segment from P(true=i | predicted=j) ~ p_draws[d, i, j] * prior[i].
    Returns int array (n_imputations, n_segments) of indices into ALL_TYPES.
    Segmentation (which segments exist at all) is held fixed at its
    predicted boundaries -- see sample_true_types_and_segments to also
    correct for start_page uncertainty.
    """
    rng = np.random.default_rng(RNG_SEED)
    pred_idx = segments["type_idx"].to_numpy()
    n_segments = len(segments)
    n_imputations = p_draws.shape[0]

    imputed = np.empty((n_imputations, n_segments), dtype=np.int8)
    for d in range(n_imputations):
        imputed[d] = _sample_categorical(pred_idx, p_draws[d], prior_true_type, rng)

    return imputed


def sample_segments_for_draw(pages: pd.DataFrame, type_p_draw: np.ndarray, type_prior: np.ndarray,
                               start_p_draw: np.ndarray, start_prior: np.ndarray,
                               rng: np.random.Generator) -> pd.DataFrame:
    """One full joint imputation draw, at the SEGMENT level: resample
    start_page per page from the 2-class confusion matrix, reconstruct
    segment boundaries from the corrected start sequence
    (reconstruct_segments_from_start), majority-vote each NEW segment's
    predicted type, then resample ITS true type from the document-type
    confusion matrix. Returns a segments-shaped DataFrame -- same schema as
    load_segments's output (pdf_name, segment_id, type_mapped, type_idx,
    order_in_dossier, n_segments_in_dossier, norm_position, mean_confidence,
    n_pages, first_page_num) -- but built from this draw's corrected
    boundaries, not the fixed predicted ones. Segment count and boundaries
    genuinely vary draw to draw, so unlike sample_true_types there's no
    fixed-length array to accumulate across draws -- callers that need
    per-dossier order (dispersion.py, order.py) use this directly, one draw
    at a time; callers that only need counts use
    sample_dossier_type_counts_for_draw below.
    """
    start_pred_idx = (pages["predicted_start_page"] == "yes").astype(int).to_numpy()  # 0=no, 1=yes
    corrected_start_idx = _sample_categorical(start_pred_idx, start_p_draw, start_prior, rng)
    start_bool = corrected_start_idx.astype(bool)

    new_segment_id = reconstruct_segments_from_start(pages, start_bool)
    segments = pages_to_segments(pages.assign(_rseg=new_segment_id), "_rseg")

    type_pred_idx = segments["type_idx"].to_numpy()
    corrected_type_idx = _sample_categorical(type_pred_idx, type_p_draw, type_prior, rng)
    segments["type_idx"] = corrected_type_idx
    segments["type_mapped"] = [ALL_TYPES[i] for i in corrected_type_idx]
    return segments


def sample_dossier_type_counts_for_draw(pages: pd.DataFrame, type_p_draw: np.ndarray, type_prior: np.ndarray,
                                          start_p_draw: np.ndarray, start_prior: np.ndarray,
                                          rng: np.random.Generator, dossier_order) -> np.ndarray:
    """Convenience wrapper around sample_segments_for_draw for callers that
    only need a (n_dossiers, n_types) count matrix for this one draw
    (occurrence.py, co_occurrence.py) -- reindexed to `dossier_order` so
    array positions align across draws regardless of incidental groupby
    ordering (every dossier has >=1 page and page 1 always forces a segment
    start, so every dossier appears in every draw; this is a safety net,
    not a correction for missing dossiers).
    """
    segments = sample_segments_for_draw(pages, type_p_draw, type_prior, start_p_draw, start_prior, rng)
    n_types = len(ALL_TYPES)
    dossier_to_idx = {d: i for i, d in enumerate(dossier_order)}
    dossier_idx = segments["pdf_name"].map(dossier_to_idx).to_numpy()
    combined = dossier_idx * n_types + segments["type_idx"].to_numpy()
    return np.bincount(combined, minlength=len(dossier_order) * n_types).reshape(len(dossier_order), n_types)


def _self_test_reconstruct_segments_from_start():
    """Two checks: (1) resegmentation arithmetic against a hand-computed
    example (including a dossier whose first page's start_bool is False,
    which must still be forced to a segment start), (2) the SAME example
    but with a deliberately non-contiguous pandas index -- this is the case
    that let ~60% of dossiers silently vanish from a real downstream count
    table before this test existed (pandas Series `+` aligns by index
    label; the fix works in plain numpy row-position instead)."""
    pages = pd.DataFrame({
        "pdf_name": ["A"] * 5 + ["B"] * 3 + ["C"] * 4,
        "page_num": [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4],
    })
    start_bool = np.array(
        [True, False, True, False, False, True, False, True, False, False, True, False]
    )
    expected_local_idx = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]

    for label, test_pages, test_start in [
        ("contiguous index", pages, start_bool),
        # simulate what a boolean-mask filter does: keep row order, drop the
        # contiguous 0..n-1 index (here: an arbitrary reindex, standing in for
        # "some rows of a larger frame got filtered out upstream")
        ("non-contiguous index", pages.set_axis([100, 205, 3, 47, 900, 12, 88, 5, 640, 71, 2, 33]), start_bool),
    ]:
        seg_id = reconstruct_segments_from_start(test_pages, test_start)
        local_idx = seg_id.str.split("::rseg").str[1].astype(int).to_numpy()
        assert list(local_idx) == expected_local_idx, (
            f"[{label}] resegmentation mismatch: {list(local_idx)} != {expected_local_idx}"
        )
        n_unique = seg_id.nunique()
        assert n_unique == 6, f"[{label}] expected 6 unique segments (2+2+2 across the 3 dossiers), got {n_unique}"
        assert not seg_id.isna().any(), f"[{label}] NaN segment id(s) produced -- the index-alignment bug is back"
    print("_self_test_reconstruct_segments_from_start: passed (contiguous and non-contiguous index)")


if __name__ == "__main__":
    _self_test_reconstruct_segments_from_start()
