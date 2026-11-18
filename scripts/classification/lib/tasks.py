"""Task definitions for the two label sets these scripts train against:
start_page (binary, every annotated page) and doc_type (multi-class document
type, start pages only). Everything that differs between the two original
notebooks - which TSV column holds the dossier id, how the label column is
built, where the train/val/test split comes from - is captured here so the
rest of the pipeline (lib/labels.py and every script) is written once,
generically.

Both tasks read page text from markdown-per-page/**/*.markdown.md: the two
original notebooks disagreed here (doc_type_start_page_classifier_qwen.ipynb
read text-per-page/**/*.txt instead) despite both being titled "Qwen-7B
Transcriptions" - confirmed to be a mistake in the original, not an
intentional difference, so both tasks now use the same source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    labels_tsv: str          # relative to --data-root
    dossier_column: str      # column in labels_tsv holding the dossier/pdf id
    png_root: str             # relative to --data-root
    text_root: str             # relative to --data-root
    text_extension: str         # appended to "{trans_name}_page_{n:04d}"
    cache_suffix: str            # used in feature-cache filenames
    filter_start_pages_only: bool
    split_source: str             # default for --split-source: "computed" (70/15/15 dossier split) or
                                   # "tsv_column" (the labels TSV's own 'split' column) - every script's
                                   # --split-source flag can override this per-run, see lib/labels.py


TASKS: dict[str, TaskConfig] = {
    "start_page": TaskConfig(
        name="start_page",
        labels_tsv="labels/dossier_labels_merged_pdf12_stratified.tsv",
        dossier_column="pdf_name",
        png_root="image-per-page",
        text_root="markdown-per-page",
        text_extension=".markdown.md",
        cache_suffix="start_page",
        filter_start_pages_only=False,
        split_source="computed",
    ),
    "doc_type": TaskConfig(
        name="doc_type",
        labels_tsv="labels/dossier_labels_merged_pdf12_stratified.tsv",
        dossier_column="pdf_id",
        png_root="image-per-page",
        text_root="markdown-per-page",
        text_extension=".markdown.md",
        cache_suffix="doctype",
        filter_start_pages_only=True,
        split_source="tsv_column",
    ),
}


def get_task(name: str) -> TaskConfig:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r} - expected one of {sorted(TASKS)}")
    return TASKS[name]
