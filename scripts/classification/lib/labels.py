"""Label loading, path resolution, and train/val/test split - the
task-generic replacement for notebook sections 2-4 of both notebooks.

Ports only the code path each notebook actually used: page_start_classifier_qwen.ipynb's
sections 2a/2b (xlsx-per-annotator majority-vote labels, cells 4-5 of that
notebook) were dead code - overwritten by section 2c (the merged TSV, cell 6)
before anything downstream read them - so that path is dropped here rather
than ported.

Two ways to get a train/val/test split, chosen per call via split_source
("computed" or "tsv_column" - see load_labels()), independently of which
task is being loaded:
  - "computed": a fresh dossier-level 70/15/15 split, seeded by random_seed
    (see _assign_computed_split) - ignores any 'split' column in the TSV.
  - "tsv_column": use the labels TSV's own 'split' column as-is (its values
    were presumably chosen once, deliberately, e.g. to keep the split
    stable across dataset revisions or to match a split used elsewhere) -
    random_seed is irrelevant to the split in this mode.
Each TaskConfig in lib/tasks.py has a split_source default (doc_type
defaults to "tsv_column" since it's what its label TSV already provides;
start_page defaults to "computed"), but every script accepts a
--split-source override - see e.g. train_baseline.py.

Path validation: every img_path/text_path this module builds is checked
against disk (via common.py's validate_manifest_paths, the same check
joint/train.py and joint/precompute_embeddings.py use) before any other
work happens - a missing file raises by default, so a wrong --data-root or
a systematic path-formula mistake is caught immediately rather than
surfacing as a confusing crash (or, worse, a silently-shrunk dataset) deep
into training. Pass allow_missing_files=True (every script's
--allow-missing-files flag) to instead drop rows with a missing image and
continue - never the default, since silently training on fewer pages than
you think you have is exactly the failure mode this check exists to catch.
Missing *text* files are never dropped, allow_missing_files or not - a
page legitimately having no transcription (e.g. a photo) is normal, and
lib/features.py's safe_read_text already falls back to "" for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import validate_manifest_paths
from tasks import TaskConfig


@dataclass
class LabelData:
    df: pd.DataFrame          # dossier, page_num, label, img_path, text_path, split
    class_names: list[str]
    num_classes: int


def ann_to_trans_name(dossier: str) -> str:
    """Map an annotation dossier name (dash-separated, e.g.
    'a2478-smith-a-0000001') to the Qwen transcription folder/file naming
    convention (underscore-joined middle segment, e.g. 'a2478-smith_a-0000001')."""
    parts = dossier.split("-")
    return f"{parts[0]}-{'_'.join(parts[1:-1])}-{parts[-1]}"


def page_png_path(data_root: Path, task: TaskConfig, dossier: str, page_num: int) -> Path:
    return data_root / task.png_root / dossier / f"{dossier}_page_{int(page_num):04d}.png"


def page_text_path(data_root: Path, task: TaskConfig, dossier: str, page_num: int) -> Path:
    trans_name = ann_to_trans_name(dossier)
    return data_root / task.text_root / trans_name / f"{trans_name}_page_{int(page_num):04d}{task.text_extension}"


def _attach_paths(df: pd.DataFrame, data_root: Path, task: TaskConfig) -> pd.DataFrame:
    df = df.copy()
    df["img_path"] = df.apply(lambda r: page_png_path(data_root, task, r["dossier"], r["page_num"]), axis=1)
    df["text_path"] = df.apply(lambda r: page_text_path(data_root, task, r["dossier"], r["page_num"]), axis=1)
    return df


def _assign_computed_split(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    """Dossier-level 70/15/15 train/val/test split, so all pages of a
    dossier land in the same split (avoids leakage)."""
    dossiers_all = df["dossier"].unique()
    train_dos, tmp_dos = train_test_split(dossiers_all, test_size=0.30, random_state=random_seed)
    val_dos, test_dos = train_test_split(tmp_dos, test_size=0.50, random_state=random_seed)
    train_set, val_set = set(train_dos), set(val_dos)

    def split_label(d):
        if d in train_set:
            return "train"
        if d in val_set:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["dossier"].map(split_label)
    return df


def _select_columns(df: pd.DataFrame, labels_tsv: str, split_source: str) -> list[str]:
    cols = ["dossier", "page_num", "label"]
    if split_source == "tsv_column":
        if "split" not in df.columns:
            raise SystemExit(
                f"--split-source tsv_column requires a 'split' column in {labels_tsv}, but none was found "
                f"(columns: {list(df.columns)}) - use --split-source computed instead, or add a 'split' "
                f"column to the TSV."
            )
        cols.append("split")
    return cols


def _finalize_split(labels: pd.DataFrame, split_source: str, random_seed: int) -> pd.DataFrame:
    if split_source == "computed":
        return _assign_computed_split(labels, random_seed)
    return labels  # tsv_column: 'split' was already carried through by _select_columns


def _validate_and_drop_missing_images(df: pd.DataFrame, allow_missing_files: bool) -> pd.DataFrame:
    """Raises (via validate_manifest_paths) if any img_path/text_path is
    missing, unless allow_missing_files - in which case rows with a missing
    *image* are dropped (there's no fallback for those) and a count is
    printed; missing text is never a reason to drop a row (safe_read_text
    handles that itself). See module docstring."""
    validate_manifest_paths(
        df, Path(""), image_col="img_path", text_col="text_path", allow_missing=allow_missing_files,
    )
    exists_mask = df["img_path"].map(lambda p: p.exists())
    if (~exists_mask).any():
        print(f"Dropping {(~exists_mask).sum()}/{len(df)} pages with a missing image (--allow-missing-files).")
        df = df[exists_mask].reset_index(drop=True)
    return df


def _load_start_page(
    data_root: Path, task: TaskConfig, random_seed: int, split_source: str, allow_missing_files: bool = False,
) -> LabelData:
    df_tsv = pd.read_csv(data_root / task.labels_tsv, sep="\t")
    df_tsv["label"] = df_tsv["start_page"].apply(lambda x: 1 if x == "yes" else 0)
    df_tsv = df_tsv.rename(columns={task.dossier_column: "dossier"})

    labels = df_tsv[_select_columns(df_tsv, task.labels_tsv, split_source)]
    labels = _attach_paths(labels, data_root, task)
    labels = _validate_and_drop_missing_images(labels, allow_missing_files)

    labels = _finalize_split(labels, split_source, random_seed)
    for s in ("train", "val", "test"):
        sub = labels[labels["split"] == s]
        print(f"{s:6s}: {len(sub):4d} pages  |  start-pages: {sub['label'].sum():3d} ({sub['label'].mean()*100:.1f}%)")

    return LabelData(df=labels, class_names=["not-start", "start"], num_classes=2)


def _load_doc_type(
    data_root: Path, task: TaskConfig, random_seed: int, split_source: str, allow_missing_files: bool = False,
) -> LabelData:
    df_tsv = pd.read_csv(data_root / task.labels_tsv, sep="\t")
    start_pages = (
        df_tsv[df_tsv["start_page"] == "yes"].dropna(subset=["document_type"]).copy().reset_index(drop=True)
    )
    start_pages = start_pages.rename(columns={task.dossier_column: "dossier"})

    class_names = sorted(start_pages["document_type"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    start_pages["label"] = start_pages["document_type"].map(class_to_idx).astype(int)

    labels = start_pages[_select_columns(start_pages, task.labels_tsv, split_source)]
    labels = _attach_paths(labels, data_root, task)
    labels = _validate_and_drop_missing_images(labels, allow_missing_files)

    labels = _finalize_split(labels, split_source, random_seed)
    for s in ("train", "val", "test"):
        sub = labels[labels["split"] == s]
        print(f"{s:6s}: {len(sub):4d} start pages  |  {sub['label'].nunique()} distinct doc-types")

    return LabelData(df=labels, class_names=class_names, num_classes=len(class_names))


def build_extraction_manifest(data_root: Path, task: TaskConfig, allow_missing_files: bool = False) -> pd.DataFrame:
    """Every annotated page (regardless of any task-specific row filtering -
    doc_type's own loader above drops non-start pages, but a feature cache
    should cover the whole corpus so it can serve every task), with freshly
    built img_path/text_path (via page_png_path/page_text_path, so
    correctness doesn't depend on the labels TSV's own path columns - if it
    has any - already matching this --data-root).

    Keeps every other column from the raw TSV as-is (document_type,
    layout_type, functional_category, start_page, split, ...) - used by
    extract_features.py to build a manifest for
    scripts/classification/joint/precompute_embeddings.py, which expects
    exactly those column names by default."""
    df = pd.read_csv(data_root / task.labels_tsv, sep="\t")
    df = df.rename(columns={task.dossier_column: "dossier"})
    df = _attach_paths(df, data_root, task)
    return _validate_and_drop_missing_images(df, allow_missing_files)


def load_labels(
    task: TaskConfig, data_root: Path, random_seed: int = 42, split_source: str | None = None,
    allow_missing_files: bool = False,
) -> LabelData:
    """split_source: "computed" or "tsv_column", or None to use task.split_source
    (each TaskConfig's default - see lib/tasks.py). allow_missing_files: see
    module docstring - default False (raise immediately on any missing
    img_path/text_path)."""
    split_source = split_source or task.split_source
    if split_source not in ("computed", "tsv_column"):
        raise ValueError(f"unknown split_source {split_source!r} - expected 'computed' or 'tsv_column'")
    if task.name == "start_page":
        return _load_start_page(data_root, task, random_seed, split_source, allow_missing_files)
    if task.name == "doc_type":
        return _load_doc_type(data_root, task, random_seed, split_source, allow_missing_files)
    raise ValueError(f"no label loader registered for task {task.name!r}")
