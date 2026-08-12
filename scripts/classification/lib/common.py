"""Shared utilities for both pipelines under scripts/classification/:
scripts/classification/joint/ (train.py's joint 4-task training, page and
sequence mode) and scripts/classification/sequential/ (independently
trained single-task models, chained at evaluation time).

torch/torchvision are imported lazily, inside each function, rather than at
module level: scripts/classification/sequential/evaluate_models.py runs
each saved model's evaluation in its own subprocess, and a process
evaluating a KNN/XGBoost baseline should never end up loading torch at all
- torch and xgboost coexisting in the same process was observed to segfault
non-deterministically on macOS (colliding bundled OpenMP runtimes), and
that's only avoidable if a torch-free code path actually stays torch-free,
which a module-level `import torch` here would silently defeat for every
caller, baseline or not. (get_text_extractor/validate_manifest_paths/
format_confusion_matrix/assign_stratified_splits never needed torch to
begin with - only pick_device/build_transforms/build_image_transforms/
set_seed/class_weights_tensor do.)

ManifestImageDataset and the page-mode manifest-dataloader builder that
uses it live in manifest_data.py instead of here, for the same reason:
subclassing torch.utils.data.Dataset requires torch.utils.data to be
importable at class-definition time (module load), which can't be made
lazy the way a plain function call can - keeping that class out of this
module is what lets this module itself stay import-safe for a torch-free
caller.

build_transforms vs build_image_transforms: two deliberately different
functions, not a duplicate to be merged into one. build_transforms (used by
joint/train.py, any HuggingFace ViT-style backbone) normalizes to
mean=std=0.5 and layers on rotation/color-jitter/blur/erasing tuned for
scanned documents. build_image_transforms (used by sequential/, VGG16/
EfficientNet-B0 fine-tuning) normalizes to true ImageNet mean/std, matching
what those backbones were actually pretrained with - swapping in
build_transforms' 0.5/0.5/0.5 normalization there would silently feed a
pretrained-ImageNet-normalization backbone out-of-distribution inputs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from markdown_text import extract_text as _extract_markdown_text
from pagexml import extract_text as _extract_pagexml_text

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_text_extractor(text_source: str) -> Callable[[str], str]:
    """extract_text(path) -> str for --text-source ("pagexml" or
    "markdown") - shared by precompute_embeddings.py, train.py, and
    predict.py so all three resolve a text source identically. pagexml:
    PageXML transcriptions (lib/pagexml.py). markdown: Qwen2.5-VL
    markdown-mode OCR output, stripped of markup (lib/markdown_text.py) -
    see that module's docstring for why markup is stripped, and why
    markdown is this project's default text source."""
    if text_source == "markdown":
        return _extract_markdown_text
    if text_source == "pagexml":
        return _extract_pagexml_text
    raise ValueError(f"unknown text_source {text_source!r} - expected 'pagexml' or 'markdown'")


def validate_manifest_paths(
    manifest: pd.DataFrame, image_root: Path, image_col: str | None, text_col: str | None,
    allow_missing: bool = False, max_examples: int = 10,
) -> None:
    """Checks every non-null path in image_col and text_col (pass None for
    either to skip it - e.g. image_col=None for --modality text) resolves
    to an existing file under image_root, before any (slow) model loading
    or data extraction happens. Call this right after reading the manifest,
    in every script that reads raw images/text (train.py, precompute_
    embeddings.py) - NOT needed for train.py --cached-embeddings, which
    never touches raw files at all.

    Raises SystemExit listing example missing paths if any are found and
    allow_missing is False (the default). This exists because silently
    tolerating broken paths - which is what extract_text()'s "" fallback,
    combined with only a >50%-empty-text WARNING, previously allowed - can
    produce a manifest where every row's text is empty, and therefore every
    text embedding is near-identical, without erroring at all; this project
    has been bitten by exactly that failure mode more than once (see
    precompute_embeddings.py's module docstring), from different root
    causes each time (a wrong .txt/.xml path, and a wrong markdown OCR
    path) - a strict, load-time check catches the general problem instead
    of each specific instance of it after the fact.

    A NaN/missing *value* in a column (no path given at all) is not an
    error - some pages legitimately have no transcription (e.g. photos);
    only a given, non-null path that doesn't resolve to an existing file is
    treated as a mistake.

    allow_missing=True downgrades this to a warning and continues - for
    text_col, that's coherent (matches extract_text's own defined ""
    fallback for a missing page); for image_col there's no such fallback
    anywhere in this codebase, so allowing a missing image just defers the
    failure to a harder-to-diagnose crash later, during actual data
    loading, rather than avoiding it - allow_missing is meant for
    tolerating a few genuinely-expected gaps, not as a way to skip fixing a
    systematic path mistake."""
    problems = []
    for col in (image_col, text_col):
        if not col or col not in manifest.columns:
            continue
        missing = [
            str(image_root / p) for p in manifest[col]
            if pd.notna(p) and not (image_root / p).exists()
        ]
        if missing:
            problems.append((col, missing))

    if not problems:
        return

    total_missing = sum(len(missing) for _, missing in problems)
    lines = [f"{total_missing} file(s) referenced in the manifest do not exist on disk:"]
    for col, missing in problems:
        lines.append(f"  column {col!r}: {len(missing)} missing, e.g.:")
        for p in missing[:max_examples]:
            lines.append(f"    {p}")
        if len(missing) > max_examples:
            lines.append(f"    ... and {len(missing) - max_examples} more")
    message = "\n".join(lines)

    if allow_missing:
        print(f"WARNING: {message}\n(continuing anyway - --allow-missing-files was set)")
        return
    raise SystemExit(
        f"{message}\n\nThis usually means a wrong --image-root, a wrong --*-col, or (for text) a wrong "
        f"--text-source/OCR output directory - fix the paths, or pass --allow-missing-files to proceed anyway "
        f"(missing text paths fall back to empty text for that page; missing image paths will still fail "
        f"later instead, when that page is actually loaded, since there's no fallback for a missing image)."
    )


def load_page_manifest(
    manifest_path: Path, pdf_col: str, page_col: str, image_col: str, text_col: str,
    image_root: Path = Path(""), limit_dossiers: int | None = None, allow_missing_files: bool = False,
) -> pd.DataFrame:
    """Reads an arbitrary page manifest (TSV/CSV, any column names) for a
    corpus that has no labels TSV at all - e.g. a large corpus of genuinely
    unlabeled pages to run inference over, as opposed to every other loader
    in this project (lib/labels.py's build_extraction_manifest/load_labels),
    which all read one of this project's own labels TSVs. Shared by
    sequential/predict.py and sequential/extract_features.py --manifest.

    Renames to this project's canonical dossier/page_num/img_path/text_path
    names, resolves image_col/text_col against image_root (a no-op if the
    manifest's own paths are already fully resolved - image_root defaults
    to Path("")), and applies the same fail-fast path validation every
    other script in this pipeline uses (see validate_manifest_paths above) -
    missing files here should fail loudly before any embedding/prediction
    work starts, not silently produce a smaller-than-expected cache."""
    sep = "\t" if str(manifest_path).endswith(".tsv") else ","
    df = pd.read_csv(manifest_path, sep=sep)
    df = df.rename(columns={pdf_col: "dossier", page_col: "page_num", image_col: "img_path", text_col: "text_path"})
    for col in ("img_path", "text_path"):
        df[col] = df[col].map(lambda p: str(image_root / p) if pd.notna(p) else p)

    if limit_dossiers:
        keep = df["dossier"].drop_duplicates().head(limit_dossiers)
        df = df[df["dossier"].isin(keep)]

    df = df.sort_values(["dossier", "page_num"]).reset_index(drop=True)

    validate_manifest_paths(df, Path(""), image_col="img_path", text_col="text_path", allow_missing=allow_missing_files)
    if allow_missing_files:
        exists_mask = df["img_path"].map(lambda p: Path(p).exists())
        if (~exists_mask).any():
            print(f"Dropping {(~exists_mask).sum()}/{len(df)} pages with a missing image (--allow-missing-files).")
            df = df[exists_mask].reset_index(drop=True)

    return df


def format_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    """A readable aligned text grid (true label = row, predicted = column).
    Can get wide for many classes, but that's inherent to confusion
    matrices - fine once redirected to a file."""
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "true \\ pred"
    return df.to_string()


def pick_device(prefer: str | None = None) -> "torch.device":  # noqa: F821
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(image_size: int, train: bool, augment_strength: str = "moderate") -> "transforms.Compose":  # noqa: F821
    """Augmentations chosen for the described material: varying paper colour and
    background, rotation/skew, stains/tears, mixed print/handwriting, uneven
    lighting. Colour jitter and random erasing matter more here than the mild
    crops/flips typical of natural-image pipelines - documents are not
    rotation- or flip-invariant in the usual sense (a form upside down is a
    different signal), so we keep rotation small and never flip.
    """
    from torchvision import transforms

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    strong = augment_strength == "strong"
    ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(degrees=8 if not strong else 12, fill=255),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2 if not strong else 0.35, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3 if not strong else 0.5, scale=(0.02, 0.12)),  # simulates stains/tears
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


def build_image_transforms(image_size: int = 224, train: bool = False) -> "transforms.Compose":  # noqa: F821
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def class_weights_tensor(y: np.ndarray, n_classes: int, device: "torch.device") -> "torch.Tensor":  # noqa: F821
    """Inverse-frequency class weights, normalised to sum to n_classes - used
    by every nn.CrossEntropyLoss(weight=...) in this pipeline so rare classes
    (e.g. the minority "start page" class, or rare document types) aren't
    drowned out by the majority class(es)."""
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32).to(device)


def trainable_param_count(module: "nn.Module") -> int:  # noqa: F821
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assign_stratified_splits(labels: pd.Series, ratios=(0.7, 0.15, 0.15), seed: int = 0) -> pd.Series:
    """A 70/15/15 split, stratified per class; classes too small to appear in
    every split (e.g. singleton real-world labels) fall back to train-only."""
    rng = random.Random(seed)
    splits = pd.Series(index=labels.index, dtype=object)
    for label, idx in labels.groupby(labels).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            splits.loc[idx] = "train"
            continue
        n_train = max(1, round(n * ratios[0]))
        n_val = max(1, round(n * ratios[1])) if n - n_train >= 2 else 0
        assigned = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        splits.loc[idx] = assigned
    return splits
