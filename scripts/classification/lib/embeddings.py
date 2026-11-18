"""Reads the embeddings.npy + embeddings_manifest.tsv cache format written
by extract_features.py (via ../precompute_embeddings.py) - the shared
feature-cache format for this project. Every consumer here
(train_baseline.py, train_sequence.py, train_fusion.py, lib/predict.py)
joins the cache against its own dataframe by (dossier, page_num) rather
than by row position: a cache is built once for the whole corpus (see
extract_features.py's module docstring), so it generally covers more rows,
and in a different order, than any one caller's df needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def backbone_cache_dir(cache_dir: Path, backbone: str) -> Path:
    """Matches extract_features.py's own slugify_backbone() - a HuggingFace
    checkpoint name (e.g. 'facebook/dinov2-small') isn't a valid single
    path component as-is; torchvision names (vgg16, efficientnet_b0)
    already are and pass through unchanged."""
    return cache_dir / backbone.replace("/", "__")


def load_embeddings_cache(cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    if not (cache_dir / "embeddings.npy").exists():
        raise SystemExit(f"no embeddings cache at {cache_dir} - run extract_features.py for this backbone first")
    embeddings = np.load(cache_dir / "embeddings.npy")
    manifest = pd.read_csv(cache_dir / "embeddings_manifest.tsv", sep="\t")
    return embeddings, manifest


def select_features(
    embeddings: np.ndarray, manifest: pd.DataFrame, df: pd.DataFrame,
    dossier_col: str = "dossier", page_col: str = "page_num",
) -> np.ndarray:
    """Returns embeddings rows in df's order, joined against manifest via
    (pdf_id, page_number). Raises with a clear message if any (dossier,
    page_num) pair in df isn't present in the cache, rather than silently
    misaligning or producing NaNs."""
    key_to_row = {
        (pdf, int(page)): i for i, (pdf, page) in enumerate(zip(manifest["pdf_id"], manifest["page_number"]))
    }
    keys = list(zip(df[dossier_col], df[page_col].astype(int)))
    missing = [k for k in keys if k not in key_to_row]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(keys)} pages missing from the embeddings cache (e.g. {missing[0]!r}) - "
            f"re-run extract_features.py for this backbone, or check --cache-dir."
        )
    return embeddings[[key_to_row[k] for k in keys]]


def load_backbone_features(
    cache_dir: Path, backbone: str, df: pd.DataFrame, dossier_col: str = "dossier", page_col: str = "page_num",
) -> np.ndarray:
    """load_embeddings_cache + select_features in one call, for the common
    case of reading one backbone's raw (not normalized) features for df's rows."""
    embeddings, manifest = load_embeddings_cache(backbone_cache_dir(cache_dir, backbone))
    return select_features(embeddings, manifest, df, dossier_col, page_col)


def load_baseline_features(
    cache_dir: Path, backbones: list[str], df: pd.DataFrame, dossier_col: str = "dossier", page_col: str = "page_num",
) -> np.ndarray:
    """L2-normalised features for the KNN/XGBoost baselines and the
    early-fusion MLP - a single backbone's normalised vector, or several
    backbones' normalised vectors concatenated (the generalised replacement
    for the old --features {vgg16,bert,ensemble} three-way enum: any list
    of one or more cached backbone names works the same way now)."""
    parts = [normalize(load_backbone_features(cache_dir, b, df, dossier_col, page_col)) for b in backbones]
    return np.hstack(parts) if len(parts) > 1 else parts[0]
