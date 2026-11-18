"""Trains a KNN or XGBoost baseline on cached embeddings from one or more
backbones, and saves it - test-set evaluation happens separately, in
evaluate_models.py.

Requires extract_features.py to have been run first for every backbone
named in --features (a single backbone, or several - e.g.
--features vgg16 bert-base-uncased - concatenates their normalised
features, the "ensemble" case).

KNN always sweeps k in [3, 5, 7, 11, 15] on the validation set (cosine
distance, L2-normalised features) and keeps the best-k model - both
notebooks did this for VGG16, but only doc_type_start_page_classifier_qwen.ipynb
did it for BERT/ensemble too (page_start_classifier_qwen.ipynb fixed k=7
there); this always sweeps, for consistency across --features and --task.

XGBoost hyperparameters default to values in the middle of what the two
notebooks used across their VGG16/BERT/ensemble/binary/multi-class variants
- override via the --xgb-* flags if you want to reproduce a specific
notebook cell's exact numbers.

Usage:
    python scripts/classification/sequential/train_baseline.py \\
        --task start_page --model knn --features vgg16 \\
        --data-root data --run-dir runs

    python scripts/classification/sequential/train_baseline.py \\
        --task start_page --model xgboost --features vgg16 bert-base-uncased \\
        --data-root data --run-dir runs
"""

from __future__ import annotations

# Must be imported before torch (pulled in transitively below, via labels.py) -
# importing xgboost after torch has already loaded its own bundled OpenMP
# runtime crashes with an OMP Error #179 (pthread_mutex_init) segfault on
# macOS, since the two packages ship separate libomp builds that collide once
# both are initialized in the same process.
import xgboost  # noqa: F401,E402

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import save_sklearn_model
from common import set_seed
from embeddings import load_baseline_features
from labels import load_labels
from model_naming import baseline_model_name
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from tasks import get_task


def train_knn(X_tr, y_tr, X_va, y_va, num_classes: int) -> KNeighborsClassifier:
    average = "binary" if num_classes == 2 else "macro"
    k_grid = [k for k in (3, 5, 7, 11, 15) if k <= len(X_tr)] or [len(X_tr)]
    best_k, best_f1 = k_grid[0], -1.0
    for k in k_grid:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(X_tr, y_tr)
        f1 = f1_score(y_va, knn.predict(X_va), average=average, zero_division=0)
        print(f"  k={k:2d}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_k = f1, k
    print(f"  best k={best_k} (val_f1={best_f1:.4f})")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    return knn


def train_xgboost(X_tr, y_tr, X_va, y_va, num_classes: int, args) -> "XGBClassifier":  # noqa: F821
    from xgboost import XGBClassifier

    kwargs = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        random_state=args.random_seed,
        early_stopping_rounds=20,
        verbosity=0,
        n_jobs=-1,
    )
    if num_classes == 2:
        pos_weight = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
        kwargs.update(eval_metric="logloss", scale_pos_weight=pos_weight)
    else:
        kwargs.update(objective="multi:softmax", num_class=num_classes, eval_metric="mlogloss")

    model = XGBClassifier(**kwargs)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["start_page", "doc_type"], required=True)
    parser.add_argument("--model", choices=["knn", "xgboost"], required=True)
    parser.add_argument(
        "--features", nargs="+", required=True,
        help="one or more cached backbone names (e.g. vgg16, bert-base-uncased, facebook/dinov2-small) - "
             "extract_features.py must have been run for each first; several are concatenated "
             "(normalised first) into one feature vector",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--split-source", choices=["computed", "tsv_column"], default=None,
        help="'computed' or 'tsv_column' - defaults to the task's usual choice (see lib/tasks.py); must "
             "match extract_features.py and evaluate_models.py for this run",
    )
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="don't error on a missing image (or transcription) file - drop rows with a missing image and "
             "continue instead (missing text always falls back to empty text, regardless). Off by default: "
             "a wrong --data-root or a path-formula mistake should fail fast, before any heavy lifting, not "
             "silently shrink the dataset.",
    )
    args = parser.parse_args()

    set_seed(args.random_seed)

    task = get_task(args.task)
    label_data = load_labels(
        task, args.data_root, random_seed=args.random_seed, split_source=args.split_source,
        allow_missing_files=args.allow_missing_files,
    )
    df = label_data.df

    cache_dir = args.cache_dir or (args.data_root / "embeddings")
    X = load_baseline_features(cache_dir, args.features, df)

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    X_tr, y_tr = X[train_mask], df.loc[train_mask, "label"].values
    X_va, y_va = X[val_mask], df.loc[val_mask, "label"].values
    print(f"train={len(X_tr)}  val={len(X_va)}")

    model_name = baseline_model_name(args.model, args.features)
    print(f"Training {model_name} …")
    if args.model == "knn":
        model = train_knn(X_tr, y_tr, X_va, y_va, label_data.num_classes)
    else:
        model = train_xgboost(X_tr, y_tr, X_va, y_va, label_data.num_classes, args)

    config = {
        "model_family": "baseline",
        "task": args.task,
        "algorithm": args.model,
        "features": args.features,
        "num_classes": label_data.num_classes,
        "class_names": label_data.class_names,
    }
    save_sklearn_model(args.run_dir, args.task, model_name, model, config)


if __name__ == "__main__":
    main()
