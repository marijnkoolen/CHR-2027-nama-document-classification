"""Trains the LSTM+<backbone> sequence model: each dossier is a sequence of
pages, one cached backbone's features per page feed a bidirectional LSTM
that predicts a label per page. Requires extract_features.py to have been
run first for --features-backbone.

Saves the trained model + a model_config.json describing how to rebuild its
exact architecture (see lib/checkpoints.py) - test-set evaluation happens
separately, in evaluate_models.py.

Usage:
    python scripts/classification/sequential/train_sequence.py \\
        --task start_page --features-backbone vgg16 --data-root data --run-dir runs

    python scripts/classification/sequential/train_sequence.py \\
        --task start_page --features-backbone facebook/dinov2-small --data-root data --run-dir runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import save_torch_model
from common import pick_device, set_seed
from datasets import DossierSequenceDataset, sequence_pad_collate
from embeddings import load_backbone_features
from labels import load_labels
from model_naming import sequence_model_name
from models import LSTMClassifier
from tasks import get_task
from torch.utils.data import DataLoader
from train_loops import train_lstm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["start_page", "doc_type"], required=True)
    parser.add_argument(
        "--features-backbone", default="vgg16",
        help="a cached backbone name (vgg16, efficientnet_b0, or any HuggingFace checkpoint, e.g. "
             "facebook/dinov2-small) - extract_features.py must have been run for it first",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4, help="dossiers per batch")
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
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.random_seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    task = get_task(args.task)
    label_data = load_labels(
        task, args.data_root, random_seed=args.random_seed, split_source=args.split_source,
        allow_missing_files=args.allow_missing_files,
    )
    df = label_data.df

    cache_dir = args.cache_dir or (args.data_root / "embeddings")
    X = load_backbone_features(cache_dir, args.features_backbone, df)

    tr_df, va_df = (df[df["split"] == s] for s in ("train", "val"))
    tr_ds = DossierSequenceDataset(tr_df, X)
    va_ds = DossierSequenceDataset(va_df, X)
    print(f"dossiers: train={len(tr_ds)}  val={len(va_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=sequence_pad_collate)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=sequence_pad_collate)

    model = LSTMClassifier(
        input_dim=X.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_classes=label_data.num_classes, dropout=args.dropout,
    ).to(device)

    model_name = sequence_model_name(args.features_backbone)
    print(f"Training {model_name} (task={args.task}) …")
    model = train_lstm(
        model, tr_loader, va_loader, tr_df["label"].values, label_data.num_classes, device,
        n_epochs=args.n_epochs, lr=args.lr,
    )

    config = {
        "model_family": "sequence_lstm",
        "task": args.task,
        "features_backbone": args.features_backbone,
        "input_dim": X.shape[1],
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_classes": label_data.num_classes,
        "class_names": label_data.class_names,
    }
    save_torch_model(args.run_dir, args.task, model_name, model, config)


if __name__ == "__main__":
    main()
