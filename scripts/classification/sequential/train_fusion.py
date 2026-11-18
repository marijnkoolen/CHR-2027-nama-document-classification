"""Trains the early-fusion model: a small MLP over precomputed, frozen
image + text feature vectors from two cached backbones. Requires
extract_features.py to have been run first for both --image-backbone and
--text-backbone.

Saves the trained model + a model_config.json describing how to rebuild its
exact architecture (see lib/checkpoints.py) - test-set evaluation happens
separately, in evaluate_models.py.

Late fusion (averaging two fine-tuned models' softmax outputs) has no model
of its own to train - there's nothing here to save, since it's just an
average of two other models' saved predictions - so it's computed directly
by evaluate_models.py instead of by a train_*.py script.

Usage:
    python scripts/classification/sequential/train_fusion.py \\
        --task start_page --data-root data --run-dir runs

    python scripts/classification/sequential/train_fusion.py \\
        --task start_page --image-backbone microsoft/dit-large-finetuned-rvlcdip \\
        --text-backbone xlm-roberta-base --data-root data --run-dir runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import save_torch_model
from common import pick_device, set_seed
from datasets import EarlyFusionDataset
from embeddings import load_backbone_features
from labels import load_labels
from model_naming import fusion_model_name
from models import EarlyFusionMLP
from tasks import get_task
from torch.utils.data import DataLoader
from train_loops import train_fusion_mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["start_page", "doc_type"], required=True)
    parser.add_argument(
        "--image-backbone", default="efficientnet_b0",
        help="a cached image backbone name (vgg16, efficientnet_b0, or any HuggingFace checkpoint, e.g. "
             "facebook/dinov2-small) - extract_features.py must have been run for it first",
    )
    parser.add_argument(
        "--text-backbone", default="bert-base-uncased",
        help="a cached text backbone name (any HuggingFace checkpoint) - extract_features.py must have "
             "been run for it first",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
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
    X_img = load_backbone_features(cache_dir, args.image_backbone, df)
    X_text = load_backbone_features(cache_dir, args.text_backbone, df)

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    y_tr, y_va = df.loc[train_mask, "label"].values, df.loc[val_mask, "label"].values

    tr_loader = DataLoader(
        EarlyFusionDataset(X_img[train_mask], X_text[train_mask], y_tr), batch_size=args.batch_size, shuffle=True
    )
    va_loader = DataLoader(
        EarlyFusionDataset(X_img[val_mask], X_text[val_mask], y_va), batch_size=args.batch_size, shuffle=False
    )

    model = EarlyFusionMLP(
        img_dim=X_img.shape[1], text_dim=X_text.shape[1], hidden=args.hidden,
        num_classes=label_data.num_classes, dropout=args.dropout,
    ).to(device)

    model_name = fusion_model_name(args.image_backbone, args.text_backbone)
    print(f"Training {model_name} (task={args.task}) …")
    model = train_fusion_mlp(
        model, tr_loader, va_loader, y_tr, label_data.num_classes, device,
        n_epochs=args.n_epochs, lr=args.lr,
    )

    config = {
        "model_family": "fusion_early",
        "task": args.task,
        "image_backbone": args.image_backbone,
        "text_backbone": args.text_backbone,
        "img_dim": X_img.shape[1],
        "text_dim": X_text.shape[1],
        "hidden": args.hidden,
        "dropout": args.dropout,
        "num_classes": label_data.num_classes,
        "class_names": label_data.class_names,
    }
    save_torch_model(args.run_dir, args.task, model_name, model, config)


if __name__ == "__main__":
    main()
