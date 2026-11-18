"""End-to-end fine-tuning of a single backbone: VGG16, EfficientNet-B0, or
any HuggingFace image checkpoint (facebook/dinov2-small, microsoft/dit-large-
finetuned-rvlcdip, ...) for image; TextCNN (frozen-BERT-embeddings + 1D-CNN)
or full BERT-style fine-tuning (--backbone bert --bert-model <any HF text
checkpoint>) for text. Unlike train_baseline.py, these read raw images/text
directly (no cached features) since the backbone itself is being trained.

--backbone dispatches on the string itself: "textcnn"/"bert" mean text
mode (see run_text_backbone) - anything else is treated as an image
backbone (vgg16/efficientnet_b0 use their own fixed freeze policy; any
other string is loaded as a HuggingFace checkpoint via lib/models.py's
BackboneClassifier, --unfreeze-blocks applies - see build_image_classifier).

Saves the trained model + a model_config.json describing how to rebuild its
exact architecture (see lib/checkpoints.py) - test-set evaluation happens
separately, in evaluate_models.py, which uses that config to reload and
re-score the model without retraining.

Usage:
    python scripts/classification/sequential/train_finetune.py \\
        --task start_page --backbone vgg16 --data-root data --run-dir runs

    python scripts/classification/sequential/train_finetune.py \\
        --task start_page --backbone facebook/dinov2-small --data-root data --run-dir runs

    python scripts/classification/sequential/train_finetune.py \\
        --task doc_type --backbone bert --data-root data --run-dir runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import save_torch_model
from common import build_image_transforms, pick_device, set_seed
from datasets import PageImageDataset, TextDataset
from labels import load_labels
from model_naming import finetune_model_name
from models import BERTClassifier, TextCNN, build_image_classifier
from tasks import get_task
from torch.utils.data import DataLoader
from train_loops import train_image_model, train_text_model

TEXT_BACKBONES = {"textcnn", "bert"}

# n_epochs/lr/batch_size the two notebooks used for each backbone; anything
# not listed here (any other image or text backbone) uses GENERIC_DEFAULTS.
BACKBONE_DEFAULTS = {
    "vgg16": dict(n_epochs=15, lr=1e-4, batch_size=16),
    "efficientnet": dict(n_epochs=15, lr=5e-5, batch_size=16),
    "efficientnet_b0": dict(n_epochs=15, lr=5e-5, batch_size=16),
    "textcnn": dict(n_epochs=15, lr=1e-3, batch_size=16),
    "bert": dict(n_epochs=10, lr=2e-5, batch_size=8),
}
GENERIC_IMAGE_DEFAULTS = dict(n_epochs=15, lr=1e-4, batch_size=16)


def run_image_backbone(args, label_data, device):
    df = label_data.df
    tr_df, va_df = (df[df["split"] == s] for s in ("train", "val"))
    train_tf = build_image_transforms(train=True)
    eval_tf = build_image_transforms(train=False)
    tr_loader = DataLoader(PageImageDataset(tr_df, train_tf), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(PageImageDataset(va_df, eval_tf), batch_size=args.eval_batch_size, shuffle=False)

    model = build_image_classifier(
        args.backbone, label_data.num_classes, device, unfreeze_last_n_blocks=args.unfreeze_blocks
    )
    return train_image_model(
        model, tr_loader, va_loader, tr_df["label"].values, label_data.num_classes, device,
        n_epochs=args.n_epochs, lr=args.lr,
    )


def run_text_backbone(args, label_data, device):
    from transformers import AutoModel, AutoTokenizer

    df = label_data.df
    tr_df, va_df = (df[df["split"] == s] for s in ("train", "val"))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)

    if args.backbone == "textcnn":
        bert_base = AutoModel.from_pretrained(args.bert_model).to(device)
        for p in bert_base.parameters():
            p.requires_grad = False
        model = TextCNN(bert_base, num_classes=label_data.num_classes).to(device)
    else:
        model = BERTClassifier(args.bert_model, num_classes=label_data.num_classes).to(device)

    tr_loader = DataLoader(
        TextDataset(tr_df, tokenizer, args.max_text_length), batch_size=args.batch_size, shuffle=True
    )
    va_loader = DataLoader(
        TextDataset(va_df, tokenizer, args.max_text_length), batch_size=args.eval_batch_size, shuffle=False
    )

    return train_text_model(
        model, tr_loader, va_loader, tr_df["label"].values, label_data.num_classes, device,
        n_epochs=args.n_epochs, lr=args.lr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["start_page", "doc_type"], required=True)
    parser.add_argument(
        "--backbone", required=True,
        help="vgg16, efficientnet_b0, textcnn, bert, or any HuggingFace image checkpoint (e.g. "
             "facebook/dinov2-small, microsoft/dit-large-finetuned-rvlcdip) - 'textcnn'/'bert' select text "
             "mode (see --bert-model for the actual text checkpoint), everything else is image mode",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--bert-model", default="bert-base-uncased", help="used by --backbone textcnn/bert")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument(
        "--unfreeze-blocks", type=int, default=2,
        help="image mode only, and only for backbones other than vgg16/efficientnet_b0 (which use their own "
             "fixed policy) - how many trailing transformer blocks to fine-tune (0 = fully frozen)",
    )
    parser.add_argument("--n-epochs", type=int, default=None, help="default depends on --backbone")
    parser.add_argument("--lr", type=float, default=None, help="default depends on --backbone")
    parser.add_argument("--batch-size", type=int, default=None, help="default depends on --backbone")
    parser.add_argument("--eval-batch-size", type=int, default=32)
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

    defaults = BACKBONE_DEFAULTS.get(args.backbone, GENERIC_IMAGE_DEFAULTS)
    args.n_epochs = args.n_epochs or defaults["n_epochs"]
    args.lr = args.lr or defaults["lr"]
    args.batch_size = args.batch_size or defaults["batch_size"]

    set_seed(args.random_seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    task = get_task(args.task)
    label_data = load_labels(
        task, args.data_root, random_seed=args.random_seed, split_source=args.split_source,
        allow_missing_files=args.allow_missing_files,
    )

    is_text = args.backbone in TEXT_BACKBONES
    model_name = finetune_model_name(args.backbone, args.bert_model)
    print(f"Training {model_name} (task={args.task}) …")
    if is_text:
        model = run_text_backbone(args, label_data, device)
        config = {
            "model_family": "finetune_text",
            "task": args.task,
            "backbone": args.backbone,
            "bert_model": args.bert_model,
            "max_text_length": args.max_text_length,
            "num_classes": label_data.num_classes,
            "class_names": label_data.class_names,
        }
    else:
        model = run_image_backbone(args, label_data, device)
        config = {
            "model_family": "finetune_image",
            "task": args.task,
            "backbone": args.backbone,
            "num_classes": label_data.num_classes,
            "class_names": label_data.class_names,
        }

    save_torch_model(args.run_dir, args.task, model_name, model, config)


if __name__ == "__main__":
    main()
