"""Reproduces a saved checkpoint's predictions for an arbitrary set of rows
- the single place model-family dispatch (rebuild skeleton from
model_config.json, load weights, run inference) lives, shared by
evaluate_models.py (always predicts on --task's own test split) and
evaluate_pipeline.py (predicts on the start_page+doc_type joint test set,
and later feeds a doc_type model only the pages a start_page model actually
predicted as start pages - not necessarily --task doc_type's own test
split).

Every predict_* function returns (keys, preds, probs) where keys is a list
of (dossier, page_num) tuples in the same row order as preds/probs. Callers
should look up ground truth (or merge predictions back into a bigger
dataframe) by joining on keys rather than assuming preds line up with the
input df's own row order - true for every family except sequence_lstm,
whose DataLoader iterates dossiers in sorted order (via
lib/datasets.py's DossierSequenceDataset) and pages within a dossier
sorted by page_num, not necessarily the input df's order.

Feature-cache lookups (baseline/sequence_lstm/fusion_early) go through
lib/embeddings.py, joining by (dossier, page_num) against whichever
backbone(s) each checkpoint's own model_config.json says it was trained
on - unlike the old per-task npz caches, extract_features.py's cache now
covers every annotated page regardless of task (see its module docstring),
so evaluate_pipeline.py feeding a doc_type model pages it never saw as
*true* start pages needs no special-casing here anymore: the same cache
already has them.

ctx is a dict of: run_dir, task_name (where the checkpoint lives -
run_dir/task_name/model_name/), cache_dir, device, batch_size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from checkpoints import load_config, load_sklearn_model, load_torch_state_dict
from embeddings import load_backbone_features, load_baseline_features


def _keys(df: pd.DataFrame) -> list[tuple]:
    return list(zip(df["dossier"], df["page_num"]))


def predict_baseline(ctx, model_name, config, df):
    """sklearn/xgboost - deliberately does not touch torch (not imported by
    this function, lib/checkpoints.py's load_sklearn_model, or
    lib/embeddings.py) so a worker process evaluating one of these never
    loads torch at all - torch and xgboost coexisting in the same process
    was observed to segfault on macOS, see evaluate_models.py's module
    docstring."""
    model = load_sklearn_model(ctx["run_dir"], ctx["task_name"], model_name)
    X = load_baseline_features(ctx["cache_dir"], config["features"], df)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return _keys(df), preds, probs


def predict_finetune_image(ctx, model_name, config, df):
    from common import build_image_transforms
    from datasets import PageImageDataset
    from models import build_image_classifier
    from torch.utils.data import DataLoader
    from train_loops import predict_image_model

    eval_tf = build_image_transforms(train=False)
    loader = DataLoader(PageImageDataset(df, eval_tf), batch_size=ctx["batch_size"], shuffle=False)

    # unfreeze_last_n_blocks doesn't affect the architecture's shape (only
    # which params get grad at train time), so the default here is fine
    # regardless of what --unfreeze-blocks train_finetune.py was run with.
    model = build_image_classifier(config["backbone"], config["num_classes"], ctx["device"])
    model.load_state_dict(load_torch_state_dict(ctx["run_dir"], ctx["task_name"], model_name))
    model.eval()

    preds, probs = predict_image_model(model, loader, ctx["device"])
    return _keys(df), preds, probs


def predict_finetune_text(ctx, model_name, config, df):
    from datasets import TextDataset
    from models import BERTClassifier, TextCNN
    from torch.utils.data import DataLoader
    from train_loops import predict_text_model
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"], use_fast=True)
    if config["backbone"] == "textcnn":
        bert_base = AutoModel.from_pretrained(config["bert_model"]).to(ctx["device"])
        for p in bert_base.parameters():
            p.requires_grad = False
        model = TextCNN(bert_base, num_classes=config["num_classes"]).to(ctx["device"])
    else:
        model = BERTClassifier(config["bert_model"], num_classes=config["num_classes"]).to(ctx["device"])
    model.load_state_dict(load_torch_state_dict(ctx["run_dir"], ctx["task_name"], model_name))
    model.eval()

    loader = DataLoader(
        TextDataset(df, tokenizer, config["max_text_length"]), batch_size=ctx["batch_size"], shuffle=False
    )
    preds, probs = predict_text_model(model, loader, ctx["device"])
    return _keys(df), preds, probs


def predict_sequence_lstm(ctx, model_name, config, df):
    from datasets import DossierSequenceDataset, sequence_pad_collate
    from models import LSTMClassifier
    from torch.utils.data import DataLoader
    from train_loops import predict_lstm

    X = load_backbone_features(ctx["cache_dir"], config["features_backbone"], df)
    # DossierSequenceDataset indexes X by df's pandas index (features[grp.index]),
    # so it must be a fresh 0..len(df)-1 range matching X's own row order - df
    # arrives here as an arbitrary caller-chosen subset (evaluate_models.py's test
    # split, or evaluate_pipeline.py's predicted-head-pages subset), which generally
    # still carries its *original* (non-contiguous) index from a larger dataframe.
    df = df.reset_index(drop=True)
    loader = DataLoader(
        DossierSequenceDataset(df, X), batch_size=ctx["batch_size"], shuffle=False, collate_fn=sequence_pad_collate
    )

    model = LSTMClassifier(
        input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
        num_classes=config["num_classes"], dropout=config["dropout"],
    ).to(ctx["device"])
    model.load_state_dict(load_torch_state_dict(ctx["run_dir"], ctx["task_name"], model_name))
    model.eval()

    _, preds, probs = predict_lstm(model, loader, ctx["device"])

    keys = []
    for _, grp in df.groupby("dossier"):
        grp = grp.sort_values("page_num")
        keys.extend(zip(grp["dossier"], grp["page_num"]))
    return keys, preds, probs


def predict_fusion_early(ctx, model_name, config, df):
    from datasets import EarlyFusionDataset
    from models import EarlyFusionMLP
    from torch.utils.data import DataLoader
    from train_loops import predict_fusion_mlp

    X_img = load_backbone_features(ctx["cache_dir"], config["image_backbone"], df)
    X_text = load_backbone_features(ctx["cache_dir"], config["text_backbone"], df)
    y_dummy = np.zeros(len(df), dtype=int)  # EarlyFusionDataset requires a labels array; unused for inference
    loader = DataLoader(EarlyFusionDataset(X_img, X_text, y_dummy), batch_size=ctx["batch_size"], shuffle=False)

    model = EarlyFusionMLP(
        img_dim=config["img_dim"], text_dim=config["text_dim"], hidden=config["hidden"],
        num_classes=config["num_classes"], dropout=config["dropout"],
    ).to(ctx["device"])
    model.load_state_dict(load_torch_state_dict(ctx["run_dir"], ctx["task_name"], model_name))
    model.eval()

    preds, probs = predict_fusion_mlp(model, loader, ctx["device"])
    return _keys(df), preds, probs


def predict_late_fusion(ctx, model_name, config, df):
    """Averages two independently fine-tuned models' softmax outputs -
    config["vision_model"]/config["text_model"] name the two constituent
    finetune_image/finetune_text checkpoints (under this same
    run_dir/task_name), each reloaded and run through its own predictor,
    then combined by (dossier, page_num) before averaging.

    Generalizes evaluate_models.py's compute_late_fusion (which only ever
    averaged two --task test-split predictions already sitting on disk) to
    an arbitrary df - in particular evaluate_pipeline.py's predicted-head-
    pages subset, which isn't necessarily --task's own test split (a page
    a start_page model false-positives on a doc_type model's own test
    split never scored). That's the actual reason late-fusion couldn't be
    a pipeline candidate before this existed: not merely "no checkpoint",
    but no way to get either constituent's prediction on a page outside
    its own pre-computed test set."""
    vision_config = load_config(ctx["run_dir"], ctx["task_name"], config["vision_model"])
    text_config = load_config(ctx["run_dir"], ctx["task_name"], config["text_model"])

    vis_keys, _, vis_probs = predict_finetune_image(ctx, config["vision_model"], vision_config, df)
    txt_keys, _, txt_probs = predict_finetune_text(ctx, config["text_model"], text_config, df)
    if vis_keys != txt_keys:
        raise SystemExit(
            f"late-fusion {model_name!r}: {config['vision_model']!r} and {config['text_model']!r} "
            f"returned predictions in different row order - shouldn't happen, both are called with the same df."
        )

    probs = (vis_probs + txt_probs) / 2.0
    preds = probs.argmax(axis=1)
    return vis_keys, preds, probs


# Families that need a torch device at all - "baseline" (sklearn/xgboost)
# deliberately isn't here, see predict_baseline's docstring.
TORCH_FAMILIES = {"finetune_image", "finetune_text", "sequence_lstm", "fusion_early", "late_fusion"}

FAMILY_PREDICTORS = {
    "baseline": predict_baseline,
    "finetune_image": predict_finetune_image,
    "finetune_text": predict_finetune_text,
    "sequence_lstm": predict_sequence_lstm,
    "fusion_early": predict_fusion_early,
    "late_fusion": predict_late_fusion,
}


def predict_with_checkpoint(ctx, model_name: str, config: dict, df: pd.DataFrame):
    """Dispatches on config["model_family"]. Returns (keys, preds, probs) -
    see module docstring."""
    family = config["model_family"]
    fn = FAMILY_PREDICTORS.get(family)
    if fn is None:
        raise ValueError(f"unknown model_family {family!r} for {model_name!r}")
    return fn(ctx, model_name, config, df)
