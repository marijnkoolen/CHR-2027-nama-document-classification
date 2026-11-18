"""Model-checkpoint naming, shared between each train_*.py script (which
computes its own model_name to save its checkpoint under) and train_all.py
(which needs the exact same names to check whether a job's checkpoint
already exists, for idempotent resume of a big batch run).

No torch/xgboost imports here on purpose - trivial string formatting only,
so it's always safe to import regardless of which heavy libraries the
importing process has or hasn't loaded yet (see lib/common.py's docstring
for why that ordering matters in this project).
"""

from __future__ import annotations


def slug(name: str) -> str:
    return name.replace("/", "__")


def baseline_model_name(model: str, features: list[str]) -> str:
    return f"{model}-{'+'.join(slug(f) for f in features)}"


def sequence_model_name(features_backbone: str) -> str:
    return f"lstm-{slug(features_backbone)}"


def fusion_model_name(image_backbone: str, text_backbone: str) -> str:
    return f"early-fusion-{slug(image_backbone)}+{slug(text_backbone)}"


def late_fusion_model_name(image_backbone: str, text_backbone: str) -> str:
    """late-fusion has no checkpoint of its own (see evaluate_models.py's
    compute_late_fusion) - this names its derived result directory, using
    the same image+text backbone naming as fusion_model_name so the two
    regimes read consistently side by side."""
    return f"late-fusion-{slug(image_backbone)}+{slug(text_backbone)}"


FINETUNE_IMAGE_MODEL_NAMES = {
    "vgg16": "vgg16-ft",
    "efficientnet": "efficientnet-ft",
    "efficientnet_b0": "efficientnet-ft",
}
FINETUNE_TEXT_BASE_NAMES = {
    "textcnn": "textcnn",
    "bert": "bert-ft",
}


def finetune_model_name(backbone: str, bert_model: str | None = None) -> str:
    """--backbone textcnn/bert are always suffixed with --bert-model (even
    when it's the default bert-base-uncased) - otherwise two runs that only
    differ in --bert-model would compute the identical model_name and
    silently overwrite each other's checkpoint."""
    if backbone in FINETUNE_TEXT_BASE_NAMES:
        return f"{FINETUNE_TEXT_BASE_NAMES[backbone]}-{slug(bert_model)}"
    return FINETUNE_IMAGE_MODEL_NAMES.get(backbone, f"{slug(backbone)}-ft")
