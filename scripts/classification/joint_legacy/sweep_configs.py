"""
Shared config list for train_sweep.py / evaluate_sweep.py - the 12
vision/text/multimodal x efficient/quality sequence-mode configs this
project's results tables are built from. Edit this list (not
train_sweep.py or evaluate_sweep.py) to add, remove, or rename configs -
both scripts read from here, so they can't drift out of sync with each
other the way two independently-maintained lists would.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SweepConfig:
    name: str
    modality: str  # vision | multimodal | text
    scenario: str  # efficient | quality
    cached_embeddings: str  # a precompute_embeddings.py output dir, relative to repo root
    image_backbone: str | None = None
    text_backbone: str | None = None
    n_heads: int | None = None  # None = use --scenario's own preset default
    n_layers: int | None = None

    def train_args(self) -> list[str]:
        args = [
            "--scenario", self.scenario, "--modality", self.modality,
            "--cached-embeddings", self.cached_embeddings,
        ]
        if self.image_backbone:
            args += ["--image-backbone", self.image_backbone]
        if self.text_backbone:
            args += ["--text-backbone", self.text_backbone]
        if self.n_heads is not None:
            args += ["--n-heads", str(self.n_heads)]
        if self.n_layers is not None:
            args += ["--n-layers", str(self.n_layers)]
        return args


SWEEP_CONFIGS: list[SweepConfig] = [
    SweepConfig(
        "vision_efficient", "vision", "efficient",
        "data/embeddings_vision_dino_aug4", image_backbone="facebook/dinov2-small",
    ),
    SweepConfig(
        "vision_quality_8_4", "vision", "quality",
        "data/embeddings_vision_dit_aug4", image_backbone="microsoft/dit-large-finetuned-rvlcdip",
        n_heads=8, n_layers=4,
    ),
    SweepConfig(
        "vision_quality_4_2", "vision", "quality",
        "data/embeddings_vision_dit_aug4", image_backbone="microsoft/dit-large-finetuned-rvlcdip",
        n_heads=4, n_layers=2,
    ),
    SweepConfig(
        "text_bert_base_uncased", "text", "efficient",
        "data/embeddings_text_bert_base_uncased", text_backbone="bert-base-uncased",
    ),
    SweepConfig(
        "text_bert_base_dutch_cased", "text", "efficient",
        "data/embeddings_text_bert_base_dutch_cased", text_backbone="GroNLP/bert-base-dutch-cased",
    ),
    SweepConfig(
        "text_paraphrase_multilingual", "text", "efficient",
        "data/embeddings_text_paraphrase_multilingual",
        text_backbone="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ),
    SweepConfig(
        "text_xlm_roberta_base", "text", "efficient",
        "data/embeddings_text_xlm_roberta_base", text_backbone="xlm-roberta-base",
    ),
    SweepConfig(
        "text_xlm_roberta_large_4_2", "text", "quality",
        "data/embeddings_text_xlm_roberta_large", text_backbone="xlm-roberta-large",
        n_heads=4, n_layers=2,
    ),
    SweepConfig(
        "text_xlm_roberta_large_8_4", "text", "quality",
        "data/embeddings_text_xlm_roberta_large", text_backbone="xlm-roberta-large",
        n_heads=8, n_layers=4,
    ),
    SweepConfig(
        "multimodal_efficient", "multimodal", "efficient",
        "data/embeddings_multimodal_dino_aug4",
        image_backbone="facebook/dinov2-small", text_backbone="bert-base-uncased",
    ),
    SweepConfig(
        "multimodal_quality_8_4", "multimodal", "quality",
        "data/embeddings_multimodal_dit_aug4",
        image_backbone="microsoft/dit-large-finetuned-rvlcdip", text_backbone="bert-base-uncased",
        n_heads=8, n_layers=4,
    ),
    SweepConfig(
        "multimodal_quality_4_2", "multimodal", "quality",
        "data/embeddings_multimodal_dit_aug4",
        image_backbone="microsoft/dit-large-finetuned-rvlcdip", text_backbone="bert-base-uncased",
        n_heads=4, n_layers=2,
    ),
]
