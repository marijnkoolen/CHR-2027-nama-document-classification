"""torch Dataset classes shared by the fine-tuning/sequence/fusion scripts.

PageImageDataset/TextDataset/DossierSequenceDataset all default to a
zero-filled "label" column when df doesn't have one: train_*.py always
passes labeled data, but lib/predict.py's prediction-only callers (used by
evaluate_pipeline.py to run a model on an arbitrary page subset it built
itself, e.g. another model's predicted start pages) generally don't have
real labels for whatever subset they're predicting on - the label is only
ever used to build a (T,) tensor alongside the real inputs (and, for
DossierSequenceDataset, to mask padded sequence positions at predict time),
never read back out of these datasets by anything a prediction-only caller
would run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from features import safe_read_text


def _labels_or_zeros(df: pd.DataFrame) -> list:
    if "label" in df.columns:
        return df["label"].tolist()
    return [0] * len(df)


class PageImageDataset(Dataset):
    """Raw page images, for end-to-end fine-tuning (VGG16-FT, EfficientNet-FT)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["img_path"].tolist()
        self.labels = _labels_or_zeros(df)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as img:
            tensor = self.transform(img)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


class TextDataset(Dataset):
    """Tokenized page text, for TextCNN and BERT fine-tuning."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = [safe_read_text(p) for p in df["text_path"].tolist()]
        self.labels = _labels_or_zeros(df)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class DossierSequenceDataset(Dataset):
    """Groups pages by dossier (sorted by page_num) into (T, D) feature
    sequences + (T,) label sequences, for the LSTM+VGG16 sequence model."""

    def __init__(self, df: pd.DataFrame, features: np.ndarray):
        if "label" not in df.columns:
            df = df.assign(label=0)
        self.sequences = []
        for _, grp in df.groupby("dossier"):
            grp = grp.sort_values("page_num")
            feats = features[grp.index]
            labels = grp["label"].values
            self.sequences.append((
                torch.tensor(feats, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]


def sequence_pad_collate(batch):
    """Pads a batch of (feats, labels) sequences to the same length; padded
    label positions get -1 so the training loop's CrossEntropyLoss(ignore_index=-1)
    excludes them."""
    feats_list, labels_list = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in feats_list])
    feats_padded = nn.utils.rnn.pad_sequence(feats_list, batch_first=True)
    labels_padded = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-1)
    return feats_padded, labels_padded, lengths


class EarlyFusionDataset(Dataset):
    """Precomputed image-backbone + BERT [CLS] feature vectors, for the
    early-fusion MLP. Both notebooks eventually only need frozen features
    here (no gradient ever reaches either backbone) - this uses the
    page_start_classifier_qwen.ipynb approach (precomputed BERT features)
    rather than doc_type_start_page_classifier_qwen.ipynb's (re-tokenizing
    and re-running a frozen BERT forward pass every batch): same result,
    without the repeated forward pass."""

    def __init__(self, img_feats: np.ndarray, text_feats: np.ndarray, labels: np.ndarray):
        self.img = torch.tensor(img_feats, dtype=torch.float32)
        self.text = torch.tensor(text_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.img[idx], self.text[idx], self.labels[idx]
