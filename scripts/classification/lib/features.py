"""Text reading for on-the-fly (uncached) training/prediction -
sequential/train_finetune.py's TextDataset (TextCNN/BERT-FT) reads raw page
text directly rather than from a cached feature vector, since the backbone
itself is being trained/fine-tuned there.

Cached, precomputed features (used by sequential/train_baseline.py,
train_sequence.py, train_fusion.py, and lib/predict.py's baseline/
sequence_lstm/fusion_early families) come from sequential/extract_features.py
+ lib/embeddings.py instead - see extract_features.py's module docstring
for why that extraction delegates to joint/precompute_embeddings.py rather
than living here.

safe_read_text is markdown_text.py's extract_text under the name this
project's sequential/ scripts already call it by - historically a separate
ported copy (before this project's two lib/ directories merged into one),
now a plain re-export of the single implementation both pipelines share.
"""

from __future__ import annotations

from markdown_text import extract_text as safe_read_text  # noqa: F401
