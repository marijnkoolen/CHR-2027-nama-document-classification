# Model Evaluation Report

Every trained checkpoint — baseline, sequence, fusion and finetune regimes, six backbones, two tasks — scored on the held-out test split, plus the best start_page → document_type pipeline combinations.

**56** start_page models &nbsp;·&nbsp; **56** doc_type models &nbsp;·&nbsp; **338** test pages &nbsp;·&nbsp; **1,717 / 367** train / val pages

## Per-task leaderboards

Ranked by macro F1 · filter by training regime

### start_page

| # | Model | Regime | Macro F1 |
|---|---|---|---|
| 1 | KNN (DINOv2-S + mpnet-multilingual) | baseline | 0.953 |
| 2 | LSTM (mpnet-multilingual) | sequence | 0.950 |
| 3 | LSTM (DINOv2-S) | sequence | 0.950 |
| 4 | KNN (EfficientNet-B0 + mpnet-multilingual) | baseline | 0.949 |
| 5 | LSTM (VGG16) | sequence | 0.945 |
| 6 | LSTM (EfficientNet-B0) | sequence | 0.940 |
| 7 | Late fusion (DINOv2-S + BERT-base) | fusion | 0.938 |
| 8 | Late fusion (DINOv2-S + mpnet-multilingual) | fusion | 0.938 |
| 9 | KNN (DINOv2-S) | baseline | 0.936 |
| 10 | KNN (EfficientNet-B0) | baseline | 0.935 |
| 11 | KNN (DINOv2-S + BERT-base) | baseline | 0.935 |
| 12 | KNN (EfficientNet-B0 + BERT-base) | baseline | 0.934 |
| 13 | Fine-tune (DINOv2-S) | finetune | 0.931 |
| 14 | Early fusion (DINOv2-S + mpnet-multilingual) | fusion | 0.928 |
| 15 | Early fusion (EfficientNet-B0 + mpnet-multilingual) | fusion | 0.924 |
| 16 | Early fusion (DINOv2-S + BERT-base) | fusion | 0.921 |
| 17 | Late fusion (VGG16 + BERT-base) | fusion | 0.919 |
| 18 | Late fusion (VGG16 + mpnet-multilingual) | fusion | 0.915 |
| 19 | XGBoost (DINOv2-S) | baseline | 0.914 |
| 20 | Fine-tune (EfficientNet-B0) | finetune | 0.913 |
| 21 | Late fusion (EfficientNet-B0 + BERT-base) | fusion | 0.912 |
| 22 | Late fusion (DiT-Large + BERT-base) | fusion | 0.912 |
| 23 | BERT fine-tune (BERT-base) | finetune | 0.912 |
| 24 | Fine-tune (VGG16) | finetune | 0.910 |
| 25 | Early fusion (EfficientNet-B0 + BERT-base) | fusion | 0.910 |
| 26 | Late fusion (EfficientNet-B0 + mpnet-multilingual) | fusion | 0.908 |
| 27 | KNN (BERT-base) | baseline | 0.908 |
| 28 | Early fusion (VGG16 + mpnet-multilingual) | fusion | 0.907 |
| 29 | KNN (VGG16 + mpnet-multilingual) | baseline | 0.903 |
| 30 | Early fusion (DiT-Large + mpnet-multilingual) | fusion | 0.903 |
| 31 | XGBoost (DINOv2-S + BERT-base) | baseline | 0.901 |
| 32 | XGBoost (DiT-Large + mpnet-multilingual) | baseline | 0.900 |
| 33 | KNN (VGG16 + BERT-base) | baseline | 0.900 |
| 34 | KNN (DiT-Large + mpnet-multilingual) | baseline | 0.899 |
| 35 | Early fusion (VGG16 + BERT-base) | fusion | 0.898 |
| 36 | XGBoost (EfficientNet-B0 + mpnet-multilingual) | baseline | 0.898 |
| 37 | XGBoost (EfficientNet-B0 + BERT-base) | baseline | 0.898 |
| 38 | XGBoost (DINOv2-S + mpnet-multilingual) | baseline | 0.897 |
| 39 | BERT fine-tune (mpnet-multilingual) | finetune | 0.897 |
| 40 | LSTM (BERT-base) | sequence | 0.894 |
| 41 | Late fusion (DiT-Large + mpnet-multilingual) | fusion | 0.893 |
| 42 | LSTM (DiT-Large) | sequence | 0.892 |
| 43 | Early fusion (DiT-Large + BERT-base) | fusion | 0.890 |
| 44 | KNN (VGG16) | baseline | 0.889 |
| 45 | XGBoost (VGG16 + mpnet-multilingual) | baseline | 0.889 |
| 46 | XGBoost (EfficientNet-B0) | baseline | 0.888 |
| 47 | XGBoost (mpnet-multilingual) | baseline | 0.878 |
| 48 | XGBoost (VGG16 + BERT-base) | baseline | 0.878 |
| 49 | KNN (mpnet-multilingual) | baseline | 0.876 |
| 50 | XGBoost (DiT-Large + BERT-base) | baseline | 0.872 |
| 51 | KNN (DiT-Large + BERT-base) | baseline | 0.871 |
| 52 | XGBoost (DiT-Large) | baseline | 0.870 |
| 53 | XGBoost (VGG16) | baseline | 0.865 |
| 54 | XGBoost (BERT-base) | baseline | 0.855 |
| 55 | KNN (DiT-Large) | baseline | 0.786 |
| 56 | Fine-tune (DiT-Large) | finetune | 0.677 |

### doc_type

| # | Model | Regime | Macro F1 |
|---|---|---|---|
| 1 | Early fusion (DiT-Large + mpnet-multilingual) | fusion | 0.730 |
| 2 | Late fusion (EfficientNet-B0 + BERT-base) | fusion | 0.730 |
| 3 | Late fusion (DiT-Large + BERT-base) | fusion | 0.715 |
| 4 | BERT fine-tune (BERT-base) | finetune | 0.714 |
| 5 | Early fusion (EfficientNet-B0 + BERT-base) | fusion | 0.704 |
| 6 | Early fusion (VGG16 + mpnet-multilingual) | fusion | 0.682 |
| 7 | Late fusion (VGG16 + BERT-base) | fusion | 0.678 |
| 8 | Late fusion (DINOv2-S + BERT-base) | fusion | 0.673 |
| 9 | Early fusion (DINOv2-S + mpnet-multilingual) | fusion | 0.665 |
| 10 | BERT fine-tune (mpnet-multilingual) | finetune | 0.662 |
| 11 | Early fusion (DiT-Large + BERT-base) | fusion | 0.661 |
| 12 | Early fusion (EfficientNet-B0 + mpnet-multilingual) | fusion | 0.655 |
| 13 | Late fusion (DiT-Large + mpnet-multilingual) | fusion | 0.654 |
| 14 | XGBoost (DINOv2-S + mpnet-multilingual) | baseline | 0.652 |
| 15 | XGBoost (mpnet-multilingual) | baseline | 0.650 |
| 16 | XGBoost (EfficientNet-B0 + mpnet-multilingual) | baseline | 0.647 |
| 17 | Early fusion (VGG16 + BERT-base) | fusion | 0.642 |
| 18 | Late fusion (EfficientNet-B0 + mpnet-multilingual) | fusion | 0.637 |
| 19 | XGBoost (DiT-Large + BERT-base) | baseline | 0.636 |
| 20 | XGBoost (EfficientNet-B0 + BERT-base) | baseline | 0.635 |
| 21 | Late fusion (VGG16 + mpnet-multilingual) | fusion | 0.633 |
| 22 | XGBoost (DiT-Large + mpnet-multilingual) | baseline | 0.627 |
| 23 | XGBoost (VGG16 + mpnet-multilingual) | baseline | 0.627 |
| 24 | Early fusion (DINOv2-S + BERT-base) | fusion | 0.622 |
| 25 | XGBoost (DINOv2-S + BERT-base) | baseline | 0.616 |
| 26 | XGBoost (VGG16 + BERT-base) | baseline | 0.613 |
| 27 | XGBoost (BERT-base) | baseline | 0.612 |
| 28 | KNN (BERT-base) | baseline | 0.603 |
| 29 | KNN (mpnet-multilingual) | baseline | 0.599 |
| 30 | Late fusion (DINOv2-S + mpnet-multilingual) | fusion | 0.598 |
| 31 | KNN (EfficientNet-B0 + mpnet-multilingual) | baseline | 0.596 |
| 32 | KNN (VGG16 + mpnet-multilingual) | baseline | 0.593 |
| 33 | KNN (DINOv2-S + mpnet-multilingual) | baseline | 0.586 |
| 34 | Fine-tune (DINOv2-S) | finetune | 0.553 |
| 35 | Fine-tune (VGG16) | finetune | 0.552 |
| 36 | LSTM (mpnet-multilingual) | sequence | 0.545 |
| 37 | Fine-tune (EfficientNet-B0) | finetune | 0.532 |
| 38 | KNN (VGG16 + BERT-base) | baseline | 0.529 |
| 39 | KNN (DiT-Large + mpnet-multilingual) | baseline | 0.527 |
| 40 | KNN (EfficientNet-B0 + BERT-base) | baseline | 0.524 |
| 41 | LSTM (DINOv2-S) | sequence | 0.514 |
| 42 | LSTM (EfficientNet-B0) | sequence | 0.491 |
| 43 | XGBoost (DiT-Large) | baseline | 0.490 |
| 44 | XGBoost (DINOv2-S) | baseline | 0.482 |
| 45 | KNN (VGG16) | baseline | 0.482 |
| 46 | LSTM (BERT-base) | sequence | 0.482 |
| 47 | KNN (DINOv2-S + BERT-base) | baseline | 0.477 |
| 48 | XGBoost (EfficientNet-B0) | baseline | 0.472 |
| 49 | KNN (DiT-Large + BERT-base) | baseline | 0.468 |
| 50 | KNN (EfficientNet-B0) | baseline | 0.466 |
| 51 | XGBoost (VGG16) | baseline | 0.453 |
| 52 | LSTM (DiT-Large) | sequence | 0.412 |
| 53 | KNN (DINOv2-S) | baseline | 0.412 |
| 54 | LSTM (VGG16) | sequence | 0.392 |
| 55 | KNN (DiT-Large) | baseline | 0.341 |
| 56 | Fine-tune (DiT-Large) | finetune | 0.231 |

## Best pipeline combinations

Top 3 start_page × 5 doc_type models (3 standalone leaders + 2 late-fusion candidates competitive enough standalone to be worth checking end-to-end), chained and scored end-to-end · ranked by LAS

| # | Pipeline (start_page → doc_type) | start Macro F1 | doc Macro F1 (e2e) | PQ | UAS | LAS |
|---|---|---|---|---|---|---|
| 1 | KNN (DINOv2-S + mpnet-multilingual) → Late fusion (EfficientNet-B0 + BERT-base) **(best)** | 0.953 | 0.796 | 0.947 | 0.959 | 0.953 |
| 2 | KNN (DINOv2-S + mpnet-multilingual) → BERT fine-tune (BERT-base) | 0.953 | 0.785 | 0.947 | 0.959 | 0.950 |
| 3 | KNN (DINOv2-S + mpnet-multilingual) → Late fusion (DiT-Large + BERT-base) | 0.953 | 0.785 | 0.947 | 0.959 | 0.950 |
| 4 | LSTM (DINOv2-S) → Late fusion (EfficientNet-B0 + BERT-base) | 0.950 | 0.748 | 0.949 | 0.953 | 0.941 |
| 5 | LSTM (DINOv2-S) → BERT fine-tune (BERT-base) | 0.950 | 0.737 | 0.949 | 0.953 | 0.938 |
| 6 | LSTM (DINOv2-S) → Late fusion (DiT-Large + BERT-base) | 0.950 | 0.737 | 0.949 | 0.953 | 0.938 |
| 7 | LSTM (mpnet-multilingual) → Late fusion (EfficientNet-B0 + BERT-base) | 0.950 | 0.757 | 0.934 | 0.941 | 0.935 |
| 8 | LSTM (mpnet-multilingual) → BERT fine-tune (BERT-base) | 0.950 | 0.746 | 0.934 | 0.941 | 0.932 |
| 9 | LSTM (mpnet-multilingual) → Late fusion (DiT-Large + BERT-base) | 0.950 | 0.746 | 0.934 | 0.941 | 0.932 |
| 10 | KNN (DINOv2-S + mpnet-multilingual) → Early fusion (DiT-Large + mpnet-multilingual) | 0.953 | 0.743 | 0.947 | 0.959 | 0.929 |
| 11 | KNN (DINOv2-S + mpnet-multilingual) → Early fusion (EfficientNet-B0 + BERT-base) | 0.953 | 0.759 | 0.947 | 0.959 | 0.923 |
| 12 | LSTM (DINOv2-S) → Early fusion (DiT-Large + mpnet-multilingual) | 0.950 | 0.691 | 0.949 | 0.953 | 0.917 |
| 13 | LSTM (DINOv2-S) → Early fusion (EfficientNet-B0 + BERT-base) | 0.950 | 0.723 | 0.949 | 0.953 | 0.914 |
| 14 | LSTM (mpnet-multilingual) → Early fusion (DiT-Large + mpnet-multilingual) | 0.950 | 0.704 | 0.934 | 0.941 | 0.911 |
| 15 | LSTM (mpnet-multilingual) → Early fusion (EfficientNet-B0 + BERT-base) | 0.950 | 0.720 | 0.934 | 0.941 | 0.905 |

## Methodology & notes

- Every score on this page is macro-averaged and end-to-end — start F1 is the mean of both classes' F1 (start + not-start), matching the standalone start_page leaderboard's own macro F1; doc macro F1 is scored over every true start page in the test set, matching the standalone doc_type leaderboard's own denominator — a page whose start wasn't detected counts as wrong, not as excluded. Positive-class-only F1 (start_page's binary "start"-class-only score) and document_type accuracy/F1 conditioned on the start_page model having actually detected the page are deliberately not reported anywhere in this report: both silently score an easier, differently-scoped question than the real inference task, and have been mistaken downstream for the honest number — including for document-type count corrections across the full corpus, which is not a valid use of either.
- PQ (panoptic quality), UAS (unlabeled attachment score) and LAS (labeled attachment score) treat each dossier as a sequence of document segments: PQ scores how well predicted segment boundaries match ground truth, UAS whether pages are grouped into the right segments regardless of label, LAS whether they're grouped and correctly typed.
