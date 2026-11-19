.PHONY: help extract-features train evaluate-models evaluate-pipeline predict-corpus report all

# Page-classification pipeline: start_page (binary) and document_type
# (multi-class, start pages only). See README.md for the full narrative.
#
# This repo ships no data - point DATA_ROOT at your own corpus, laid out
# the way lib/labels.py expects (see its module docstring): a labels TSV
# with dossier/page_num/document_type/start_page/img_path/text_path
# columns, plus split assignment (a `split` column, or let lib/labels.py
# compute one - see --split-source).
#
# joint_legacy/ is an older pipeline variant, fully superseded by
# sequential/ for start_page/doc_type - kept only because
# sequential/extract_features.py and lib/segmentation_metrics.py still
# genuinely reuse a few of its files. No Makefile targets of its own.

DATA_ROOT ?= data
RUN_DIR ?= runs
DEVICE ?= cpu
SEQ := scripts/classification/sequential

VISION_BACKBONES ?= vgg16 efficientnet_b0 facebook/dinov2-small microsoft/dit-large-finetuned-rvlcdip
TEXT_BACKBONES ?= bert-base-uncased sentence-transformers/paraphrase-multilingual-mpnet-base-v2

help:
	@echo "Pipeline stages, in order:"
	@echo "  extract-features     cache per-backbone embeddings (needed by baseline/sequence/fusion)"
	@echo "  train                train every regime x backbone x task combination (idempotent)"
	@echo "  evaluate-models      score every trained model on its own task's held-out test set"
	@echo "  evaluate-pipeline    chain top start_page/doc_type models, score end-to-end"
	@echo "  predict-corpus       run a chosen pipeline combination over an unlabeled corpus"
	@echo "  report               build eval_report.html from whatever's under RUN_DIR"
	@echo "  all                  extract-features through evaluate-pipeline"
	@echo ""
	@echo "Override DATA_ROOT / RUN_DIR / DEVICE / VISION_BACKBONES / TEXT_BACKBONES as needed."

all: extract-features train evaluate-models evaluate-pipeline

extract-features:
	@for b in $(VISION_BACKBONES); do \
		python $(SEQ)/extract_features.py --data-root $(DATA_ROOT) --modality vision --image-backbone $$b --device $(DEVICE); \
	done
	@for b in $(TEXT_BACKBONES); do \
		python $(SEQ)/extract_features.py --data-root $(DATA_ROOT) --modality text --text-backbone $$b --device $(DEVICE); \
	done

JOBS ?= 1

train:
	python $(SEQ)/train_all.py --data-root $(DATA_ROOT) --run-dir $(RUN_DIR) --device $(DEVICE) --jobs $(JOBS) \
		--vision-backbones $(VISION_BACKBONES) --text-backbones $(TEXT_BACKBONES)

evaluate-models:
	python $(SEQ)/evaluate_models.py --task start_page --data-root $(DATA_ROOT) --run-dir $(RUN_DIR) --device $(DEVICE)
	python $(SEQ)/evaluate_models.py --task doc_type --data-root $(DATA_ROOT) --run-dir $(RUN_DIR) --device $(DEVICE)

TOP_N ?= 3

evaluate-pipeline:
	python $(SEQ)/evaluate_pipeline.py --data-root $(DATA_ROOT) --run-dir $(RUN_DIR) --top-n $(TOP_N) --device $(DEVICE)

# No defaults for these three - there's no "the" corpus/model pair without
# your own data and your own evaluate-models/evaluate-pipeline results to
# pick a winning combination from.
predict-corpus:
	python $(SEQ)/predict.py \
		--manifest $(MANIFEST) --image-root $(IMAGE_ROOT) \
		--run-dir $(RUN_DIR) --cache-dir $(CACHE_DIR) \
		--start-model $(START_MODEL) --doc-model $(DOC_MODEL) --device $(DEVICE) \
		--out $(PREDICT_OUT)

report:
	python scripts/classification/build_report.py --run-dir $(RUN_DIR)
