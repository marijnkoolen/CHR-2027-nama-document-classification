# Dossier-composition analysis pipeline (occurrence, and later co-occurrence
# / dispersion / order), built on classifier predictions + a confusion-matrix
# correction for uneven per-type reliability.
#
# Run from anywhere with `make -C scripts/dossier_composition <target>`, or
# `cd scripts/dossier_composition && make <target>`.
#
# To rerun everything against a NEW model's predictions (e.g. once the
# better-performing model finishes), override PREDICTIONS and
# TEST_PREDICTIONS on the command line -- no script edits needed:
#
#   make -C scripts/dossier_composition all \
#       PREDICTIONS=data/predictions-new-model.tsv \
#       TEST_PREDICTIONS=runs/new_model/test_predictions.tsv

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(MAKEFILE_DIR)/../..)
PY := cd $(ROOT_DIR) && python3

PREDICTIONS ?= data/predictions/predictions-top_combo_latefusion-114k-fixed.tsv
TEST_PREDICTIONS ?= runs/per_task/pipeline_latefusion_check/knn-facebook__dinov2-small+sentence-transformers__paraphrase-multilingual-mpnet-base-v2__late-fusion-efficientnet_b0+bert-base-uncased/predictions.tsv

.PHONY: all help confusion_matrix start_page_confusion_matrix occurrence occurrence_plot co_occurrence co_occurrence_plot dispersion dispersion_plot order order_plot order_selftest report

help:
	@echo "Targets:"
	@echo "  confusion_matrix            fit the document-type Bayesian confusion matrix from the test set"
	@echo "  start_page_confusion_matrix fit the start_page (segmentation) confusion matrix from the test set"
	@echo "  occurrence          occurrence analysis (naive vs. corrected), depends on both confusion matrices"
	@echo "  occurrence_plot     plot occurrence.py's output, depends on occurrence"
	@echo "  co_occurrence       pairwise co-occurrence (naive vs. corrected), depends on both confusion matrices"
	@echo "  co_occurrence_plot  plot co_occurrence.py's output, depends on co_occurrence"
	@echo "  dispersion          contiguity vs. random-placement null (naive vs. corrected), depends on both confusion matrices"
	@echo "  dispersion_plot     plot dispersion.py's output, depends on dispersion"
	@echo "  order_selftest      validate the Bradley-Terry sign convention against synthetic data (no data needed)"
	@echo "  order               position + Bradley-Terry pairwise order (naive vs. corrected), depends on both confusion matrices"
	@echo "  order_plot          plot order.py's output, depends on order"
	@echo "  report              styled HTML report (data/dossier_composition/report.html) with full tables, depends on all four *_plot targets"
	@echo "  all                 both confusion matrices -> occurrence_plot -> co_occurrence_plot -> dispersion_plot -> order_plot -> report"
	@echo
	@echo "'Corrected' in every analysis now means joint segmentation + document-type"
	@echo "correction (see imputation.py's sample_segments_for_draw) -- each imputation"
	@echo "draw resamples start_page per page, rebuilds segment boundaries, then resamples"
	@echo "type on the new segments. Both confusion matrices must be fit from the SAME"
	@echo "model's test set as PREDICTIONS came from."
	@echo
	@echo "Override PREDICTIONS / TEST_PREDICTIONS to point at a new model's output, e.g.:"
	@echo "  make occurrence PREDICTIONS=data/predictions-new-model.tsv TEST_PREDICTIONS=runs/new_model/test_predictions.tsv"

all: report

confusion_matrix:
	$(PY) scripts/dossier_composition/confusion_matrix.py --test-predictions $(TEST_PREDICTIONS)

start_page_confusion_matrix:
	$(PY) scripts/dossier_composition/start_page_confusion_matrix.py --test-predictions $(TEST_PREDICTIONS)

occurrence: confusion_matrix start_page_confusion_matrix
	$(PY) scripts/dossier_composition/occurrence.py --predictions $(PREDICTIONS) --test-predictions $(TEST_PREDICTIONS)

occurrence_plot: occurrence
	$(PY) scripts/dossier_composition/plot_occurrence.py

co_occurrence: confusion_matrix start_page_confusion_matrix
	$(PY) scripts/dossier_composition/co_occurrence.py --predictions $(PREDICTIONS) --test-predictions $(TEST_PREDICTIONS)

co_occurrence_plot: co_occurrence
	$(PY) scripts/dossier_composition/plot_co_occurrence.py

dispersion: confusion_matrix start_page_confusion_matrix
	$(PY) scripts/dossier_composition/dispersion.py --predictions $(PREDICTIONS) --test-predictions $(TEST_PREDICTIONS)

dispersion_plot: dispersion
	$(PY) scripts/dossier_composition/plot_dispersion.py

order_selftest:
	$(PY) scripts/dossier_composition/order.py --self-test

order: confusion_matrix start_page_confusion_matrix order_selftest
	$(PY) scripts/dossier_composition/order.py --predictions $(PREDICTIONS) --test-predictions $(TEST_PREDICTIONS)

order_plot: order
	$(PY) scripts/dossier_composition/plot_order.py

report: occurrence_plot co_occurrence_plot dispersion_plot order_plot
	$(PY) scripts/dossier_composition/build_report.py --test-predictions $(TEST_PREDICTIONS)
