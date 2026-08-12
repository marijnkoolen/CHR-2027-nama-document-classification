# Dossier-size / family-composition Bayesian analysis pipeline.
#
# Run from anywhere with `make -C scripts/dossier_size_model <target>`, or
# `cd scripts/dossier_size_model && make <target>`. Recipes cd to the repo
# root before running anything, since every script's DATA_PATH/OUT_DIR are
# relative to the repo root (data/ages_num_docs_pages.tsv, data/dossier_size_model).
#
# Each `_plot` target depends on its model target, so `make X_plot` reruns
# the model first. `make all` reruns the entire pipeline end to end --
# expect on the order of 1.5-2 hours; the two doc-type hierarchical models
# alone (doc_types_three_groups, doc_types_count_vs_presence) are ~30 min
# each. Prefer running individual targets unless you actually need everything.
#
# model_dossier_size / dossier_size_plot are SUPERSEDED (see their docstrings)
# by three_groups / three_groups_plot -- kept for reproducibility of the
# original (since-corrected) result, not as the current account of the data.

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(MAKEFILE_DIR)/../..)
PY := cd $(ROOT_DIR) && python3

.PHONY: all help \
	dossier_size dossier_size_plot \
	doc_types doc_types_plot \
	dossier_size_temporal dossier_size_temporal_plot \
	family_size_temporal family_size_temporal_plot \
	adults_binomial adults_binomial_plot \
	three_groups three_groups_plot \
	doc_types_three_groups doc_types_three_groups_plot \
	doc_types_count_vs_presence \
	doc_type_period_diff report \
	doc_type_uncertainty_point doc_type_uncertainty_pool doc_type_period_diff_uncertainty \
	aggregate_uncertainty

help:
	@echo "Targets (model / model_plot pairs unless noted):"
	@echo "  dossier_size, dossier_size_plot                  ~1 min   (SUPERSEDED, see docstring)"
	@echo "  doc_types, doc_types_plot                        ~14 min"
	@echo "  dossier_size_temporal, dossier_size_temporal_plot ~2 min"
	@echo "  family_size_temporal, family_size_temporal_plot   ~8 min"
	@echo "  adults_binomial, adults_binomial_plot             ~3 min"
	@echo "  three_groups, three_groups_plot                   ~1 min   (corrected minor/pre-adult/adult model)"
	@echo "  doc_types_three_groups, ..._plot                  ~25 min  (full + pre-1956 + post-1956 fits)"
	@echo "  doc_types_count_vs_presence                       ~30 min  (no plot script)"
	@echo "  doc_type_period_diff                              ~5s      pre/post-1956 diff per type, depends on doc_types_three_groups"
	@echo "  report                                            styled HTML report, depends on everything above"
	@echo "  all                                               ~1.5-2 hours, everything above in order, ending with report"
	@echo ""
	@echo "  doc_type_uncertainty_point         ~25 min, CPU  bias-corrected point estimate (both doc-type analyses)"
	@echo "  doc_type_period_diff_uncertainty   ~5s           point-corrected pre/post-1956 diff, depends on doc_type_uncertainty_point"
	@echo "  doc_type_uncertainty_pool          fast          pool GPU-box multiple-imputation results after copying them back"
	@echo "  aggregate_uncertainty              ~2 min, CPU   segmentation-only correction for the AGGREGATE num_docs models"
	@echo "  Full multiple-imputation run (doc-type analyses) is GPU-only and sharded across 2 processes -- not a single"
	@echo "  make target, see doc_type_three_groups_uncertainty.py / doc_type_count_vs_presence_uncertainty.py docstrings"
	@echo "  for the exact commands to run on the GPU box. aggregate_uncertainty has no mi/GPU mode (point only)."
	@echo ""
	@echo "  report now depends on doc_type_uncertainty_point, doc_type_period_diff_uncertainty, and aggregate_uncertainty"
	@echo "  too (build_report.py's sect;3/6/7 read their output) -- 'make all' picks these up automatically."

all: report

dossier_size:
	$(PY) scripts/dossier_size_model/model_dossier_size.py

dossier_size_plot: dossier_size
	$(PY) scripts/dossier_size_model/plot_dossier_size_results.py

doc_types:
	$(PY) scripts/dossier_size_model/model_doc_types.py

doc_types_plot: doc_types
	$(PY) scripts/dossier_size_model/plot_doc_type_effects.py

dossier_size_temporal:
	$(PY) scripts/dossier_size_model/model_dossier_size_temporal.py

dossier_size_temporal_plot: dossier_size_temporal
	$(PY) scripts/dossier_size_model/plot_dossier_size_temporal.py

family_size_temporal:
	$(PY) scripts/dossier_size_model/model_family_size_temporal.py

family_size_temporal_plot: family_size_temporal
	$(PY) scripts/dossier_size_model/plot_family_size_temporal.py

adults_binomial:
	$(PY) scripts/dossier_size_model/model_num_adults_binomial.py

adults_binomial_plot: adults_binomial
	$(PY) scripts/dossier_size_model/plot_adults_binomial.py

three_groups:
	$(PY) scripts/dossier_size_model/model_dossier_size_three_groups.py

three_groups_plot: three_groups
	$(PY) scripts/dossier_size_model/plot_three_group_effects.py

doc_types_three_groups:
	$(PY) scripts/dossier_size_model/model_doc_types_three_groups.py

doc_types_three_groups_plot: doc_types_three_groups
	$(PY) scripts/dossier_size_model/plot_doc_types_three_groups.py

doc_types_count_vs_presence:
	$(PY) scripts/dossier_size_model/model_doc_types_count_vs_presence.py

doc_type_period_diff: doc_types_three_groups
	$(PY) scripts/dossier_size_model/doc_type_period_diff.py

report: dossier_size_plot doc_types_plot dossier_size_temporal_plot family_size_temporal_plot \
	adults_binomial_plot three_groups_plot doc_types_three_groups_plot doc_types_count_vs_presence \
	doc_type_period_diff doc_type_uncertainty_point doc_type_period_diff_uncertainty aggregate_uncertainty
	$(PY) scripts/dossier_size_model/build_report.py

# Classifier-uncertainty correction for the two doc-type analyses (see
# doc_type_uncertainty_common.py's docstring). Requires
# data/dossier_composition/idata_confusion_matrix.nc to already exist --
# run `make -C scripts/dossier_composition confusion_matrix` first if not.
doc_type_uncertainty_point:
	$(PY) scripts/dossier_size_model/doc_type_three_groups_uncertainty.py --mode point
	$(PY) scripts/dossier_size_model/doc_type_count_vs_presence_uncertainty.py --mode point

doc_type_period_diff_uncertainty: doc_type_uncertainty_point
	$(PY) scripts/dossier_size_model/doc_type_period_diff_uncertainty.py

doc_type_uncertainty_pool:
	$(PY) scripts/dossier_size_model/doc_type_three_groups_uncertainty.py --mode pool --n-imputations $(N_IMPUTATIONS)
	$(PY) scripts/dossier_size_model/doc_type_count_vs_presence_uncertainty.py --mode pool --n-imputations $(N_IMPUTATIONS)

# Segmentation-only classifier-uncertainty correction for the AGGREGATE
# num_docs models (three_groups, dossier_size_temporal) -- see
# aggregate_uncertainty.py's docstring. Requires
# data/dossier_composition/idata_start_page_confusion_matrix.nc to already
# exist -- run `make -C scripts/dossier_composition start_page_confusion_matrix`
# first if not. Point mode only (~2 min, CPU) -- no GPU/mi mode implemented.
aggregate_uncertainty:
	$(PY) scripts/dossier_size_model/aggregate_uncertainty.py

N_IMPUTATIONS ?= 20
