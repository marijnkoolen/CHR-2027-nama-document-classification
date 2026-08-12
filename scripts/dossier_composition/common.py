"""
Shared constants and the predicted-type -> tracked-type mapping used across
the dossier-composition analysis (occurrence, co-occurrence, dispersion,
order). Same 11 tracked types as scripts/dossier_size_model; everything else
buckets into "Other".

"Testimonial medical letter" used to be merged into "Testimonial medical
form" (thin/unreliable support at the time). Un-merged per user decision:
it turned out to be prevalent and to have good precision/recall on the test
set once evaluated on its own, so it's now tracked as its own type rather
than folded into medical form.
"""

TRACKED_TYPES = [
    "Approval notice",
    "D.1",
    "D.2",
    "DM.1",
    "Judicial and political background check",
    "NAMA agreement",
    "Registration card",
    "Report of selection and medical officers",
    "Testimonial labour (Qualification & Employment Proof)",
    "Testimonial medical form (Medical & Health Documents)",
    "Testimonial medical letter (Medical & Health Documents)",
]

OTHER = "Other"
ALL_TYPES = TRACKED_TYPES + [OTHER]

# Default input paths -- override via CLI flags (see each script's --help)
# when a new model's predictions/test evaluation become available; nothing
# else in the pipeline needs to change. Also update dossier_composition/
# Makefile's PREDICTIONS/TEST_PREDICTIONS `?=` defaults, which are a SEPARATE
# hardcoded fallback, not read from here (learned the hard way once already).
DEFAULT_PREDICTIONS_PATH = "data/predictions/predictions-top_combo_latefusion-114k-fixed.tsv"
DEFAULT_TEST_PREDICTIONS_PATH = (
    "runs/per_task/pipeline_latefusion_check/"
    "knn-facebook__dinov2-small+sentence-transformers__paraphrase-multilingual-mpnet-base-v2__"
    "late-fusion-efficientnet_b0+bert-base-uncased/"
    "predictions.tsv"
)
OUT_DIR = "data/dossier_composition"


def map_type(raw_type: str) -> str:
    """Map a raw predicted/true document_type string to one of ALL_TYPES."""
    return raw_type if raw_type in TRACKED_TYPES else OTHER
