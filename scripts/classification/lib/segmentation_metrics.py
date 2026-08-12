"""Document-segmentation metrics for the sequential pipeline's start_page ->
doc_type evaluation (sequential/evaluate_pipeline.py): Panoptic Quality and
Unlabeled/Labeled Attachment Score, plus the segment-construction helpers
both depend on.

Genuinely re-uses ../joint_legacy/evaluate_segmentation.py and
../joint_legacy/flag_prediction_errors.py (the joint pipeline's own,
actively-maintained implementation) rather than keeping a hand-ported
duplicate in sync by hand - loaded via importlib under controlled module
names instead of a plain `sys.path.insert` + `import`, because a plain
import would collide: joint/ has its own predict.py (a standalone
sequence-mode inference script) and this lib/ has its own, differently-
purposed predict.py (sequential/'s checkpoint-family prediction dispatch,
imported by sequential/evaluate_models.py and evaluate_pipeline.py) - if
joint/ and lib/ were ever both on sys.path in the same process, `from
predict import ...` would silently resolve to whichever one Python
happened to find first. Loading these two specific files by explicit path,
under names nothing else in this project uses, sidesteps that entirely:
evaluate_segmentation.py itself does `from flag_prediction_errors import
segment_ids_from_start_col`, so flag_prediction_errors.py is loaded first
and registered in sys.modules under exactly that bare name, letting
evaluate_segmentation.py's own import resolve via the already-cached
module rather than needing joint/ on sys.path at all.

See ../joint_legacy/evaluate_segmentation.py's module docstring for the citations
behind PQ (van Heusden, Kamps & Marx 2022/2024) and LAS/UAS (Demirtas et
al. 2022) - unchanged here, since this module doesn't re-implement them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_JOINT_DIR = Path(__file__).resolve().parent.parent / "joint_legacy"


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec so internal same-name imports resolve to it
    spec.loader.exec_module(module)
    return module


_load_module("flag_prediction_errors", _JOINT_DIR / "flag_prediction_errors.py")
_evaluate_segmentation = _load_module("evaluate_segmentation", _JOINT_DIR / "evaluate_segmentation.py")

segments_from_start_col = _evaluate_segmentation.segments_from_start_col
segments_from_id_col = _evaluate_segmentation.segments_from_id_col
majority_label_from_segments = _evaluate_segmentation.majority_label_from_segments
head_page_lookup = _evaluate_segmentation.head_page_lookup
match_segments = _evaluate_segmentation.match_segments
panoptic_quality = _evaluate_segmentation.panoptic_quality
attachment_scores = _evaluate_segmentation.attachment_scores
macro_f1_report = _evaluate_segmentation.macro_f1_report
start_page_report = _evaluate_segmentation.start_page_report
