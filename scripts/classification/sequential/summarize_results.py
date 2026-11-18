"""Standalone regeneration of the combined comparison table + bar chart
from whatever metrics.json files already exist under <run-dir>/<task>/ -
the same report evaluate_models.py writes at the end of a full run. Useful
if you've hand-edited/removed a model's results and just want the report
rebuilt, without re-running evaluation.

Usage:
    python scripts/classification/sequential/summarize_results.py \\
        --task start_page --run-dir runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from summarize import build_and_write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["start_page", "doc_type"], required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <run-dir>/<task>")
    args = parser.parse_args()

    build_and_write_summary(args.run_dir, args.task, args.out_dir)


if __name__ == "__main__":
    main()
