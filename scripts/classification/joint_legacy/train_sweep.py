"""
Trains every config in sweep_configs.py (train.py --mode sequence
--cached-embeddings), writing each to <out-dir>/<config-name>/.

Deliberately a separate script/step from evaluate_sweep.py, not bundled
into one: re-running the evaluation (e.g. after a metric-code fix, or just
to regenerate the results table) never needs to retrain anything, and
retraining after new embeddings/labels doesn't require re-deriving every
metric by hand. See the Makefile for wiring both together.

Usage:
    python scripts/classification/train_sweep.py --out-dir runs/from_embeddings

    # only a subset, e.g. while iterating on one config:
    python scripts/classification/train_sweep.py --out-dir runs/from_embeddings \\
        --configs vision_efficient text_bert_base_uncased

    # MPS has a known, previously-confirmed inf/nan-gradient bug in this
    # project (sequence-mode training can silently collapse to loss=nan
    # from epoch 1) - pass --device cpu explicitly if you hit it; a clean,
    # non-nan loss curve in the first few epochs is enough to confirm the
    # device is safe before committing to a full sweep:
    python scripts/classification/train_sweep.py --out-dir runs/from_embeddings --device cpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sweep_configs import SWEEP_CONFIGS


def select_configs(names: list[str] | None):
    if not names:
        return SWEEP_CONFIGS
    wanted = set(names)
    configs = [c for c in SWEEP_CONFIGS if c.name in wanted]
    missing = wanted - {c.name for c in configs}
    if missing:
        valid = ", ".join(c.name for c in SWEEP_CONFIGS)
        raise SystemExit(f"unknown --configs name(s): {sorted(missing)}\nvalid names: {valid}")
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/from_embeddings"))
    parser.add_argument("--configs", nargs="+", default=None,
                         help="only train these configs by name (default: all - see sweep_configs.py)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--project-to", type=int, default=384)
    parser.add_argument("--loss-weight-start", type=float, default=1.0)
    parser.add_argument("--loss-weight-doctype", type=float, default=1.0)
    parser.add_argument("--loss-weight-layout", type=float, default=1.0)
    parser.add_argument("--loss-weight-functional", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="default: let train.py auto-pick - see module docstring "
                                                         "re: MPS's known nan-gradient bug")
    args = parser.parse_args()

    configs = select_configs(args.configs)
    train_py = Path(__file__).parent / "train.py"

    for i, config in enumerate(configs):
        out_dir = args.out_dir / config.name
        print(f"\n{'=' * 60}\n[{i + 1}/{len(configs)}] TRAIN: {config.name} -> {out_dir}\n{'=' * 60}")
        cmd = [
            sys.executable, str(train_py), "--mode", "sequence", "--out-dir", str(out_dir),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--project-to", str(args.project_to),
            "--loss-weight-start", str(args.loss_weight_start), "--loss-weight-doctype", str(args.loss_weight_doctype),
            "--loss-weight-layout", str(args.loss_weight_layout), "--loss-weight-functional", str(args.loss_weight_functional),
            *config.train_args(),
        ]
        if args.device:
            cmd += ["--device", args.device]
        subprocess.run(cmd, check=True)

    print(f"\nAll {len(configs)} config(s) trained under {args.out_dir}")


if __name__ == "__main__":
    main()
