"""Loads every saved model checkpoint for a task, reproduces its test-set
predictions, and writes a per-model report + a combined comparison table -
the single place test-set numbers get computed, kept separate from the
train_*.py scripts so the test set is only ever looked at deliberately,
here, and models can be re-scored (e.g. after a metric change) without
retraining.

Dispatches on each checkpoint's model_config.json ("model_family" field,
written by whichever train_*.py script produced it) to rebuild the exact
architecture, load its weights, and run inference - see lib/checkpoints.py
and lib/models.py.

late-fusion-<vision>+<text> is a special case: it isn't a trained model
(there's nothing train_fusion.py could save for it - see its docstring),
so instead of loading a checkpoint this derives one result per
(vision-finetune, text-finetune) pair from their already-evaluated
test-set probabilities (either just written by this same run, or left
over from an earlier one) and evaluates each average like any other
model's predictions - see compute_late_fusion below.

Each requested model is evaluated in its own subprocess (see main()/
evaluate_one()/--_worker below), for two reasons: loading several
from_pretrained() BERT checkpoints back-to-back in one long-lived process
was observed to segfault non-deterministically on macOS, and separately,
torch and xgboost coexisting in the *same* process at all (regardless of
import order) was observed to segfault too, once torch does anything
beyond trivial work - both are native-library interaction bugs outside
this script's control. Per-model subprocesses sidestep both: a worker
evaluating a KNN/XGBoost baseline never imports torch (see lib/common.py,
lib/checkpoints.py and lib/features.py, which all import torch lazily), and
a worker evaluating a torch model only ever loads one checkpoint.

Usage:
    # evaluate every saved model for a task
    python scripts/classification/sequential/evaluate_models.py \\
        --task start_page --data-root data --run-dir runs

    # evaluate a subset
    python scripts/classification/sequential/evaluate_models.py \\
        --task start_page --data-root data --run-dir runs \\
        --models vgg16-ft bert-ft-bert-base-uncased early-fusion-vgg16+bert-base-uncased

Writes, per model, under <run-dir>/<task>/<model_name>/:
    metrics.json  preds_test.npy  probs_test.npy  confusion_matrix_test.png
and, once every requested model (+ every derivable late-fusion-<vision>+<text>
pair) is done:
    <run-dir>/<task>/summary.tsv  summary.json  classifier_comparison.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from checkpoints import checkpoint_dir, discover_models, load_config, save_config
from evaluate import evaluate_predictions
from labels import load_labels
from model_naming import late_fusion_model_name
from predict import FAMILY_PREDICTORS, TORCH_FAMILIES, predict_with_checkpoint
from results_io import load_result, save_result
from summarize import build_and_write_summary
from tasks import get_task


def compute_late_fusion(ctx, label_data):
    """Derives late-fusion for every (vision-finetune, text-finetune) pair
    whose probs_test.npy are both already on disk (from earlier in this run,
    or a previous evaluate_models.py run) - averaging two independently
    fine-tuned models' softmax outputs, no training involved (there's
    nothing train_fusion.py could save for it - see its docstring).

    Loops over every finetune_image/finetune_text checkpoint actually
    present rather than a hardcoded pair: an earlier hardcoded version
    (looking only for checkpoints literally named "efficientnet-ft" and
    "bert-ft") broke silently the moment a naming fix elsewhere renamed
    text-finetune checkpoints to bert-ft-<bert-model> - discovering pairs
    dynamically means this can't drift out of sync with checkpoint naming
    again. Text side is restricted to --backbone bert checkpoints (not
    textcnn), matching what the original hardcoded pairing used.

    Also writes model_config.json (model_family "late_fusion", naming its
    two constituent checkpoints) - the piece that used to be missing and
    is why late-fusion couldn't be used anywhere beyond this exact --task
    test split: lib/predict.py's predict_late_fusion reads this same
    config to reload and average both constituents' predictions on an
    arbitrary df, which is what makes late-fusion usable as a genuine
    evaluate_pipeline.py candidate (see that script's own docstring)."""
    run_dir, task_name = ctx["run_dir"], ctx["task_name"]

    vision_models, text_models = [], []
    for model_name in discover_models(run_dir, task_name):
        config = load_config(run_dir, task_name, model_name)
        if config["model_family"] == "finetune_image":
            vision_models.append((model_name, config["backbone"]))
        elif config["model_family"] == "finetune_text" and config["backbone"] == "bert":
            text_models.append((model_name, config["bert_model"]))

    if not vision_models or not text_models:
        print("\nSkipping late-fusion: no finetune_image + finetune_text (--backbone bert) pair found.")
        return

    te_df = ctx["df"][ctx["df"]["split"] == "test"]
    y_te = np.array(te_df["label"])

    for vis_name, vis_backbone in vision_models:
        vis_probs_path = checkpoint_dir(run_dir, task_name, vis_name) / "probs_test.npy"
        if not vis_probs_path.exists():
            continue
        vis = load_result(run_dir, task_name, vis_name)
        for txt_name, txt_backbone in text_models:
            txt_probs_path = checkpoint_dir(run_dir, task_name, txt_name) / "probs_test.npy"
            if not txt_probs_path.exists():
                continue
            txt = load_result(run_dir, task_name, txt_name)
            if vis["probs"].shape != txt["probs"].shape:
                print(
                    f"\nSkipping late-fusion {vis_name}+{txt_name}: probs shapes "
                    f"{vis['probs'].shape} vs {txt['probs'].shape} don't match."
                )
                continue
            if len(y_te) != len(vis["probs"]):
                print(
                    f"\nSkipping late-fusion {vis_name}+{txt_name}: test-set size mismatch against the "
                    f"current --task/--random-seed."
                )
                continue

            model_name = late_fusion_model_name(vis_backbone, txt_backbone)
            print(f"\nEvaluating {model_name} ({vis_name} + {txt_name} average) …")
            probs = (vis["probs"] + txt["probs"]) / 2.0
            preds = probs.argmax(axis=1)
            run_dir_task = run_dir / task_name / model_name
            metrics = evaluate_predictions(
                model_name, y_te, preds, label_data.num_classes, probs=probs,
                class_names=label_data.class_names, split="test", out_dir=run_dir_task,
            )
            save_result(run_dir, task_name, model_name, metrics, preds, probs)
            save_config(run_dir, task_name, model_name, {
                "model_family": "late_fusion",
                "task": task_name,
                "vision_model": vis_name,
                "text_model": txt_name,
                "num_classes": label_data.num_classes,
                "class_names": label_data.class_names,
            })


def evaluate_one(args, label_data, model_name: str) -> None:
    """Evaluates exactly one model - see module docstring for why this
    always runs as its own process (--_worker) rather than being called in
    a loop from main()."""
    config = load_config(args.run_dir, args.task, model_name)
    family = config["model_family"]
    if family not in FAMILY_PREDICTORS:
        print(f"Skipping {model_name}: unknown model_family {family!r}")
        return

    device = None
    if family in TORCH_FAMILIES:
        from common import pick_device

        device = pick_device(args.device)
        print(f"device: {device}")

    ctx = {
        "run_dir": args.run_dir, "task_name": args.task,
        "cache_dir": args.cache_dir or (args.data_root / "embeddings"), "device": device,
        "batch_size": args.batch_size,
    }
    te_df = label_data.df[label_data.df["split"] == "test"]
    print(f"Evaluating {model_name} ({family}) …")
    keys, preds, probs = predict_with_checkpoint(ctx, model_name, config, te_df)

    # keys may not be in te_df's own row order (sequence_lstm groups/sorts by
    # dossier - see lib/predict.py), so ground truth is looked up per key
    # rather than assumed to line up with te_df positionally.
    label_lookup = te_df.set_index(["dossier", "page_num"])["label"]
    y_te = np.array([label_lookup[k] for k in keys])

    run_dir_task = args.run_dir / args.task / model_name
    metrics = evaluate_predictions(
        model_name, y_te, preds, config["num_classes"], probs=probs,
        class_names=config.get("class_names"), split="test", out_dir=run_dir_task,
    )
    save_result(args.run_dir, args.task, model_name, metrics, preds, probs)


def build_worker_command(args, model_name: str) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--task", args.task, "--data-root", str(args.data_root), "--run-dir", str(args.run_dir),
        "--batch-size", str(args.batch_size), "--random-seed", str(args.random_seed),
        "--_worker", model_name,
    ]
    if args.cache_dir is not None:
        cmd += ["--cache-dir", str(args.cache_dir)]
    if args.split_source is not None:
        cmd += ["--split-source", args.split_source]
    if args.allow_missing_files:
        cmd += ["--allow-missing-files"]
    if args.device is not None:
        cmd += ["--device", args.device]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["start_page", "doc_type"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="model names to evaluate (default: every checkpoint found under <run-dir>/<task>)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="must match the seed used at train time - only matters when --split-source computed is in "
             "effect (its split is derived from this seed); irrelevant under --split-source tsv_column",
    )
    parser.add_argument(
        "--split-source", choices=["computed", "tsv_column"], default=None,
        help="'computed' or 'tsv_column' - defaults to the task's usual choice (see lib/tasks.py); must "
             "match whatever extract_features.py/train_*.py used for this run",
    )
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="don't error on a missing image (or transcription) file - drop rows with a missing image and "
             "continue instead (missing text always falls back to empty text, regardless). Off by default: "
             "a wrong --data-root or a path-formula mistake should fail fast, before any heavy lifting, not "
             "silently shrink the dataset.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # No set_seed()/torch.manual_seed() here: every model is loaded already-trained
    # and run in eval() mode (no dropout, no augmentation), so there's nothing
    # stochastic left at evaluation time to seed - the only place --random-seed
    # matters is load_labels()'s dossier split (under --split-source computed),
    # which takes it as an explicit parameter rather than depending on global
    # random state.
    task = get_task(args.task)
    label_data = load_labels(
        task, args.data_root, random_seed=args.random_seed, split_source=args.split_source,
        allow_missing_files=args.allow_missing_files,
    )

    if args._worker:
        evaluate_one(args, label_data, args._worker)
        return

    model_names = [
        m for m in (args.models or discover_models(args.run_dir, args.task))
        if not m.startswith("late-fusion")
    ]
    if not model_names:
        raise SystemExit(
            f"no saved checkpoints found under {args.run_dir / args.task} - run a train_*.py script first"
        )
    for m in model_names:
        if not (checkpoint_dir(args.run_dir, args.task, m) / "model_config.json").exists():
            raise SystemExit(f"no model_config.json for {m!r} under {args.run_dir / args.task} - typo, or not trained yet?")

    failed = []
    for model_name in model_names:
        print()
        result = subprocess.run(build_worker_command(args, model_name))
        if result.returncode != 0:
            print(f"** {model_name} failed (exit {result.returncode}) - continuing with the rest **")
            failed.append(model_name)

    ctx = {"run_dir": args.run_dir, "task_name": args.task, "df": label_data.df}
    compute_late_fusion(ctx, label_data)

    print()
    build_and_write_summary(args.run_dir, args.task)

    if failed:
        raise SystemExit(f"\n{len(failed)} model(s) failed to evaluate: {', '.join(failed)}")


if __name__ == "__main__":
    main()
