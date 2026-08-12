"""Runs every train_*.py regime x backbone x modality x task combination
the sequential pipeline supports, as one batch.

Not every (regime, modality) pair has a meaningful job with the current
train_*.py scripts, so this only builds jobs for the pairs that exist:

  - baseline (train_baseline.py):  text, vision, multimodal - multimodal
    concatenates one vision + one text backbone's cached features
    (train_baseline.py's --features takes a list; this script always
    passes exactly one backbone for text/vision, two for multimodal)
  - sequence (train_sequence.py):  text, vision only - the LSTM reads ONE
    cached backbone's per-page features (lib/datasets.py's
    DossierSequenceDataset takes a single feature matrix), so there's no
    multimodal sequence job
  - fusion   (train_fusion.py):    multimodal only - early fusion is
    inherently a (image backbone, text backbone) pair; there's no
    single-modality fusion job
  - finetune (train_finetune.py):  text, vision only - it fine-tunes
    exactly ONE backbone end-to-end, so there's no multimodal finetune job

Regimes run cheapest-first by default (baseline, sequence, fusion,
finetune): baseline/sequence/fusion all train a small model on top of
frozen, pre-extracted features (seconds to low minutes each); finetune
backpropagates through the whole backbone from raw images/text and is by
far the most expensive part of the matrix, especially the two transformer
vision backbones (facebook/dinov2-small, and particularly
microsoft/dit-large-finetuned-rvlcdip, on CPU) - if you need to stop partway
through an overnight run, this ordering means you keep the cheap results.

Runs on --device cpu by default: mps has known correctness bugs (silent
NaNs, collapsed training) for some ops in the PyTorch versions this project
has hit in practice, and a many-hour unattended batch run is exactly the
case where a silent NaN collapse is worst to discover after the fact -
pass --device mps/cuda to override if you've verified it's safe on your
setup.

Idempotent: before each job, checks whether
<run-dir>/<task>/<model_name>/model_config.json already exists (using the
same model-naming functions - lib/model_naming.py - that each train_*.py
script itself uses to name its checkpoint) and skips it unless --force is
given. Safe to Ctrl-C and re-run; also safe to re-run after adding backbones
or regimes to a --run-dir from a previous, narrower run.

baseline/sequence/fusion all read pre-extracted per-backbone caches (see
extract_features.py) rather than raw images/text, keyed only by backbone
name (not by task or modality) - so one cache per backbone covers every job
that needs it. This script runs extract_features.py once per backbone
first; it's a no-op if that backbone's cache already exists (e.g. from a
previous run of this script, or from extract_features.py run by hand), so
it's always safe to leave this pre-pass on. finetune reads raw images/text
directly and needs no cache.

Each job's own stdout/stderr is written to
<run-dir>/_train_all_logs/<task>__<regime>__<model_name>.log (there are up
to ~100 jobs in the full matrix - dumping all of that to the console would
bury the one-line-per-job progress this prints instead). A failed job is
logged and skipped, not fatal to the rest of the batch (matching
evaluate_models.py's convention) - see the FAILED summary at the end.

--jobs N runs N jobs concurrently (default 1, sequential). This is worth
using even on CPU: KNN/XGBoost already parallelize internally
(n_jobs=-1) and are seconds-fast regardless, but a single finetune job's
own intra-op thread pool often doesn't come close to saturating all
cores - training with small batch sizes (8-16 images/texts) just doesn't
have enough per-op parallelism to keep many threads busy, so several
finetune jobs run side by side can be far faster in wall-clock time than
one job at a time. --max-threads-per-job caps OMP_NUM_THREADS (and the
other common thread-pool env vars) per job's subprocess, to keep N
concurrent jobs from oversubscribing the machine; leave it unset to let
each job use whatever it naturally uses (usually the safer starting
point, since jobs already tend to self-limit - check with `ps`/Activity
Monitor while a few are running and set this only if you see contention).

Usage:
    # see the full job matrix (regime, task, modality, model_name) without
    # running or extracting anything
    python scripts/classification/sequential/train_all.py \\
        --data-root data --run-dir runs --dry-run

    # run everything - will take a long time, see --regimes/--modalities/
    # --tasks/--text-backbones/--vision-backbones to scope down
    python scripts/classification/sequential/train_all.py \\
        --data-root data --run-dir runs

    # just the cheap regimes, to get fast results while deciding whether
    # to let the expensive finetune jobs run overnight
    python scripts/classification/sequential/train_all.py \\
        --data-root data --run-dir runs --regimes baseline sequence fusion

    # re-run, adding a backbone that wasn't in the original sweep - already
    # -trained jobs are skipped automatically
    python scripts/classification/sequential/train_all.py \\
        --data-root data --run-dir runs --vision-backbones facebook/dinov2-small

    # resume the rest of an in-progress run, 4 finetune jobs at a time
    python scripts/classification/sequential/train_all.py \\
        --data-root data --run-dir runs --regimes finetune --jobs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEQUENTIAL_DIR = Path(__file__).resolve().parent
LIB_DIR = SEQUENTIAL_DIR.parent / "lib"
sys.path.insert(0, str(LIB_DIR))

from checkpoints import checkpoint_dir  # noqa: E402
from model_naming import (  # noqa: E402
    baseline_model_name,
    finetune_model_name,
    fusion_model_name,
    sequence_model_name,
    slug,
)

TEXT_BACKBONES = ["bert-base-uncased", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]
VISION_BACKBONES = ["vgg16", "efficientnet_b0", "facebook/dinov2-small", "microsoft/dit-large-finetuned-rvlcdip"]
BASELINE_MODELS = ["knn", "xgboost"]
TASKS = ["start_page", "doc_type"]
REGIMES = ["baseline", "sequence", "fusion", "finetune"]  # cheapest first, see module docstring
MODALITIES = ["text", "vision", "multimodal"]


def build_jobs(
    tasks: list[str], regimes: list[str], modalities: list[str], baseline_models: list[str],
    text_backbones: list[str], vision_backbones: list[str],
) -> list[dict]:
    jobs = []

    def add(task, regime, modality, model_name, script, extra_args):
        jobs.append(dict(
            task=task, regime=regime, modality=modality, model_name=model_name,
            script=script, extra_args=extra_args,
        ))

    for task in tasks:
        if "baseline" in regimes:
            for model in baseline_models:
                if "text" in modalities:
                    for tb in text_backbones:
                        add(task, "baseline", "text", baseline_model_name(model, [tb]),
                            "train_baseline.py", ["--model", model, "--features", tb])
                if "vision" in modalities:
                    for vb in vision_backbones:
                        add(task, "baseline", "vision", baseline_model_name(model, [vb]),
                            "train_baseline.py", ["--model", model, "--features", vb])
                if "multimodal" in modalities:
                    for vb in vision_backbones:
                        for tb in text_backbones:
                            add(task, "baseline", "multimodal", baseline_model_name(model, [vb, tb]),
                                "train_baseline.py", ["--model", model, "--features", vb, tb])

        if "sequence" in regimes:
            if "text" in modalities:
                for tb in text_backbones:
                    add(task, "sequence", "text", sequence_model_name(tb),
                        "train_sequence.py", ["--features-backbone", tb])
            if "vision" in modalities:
                for vb in vision_backbones:
                    add(task, "sequence", "vision", sequence_model_name(vb),
                        "train_sequence.py", ["--features-backbone", vb])

        if "fusion" in regimes and "multimodal" in modalities:
            for vb in vision_backbones:
                for tb in text_backbones:
                    add(task, "fusion", "multimodal", fusion_model_name(vb, tb),
                        "train_fusion.py", ["--image-backbone", vb, "--text-backbone", tb])

        if "finetune" in regimes:
            if "text" in modalities:
                for tb in text_backbones:
                    add(task, "finetune", "text", finetune_model_name("bert", tb),
                        "train_finetune.py", ["--backbone", "bert", "--bert-model", tb])
            if "vision" in modalities:
                for vb in vision_backbones:
                    add(task, "finetune", "vision", finetune_model_name(vb),
                        "train_finetune.py", ["--backbone", vb])

    return jobs


def already_trained(run_dir: Path, task: str, model_name: str) -> bool:
    return (checkpoint_dir(run_dir, task, model_name) / "model_config.json").exists()


def run_extraction_pass(args, text_backbones: list[str], vision_backbones: list[str]) -> None:
    print(f"\n=== extracting features ({len(vision_backbones)} vision + {len(text_backbones)} text backbones) ===")
    extract_script = SEQUENTIAL_DIR / "extract_features.py"
    for modality, backbones, flag in [("vision", vision_backbones, "--image-backbone"), ("text", text_backbones, "--text-backbone")]:
        for backbone in backbones:
            cmd = [
                sys.executable, str(extract_script),
                "--data-root", str(args.data_root), "--modality", modality, flag, backbone,
                "--device", args.device,
            ]
            if args.cache_dir is not None:
                cmd += ["--cache-dir", str(args.cache_dir)]
            if args.allow_missing_files:
                cmd += ["--allow-missing-files"]
            print(f"  {modality}: {backbone}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise SystemExit(
                    f"extract_features.py failed for {modality} backbone {backbone!r} (exit {result.returncode}):\n"
                    f"{result.stdout}\n{result.stderr}"
                )


def build_job_command(args, job: dict) -> list[str]:
    cmd = [
        sys.executable, str(SEQUENTIAL_DIR / job["script"]),
        "--task", job["task"], "--data-root", str(args.data_root), "--run-dir", str(args.run_dir),
        *job["extra_args"],
    ]
    # train_baseline.py has no --device (KNN/XGBoost don't use one) - every other script does.
    if job["script"] != "train_baseline.py":
        cmd += ["--device", args.device]
    if args.cache_dir is not None and job["script"] != "train_finetune.py":
        cmd += ["--cache-dir", str(args.cache_dir)]
    if args.allow_missing_files:
        cmd += ["--allow-missing-files"]
    return cmd


def run_job(args, job: dict, log_dir: Path) -> tuple[dict, int, float, Path]:
    log_path = log_dir / f"{job['task']}__{job['regime']}__{slug(job['model_name'])}.log"
    cmd = build_job_command(args, job)

    env = None
    if args.max_threads_per_job is not None:
        env = os.environ.copy()
        # covers PyTorch's OpenMP backend, MKL, and macOS's Accelerate/vecLib (what
        # numpy/torch fall back to on Apple Silicon) - whichever the job's actual BLAS
        # backend turns out to be, one of these is the one it reads.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(args.max_threads_per_job)

    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start
    return job, result.returncode, elapsed, log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=MODALITIES)
    parser.add_argument("--baseline-models", nargs="+", choices=BASELINE_MODELS, default=BASELINE_MODELS)
    parser.add_argument("--text-backbones", nargs="+", default=TEXT_BACKBONES)
    parser.add_argument("--vision-backbones", nargs="+", default=VISION_BACKBONES)
    parser.add_argument(
        "--device", default="cpu",
        help="forwarded to every job (and to the extraction pre-pass) - defaults to cpu, see module docstring",
    )
    parser.add_argument("--force", action="store_true", help="retrain even if a checkpoint already exists")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="run this many jobs concurrently (default: 1, sequential) - see module docstring for why this can "
             "help a lot even on CPU, especially for finetune jobs",
    )
    parser.add_argument(
        "--max-threads-per-job", type=int, default=None,
        help="caps each job's own thread pool (OMP_NUM_THREADS and friends) - only meaningful with --jobs > 1; "
             "unset by default, letting each job use whatever it naturally uses (see module docstring)",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="skip the extract_features.py pre-pass entirely (assumes every needed backbone is already cached)",
    )
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="forwarded to every job and to the extraction pre-pass - see each script's own --allow-missing-files "
             "help for what this does; off by default, matching every other script in this pipeline",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the job matrix and exit, run nothing")
    args = parser.parse_args()

    jobs = build_jobs(
        args.tasks, args.regimes, args.modalities, args.baseline_models, args.text_backbones, args.vision_backbones,
    )
    if not jobs:
        raise SystemExit("no jobs to run for this --regimes/--modalities combination - see module docstring")

    print(f"{len(jobs)} candidate jobs ({len(args.tasks)} tasks x {len(args.regimes)} regimes x ...):")
    for j in jobs:
        print(f"  [{j['task']}] {j['regime']:9s} {j['modality']:10s} {j['model_name']}")

    if args.dry_run:
        return

    cache_jobs = [j for j in jobs if j["regime"] in ("baseline", "sequence", "fusion")]
    if cache_jobs and not args.skip_extract:
        needed_vision = sorted({b for j in cache_jobs for b in j["extra_args"] if b in args.vision_backbones})
        needed_text = sorted({b for j in cache_jobs for b in j["extra_args"] if b in args.text_backbones})
        run_extraction_pass(args, needed_text, needed_vision)

    log_dir = args.run_dir / "_train_all_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    skipped, succeeded, failed = [], [], []
    pending = []
    for job in jobs:
        if not args.force and already_trained(args.run_dir, job["task"], job["model_name"]):
            print(f"[{job['task']}] {job['regime']} {job['model_name']} ... skipped (already trained, pass --force to retrain)")
            skipped.append(job)
        else:
            pending.append(job)

    def report(job: dict, count: int, returncode: int, elapsed: float, log_path: Path) -> None:
        tag = f"[{count}/{len(pending)}] {job['task']} {job['regime']} {job['model_name']}"
        if returncode == 0:
            print(f"{tag} ... OK ({elapsed:.0f}s)")
        else:
            print(f"{tag} ... FAILED (exit {returncode}, {elapsed:.0f}s) - see {log_path}")

    print(f"\n=== running {len(pending)} jobs (device={args.device}, jobs={args.jobs}) ===")
    if args.jobs <= 1:
        for i, job in enumerate(pending, 1):
            _, returncode, elapsed, log_path = run_job(args, job, log_dir)
            report(job, i, returncode, elapsed, log_path)
            (succeeded if returncode == 0 else failed).append(job)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_job, args, job, log_dir) for job in pending]
            for i, future in enumerate(as_completed(futures), 1):
                job, returncode, elapsed, log_path = future.result()
                report(job, i, returncode, elapsed, log_path)
                (succeeded if returncode == 0 else failed).append(job)

    print(f"\n=== done: {len(succeeded)} trained, {len(skipped)} skipped, {len(failed)} failed ===")
    if failed:
        print("failed jobs:")
        for j in failed:
            print(f"  [{j['task']}] {j['regime']} {j['model_name']}")


if __name__ == "__main__":
    main()
