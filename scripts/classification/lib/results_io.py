"""Persists each model's test-set metrics/predictions/probabilities to disk
under <run_dir>/<task>/<model_name>/ - the replacement for the notebooks'
in-memory results_probs/results_summary dicts, which only worked because
every cell shared one kernel. Scripts don't share a process, so late fusion
(train_fusion.py --fusion late) and summarize_results.py read these files
back in rather than reaching into another script's variables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def model_dir(run_dir: Path, task: str, model_name: str) -> Path:
    return run_dir / task / model_name


def save_result(
    run_dir: Path, task: str, model_name: str, metrics: dict,
    preds: np.ndarray, probs: np.ndarray | None = None,
) -> None:
    d = model_dir(run_dir, task, model_name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(d / "preds_test.npy", preds)
    if probs is not None:
        np.save(d / "probs_test.npy", probs)
    print(f"  wrote results -> {d}")


def load_result(run_dir: Path, task: str, model_name: str) -> dict:
    d = model_dir(run_dir, task, model_name)
    with open(d / "metrics.json") as f:
        metrics = json.load(f)
    preds = np.load(d / "preds_test.npy")
    probs_path = d / "probs_test.npy"
    probs = np.load(probs_path) if probs_path.exists() else None
    return {"model_name": model_name, "metrics": metrics, "preds": preds, "probs": probs}


def load_all_results(run_dir: Path, task: str) -> dict:
    task_dir = run_dir / task
    if not task_dir.exists():
        return {}
    results = {}
    for d in sorted(task_dir.iterdir()):
        if (d / "metrics.json").exists():
            results[d.name] = load_result(run_dir, task, d.name)
    return results
