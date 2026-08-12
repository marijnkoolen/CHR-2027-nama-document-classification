"""Model checkpoint save/load - the contract between the train_*.py scripts
(which each save a checkpoint + a model_config.json describing how to
rebuild that exact architecture) and evaluate_models.py (which reads
model_config.json to rebuild the untrained skeleton via lib/models.py's
builder functions, loads the checkpoint into it, and reproduces test-set
predictions without retraining).

Two checkpoint kinds:
  - torch models (finetune_image, finetune_text, sequence_lstm, fusion_early):
    model.pt (full state_dict - including any frozen submodules, e.g.
    TextCNN's frozen BERT backbone - so reloading never depends on a
    separately-tracked "which params were frozen" list, at the cost of a
    larger file than saving only the trainable subset would be).
  - sklearn/xgboost baselines (knn, xgboost): model.pkl (plain pickle -
    both KNeighborsClassifier and XGBClassifier pickle/unpickle cleanly;
    for xgboost specifically this isn't guaranteed stable across major
    xgboost version upgrades, unlike its own native .save_model() format,
    but that cross-version guarantee isn't needed for reproducing a result
    within the same project/environment).

Every checkpoint directory (run_dir/<task>/<model_name>/) also gets a
metrics.json/preds_test.npy/probs_test.npy/confusion_matrix_test.png once
evaluate_models.py has run - see lib/results_io.py and lib/evaluate.py.

torch is imported lazily, only inside save_torch_model/load_torch_state_dict
- a process that only ever calls save_sklearn_model/load_sklearn_model
(a KNN/XGBoost baseline) should never import torch at all, see
lib/common.py's docstring for why that matters.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path


def checkpoint_dir(run_dir: Path, task: str, model_name: str) -> Path:
    return run_dir / task / model_name


def save_config(run_dir: Path, task: str, model_name: str, config: dict) -> None:
    d = checkpoint_dir(run_dir, task, model_name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_config(run_dir: Path, task: str, model_name: str) -> dict:
    d = checkpoint_dir(run_dir, task, model_name)
    with open(d / "model_config.json") as f:
        return json.load(f)


def save_torch_model(run_dir: Path, task: str, model_name: str, model: "nn.Module", config: dict) -> None:  # noqa: F821
    import torch

    d = checkpoint_dir(run_dir, task, model_name)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    save_config(run_dir, task, model_name, config)
    print(f"  wrote checkpoint -> {d / 'model.pt'}")


def load_torch_state_dict(run_dir: Path, task: str, model_name: str) -> dict:
    import torch

    d = checkpoint_dir(run_dir, task, model_name)
    return torch.load(d / "model.pt", map_location="cpu")


def save_sklearn_model(run_dir: Path, task: str, model_name: str, model, config: dict) -> None:
    d = checkpoint_dir(run_dir, task, model_name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    save_config(run_dir, task, model_name, config)
    print(f"  wrote checkpoint -> {d / 'model.pkl'}")


def load_sklearn_model(run_dir: Path, task: str, model_name: str):
    d = checkpoint_dir(run_dir, task, model_name)
    with open(d / "model.pkl", "rb") as f:
        return pickle.load(f)


def discover_models(run_dir: Path, task: str) -> list[str]:
    """Model names with a saved model_config.json under run_dir/task -
    i.e. every model a train_*.py script has actually trained and saved.
    Derived entries like late-fusion (computed from two other models'
    saved probabilities, not from its own checkpoint) intentionally don't
    have a model_config.json, so they're never "discovered" here - see
    evaluate_models.py, which adds late-fusion as an extra step instead."""
    task_dir = run_dir / task
    if not task_dir.exists():
        return []
    return sorted(d.name for d in task_dir.iterdir() if (d / "model_config.json").exists())
