"""Generic training loops shared by train_finetune.py, train_sequence.py and
train_fusion.py --fusion early. Each loop trains for a fixed number of epochs
with an Adam + StepLR schedule, keeps the state_dict with the best
validation F1 (macro-F1 for multi-class tasks, positive-class F1 for binary
- see _val_f1), and loads that state back into the model before returning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn, optim


def _val_f1(y_true, y_pred, num_classes: int) -> float:
    average = "binary" if num_classes == 2 else "macro"
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def _snapshot(model: nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def predict_image_model(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds, probs_list = [], []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(preds), np.vstack(probs_list)


def train_image_model(
    model: nn.Module, tr_loader, va_loader, train_labels: np.ndarray, num_classes: int, device: torch.device,
    n_epochs: int = 15, lr: float = 1e-4, weight_decay: float = 1e-5, step_size: int = 5, gamma: float = 0.5,
) -> nn.Module:
    from common import class_weights_tensor

    cw = class_weights_tensor(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    val_labels = np.array(va_loader.dataset.labels)

    best_f1, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        preds, _ = predict_image_model(model, va_loader, device)
        f1 = _val_f1(val_labels, preds, num_classes)
        print(f"  Epoch {epoch:2d}  val_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1, best_state = f1, _snapshot(model)

    model.load_state_dict(best_state)
    return model


# Text models (TextCNN, BERTClassifier) share the exact same (ids, mask, y)
# batch shape and forward(ids, mask) signature, so one pair of functions
# covers both.


def predict_text_model(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds, probs_list = [], []
    with torch.no_grad():
        for ids, mask, _ in loader:
            logits = model(ids.to(device), mask.to(device))
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(preds), np.vstack(probs_list)


def train_text_model(
    model: nn.Module, tr_loader, va_loader, train_labels: np.ndarray, num_classes: int, device: torch.device,
    n_epochs: int = 15, lr: float = 1e-3, weight_decay: float = 1e-5, step_size: int = 5, gamma: float = 0.5,
    clip_norm: float = 1.0,
) -> nn.Module:
    from common import class_weights_tensor

    cw = class_weights_tensor(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    val_labels = np.array(va_loader.dataset.labels)

    best_f1, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for ids, mask, yb in tr_loader:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        scheduler.step()

        preds, _ = predict_text_model(model, va_loader, device)
        f1 = _val_f1(val_labels, preds, num_classes)
        print(f"  Epoch {epoch:2d}  val_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1, best_state = f1, _snapshot(model)

    model.load_state_dict(best_state)
    return model


def predict_lstm(model: nn.Module, loader, device: torch.device):
    """Returns flattened (non-padding) (y_true, y_pred, probs) across every
    dossier/page in loader - probs has shape (N_pages, num_classes)."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for feats, labels, lengths in loader:
            feats = feats.to(device)
            logits = model(feats, lengths)  # (B, T, C)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            truth = labels.numpy()
            mask = truth != -1
            all_true.extend(truth[mask].tolist())
            all_pred.extend(preds.cpu().numpy()[mask].tolist())
            all_prob.extend(probs.cpu().numpy()[mask].tolist())
    return np.array(all_true), np.array(all_pred), np.array(all_prob)


def train_lstm(
    model: nn.Module, tr_loader, va_loader, train_labels: np.ndarray, num_classes: int, device: torch.device,
    n_epochs: int = 20, lr: float = 1e-3, weight_decay: float = 1e-5, step_size: int = 7, gamma: float = 0.5,
    clip_norm: float = 1.0,
) -> nn.Module:
    from common import class_weights_tensor

    cw = class_weights_tensor(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw, ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_f1, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for feats, labels, lengths in tr_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats, lengths)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), labels.view(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            mask = labels.view(B * T) != -1
            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()
        scheduler.step()

        va_true, va_pred, _ = predict_lstm(model, va_loader, device)
        f1 = _val_f1(va_true, va_pred, num_classes)
        print(f"  Epoch {epoch:2d}  loss={total_loss / max(n, 1):.4f}  val_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1, best_state = f1, _snapshot(model)

    model.load_state_dict(best_state)
    return model


def predict_fusion_mlp(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds, probs_list = [], []
    with torch.no_grad():
        for img_f, txt_f, _ in loader:
            logits = model(img_f.to(device), txt_f.to(device))
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(preds), np.vstack(probs_list)


def train_fusion_mlp(
    model: nn.Module, tr_loader, va_loader, train_labels: np.ndarray, num_classes: int, device: torch.device,
    n_epochs: int = 20, lr: float = 1e-3, weight_decay: float = 1e-5, step_size: int = 5, gamma: float = 0.5,
) -> nn.Module:
    from common import class_weights_tensor

    cw = class_weights_tensor(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    val_labels = np.array(va_loader.dataset.labels)

    best_f1, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for img_f, txt_f, yb in tr_loader:
            img_f, txt_f, yb = img_f.to(device), txt_f.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img_f, txt_f), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        preds, _ = predict_fusion_mlp(model, va_loader, device)
        f1 = _val_f1(val_labels, preds, num_classes)
        print(f"  Epoch {epoch:2d}  val_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1, best_state = f1, _snapshot(model)

    model.load_state_dict(best_state)
    return model
