"""
Training loop and inference wrapper for pose sequence models.

Kept deliberately framework-light: a vanilla PyTorch training loop with
class balancing, label smoothing, gradient clipping, and a cosine LR
schedule. No external trainer dependencies — works inside a notebook,
a CLI, or as part of an MLflow run.

The InferenceEngine wraps a trained model with:
    - state dict loading
    - eval-mode + no_grad context
    - batched and live-streaming inference paths
    - confidence thresholding and label smoothing during predict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except ImportError:                              # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch")


# ================================================================ data
class PoseSequenceDataset(Dataset if _TORCH_AVAILABLE else object):
    """Wraps a list of (sequence, label) pairs.

    Sequences are numpy arrays of shape (T_i, F). They may have variable
    length — the collate function pads to the longest in the batch.
    """

    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        _require_torch()
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have same length")
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        x = torch.as_tensor(self.sequences[i], dtype=torch.float32)
        y = torch.as_tensor(self.labels[i], dtype=torch.long)
        return x, y


def pad_collate(batch):
    """Pad variable-length sequences with zeros on the right.

    Pose vectors of all zeros are not a valid pose, so the model can
    learn to ignore them (or callers can supply a mask if needed).
    """
    _require_torch()
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    T = max(lens)
    F_dim = xs[0].shape[1]
    out = torch.zeros(len(xs), T, F_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
    return out, torch.stack(list(ys)), torch.as_tensor(lens, dtype=torch.long)


# =========================================================== training
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    device: str = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    checkpoint_path: Optional[str] = None


def _class_weights(labels: List[int], num_classes: int) -> "torch.Tensor":
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def train_model(model, train_ds, val_ds, num_classes: int,
                cfg: TrainConfig = TrainConfig()) -> Dict[str, list]:
    """Train a sequence model. Returns history dict {train_loss, val_acc}.

    Handles class imbalance (weighted loss), gradient clipping, cosine
    LR decay. Saves the best-val-acc checkpoint to cfg.checkpoint_path.
    """
    _require_torch()
    device = cfg.device
    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, collate_fn=pad_collate)

    class_w = _class_weights(train_ds.labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w,
                                    label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=cfg.epochs)

    history = {"train_loss": [], "val_acc": []}
    best_acc = -1.0

    for epoch in range(cfg.epochs):
        # ---- train ---------------------------------------------------
        model.train()
        running_loss = 0.0
        n = 0
        for x, y, _lens in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        history["train_loss"].append(running_loss / max(n, 1))

        # ---- validate ------------------------------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, _lens in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        history["val_acc"].append(acc)

        if cfg.checkpoint_path and acc > best_acc:
            best_acc = acc
            Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.checkpoint_path)

    return history


# ========================================================== inference
class InferenceEngine:
    """Live or batched inference for a trained sequence model.

    Designed for the websocket frame-loop: the engine keeps no internal
    buffer, you pass it the current SequenceBuffer.feature_tensor() and
    it returns (label, confidence).
    """

    def __init__(self, model, labels: List[str],
                 device: Optional[str] = None,
                 min_confidence: float = 0.0):
        _require_torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.labels = labels
        self.min_confidence = min_confidence

    @classmethod
    def from_checkpoint(cls, model, checkpoint_path: str,
                        labels: List[str], **kwargs) -> "InferenceEngine":
        _require_torch()
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        return cls(model, labels, **kwargs)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[Optional[str], float]:
        """Predict from a (T, F) feature tensor. Returns (label, prob)."""
        if features.size == 0:
            return None, 0.0
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)                                     # (1, T, F)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        conf, idx = torch.max(probs, dim=-1)
        conf = float(conf.item())
        if conf < self.min_confidence:
            return None, conf
        return self.labels[int(idx.item())], conf

    @torch.no_grad()
    def predict_batch(self, sequences: List[np.ndarray]
                       ) -> List[Tuple[Optional[str], float]]:
        if not sequences:
            return []
        # Pad to longest
        T = max(s.shape[0] for s in sequences)
        F_dim = sequences[0].shape[1]
        x = torch.zeros(len(sequences), T, F_dim,
                        dtype=torch.float32, device=self.device)
        for i, s in enumerate(sequences):
            x[i, : s.shape[0]] = torch.as_tensor(s, dtype=torch.float32)
        probs = torch.softmax(self.model(x), dim=-1)
        out = []
        for row in probs:
            conf, idx = torch.max(row, dim=-1)
            conf = float(conf.item())
            label = self.labels[int(idx.item())] if conf >= self.min_confidence else None
            out.append((label, conf))
        return out
