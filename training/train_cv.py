"""
training/train_cv.py
──────────────────────
PyTorch training pipeline for ExerciseNet (exercise classifier on pose keypoints).

Dataset: datasets/pose_keypoints.csv
Schema:
    - exercise (string class label)
    - 51 keypoint columns: kp0_x, kp0_y, kp0_vis, kp1_x, ..., kp16_vis
    Any reasonable column order works — we detect by name.

Pipeline:
  1. Load CSV → pandas
  2. Build (X, y): X is 61-dim (51 keypoints + 10 joint angles), y is class index
  3. Fit StandardScaler on train split; save alongside model
  4. Stratified 80/20 split
  5. Train ExerciseNet with:
       - CrossEntropyLoss with class weights
       - AdamW optimiser
       - Cosine LR schedule with warm restart
       - Early stopping on validation F1
       - Mixed precision (GradScaler) when CUDA is available
  6. Save checkpoint + config + scaler + training report

CLI:
    python -m training.train_cv --epochs 60 --batch-size 128
    python -m training.train_cv --resume ai_models/dl_models/exercise_classifier.pth
    python -m training.train_cv --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.config import settings
from backend.core.logging import setup_logging
from backend.cv.exercise_classifier import _build_mlp, compute_joint_angles


# ─── DATASET ──────────────────────────────────────────────────────────────────

class KeypointDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

KEYPOINT_COLS_TEMPLATE = [f"kp{i}_{axis}" for i in range(17) for axis in ("x", "y", "vis")]


def load_keypoint_csv(path: Path):
    df = pd.read_csv(path)
    if "exercise" not in df.columns:
        raise ValueError(f"{path}: no 'exercise' column")

    kp_cols = [c for c in KEYPOINT_COLS_TEMPLATE if c in df.columns]
    if len(kp_cols) != 51:
        # Tolerate legacy layouts: try numeric 0..50
        numeric_cols = [c for c in df.columns if c not in ("exercise",)
                        and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 51:
            kp_cols = numeric_cols[:51]
        else:
            raise ValueError(
                f"Expected 51 keypoint columns, found {len(kp_cols)}. "
                f"Rename to kp0_x, kp0_y, kp0_vis, …, kp16_vis."
            )

    X_kp = df[kp_cols].to_numpy(dtype=np.float32)

    # Add 10 joint angles per row
    angles = np.zeros((len(df), 10), dtype=np.float32)
    for i, row in enumerate(X_kp):
        angles[i] = compute_joint_angles(row.reshape(17, 3))
    X = np.hstack([X_kp, angles])

    # Label encoding
    classes = sorted(df["exercise"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["exercise"].map(class_to_idx).to_numpy(dtype=np.int64)
    return X, y, classes


# ─── TRAIN CONFIG ─────────────────────────────────────────────────────────────

@dataclass
class CVTrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    seed: int = 42
    device: str = "auto"


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    losses, preds, labels = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train: optimizer.zero_grad(set_to_none=True)

            # Mixed precision only when on CUDA
            if train and scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x); loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(x); loss = criterion(logits, y)
                if train:
                    loss.backward(); optimizer.step()

            losses.append(loss.item())
            preds.append(logits.argmax(1).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds); labels = np.concatenate(labels)
    return (float(np.mean(losses)),
            float(accuracy_score(labels, preds)),
            float(f1_score(labels, preds, average="macro")))


def train_exercisenet(X: np.ndarray, y: np.ndarray, classes: list,
                       cfg: CVTrainConfig, out_dir: Path, resume: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # Device
    if cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Training device: {device}")

    # Split + scale
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg.seed)
    scaler_fs = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler_fs.transform(Xtr), scaler_fs.transform(Xva)

    tr_loader = DataLoader(KeypointDataset(Xtr_s, ytr), batch_size=cfg.batch_size,
                           shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    va_loader = DataLoader(KeypointDataset(Xva_s, yva), batch_size=cfg.batch_size,
                           shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    model = _build_mlp(in_dim=X.shape[1], num_classes=len(classes)).to(device)
    if resume and Path(resume).exists():
        try:
            model.load_state_dict(torch.load(resume, map_location=device))
            logger.info(f"Resumed weights from {resume}")
        except Exception as e:
            logger.warning(f"Resume failed ({e}); training from scratch")

    # Class-weighted loss for imbalance
    class_counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    class_weights = (class_counts.sum() / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.mean()
    cw_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw_t)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    amp = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_f1, best_state, epochs_no_improve = -1.0, None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, tr_loader, criterion, optim, amp, device, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, va_loader, criterion, optim, None,  device, train=False)
        sched.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc,
                        "train_f1": tr_f1, "val_f1": va_f1,
                        "lr": sched.get_last_lr()[0]})

        logger.info(f"Epoch {epoch:>3}/{cfg.epochs} | "
                    f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                    f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

        if va_f1 > best_f1 + 1e-4:
            best_f1, best_state = va_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} (best F1={best_f1:.3f})")
                break

    # ── save best ──
    assert best_state is not None, "No checkpoint was saved — training produced nothing"
    model.load_state_dict(best_state)
    ckpt_path = out_dir / "exercise_classifier.pth"
    tmp = ckpt_path.with_suffix(".pth.tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ckpt_path)
    logger.info(f"Saved checkpoint → {ckpt_path}")

    joblib.dump(scaler_fs, out_dir / "cv_keypoint_scaler.pkl")
    (out_dir / "exercise_classifier_config.json").write_text(
        json.dumps({"classes": classes, "in_dim": X.shape[1]}, indent=2))

    # Final validation report
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(Xva_s).float().to(device)
        preds = model(xv).argmax(1).cpu().numpy()
    report = classification_report(yva, preds, target_names=classes, output_dict=True,
                                    zero_division=0)
    summary = {
        "best_val_f1": round(best_f1, 4),
        "best_val_acc": round(float(accuracy_score(yva, preds)), 4),
        "elapsed_sec": round(time.time() - t0, 1),
        "classes": classes,
        "num_train": int(len(ytr)), "num_val": int(len(yva)),
        "classification_report": report,
        "history": history,
    }
    (out_dir / "training_report.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"✅ Done · best val F1={best_f1:.3f} · acc={summary['best_val_acc']:.3f} "
                f"· {summary['elapsed_sec']}s")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/pose_keypoints.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    s = settings()
    cfg = CVTrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, device=args.device)

    X, y, classes = load_keypoint_csv(Path(args.data))
    logger.info(f"Loaded {X.shape[0]} samples · {X.shape[1]}-dim features · {len(classes)} classes")

    out_dir = Path(s.CV_MODEL_PATH).parent
    train_exercisenet(X, y, classes, cfg, out_dir, resume=args.resume)


if __name__ == "__main__":
    main()
