"""
training/train_ml.py
──────────────────────
Production ML training pipeline for the scikit-learn models.

Covers three models shipped in ai_models/ml_models/:
  - calorie_regression.pkl
  - weight_regression.pkl
  - fitness_classifier.pkl (+ fitness_classifier_labels.pkl)

Features (9):
  age, weight_kg, height_cm, activity_level, gender,
  bmi, bmr, age_squared, weight_height

Calorie/weight models also receive calories_tdee.

Pipeline:
  1. Load dataset (datasets/fitness_profiles.csv — or generate synthetic)
  2. Feature engineering (deterministic, single source of truth)
  3. Train/test split with stratification where applicable
  4. Train with GridSearchCV (small grid, 3-fold CV)
  5. Evaluate on held-out test set — print metrics + save to reports/
  6. Persist as joblib .pkl — atomic write (tmp → rename)

CLI:
    python -m training.train_ml                 # train all three models
    python -m training.train_ml --model calorie # just one
    python -m training.train_ml --no-gridsearch # faster, uses defaults
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.core.config import settings
from backend.core.logging import setup_logging


FEATURE_COLUMNS = [
    "age", "weight_kg", "height_cm", "activity_level", "gender",
    "bmi", "bmr", "age_squared", "weight_height",
]


# ─── Feature engineering (single source of truth) ────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces the 9 canonical features used by every ML model at inference.
    Mirrors backend/services/ml_service._build_features exactly.
    """
    df = df.copy()

    if "gender" in df.columns and df["gender"].dtype == object:
        df["gender"] = (df["gender"].str.lower()
                                     .map({"m": 1, "male": 1, "f": 0, "female": 0})
                                     .fillna(1).astype(int))

    h_m = df["height_cm"] / 100.0
    df["bmi"] = df["weight_kg"] / (h_m ** 2)
    df["bmr"] = (10 * df["weight_kg"] + 6.25 * df["height_cm"]
                 - 5 * df["age"] + np.where(df["gender"] == 1, 5, -161))
    df["age_squared"] = df["age"] ** 2
    df["weight_height"] = df["weight_kg"] / df["height_cm"]

    activity_to_tdee = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    df["calories_tdee"] = df["bmr"] * df["activity_level"].map(activity_to_tdee).fillna(1.375)
    return df


# ─── Data loading ────────────────────────────────────────────────────────────

DATASET_PATH = Path("datasets/fitness_profiles.csv")


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        logger.warning(f"{DATASET_PATH} missing — generating synthetic data")
        return generate_synthetic(n=3000)
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded {len(df)} rows from {DATASET_PATH}")
    return df


def generate_synthetic(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    gender = rng.integers(0, 2, n)
    height = np.where(gender == 1,
                      rng.normal(176, 7, n),
                      rng.normal(163, 6, n)).clip(150, 200)
    weight = np.where(gender == 1,
                      rng.normal(80, 12, n),
                      rng.normal(65, 11, n)).clip(40, 150)
    activity = rng.integers(1, 6, n)

    h_m = height / 100
    bmi = weight / (h_m ** 2)
    bmr = 10 * weight + 6.25 * height - 5 * age + np.where(gender == 1, 5, -161)
    mult = np.array([1.2, 1.375, 1.55, 1.725, 1.9])[activity - 1]
    tdee = bmr * mult + rng.normal(0, 60, n)

    # target weight change over 30 days at current intake (label noise)
    daily_deficit = rng.normal(-300, 400, n)
    weight_delta = daily_deficit * 30 / 7700  # kg / 7700 kcal per kg fat

    # fitness level from activity + inverse BMI pressure
    score = activity * 20 + (30 - (bmi - 22).clip(0, 10) * 2) + rng.normal(0, 5, n)
    level = np.where(score < 50, 0, np.where(score < 80, 1, 2))

    return pd.DataFrame({
        "age": age, "weight_kg": weight.round(1), "height_cm": height.round(1),
        "activity_level": activity, "gender": gender,
        "calories_target": tdee.round(),
        "weight_change_30d": weight_delta.round(2),
        "fitness_level": level,
    })


# ─── Trainer ─────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    model_dir: Path
    reports_dir: Path
    do_gridsearch: bool = True
    test_size: float = 0.2
    seed: int = 42


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _save_report(reports_dir: Path, name: str, report: dict):
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved report → {out}")


# ── calorie regression ──
def train_calorie(df: pd.DataFrame, cfg: TrainConfig):
    logger.info("━━ Training calorie regression ━━")
    X = df[FEATURE_COLUMNS + ["calories_tdee"]]
    y = df["calories_target"] if "calories_target" in df else df["calories_tdee"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("reg", Ridge(alpha=1.0))])
    if cfg.do_gridsearch:
        pipe = GridSearchCV(pipe, {"reg__alpha": [0.3, 1.0, 3.0]}, cv=3, n_jobs=-1).fit(Xtr, ytr)
        best = pipe.best_estimator_
        logger.info(f"Best params: {pipe.best_params_}")
    else:
        pipe.fit(Xtr, ytr); best = pipe

    pred = best.predict(Xte)
    report = {
        "mae": round(float(mean_absolute_error(yte, pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 2),
        "r2": round(float(r2_score(yte, pred)), 4),
        "n_train": len(ytr), "n_test": len(yte),
    }
    logger.info(f"Calorie model · MAE={report['mae']} · RMSE={report['rmse']} · R²={report['r2']}")
    _atomic_save(best, cfg.model_dir / "calorie_regression.pkl")
    (cfg.model_dir / "feature_list.json").write_text(
        json.dumps({"features": FEATURE_COLUMNS + ["calories_tdee"]}, indent=2))
    _save_report(cfg.reports_dir, "calorie", report)


# ── weight-change regression ──
def train_weight(df: pd.DataFrame, cfg: TrainConfig):
    logger.info("━━ Training weight-change regression ━━")
    if "weight_change_30d" not in df:
        logger.warning("No 'weight_change_30d' column — skipping")
        return
    X = df[FEATURE_COLUMNS + ["calories_tdee"]]
    y = df["weight_change_30d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("reg", GradientBoostingRegressor(random_state=cfg.seed))])
    if cfg.do_gridsearch:
        pipe = GridSearchCV(pipe, {
            "reg__n_estimators": [100, 200],
            "reg__max_depth": [3, 5],
            "reg__learning_rate": [0.05, 0.1],
        }, cv=3, n_jobs=-1).fit(Xtr, ytr)
        best = pipe.best_estimator_
    else:
        pipe.fit(Xtr, ytr); best = pipe

    pred = best.predict(Xte)
    report = {
        "mae": round(float(mean_absolute_error(yte, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 3),
        "r2": round(float(r2_score(yte, pred)), 4),
    }
    logger.info(f"Weight model · MAE={report['mae']} kg · R²={report['r2']}")
    _atomic_save(best, cfg.model_dir / "weight_regression.pkl")
    _save_report(cfg.reports_dir, "weight", report)


# ── fitness-level classifier ──
def train_fitness(df: pd.DataFrame, cfg: TrainConfig):
    logger.info("━━ Training fitness-level classifier ━━")
    if "fitness_level" not in df:
        logger.warning("No 'fitness_level' column — skipping")
        return
    X = df[FEATURE_COLUMNS]
    y = df["fitness_level"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", GradientBoostingClassifier(random_state=cfg.seed))])
    if cfg.do_gridsearch:
        pipe = GridSearchCV(pipe, {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5],
        }, cv=3, n_jobs=-1).fit(Xtr, ytr)
        best = pipe.best_estimator_
    else:
        pipe.fit(Xtr, ytr); best = pipe

    pred = best.predict(Xte)
    report = {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
        "report": classification_report(yte, pred, output_dict=True),
    }
    logger.info(f"Fitness clf · acc={report['accuracy']} · F1={report['f1_macro']}")
    _atomic_save(best, cfg.model_dir / "fitness_classifier.pkl")
    _atomic_save({0: "Beginner", 1: "Intermediate", 2: "Advanced"},
                 cfg.model_dir / "fitness_classifier_labels.pkl")
    _save_report(cfg.reports_dir, "fitness_classifier", report)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["all", "calorie", "weight", "fitness"], default="all")
    ap.add_argument("--no-gridsearch", action="store_true")
    args = ap.parse_args()

    s = settings()
    cfg = TrainConfig(
        model_dir=Path(s.ML_MODEL_DIR),
        reports_dir=Path(s.ML_MODEL_DIR) / "reports",
        do_gridsearch=not args.no_gridsearch,
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    df = engineer_features(load_dataset())
    logger.info(f"Feature matrix: {len(df)} rows × {len(df.columns)} cols")

    if args.model in ("all", "calorie"): train_calorie(df, cfg)
    if args.model in ("all", "weight"):  train_weight(df, cfg)
    if args.model in ("all", "fitness"): train_fitness(df, cfg)

    logger.info("✅ ML training complete")


if __name__ == "__main__":
    main()
