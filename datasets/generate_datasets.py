"""
datasets/generate_datasets.py
─────────────────────────────
Generates synthetic fitness datasets used for training the ML models.
Run once before training: `python datasets/generate_datasets.py`
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000  # number of samples

OUT = Path(__file__).parent


# ─── 1. FITNESS PROFILES DATASET ──────────────────────────────────────────────
def generate_fitness_profiles():
    """
    Features : age, weight_kg, height_cm, activity_level (1-5), gender (0/1)
    Targets  : calories_tdee, fitness_level (0=Beginner,1=Intermediate,2=Advanced),
               weight_change_30d_kg
    """
    age            = np.random.randint(16, 65, N)
    weight_kg      = np.random.uniform(45, 140, N)
    height_cm      = np.random.uniform(150, 200, N)
    activity_level = np.random.randint(1, 6, N)       # 1=sedentary … 5=athlete
    gender         = np.random.randint(0, 2, N)        # 0=female 1=male

    # BMR (Mifflin-St Jeor)
    bmr = np.where(
        gender == 1,
        10 * weight_kg + 6.25 * height_cm - 5 * age + 5,
        10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    )

    activity_mult = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    tdee = np.array([bmr[i] * activity_mult[activity_level[i]] for i in range(N)])
    tdee += np.random.normal(0, 50, N)  # add noise

    # Fitness level classification (rule-based + noise)
    fit_score = activity_level * 20 + (age < 35) * 10 - (weight_kg > 90) * 10
    fitness_level = np.where(fit_score < 30, 0, np.where(fit_score < 60, 1, 2))
    # add 10 % label noise for realism
    noise_mask = np.random.random(N) < 0.10
    fitness_level[noise_mask] = np.random.randint(0, 3, noise_mask.sum())

    # Weight change in 30 days (regression target)
    deficit = (tdee - 2000)  # positive = surplus
    weight_change_30d = (deficit / 7700) * 30  # 7700 kcal ≈ 1 kg fat
    weight_change_30d += np.random.normal(0, 0.3, N)

    df = pd.DataFrame({
        "age":              age,
        "weight_kg":        weight_kg.round(1),
        "height_cm":        height_cm.round(1),
        "activity_level":   activity_level,
        "gender":           gender,
        "calories_tdee":    tdee.round(0).astype(int),
        "fitness_level":    fitness_level,          # 0/1/2
        "weight_change_30d": weight_change_30d.round(2),
    })

    path = OUT / "fitness_profiles.csv"
    df.to_csv(path, index=False)
    print(f"✅ Saved fitness_profiles.csv  ({N} rows)")
    return df


# ─── 2. WORKOUT HISTORY DATASET ───────────────────────────────────────────────
def generate_workout_history():
    """
    Used by the recommendation engine.
    Columns: user_id, exercise, muscle_group, difficulty, duration_min, rating
    """
    exercises = [
        ("Bench Press", "chest",      3),
        ("Push-ups",    "chest",      1),
        ("Squats",      "legs",       2),
        ("Deadlifts",   "back",       3),
        ("Pull-ups",    "back",       2),
        ("Lunges",      "legs",       1),
        ("OHP",         "shoulders",  3),
        ("Plank",       "core",       1),
        ("Burpees",     "full",       2),
        ("Row",         "back",       2),
    ]

    rows = []
    for user_id in range(1, 201):          # 200 synthetic users
        n_logs = np.random.randint(5, 30)
        fitness = np.random.randint(0, 3)  # beginner / intermediate / advanced
        for _ in range(n_logs):
            ex = exercises[np.random.randint(0, len(exercises))]
            duration = np.random.randint(20, 90)
            # advanced users tend to rate harder exercises higher
            base_rating = 5 - abs(ex[2] - fitness)
            rating = int(np.clip(base_rating + np.random.randint(-1, 2), 1, 5))
            rows.append({
                "user_id":      user_id,
                "exercise":     ex[0],
                "muscle_group": ex[1],
                "difficulty":   ex[2],
                "duration_min": duration,
                "rating":       rating,
                "fitness_level": fitness,
            })

    df = pd.DataFrame(rows)
    path = OUT / "workout_history.csv"
    df.to_csv(path, index=False)
    print(f"✅ Saved workout_history.csv  ({len(df)} rows)")
    return df


# ─── 3. POSE KEYPOINTS DATASET ────────────────────────────────────────────────
def generate_pose_keypoints():
    """
    Synthetic pose keypoint dataset for the binary posture classifier.
    17 MoveNet keypoints × (x, y, confidence) = 51 features.
    Label: 0 = incorrect posture, 1 = correct posture
    """
    JOINTS = 17
    rows = []

    for _ in range(N):
        label = np.random.randint(0, 2)
        if label == 1:  # correct — joints roughly aligned
            kp = np.random.normal(loc=0.5, scale=0.05, size=(JOINTS, 2))
        else:           # incorrect — more scattered
            kp = np.random.uniform(0.1, 0.9, size=(JOINTS, 2))
        conf = np.random.uniform(0.6, 1.0, size=(JOINTS,))
        flat = np.concatenate([kp.flatten(), conf])
        row  = {f"kp_{i}": flat[i] for i in range(len(flat))}
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    path = OUT / "pose_keypoints.csv"
    df.to_csv(path, index=False)
    print(f"✅ Saved pose_keypoints.csv  ({N} rows)")
    return df


if __name__ == "__main__":
    generate_fitness_profiles()
    generate_workout_history()
    generate_pose_keypoints()
    print("\n🎉 All datasets generated in /datasets/")
