"""
training/train_dl.py
──────────────────────
DL (deep-learning) training CLI — trains the PyTorch ExerciseNet classifier.

This is the entry point the project spec calls out by name. It delegates
to the full implementation in `training.train_cv` (same thing — our only
deep-learning model IS the CV classifier).

Examples:
    python -m training.train_dl                        # sensible defaults
    python -m training.train_dl --epochs 80 --batch-size 256
    python -m training.train_dl --data datasets/pose_keypoints.csv --device cuda
    python -m training.train_dl --resume ai_models/dl_models/exercise_classifier.pth

Outputs:
    ai_models/dl_models/exercise_classifier.pth
    ai_models/dl_models/cv_keypoint_scaler.pkl
    ai_models/dl_models/exercise_classifier_config.json
    ai_models/dl_models/training_report.json
"""
from training.train_cv import main

if __name__ == "__main__":
    main()
