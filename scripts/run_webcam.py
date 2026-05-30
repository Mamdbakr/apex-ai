"""
scripts/run_webcam.py
───────────────────────
Desktop / local real-time webcam demo.

Does NOT require the API server. Opens the default webcam, runs the full
CV pipeline, and renders an overlay with the detected exercise, rep count,
form score, and live cues.

Controls:
    q  quit
    r  reset rep counter
    s  save a snapshot to ./snapshots/

Usage:
    python scripts/run_webcam.py
    python scripts/run_webcam.py --camera 1 --width 1280 --height 720
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from backend.core.logging import setup_logging
from backend.cv.pipeline import get_cv_pipeline
from backend.cv.rep_counter import get_rep_counter


# ── drawing helpers ──────────────────────────────────────────────────────────

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _draw_skeleton(frame, kp51_flat, width, height):
    if not kp51_flat:
        return
    pts = np.array(kp51_flat).reshape(17, 3)
    for i, (x, y, v) in enumerate(pts):
        if v < 0.3:
            continue
        cx, cy = int(x * width), int(y * height)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for a, b in SKELETON_EDGES:
        xa, ya, va = pts[a]
        xb, yb, vb = pts[b]
        if va < 0.3 or vb < 0.3:
            continue
        cv2.line(frame,
                 (int(xa * width), int(ya * height)),
                 (int(xb * width), int(yb * height)),
                 (0, 200, 255), 2)


def _draw_hud(frame, result):
    h, w = frame.shape[:2]
    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"{result.exercise_name}  ({result.confidence*100:.0f}%)",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame,
                f"Reps: {result.reps}   Form: {result.form_score}   FPS: {result.fps}",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    # Form cues
    if result.form_cues:
        y = h - 20 - 22 * (len(result.form_cues) - 1)
        cv2.rectangle(frame, (0, y - 30), (w, h), (0, 0, 0), -1)
        for i, cue in enumerate(result.form_cues):
            cv2.putText(frame, "• " + cue, (16, y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--session", default="webcam")
    ap.add_argument("--exercise", default=None,
                    help="Pin the rep counter to this exercise (squat, push_up, ...). "
                         "If omitted, the auto-classifier picks one each frame.")
    args = ap.parse_args()

    pipe = get_cv_pipeline()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        sys.exit(1)

    Path("snapshots").mkdir(exist_ok=True)
    print("Webcam started. Press q=quit, r=reset, s=snapshot")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                 # mirror for UX
            result = pipe.analyze_frame(frame, session_id=args.session,
                                        exercise_hint=args.exercise)
            _draw_skeleton(frame, result.keypoints, frame.shape[1], frame.shape[0])
            _draw_hud(frame, result)
            cv2.imshow("APEX AI — Live Form Coach", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                get_rep_counter().reset(args.session)
                print("Rep counter reset.")
            if key == ord("s"):
                p = Path("snapshots") / f"{int(time.time())}.jpg"
                cv2.imwrite(str(p), frame)
                print(f"Saved {p}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
