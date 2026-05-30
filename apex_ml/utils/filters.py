"""
One-Euro filter for low-latency landmark smoothing.

Standard EMA introduces a tradeoff: smooth signals = high lag. The One
Euro filter (Casiez et al., 2012) dynamically increases its cutoff
frequency in fast motion, giving us both quiet hands when still and
responsive tracking during fast reps — exactly what we want for live
form correction.

Reference: http://cristal.univ-lille.fr/~casiez/1euro/
"""

from __future__ import annotations

import math
import numpy as np


class _LowPass:
    def __init__(self, alpha: float, init: np.ndarray | None = None):
        self.alpha = alpha
        self.y = init
        self.s = init

    def __call__(self, value: np.ndarray, alpha: float | None = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s


class OneEuroFilter:
    """Per-element One Euro filter for an arbitrarily shaped tensor.

    Parameters
    ----------
    freq : float
        Approximate sample rate (Hz). Webcams are usually 25–30.
    min_cutoff : float
        Minimum cutoff frequency. Lower = smoother when still.
    beta : float
        Speed coefficient. Higher = less lag during fast motion.
    d_cutoff : float
        Cutoff for the derivative low-pass.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter: _LowPass | None = None
        self.dx_filter: _LowPass | None = None
        self.last_value: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.x_filter is None:
            # First sample: initialize without smoothing
            self.x_filter = _LowPass(self._alpha(self.min_cutoff, self.freq), value)
            self.dx_filter = _LowPass(self._alpha(self.d_cutoff, self.freq),
                                      np.zeros_like(value))
            self.last_value = value
            return value

        dx = (value - self.last_value) * self.freq
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff, self.freq))
        # Adaptive cutoff: faster movement -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(edx))
        smoothed = self.x_filter(value, self._alpha(cutoff, self.freq))
        self.last_value = value
        return smoothed

    def reset(self) -> None:
        self.x_filter = None
        self.dx_filter = None
        self.last_value = None
