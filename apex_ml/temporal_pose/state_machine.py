"""
Exercise state machine — robust rep counting via temporal state transitions.

Replaces brittle "if angle < threshold and direction == down then count++"
logic with a proper finite-state machine driven by PhaseDetector. A rep
is committed only when:

    1. The user transitions ECCENTRIC -> LOCKOUT (or equivalent for the
       exercise variant) and back up to RESET, AND
    2. The angular range between trough and peak is >= rom_min_deg.

Incomplete reps (e.g. partial squat above parallel) are detected and
reported but NOT counted — they're emitted as a `partial_rep` event for
the form-correction layer to flag.

The machine emits structured events:
    - on_phase_change(Phase)
    - on_rep_completed(RepRecord)
    - on_partial_rep(RepRecord)

These are pure callbacks; the FastAPI router converts them to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .phase_detector import Phase, PhaseConfig, PhaseDetector, RepRecord
from .sequence_buffer import SequenceBuffer


# Default phase configs per exercise. The state machine accepts an override
# so the application layer can tune for an individual user.
EXERCISE_DEFAULTS = {
    "squat":      PhaseConfig(primary_joint="left_knee",   concentric_direction=+1, rom_min_deg=50.0),
    "pushup":     PhaseConfig(primary_joint="left_elbow",  concentric_direction=+1, rom_min_deg=45.0),
    "plank":      PhaseConfig(primary_joint="left_hip",    concentric_direction=+1, rom_min_deg=0.0),
    "lunge":      PhaseConfig(primary_joint="left_knee",   concentric_direction=+1, rom_min_deg=50.0),
    "bicep_curl": PhaseConfig(primary_joint="right_elbow", concentric_direction=-1, rom_min_deg=60.0),
}
# Note on `concentric_direction`:
#   - Squat: starting standing, descending phase REDUCES knee angle.
#     We define concentric = standing up, which INCREASES the angle => +1.
#   - Push-up: lowering REDUCES elbow angle; concentric (press up)
#     INCREASES it => +1.
#   - Bicep curl: curling up REDUCES elbow angle => -1.


@dataclass
class _RepInProgress:
    start_t: float
    eccentric_start_t: Optional[float] = None
    bottom_t: Optional[float] = None
    bottom_angle: Optional[float] = None
    top_angle: Optional[float] = None
    saw_eccentric: bool = False
    saw_concentric: bool = False


class ExerciseStateMachine:
    """Drive temporal rep counting + per-rep summaries.

    Parameters
    ----------
    exercise : str
        Key into EXERCISE_DEFAULTS.
    config : PhaseConfig, optional
        Override the default phase configuration for the exercise.
    """

    def __init__(self, exercise: str, config: Optional[PhaseConfig] = None):
        if exercise not in EXERCISE_DEFAULTS and config is None:
            raise KeyError(
                f"Unknown exercise '{exercise}'. Pass an explicit "
                f"PhaseConfig or pick one of {list(EXERCISE_DEFAULTS)}."
            )
        self.exercise = exercise
        self.cfg = config or EXERCISE_DEFAULTS[exercise]
        self.detector = PhaseDetector(self.cfg)

        self.rep_count: int = 0
        self.partial_count: int = 0
        self.reps: List[RepRecord] = []
        self.partials: List[RepRecord] = []
        self._in_progress: Optional[_RepInProgress] = None
        self._prev_phase: Phase = Phase.START

        # Callback hooks (set by caller; default no-ops)
        self.on_phase_change: Callable[[Phase], None] = lambda _: None
        self.on_rep_completed: Callable[[RepRecord], None] = lambda _: None
        self.on_partial_rep:   Callable[[RepRecord], None] = lambda _: None

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        self.detector.reset()
        self.rep_count = 0
        self.partial_count = 0
        self.reps.clear()
        self.partials.clear()
        self._in_progress = None
        self._prev_phase = Phase.START

    def update(self, buf: SequenceBuffer) -> Phase:
        """Advance the state machine by one frame's worth of buffer state."""
        phase = self.detector.step(buf)
        if not buf.is_ready():
            return phase

        last = buf.last()
        now = last.t
        angle_now = last.angles[self.cfg.primary_joint]

        # Phase transition handling --------------------------------------
        if phase != self._prev_phase:
            self._handle_transition(self._prev_phase, phase, now, angle_now)
            self.on_phase_change(phase)
            self._prev_phase = phase

        # Track extrema while a rep is in progress -----------------------
        if self._in_progress is not None:
            ip = self._in_progress
            sign = self.cfg.concentric_direction
            if sign > 0:
                # concentric = increasing angle. So bottom = min, top = max.
                if ip.bottom_angle is None or angle_now < ip.bottom_angle:
                    ip.bottom_angle = angle_now
                    ip.bottom_t = now
                if ip.top_angle is None or angle_now > ip.top_angle:
                    ip.top_angle = angle_now
            else:
                # concentric = decreasing angle. bottom = max, top = min.
                if ip.bottom_angle is None or angle_now > ip.bottom_angle:
                    ip.bottom_angle = angle_now
                    ip.bottom_t = now
                if ip.top_angle is None or angle_now < ip.top_angle:
                    ip.top_angle = angle_now

        return phase

    # ----------------------------------------------------------- internals
    def _handle_transition(self, old: Phase, new: Phase,
                            t: float, angle: float) -> None:
        # Start a new rep on first eccentric movement
        if new == Phase.ECCENTRIC and self._in_progress is None:
            self._in_progress = _RepInProgress(
                start_t=t, eccentric_start_t=t,
                top_angle=angle, bottom_angle=angle,
            )
            return

        if self._in_progress is not None:
            ip = self._in_progress
            if new == Phase.ECCENTRIC:
                ip.saw_eccentric = True
            if new == Phase.CONCENTRIC:
                ip.saw_concentric = True

            # Finalize a rep when:
            #  (a) the user returns to RESET after concentric (paused reps), OR
            #  (b) a new ECCENTRIC begins after a concentric — i.e. they
            #      flowed right into the next rep without a top pause.
            # Both indicate the previous rep is complete.
            completed_via_reset = (new == Phase.RESET
                                    and ip.saw_eccentric and ip.saw_concentric)
            completed_via_chain = (old == Phase.CONCENTRIC
                                    and new == Phase.ECCENTRIC
                                    and ip.saw_eccentric and ip.saw_concentric)
            if completed_via_reset or completed_via_chain:
                # If the next eccentric is starting immediately, the new
                # rep's start should be this transition; capture it before
                # we wipe _in_progress.
                next_rep_start_t = t if completed_via_chain else None
                next_rep_start_angle = angle if completed_via_chain else None
                self._commit_rep(t)
                if next_rep_start_t is not None:
                    self._in_progress = _RepInProgress(
                        start_t=next_rep_start_t,
                        eccentric_start_t=next_rep_start_t,
                        top_angle=next_rep_start_angle,
                        bottom_angle=next_rep_start_angle,
                    )

    def _commit_rep(self, end_t: float) -> None:
        ip = self._in_progress
        if ip is None:
            return
        rom = abs((ip.top_angle or 0.0) - (ip.bottom_angle or 0.0))
        valid = rom >= self.cfg.rom_min_deg
        concentric_t = (end_t - (ip.bottom_t or ip.start_t))
        eccentric_t = ((ip.bottom_t or ip.start_t) - ip.start_t)
        rec = RepRecord(
            index=self.rep_count + 1 if valid else self.partial_count + 1,
            start_t=ip.start_t, end_t=end_t,
            rom_deg=rom,
            peak_angle=ip.top_angle or 0.0,
            trough_angle=ip.bottom_angle or 0.0,
            concentric_t=concentric_t,
            eccentric_t=eccentric_t,
            valid=valid,
        )
        if valid:
            self.rep_count += 1
            self.reps.append(rec)
            self.on_rep_completed(rec)
        else:
            self.partial_count += 1
            self.partials.append(rec)
            self.on_partial_rep(rec)
        self._in_progress = None
