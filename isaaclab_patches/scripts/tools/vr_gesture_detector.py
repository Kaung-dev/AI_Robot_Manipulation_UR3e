"""Left-hand gesture detector for VR recording controls.

Detects three gestures from the left hand joint poses (updated each frame by
the OpenXR device). Each gesture must be held for HOLD_FRAMES consecutive
frames before firing, preventing accidental triggers.

Gesture mappings:
    pinch      (thumb+index close)  — accept episode + reset
    fist       (all fingers closed) — discard episode + reset
    open_palm  (all fingers extended, held) — toggle pause/resume recording
"""

from __future__ import annotations

import numpy as np


class LeftHandGestureDetector:
    PINCH_THRESHOLD = 0.03   # m — thumb-tip to index-tip
    FIST_THRESHOLD  = 0.10   # m — avg fingertip distance from wrist (closed)
    OPEN_THRESHOLD  = 0.13   # m — avg fingertip distance from wrist (open)
    HOLD_FRAMES     = 15     # frames gesture must be held before firing (~0.5 s at 30 Hz)

    _GESTURES = ("pinch", "fist", "open_palm")

    def __init__(self):
        self._counts:    dict[str, int]  = {g: 0 for g in self._GESTURES}
        self._triggered: dict[str, bool] = {g: False for g in self._GESTURES}

    def update(self, left_poses: dict[str, np.ndarray]) -> dict[str, bool]:
        """Return which gestures fired this frame (True = just triggered).

        Args:
            left_poses: Latest left-hand joint poses from OpenXRDevice.get_hand_poses().
        """
        wrist      = left_poses.get("wrist")
        thumb_tip  = left_poses.get("thumb_tip")
        index_tip  = left_poses.get("index_tip")
        middle_tip = left_poses.get("middle_tip")
        ring_tip   = left_poses.get("ring_tip")
        little_tip = left_poses.get("little_tip")

        if wrist is None or thumb_tip is None or index_tip is None:
            return {g: False for g in self._GESTURES}

        # thumb+middle pinch avoids the Quest 2 system menu (which triggers on thumb+index)
        pinch_dist = float(np.linalg.norm(thumb_tip[:3] - middle_tip[:3])) if middle_tip is not None else 1.0
        is_pinching = pinch_dist < self.PINCH_THRESHOLD

        tips = [t for t in [thumb_tip, index_tip, middle_tip, ring_tip, little_tip] if t is not None]
        avg_dist = float(np.mean([np.linalg.norm(t[:3] - wrist[:3]) for t in tips]))
        is_fist     = avg_dist < self.FIST_THRESHOLD and not is_pinching
        is_open     = avg_dist > self.OPEN_THRESHOLD and not is_pinching

        active = {"pinch": is_pinching, "fist": is_fist, "open_palm": is_open}
        return {g: self._debounce(g, active[g]) for g in self._GESTURES}

    def _debounce(self, gesture: str, active: bool) -> bool:
        if active:
            self._counts[gesture] += 1
            if self._counts[gesture] >= self.HOLD_FRAMES and not self._triggered[gesture]:
                self._triggered[gesture] = True
                return True
        else:
            self._counts[gesture] = 0
            self._triggered[gesture] = False
        return False
