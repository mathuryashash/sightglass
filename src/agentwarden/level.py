"""Liquid level from a liquid mask + container box + row->% calibration. Pure numpy."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Box = tuple[float, float, float, float]  # (left, top, right, bottom)
Status = Literal["OK", "EMPTY", "CAMERA_BLIND", "LOW_CONFIDENCE"]


@dataclass(frozen=True)
class Calibration:
    rows: tuple[float, ...]  # ascending
    levels: tuple[float, ...]

    @classmethod
    def from_points(cls, points) -> "Calibration":
        pts = [(float(r), float(p)) for r, p in points]
        if len(pts) < 2:
            raise ValueError("need at least 2 calibration points")
        rows = [r for r, _ in pts]
        drow = [b - a for a, b in zip(rows, rows[1:])]
        if not (all(d > 0 for d in drow) or all(d < 0 for d in drow)):
            raise ValueError("calibration rows must be strictly monotonic")
        pts.sort()
        rows = [r for r, _ in pts]
        levels = [p for _, p in pts]
        steps = [b - a for a, b in zip(levels, levels[1:])]
        if not (all(s > 0 for s in steps) or all(s < 0 for s in steps)):
            raise ValueError("calibration levels must be strictly monotonic in row")
        if not all(0.0 <= p <= 100.0 for p in levels):
            raise ValueError("calibration levels must be within 0..100")
        return cls(tuple(rows), tuple(levels))

    def level_at(self, row: float) -> tuple[float, bool]:
        """(level %, extrapolated). Beyond the end points: linear on the end segment, clamped to [0, 100]."""
        rows, levels = self.rows, self.levels
        extrapolated = not rows[0] <= row <= rows[-1]
        i = min(max(int(np.searchsorted(rows, row, side="right")) - 1, 0), len(rows) - 2)
        slope = (levels[i + 1] - levels[i]) / (rows[i + 1] - rows[i])
        pct = levels[i] + slope * (row - rows[i])
        return min(100.0, max(0.0, pct)), extrapolated


@dataclass(frozen=True)
class Reading:
    level_pct: float | None
    confidence: float
    status: Status
    top_row: float | None
    extrapolated: bool = False


def estimate(
    liquid_mask: np.ndarray,
    container_box: Box | None,
    calib: Calibration,
    sam_confidence: float | None = None,
) -> Reading:
    if container_box is None:
        return Reading(None, 0.0, "CAMERA_BLIND", None)
    h, w = liquid_mask.shape
    l, t, r, b = container_box
    x0, x1 = max(0, math.floor(l)), min(w, math.ceil(r))
    y0, y1 = max(0, math.floor(t)), min(h, math.ceil(b))
    inside = liquid_mask[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((0, 0), dtype=bool)
    if not inside.any():
        return Reading(0.0, 1.0, "EMPTY", None)

    n = inside.shape[1]
    mid = inside[:, n // 4 : n - n // 4]  # middle 50 % of columns
    has = mid.any(axis=0)
    n_mid, n_with = mid.shape[1], int(has.sum())
    if n_with == 0:
        return Reading(None, 0.0, "LOW_CONFIDENCE", None)

    tops = mid[:, has].argmax(axis=0) + y0  # topmost liquid row per column, frame coordinates
    top_row = float(np.median(tops))
    pct, extrapolated = calib.level_at(top_row)
    confidence = n_with / n_mid * (1.0 if sam_confidence is None else sam_confidence)
    low = n_with * 10 < n_mid * 3  # fewer than 30 % of middle columns have liquid
    return Reading(pct, confidence, "LOW_CONFIDENCE" if low else "OK", top_row, extrapolated)


def camera_moved(box_now: Box | None, box_ref: Box | None, tol_px: float) -> bool:
    """True if any container-box edge shifted by more than tol_px, or a box is missing."""
    if box_now is None or box_ref is None:
        return True
    return max(abs(a - b) for a, b in zip(box_now, box_ref)) > tol_px
