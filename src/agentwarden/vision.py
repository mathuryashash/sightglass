"""Frame -> Reading. Live path: 2 SAM calls (liquid, container) -> parse -> pick -> estimate.

Replay path (ReplayVision) returns recorded Readings so the whole pipeline runs with no network and no key.
"""
import bisect
import json
import logging
import random
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from agentwarden.level import ALERT_MIN_CONFIDENCE, Box, Calibration, Reading, camera_moved, estimate
from agentwarden.meta import _output_text
from agentwarden.sam import parse_sam
from agentwarden.select import pick

log = logging.getLogger(__name__)

CONCEPTS = ("orange liquid", "plastic bottle")  # (liquid, container)
CAMERA_MOVED_TOL_PX = 20.0  # container-box edge shift, in 640-wide frame pixels
_NO_CALIB = Calibration((0.0, 1.0), (100.0, 0.0))  # only used for CAMERA_BLIND, which ignores it


@dataclass(frozen=True)
class Analysis:
    reading: Reading
    box: Box | None = None
    mask: np.ndarray | None = None  # liquid mask, frame shape, for the dashboard overlay
    camera_moved: bool = False


def auto_from_box(box: Box, shoulder_frac: float = 0.22) -> Calibration:
    """0 % at the container-box bottom, 100 % at the shoulder (shoulder_frac of the box height below the top).

    Only approximate: the true full line depends on the bottle's shape and how it was filled, so this
    can be off by several points. Use explicit (row, pct) points for anything that matters.
    """
    _, top, _, bottom = box
    shoulder = top + shoulder_frac * (bottom - top)
    return Calibration.from_points([(bottom, 0.0), (shoulder, 100.0)])


def _detect(frame, client, concepts, run_id):
    """Run both SAM prompts; returns (box, liquid_mask, liquid_conf, container_conf) or None (no container)."""
    _, jpg, w, h = frame
    liquid_concept, container_concept = concepts
    liq = parse_sam(_output_text(client.sam_image(jpg, liquid_concept, run_id)), w, h)
    cont = parse_sam(_output_text(client.sam_image(jpg, container_concept, run_id)), w, h)
    picked = pick(cont, liq)
    if picked is None:
        return None
    box, mask, liq_conf = picked
    return box, mask, liq_conf, max(d.confidence or 0.0 for d in cont)


def _finish(det, calib, ref_box, tol_px) -> Analysis:
    if det is None:
        return Analysis(estimate(None, None, _NO_CALIB))
    box, mask, liq_conf, cont_conf = det
    reading = estimate(mask, box, calib, liq_conf, cont_conf)
    moved = ref_box is not None and camera_moved(box, ref_box, tol_px)
    if moved:  # the calibration no longer applies: never act on this level
        reading = replace(reading, status="LOW_CONFIDENCE", alertable=False)
    return Analysis(reading, box, mask, moved)


def analyze(frame, client, calib: Calibration, concepts=CONCEPTS, ref_box: Box | None = None,
            tol_px: float = CAMERA_MOVED_TOL_PX, run_id: str = "sightglass") -> Analysis:
    return _finish(_detect(frame, client, concepts, run_id), calib, ref_box, tol_px)


def read_level(frame, client, calib: Calibration, concepts=CONCEPTS, ref_box: Box | None = None,
               tol_px: float = CAMERA_MOVED_TOL_PX, run_id: str = "sightglass") -> Reading:
    """One frame -> Reading via 2 SAM calls on `client` (a MetaClient)."""
    return analyze(frame, client, calib, concepts, ref_box, tol_px, run_id).reading


class LiveVision:
    """Stateful live reader. With no calibration, the first container box defines both the auto
    calibration (approximate, see auto_from_box) and the reference box for camera-moved detection.

    A failed SAM call (network, refusal, parse error) yields a CAMERA_BLIND reading: unverifiable, never a trip.
    """

    def __init__(self, client, calib: Calibration | None = None, concepts=CONCEPTS,
                 tol_px: float = CAMERA_MOVED_TOL_PX, run_id: str = "sightglass",
                 ref_box: Box | None = None):
        self.client, self.calib, self.concepts = client, calib, concepts
        self.tol_px, self.run_id, self.ref_box = tol_px, run_id, ref_box
        self.last_error: str | None = None

    def read(self, frame, index=None) -> Analysis:
        try:
            det = _detect(frame, self.client, self.concepts, self.run_id)
        except Exception as e:  # ponytail: any failure = camera unverified; type name only, never the body
            self.last_error = type(e).__name__
            log.warning("vision call failed: %s", self.last_error)
            return Analysis(estimate(None, None, _NO_CALIB))
        self.last_error = None
        if det is not None:
            if self.ref_box is None:
                self.ref_box = det[0]
            if self.calib is None:
                self.calib = auto_from_box(det[0])
        return _finish(det, self.calib, self.ref_box, self.tol_px)


class LatestWorker:
    """Newest-frame-wins: a bounded queue of 1. While fn is running, a newer submit replaces any waiting item."""

    def __init__(self, fn, on_result):
        self._fn, self._on_result = fn, on_result
        self._cv = threading.Condition()
        self._slot, self._closing = None, False
        self.dropped = 0
        self.errors: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, item) -> None:
        with self._cv:
            if self._slot is not None:
                self.dropped += 1
            self._slot = (item,)
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._slot is None and not self._closing:
                    self._cv.wait()
                if self._slot is None:
                    return
                (item,), self._slot = self._slot, None
            try:
                self._on_result(self._fn(item))
            except Exception as e:
                self.errors.append(type(e).__name__)

    def close(self) -> None:
        """Finish the pending item (if any), then stop."""
        with self._cv:
            self._closing = True
            self._cv.notify()
        self._thread.join()


class ReplayVision:
    """Recorded readings, no network. `source`: a JSON path, or an already-loaded list / {"readings": [...]}.

    Each record: {"t"?, "level_pct", "confidence", "status", "top_row"?, "alertable"?}. Records with "t"
    are looked up by frame time (last record at or before it); otherwise by frame index.
    """

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            source = json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, dict):
            for key in ("readings", "results", "frames"):
                if isinstance(source.get(key), list):
                    source = source[key]
                    break
            else:
                raise ValueError("replay JSON needs a list or a 'readings' list")
        if not source:
            raise ValueError("replay JSON has no readings")
        self._recs = list(source)
        self._ts = [float(r["t"]) for r in self._recs] if all("t" in r for r in self._recs) else None
        self._calls = 0

    @classmethod
    def demo(cls) -> "ReplayVision":
        """Synthetic 54 s pour, 5 % -> 100 %, one reading per second (a stand-in for a recorded clip)."""
        rng = random.Random(54)
        recs = []
        for i in range(54):
            s = i / 53
            level = 5 + 95 * (3 * s * s - 2 * s**3) + rng.gauss(0, 0.6)
            recs.append({"t": float(i), "level_pct": min(100.0, max(0.0, level)), "confidence": 0.9,
                         "status": "OK"})
        return cls(recs)

    def times(self) -> list[float]:
        return list(self._ts) if self._ts is not None else [float(i) for i in range(len(self._recs))]

    def read(self, frame, index: int | None = None) -> Analysis:
        if index is None:
            if self._ts is not None:
                index = max(bisect.bisect_right(self._ts, frame[0]) - 1, 0)
            else:
                index, self._calls = self._calls, self._calls + 1
        r = self._recs[min(index, len(self._recs) - 1)]
        level = r.get("level_pct", r.get("level", r.get("pct")))
        conf = float(r.get("confidence", r.get("conf", 0.0 if level is None else 0.9)))
        status = r.get("status", "CAMERA_BLIND" if level is None else "OK")
        alertable = r.get("alertable", level is not None and status in ("OK", "EMPTY")
                          and conf >= ALERT_MIN_CONFIDENCE)
        return Analysis(Reading(None if level is None else float(level), conf, status, r.get("top_row"),
                                bool(r.get("extrapolated", False)), bool(alertable)))
