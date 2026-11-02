"""Deterministic overflow interlock. It may only switch the pump permit OFF, and the trip latches.

step() is pure (injected time, no I/O) and can never emit WRITE_PERMIT_TRUE. Only reset(by_human=True)
can, and only once the trip condition has cleared. Documented limit: a camera that goes blind and
THEN a PLC spoof begins is undetected, because CAMERA_BLIND/LOW_CONFIDENCE only alarm and clear the
mismatch timers; it never trips.
"""
from dataclasses import dataclass, replace
from enum import Enum

from agentwarden.level import Reading

RATE_MIN_SAMPLES = 5
RATE_MIN_SPAN_FRAC = 0.9


class Action(str, Enum):
    WRITE_PERMIT_FALSE = "WRITE_PERMIT_FALSE"
    WRITE_PERMIT_TRUE = "WRITE_PERMIT_TRUE"  # produced by reset() only, never by step()
    ALARM_CAMERA_UNVERIFIED = "ALARM_CAMERA_UNVERIFIED"
    ALARM_PLC_STALE = "ALARM_PLC_STALE"
    ALARM_RESET_REFUSED = "ALARM_RESET_REFUSED"


@dataclass(frozen=True)
class Config:
    T_pts: float = 15.0  # |camera - plc| above this counts as divergence
    N_s: float = 5.0  # ...continuously for this long
    W_s: float = 20.0  # rate window
    R_pts: float = 12.0  # camera rise over W_s that must be echoed by the PLC (>= R/3)
    plc_stale_s: float = 10.0
    hh_limit: float = 95.0  # high-high: trip on camera alone
    hyst_pts: float = 3.0  # divergence timer only clears once diff <= T - hyst


@dataclass(frozen=True)
class InterlockState:
    tripped: bool = False
    trip_reason: str | None = None
    trip_t: float | None = None
    mismatch_since: float | None = None
    cam_hist: tuple = ()
    plc_hist: tuple = ()
    condition_active: bool = False
    last_t: float | None = None


def initial_state() -> InterlockState:
    return InterlockState()


def _slope(hist) -> float:
    n = len(hist)
    mt = sum(t for t, _ in hist) / n
    mv = sum(v for _, v in hist) / n
    den = sum((t - mt) ** 2 for t, _ in hist)
    return sum((t - mt) * (v - mv) for t, v in hist) / den if den else 0.0


def _rate_mismatch(cam_hist, plc_hist, cfg: Config) -> bool:
    if len(cam_hist) < RATE_MIN_SAMPLES or len(plc_hist) < RATE_MIN_SAMPLES:
        return False
    if cam_hist[-1][0] - cam_hist[0][0] < RATE_MIN_SPAN_FRAC * cfg.W_s:
        return False
    cam_rise = _slope(cam_hist) * cfg.W_s
    plc_move = abs(_slope(plc_hist)) * cfg.W_s
    return cam_rise >= cfg.R_pts and plc_move < cfg.R_pts / 3


def _push(hist, t, v, cfg: Config):
    return tuple(p for p in hist + ((t, v),) if p[0] >= t - cfg.W_s)


def step(state: InterlockState, camera: Reading, plc_level: float, plc_age_s: float, t: float,
         cfg: Config) -> tuple[InterlockState, list[Action]]:
    actions: list[Action] = []
    if state.last_t is not None and t < state.last_t:  # clock went backwards: drop stale timers
        state = replace(state, mismatch_since=None, cam_hist=(), plc_hist=())

    alertable = bool(camera.alertable) and camera.level_pct is not None
    stale = plc_age_s > cfg.plc_stale_s
    if stale:
        actions.append(Action.ALARM_PLC_STALE)

    cam_hist, plc_hist, since = state.cam_hist, state.plc_hist, state.mismatch_since
    hh = diverged = trip_div = rate = False
    if not alertable:
        actions.append(Action.ALARM_CAMERA_UNVERIFIED)
        cam_hist, plc_hist, since = (), (), None
    else:
        level = camera.level_pct
        cam_hist = _push(cam_hist, t, level, cfg)
        hh = level >= cfg.hh_limit
        if stale:  # PLC value untrusted: neither feeds nor keeps the mismatch rules
            plc_hist, since = (), None
        else:
            plc_hist = _push(plc_hist, t, plc_level, cfg)
            diff = abs(level - plc_level)
            diverged = diff > cfg.T_pts
            if diverged:
                since = t if since is None else since
            elif diff <= cfg.T_pts - cfg.hyst_pts:
                since = None
            trip_div = since is not None and t - since >= cfg.N_s
            rate = _rate_mismatch(cam_hist, plc_hist, cfg)

    tripped, reason, trip_t = state.tripped, state.trip_reason, state.trip_t
    if not tripped and (hh or trip_div or rate):
        tripped, trip_t = True, t
        reason = "HIGH_HIGH" if hh else "RATE" if rate else "DIVERGENCE"
    if tripped:
        actions.append(Action.WRITE_PERMIT_FALSE)

    # Reset may only be accepted when the condition is positively gone: refuse while unverifiable.
    condition = hh or diverged or rate or (tripped and (not alertable or stale))
    return InterlockState(tripped, reason, trip_t, since, cam_hist, plc_hist, condition, t), actions


def reset(state: InterlockState, by_human: bool = False) -> tuple[InterlockState, list[Action]]:
    """Clear a latched trip. Refused unless by a human AND the trip condition no longer holds."""
    if not state.tripped:
        return state, []
    if not by_human or state.condition_active:
        return state, [Action.ALARM_RESET_REFUSED]
    return initial_state(), [Action.WRITE_PERMIT_TRUE]
