"""Deterministic overflow interlock. It may only switch the pump permit OFF, and the trip latches.

step() is pure (injected time, no I/O) and can never emit WRITE_PERMIT_TRUE. Only reset(by_human=True)
can, and only once the trip condition has cleared. Re-arming needs a ResetGrant, which only reset_with_grant()
mints (single use; the permit module redeems it). Documented limit: a camera that goes blind and
THEN a PLC spoof begins is undetected, because CAMERA_BLIND/LOW_CONFIDENCE only alarm and clear the
mismatch timers; it never trips.
"""
import secrets
import threading
from dataclasses import dataclass, replace
from enum import Enum

from agentwarden.level import Reading

RATE_MIN_SAMPLES = 5
RATE_MIN_SPAN_FRAC = 0.9
HH_MIN_FRAMES = 2  # consecutive alertable frames at/above hh_limit before HIGH_HIGH trips (glare debounce)


class Action(str, Enum):
    WRITE_PERMIT_FALSE = "WRITE_PERMIT_FALSE"
    WRITE_PERMIT_TRUE = "WRITE_PERMIT_TRUE"  # produced by reset() only, never by step()
    ALARM_CAMERA_UNVERIFIED = "ALARM_CAMERA_UNVERIFIED"
    ALARM_PLC_STALE = "ALARM_PLC_STALE"
    ALARM_RESET_REFUSED = "ALARM_RESET_REFUSED"


@dataclass(frozen=True)
class Config:
    """Trip thresholds. The default T_pts=15 is PROVISIONAL: replace it with from_error_stats(p95) once the
    camera-vs-truth error (S2) is measured. Divergence is symmetric: camera below the PLC also trips."""

    T_pts: float = 15.0  # |camera - plc| above this counts as divergence
    N_s: float = 5.0  # ...continuously for this long
    W_s: float = 20.0  # rate window
    R_pts: float = 12.0  # camera rise over W_s that must be echoed by the PLC (>= R/3)
    plc_stale_s: float = 10.0
    hh_limit: float = 95.0  # high-high: trip on camera alone
    hyst_pts: float = 3.0  # divergence timer only clears once diff <= T - hyst

    @classmethod
    def from_error_stats(cls, p95_err: float) -> "Config":
        """T = max(3 x p95 camera error, 10) points."""
        return cls(T_pts=max(3 * p95_err, 10.0))


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
    hh_n: int = 0  # consecutive alertable frames at/above hh_limit


def initial_state() -> InterlockState:
    return InterlockState()


def unverified_reason(camera: Reading) -> str:
    """Why step() treats this reading as unverified (alarm only, timers cleared, never a trip). "" when it counts.
    OCCLUDED needs no special branch in step(): it is never alertable, exactly like LOW_CONFIDENCE."""
    if camera.alertable and camera.level_pct is not None and camera.status != "OCCLUDED":
        return ""
    return {"OCCLUDED": "occluded", "CAMERA_BLIND": "blind"}.get(camera.status, "low confidence")


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

    alertable = bool(camera.alertable) and camera.level_pct is not None and camera.status != "OCCLUDED"
    stale = plc_age_s > cfg.plc_stale_s
    if stale:
        actions.append(Action.ALARM_PLC_STALE)

    cam_hist, plc_hist, since = state.cam_hist, state.plc_hist, state.mismatch_since
    hh = diverged = trip_div = rate = False
    hh_n = 0
    if not alertable:
        actions.append(Action.ALARM_CAMERA_UNVERIFIED)
        cam_hist, plc_hist, since = (), (), None
    else:
        level = camera.level_pct
        cam_hist = _push(cam_hist, t, level, cfg)
        hh = level >= cfg.hh_limit
        hh_n = state.hh_n + 1 if hh else 0
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
    trip_hh = hh_n >= HH_MIN_FRAMES
    if not tripped and (trip_hh or trip_div or rate):
        tripped, trip_t = True, t
        reason = "HIGH_HIGH" if trip_hh else "RATE" if rate else "DIVERGENCE"
    if tripped:
        actions.append(Action.WRITE_PERMIT_FALSE)

    # Reset may only be accepted when the condition is positively gone: refuse while unverifiable.
    # (a running divergence timer in the hysteresis band also counts: the condition is not truly cleared)
    condition = hh or diverged or rate or since is not None or (tripped and (not alertable or stale))
    return InterlockState(tripped, reason, trip_t, since, cam_hist, plc_hist, condition, t, hh_n), actions


_MINT = object()  # private key: only this module can build a ResetGrant
_REDEEMED: set = set()  # ids of spent grants: a copy of a grant carries the same id, so it cannot be spent twice
_REDEEM_LOCK = threading.Lock()


class ResetGrant:
    """Proof that a human reset was accepted. Minted only by reset_with_grant(); single use.

    HONEST LIMIT: this is an in-process capability, NOT a security boundary. Python cannot stop code in the same
    process from forging one (object.__new__(ResetGrant) plus setting the slot, or calling ResetGrant(_MINT)).
    What it does: copy / deepcopy / pickle raise, `redeem` keeps a set of spent ids so a hand-made copy of a
    grant cannot be redeemed twice, and static tests keep every other module from importing the mint key.
    The real guarantee is on the wire: the Modbus server refuses any True write to the permit coil.
    """

    __slots__ = ("_id",)

    def __init__(self, key=None):
        if key is not _MINT:
            raise PermissionError("ResetGrant can only be issued by interlock.reset_with_grant")
        self._id = secrets.token_hex(16)

    def __copy__(self):
        raise PermissionError("a ResetGrant cannot be copied")

    def __deepcopy__(self, memo):
        raise PermissionError("a ResetGrant cannot be copied")

    def __reduce_ex__(self, protocol):
        raise PermissionError("a ResetGrant cannot be pickled")

    __reduce__ = __reduce_ex__


def redeem(grant) -> None:
    """Consume a grant (the permit module calls this before re-arming). Anything else is refused."""
    gid = getattr(grant, "_id", None) if isinstance(grant, ResetGrant) else None
    with _REDEEM_LOCK:
        if gid is None or gid in _REDEEMED:
            raise PermissionError("re-arm needs an unused ResetGrant from a successful human reset")
        _REDEEMED.add(gid)


def reset_with_grant(
    state: InterlockState, by_human: bool = False
) -> tuple[InterlockState, list[Action], ResetGrant | None]:
    """Clear a latched trip. Refused unless by a human AND the trip condition no longer holds."""
    if not state.tripped:
        return state, [], None
    if not by_human or state.condition_active:
        return state, [Action.ALARM_RESET_REFUSED], None
    return initial_state(), [Action.WRITE_PERMIT_TRUE], ResetGrant(_MINT)


def reset(state: InterlockState, by_human: bool = False) -> tuple[InterlockState, list[Action]]:
    """reset_with_grant() without the grant: clears the latch but cannot re-arm the permit."""
    return reset_with_grant(state, by_human)[:2]
