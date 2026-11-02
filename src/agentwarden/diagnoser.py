"""Advisory incident diagnosis by Muse Spark. SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM.

The model explains an incident to a person. Nothing in this module (or in advisor.py) can reach the pump
signal, the reset or the relay: it imports only meta (the model client) and vision (read-only recorded
readings), and a static test walks the import graph and the calls to keep it that way.

Tool use (the Spark API's native tool-calling format is not known here, so it is done in the JSON schema):
round 1 the model may return {"tool_request": [{"name": ..., "args": {...}}, ...]} instead of an answer; our code
runs at most 2 READ-ONLY lookups (get_series, recheck_frame), appends the results to the prompt, and asks once
more for the final answer (tool_request is not allowed in round 2). Two model calls at most.

A diagnosis is shown only if its evidence is grounded: every number it quotes must appear (within rounding) in
the incident data or a tool result. Anything else (refusal, timeout, bad JSON, bad schema, ungrounded) comes
back as a DiagnosisError and the UI says "diagnosis unavailable".
"""
import bisect
import json
import re
from collections import Counter
from dataclasses import dataclass

from agentwarden.meta import MetaError, MetaRefusal
from agentwarden.vision import ReplayVision

LABELS = ("SPOOF", "LEAK", "CAMERA_FAULT", "BENIGN", "UNSURE")
MAX_POINTS = 120
MAX_TOOL_CALLS = 2
EVIDENCE_MAX_N, EVIDENCE_MAX_LEN, NOTE_MAX, CHECK_MAX = 4, 120, 280, 160
DIVERGE_PTS = 10.0  # rule baseline: |plc - camera| at the end of the window that counts as "the sources disagree"
FLAT_BAND = 0.5  # a level that moved less than this is "flat"
UNVERIFIED_S = 10.0

_FINAL_PROPS = {
    "label": {"type": "string", "enum": list(LABELS)},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "evidence": {"type": "array", "minItems": 1, "maxItems": EVIDENCE_MAX_N,
                 "items": {"type": "string", "maxLength": EVIDENCE_MAX_LEN}},
    "operator_note": {"type": "string", "maxLength": NOTE_MAX},
    "suggested_check": {"type": "string", "maxLength": CHECK_MAX},
}
_REQUIRED = list(_FINAL_PROPS)
FINAL_SCHEMA = {"type": "object", "properties": _FINAL_PROPS, "required": _REQUIRED, "additionalProperties": False}
FIRST_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {**_FINAL_PROPS, "tool_request": {
        "type": "array", "maxItems": MAX_TOOL_CALLS,
        "items": {"type": "object", "properties": {
            "name": {"type": "string", "enum": ["get_series", "recheck_frame"]}, "args": {"type": "object"}}}}},
}


# ------------------------------------------------------------------ incident


def _pts(series):
    return [(float(t), None if v is None else float(v)) for t, v in series]


def _thin(pts):
    n = len(pts)
    if n <= MAX_POINTS:
        return pts
    keep = sorted({round(i * (n - 1) / (MAX_POINTS - 1)) for i in range(MAX_POINTS)})
    return [pts[i] for i in keep]


def _r(x, nd=1):
    return None if x is None else round(x, nd)


def _flat_tail(pts):
    """Seconds the series has stayed within FLAT_BAND, counted back from its newest point."""
    vals = [(t, v) for t, v in pts if v is not None]
    if not vals:
        return 0.0
    lo = hi = vals[-1][1]
    start = vals[-1][0]
    for t, v in reversed(vals):
        lo, hi = min(lo, v), max(hi, v)
        if hi - lo > FLAT_BAND:
            break
        start = t
    return vals[-1][0] - start


def _unverified_tail(evs):
    run = [e["t"] for e in evs]
    for i in range(len(evs) - 1, -1, -1):
        if "ALARM_CAMERA_UNVERIFIED" not in evs[i]["actions"]:
            run = run[i + 1:]
            break
    return run[-1] - run[0] if run else 0.0


def build_incident(events, plc_series, cam_series, window_s=60, limits=None):
    """Compact, JSON-able view of the last `window_s` seconds. Only whitelisted fields are copied, so nothing
    about how a fault was injected (scenario name, simulated truth) can reach the model."""
    plc, cam = _pts(plc_series), _pts(cam_series)
    evs = sorted(({"t": float(e["t"]), "status": str(e.get("status", "")), "pump_on": bool(e.get("pump_on")),
                   "tripped": bool(e.get("tripped")), "trip_reason": e.get("trip_reason"),
                   "actions": [str(a) for a in e.get("actions", ())]} for e in events), key=lambda e: e["t"])
    t_end = max([p[0] for p in plc] + [p[0] for p in cam] + [e["t"] for e in evs] + [0.0])
    lo = t_end - window_s
    plc, cam = _thin([p for p in plc if p[0] >= lo]), _thin([p for p in cam if p[0] >= lo])
    win = [e for e in evs if e["t"] >= lo]
    plc_r = [[round(t, 1), _r(v)] for t, v in plc]
    cam_r = [[round(t, 1), _r(v)] for t, v in cam]
    plc_v = [v for _, v in plc_r if v is not None]
    cam_v = [v for _, v in cam_r if v is not None]

    cam_t = [t for t, v in cam_r if v is not None]
    diffs = []
    for t, v in plc_r:
        i = bisect.bisect_right(cam_t, t) - 1
        if v is not None and i >= 0:
            diffs.append(abs(v - cam_v[i]))
    first_trip = next((e for e in evs if e["tripped"]), None)
    alarms = Counter(a for e in win for a in e["actions"] if a.startswith("ALARM_"))
    summary = {
        "plc_min": min(plc_v, default=None), "plc_max": max(plc_v, default=None),
        "cam_min": min(cam_v, default=None), "cam_max": max(cam_v, default=None),
        "plc_net_change": _r(plc_v[-1] - plc_v[0]) if plc_v else None,
        "camera_net_change": _r(cam_v[-1] - cam_v[0]) if cam_v else None,
        "plc_flat_s": _r(_flat_tail(plc_r)),
        "max_abs_diff": _r(max(diffs, default=None)), "end_abs_diff": _r(diffs[-1]) if diffs else None,
        "cam_missing_frac": _r(sum(v is None for _, v in cam_r) / len(cam_r), 2) if cam_r else 0.0,
        "camera_bad_frac": _r(sum(e["status"] not in ("OK", "EMPTY") for e in win) / len(win), 2) if win else 0.0,
        "pump_on_frac": _r(sum(e["pump_on"] for e in win) / len(win), 2) if win else 0.0,
        "camera_unverified_s": _r(_unverified_tail(win)),
    }
    inc = {"window_s": window_s, "t_end": round(t_end, 1), "plc": plc_r, "camera": cam_r,
           "camera_status": dict(Counter(e["status"] for e in win)), "alarms": dict(alarms),
           "trip": {"reason": first_trip["trip_reason"], "t": first_trip["t"]} if first_trip else None,
           "summary": summary}
    if limits:
        inc["limits"] = {k: float(v) for k, v in limits.items()}
    return inc


def rule_label(inc):
    """Deterministic baseline (no model): a cross-check, not a verdict."""
    s = inc["summary"]

    def num(k):
        return s.get(k) or 0.0  # None counts as 0

    if num("camera_bad_frac") >= 0.5 or num("cam_missing_frac") >= 0.5 or num("camera_unverified_s") >= UNVERIFIED_S:
        return "CAMERA_FAULT"
    cam_range = (s["cam_max"] - s["cam_min"]) if s.get("cam_max") is not None else 0.0
    if num("end_abs_diff") >= DIVERGE_PTS or (num("plc_flat_s") >= 20 and cam_range >= DIVERGE_PTS):
        return "SPOOF"
    if (num("max_abs_diff") <= 8 and num("pump_on_frac") >= 0.5 and num("plc_net_change") <= -10
            and num("camera_net_change") <= -10):
        return "LEAK"
    return "UNSURE" if inc.get("trip") else "BENIGN"


# ------------------------------------------------------------------ tools (read-only)


class Tools:
    """The only two lookups the model may request. They read the incident's own series and recorded camera
    readings; they have no handle on anything else."""

    NAMES = ("get_series", "recheck_frame")

    def __init__(self, inc, readings=None):
        self._series = {"plc": inc["plc"], "camera": inc["camera"]}
        self._replay = ReplayVision(readings) if readings else None

    @staticmethod
    def _num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def run(self, name, args):
        args = args if isinstance(args, dict) else {}
        if name == "get_series":
            lo, hi, which = args.get("from_s"), args.get("to_s"), args.get("name")
            if which not in self._series or not (self._num(lo) and self._num(hi)):
                return {"error": "get_series needs name plc|camera and numeric from_s, to_s"}
            return {"name": which, "points": [p for p in self._series[which] if lo <= p[0] <= hi][:MAX_POINTS]}
        if name == "recheck_frame":
            t = args.get("t")
            if not self._num(t):
                return {"error": "recheck_frame needs numeric t"}
            if self._replay is None:
                return {"error": "no recorded camera readings"}
            r = self._replay.read((float(t),)).reading
            return {"asked_t": round(float(t), 1), "level_pct": _r(r.level_pct), "confidence": _r(r.confidence, 2),
                    "status": r.status}
        return {"error": f"unknown tool, expected one of {list(self.NAMES)}"}

    def run_requests(self, req):
        """Run at most MAX_TOOL_CALLS requests; the rest are dropped."""
        items = [req] if isinstance(req, dict) else req if isinstance(req, list) else []
        out = []
        for it in items[:MAX_TOOL_CALLS]:
            it = it if isinstance(it, dict) else {}
            name, args = it.get("name"), it.get("args")
            out.append({"name": name if name in self.NAMES else "invalid", "args": args if isinstance(args, dict) else {},
                        "result": self.run(name, args)})
        return out


# ------------------------------------------------------------------ prompt


FRAMING = (
    "You are a diagnostic assistant for a tank-level monitor (a SIMULATED plant on one computer). "
    "You advise a person; you cannot control anything. You have no access to the pump, the pump permit, the "
    "reset or any setting, and nothing you write changes them. A fixed safety rule has already acted on its "
    "own. Your only job is to help the person understand what probably happened."
)
TASK = """Two independent sources report the tank level in percent: "plc" (what the controller reports) and
"camera" (a reading from camera images; null means no usable reading). Times are seconds.

INCIDENT_JSON: {incident}

Choose one label:
- SPOOF: the plc level no longer follows the real level (it sits at one value, drifts away slowly, or repeats old
  values) while the camera keeps tracking a real change.
- LEAK: both sources agree and the level falls although the pump ran.
- CAMERA_FAULT: the camera is the odd one out (blind or low-confidence statuses, missing readings, jumps) while the
  plc looks normal.
- BENIGN: the sources agree and nothing is unusual.
- UNSURE: the data does not support a clear answer.

Rules:
- Use only the data in this message. Never invent values or times.
- evidence: 1 to {en} short strings, at most {el} characters each. Each must quote at least one number or time
  that appears in the data (or in a tool result).
- operator_note: plain language for a non-expert, at most {nl} characters.
- suggested_check: one manual or physical check, at most {cl} characters.
- confidence: 0 to 1. Prefer UNSURE with a low confidence when the data is thin.
- Reply with one JSON object and nothing else."""
TOOL_HELP = """
Optional, before answering: you may ask for up to {n} read-only lookups by replying with only
{{"tool_request": [{{"name": "get_series", "args": {{"name": "plc" or "camera", "from_s": number, "to_s": number}}}},
{{"name": "recheck_frame", "args": {{"t": number}}}}]}}. Our code runs them and you then answer once. If you do not
need them, answer directly."""
TOOL_RESULTS = """
TOOL_RESULTS (read-only lookups run by our code): {results}
Now give the final answer as the JSON object described above. A tool_request is not allowed any more."""


def _dumps(x):
    return json.dumps(x, separators=(",", ":"), sort_keys=True)


def _prompt(inc):
    return FRAMING + "\n\n" + TASK.format(incident=_dumps(inc), en=EVIDENCE_MAX_N, el=EVIDENCE_MAX_LEN,
                                          nl=NOTE_MAX, cl=CHECK_MAX)


# ------------------------------------------------------------------ result + validation


@dataclass(frozen=True)
class Diagnosis:
    label: str
    confidence: float
    evidence: tuple
    operator_note: str
    suggested_check: str
    tool_calls: tuple = ()


class DiagnosisError(Exception):
    """Why there is no diagnosis. The UI shows "diagnosis unavailable"; the safety rule is unaffected."""

    def __init__(self, reason, detail=""):
        super().__init__(f"diagnosis unavailable ({reason})")
        self.reason, self.detail = reason, detail


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate(obj):
    if not isinstance(obj, dict) or set(obj) != set(_REQUIRED):
        raise DiagnosisError("schema", "wrong keys")
    ev = obj["evidence"]
    ok = (obj["label"] in LABELS and _is_num(obj["confidence"]) and 0 <= obj["confidence"] <= 1
          and isinstance(ev, list) and 1 <= len(ev) <= EVIDENCE_MAX_N
          and all(isinstance(x, str) and 0 < len(x) <= EVIDENCE_MAX_LEN for x in ev)
          and isinstance(obj["operator_note"], str) and 0 < len(obj["operator_note"]) <= NOTE_MAX
          and isinstance(obj["suggested_check"], str) and 0 < len(obj["suggested_check"]) <= CHECK_MAX)
    if not ok:
        raise DiagnosisError("schema", "bad value")
    return obj


_NUM = re.compile(r"\d+(?:\.\d+)?")


def _collect(x, out):
    if _is_num(x):
        out.add(abs(float(x)))
    elif isinstance(x, dict):
        for v in x.values():
            _collect(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _collect(v, out)


def _grounded(evidence, data):
    """Every evidence line quotes a number, and each quoted number is in `data` within its own rounding."""
    for line in evidence:
        quoted = _NUM.findall(line)
        if not quoted:
            return False
        for q in quoted:
            places = len(q.split(".")[1]) if "." in q else 0
            tol = max(0.5 * 10 ** -places, 0.05) + 1e-9
            if not any(abs(float(q) - x) <= tol for x in data):
                return False
    return True


def _from_exception(e):
    if isinstance(e, MetaRefusal):
        reason = "refusal"
    elif isinstance(e, MetaError):
        reason = "api"
    elif isinstance(e, TimeoutError) or "timeout" in type(e).__name__.lower():
        reason = "timeout"
    elif isinstance(e, ValueError):  # json.JSONDecodeError from the reply text
        reason = "parse"
    else:
        reason = "unavailable"
    return DiagnosisError(reason, type(e).__name__)  # the type only: never a message that could carry secrets


def diagnose(inc, client, run_id, tools=None):
    """Diagnosis, or a DiagnosisError (never raises for model/API problems). `client` is a MetaClient."""
    tools = tools or Tools(inc)
    base, calls = _prompt(inc), []
    try:
        reply = client.spark_json(base + TOOL_HELP.format(n=MAX_TOOL_CALLS), FIRST_SCHEMA, run_id)
        if isinstance(reply, dict) and "tool_request" in reply:
            calls = tools.run_requests(reply["tool_request"])
            reply = client.spark_json(base + TOOL_RESULTS.format(results=_dumps(calls)), FINAL_SCHEMA, run_id)
            if isinstance(reply, dict) and "tool_request" in reply:
                return DiagnosisError("tool", "second tool request")
        obj = _validate(reply)
    except DiagnosisError as e:
        return e
    except Exception as e:  # refusal, timeout, HTTP error, bad JSON, cassette miss, spend cap ...
        return _from_exception(e)
    data = set()
    _collect(inc, data)
    _collect([c["result"] for c in calls], data)
    if not _grounded(obj["evidence"], data):
        return DiagnosisError("ungrounded", "evidence quotes a number that is not in the data")
    return Diagnosis(obj["label"], float(obj["confidence"]), tuple(obj["evidence"]), obj["operator_note"],
                     obj["suggested_check"], tuple({"name": c["name"], "args": c["args"]} for c in calls))
