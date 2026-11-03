"""Offline evaluation of SightGlass on recorded data and the simulator. No network, no model calls, no keys.

    python -m agentwarden.evaluate --out eval/          (full run, under 3 minutes)
    python -m agentwarden.evaluate --out /tmp/e --tiny   (smoke run, a few seconds)

Writes RESULTS.md (prose, every number comes from results.json), results.json and three SVG charts.

HONESTY NOTES the script keeps: there is no tape-measured ground truth for the real pour. Level quality is judged
by a proxy (a steady pour never goes down) and by approximate eyeballed levels for the still photos. Detection and
false-alarm numbers come from a SIMULATED tank with a gaussian fake camera, so they say how the rules behave, not
how a real plant behaves. The diagnoser section keeps the recorded model calls apart from the rule baseline.
"""
import argparse
import contextlib
import html
import io
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from agentwarden import interlock
from agentwarden import evaluate_diag
from agentwarden.fault_injector import FaultInjector
from agentwarden.level import Reading
from agentwarden.permit import PermitOutput
from agentwarden.pipeline import Pipeline
from agentwarden.plc_sim import PlcController, TankSim
from agentwarden.runner import DRIFT_K, REPLAY_WINDOW_S, fake_camera
from agentwarden.vision import ReplayVision, auto_from_box

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "real"
FULL = dict(detect_seeds=200, false_seeds=200, false_minutes=30.0, diag_per_class=200, horizon_s=700.0)
TINY = dict(detect_seeds=4, false_seeds=3, false_minutes=3.0, diag_per_class=6, horizon_s=300.0)

FIRST_FRAME = 9  # frames 0-8 of the pour are a hand and bottle entering the shot
SATURATED = 99.9  # pairs touching a clamped 100 reading say nothing about noise
STILL_APPROX = (100.0, 77.0, 59.0, 44.0, 34.0, 24.0, 0.0)  # photographer's eyeballed levels, not measured
STILL_REF = "approximate reference, no tape"
CAMERA_NEAR_FULL = 95.0
CATCH_TARGET = 0.95

# (kind, sizes): size = freeze offset below truth (pts), step size (pts), drift rate (pts/s), replay has no size
DETECT_GRID = (
    ("freeze", (0.0, 5.0, 10.0, 15.0, 20.0)),
    ("step", (5.0, 10.0, 15.0, 20.0, 30.0)),
    ("drift", (0.2, 0.4, 0.6, 1.0)),
    ("slow_drift", (0.03, 0.05, 0.1)),
    ("replay", (0.0,)),
)
UNITS = {"freeze": "pts offset", "step": "pts", "drift": "pts/s", "slow_drift": "pts/s", "replay": "-"}


# ---------------------------------------------------------------- small helpers

def _pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None


def _med(xs):
    return statistics.median(xs) if xs else None


def _f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _quiet_injector(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):  # FaultInjector prints a banner on every construction
        return FaultInjector(*a, **k)


def _recs():
    return json.loads((DEMO / "pour_replay.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. level reading quality (real data)

def _series_stats(levels):
    """levels: one reading per frame (None = no reading). Backward steps, and noise = median |delta - local trend|
    where the local trend is the median of the 4 neighbouring deltas. Pairs touching a clamped 100 are left out of
    the noise figure."""
    back, deltas = [], []
    for a, b in zip(levels, levels[1:]):
        if a is None or b is None:
            continue
        if b < a:
            back.append(a - b)
        if max(a, b) < SATURATED:
            deltas.append(b - a)
    resid = []
    for k, d in enumerate(deltas):
        nb = deltas[max(0, k - 2):k] + deltas[k + 1:k + 3]
        if nb:
            resid.append(abs(d - statistics.median(nb)))
    return {"mono_violations": len(back), "mono_violations_gt2": sum(b > 2.0 for b in back),
            "max_backward_step": max(back, default=0.0), "noise_median": _med(resid),
            "no_reading_frames": sum(v is None for v in levels), "n_noise_pairs": len(resid)}


def _status_frac(recs):
    c = Counter(r["status"] for r in recs)
    return {k: c.get(k, 0) / len(recs) for k in ("OK", "EMPTY", "CAMERA_BLIND", "LOW_CONFIDENCE")}


def level_quality(recs):
    lv = [r["level_pct"] for r in recs]
    st = _series_stats(lv[FIRST_FRAME:])
    sigma_est = st["noise_median"] / (0.6745 * math.sqrt(2))  # delta of two readings has sd sigma*sqrt(2)
    return {**st, "n_frames": len(recs), "first_frame": FIRST_FRAME, "status_frac": _status_frac(recs),
            "status_frac_after_entry": _status_frac(recs[FIRST_FRAME:]),
            "sigma_est": sigma_est, "sigma_used": float(max(3, math.ceil(sigma_est))), "levels": lv}


def stills_quality():
    rows = json.loads((DEMO / "stills" / "spike_results.json").read_text(encoding="utf-8"))
    out = [{"photo": r["photo"], "status": r["status"], "reading": float(r["level"]), "approx_ref": ref,
            "abs_error": abs(float(r["level"]) - ref)} for r, ref in zip(rows, STILL_APPROX)]
    errs = [r["abs_error"] for r in out]
    return {"reference": f"{STILL_REF} (photographer's eyeballed levels)", "rows": out,
            "mean_abs_diff": statistics.mean(errs), "max_abs_diff": max(errs)}


# ---------------------------------------------------------------- simulator core

def _run(kind, size, seed, sigma, horizon_s, start=None, trace=None, stop=True, drain_lps=2.0, level0=20.0,
         pump0=False, blind=None, cfg=None):
    """One closed-loop run, 1 s ticks: TankSim + PlcController + fault + noisy fake camera + interlock.
    kind: benign | freeze | drift | slow_drift | replay | step. blind=(a, b, status): camera unusable on [a, b).
    The run ends `horizon_s` after the fault starts (or horizon_s in total without one), on the first trip when
    `stop`, or on the first overflow. `trace` gets (t, plc, camera, event) per tick."""
    cfg = cfg or interlock.Config()
    rng = random.Random(seed)
    if start is None and kind != "benign":
        start = float(round(20 + 60 * rng.random()))
    scenario = {"freeze": "freeze", "drift": "drift", "slow_drift": "drift", "replay": "replay"}.get(kind, "benign")
    inj = _quiet_injector(scenario, start_t=1e9 if start is None else start,
                          drift_k=size if "drift" in kind else DRIFT_K, replay_window_s=REPLAY_WINDOW_S)
    offset = size if kind in ("freeze", "step") else 0.0
    camera = fake_camera(sigma, rng)
    tank = TankSim(capacity_l=1000, fill_rate_lps=9, drain_rate_lps=drain_lps, level_pct=level0)
    plc = PlcController(low=30.0, high=80.0, pump_on=pump0)
    out, state = PermitOutput(None, None), interlock.initial_state()
    res = {"start": start, "trip_t": None, "reason": None, "overflow_t": None, "truth_at_trip": None}
    for i in range(int(horizon_s if start is None else start + horizon_s)):
        t = float(i)
        truth = tank.level_pct
        reported = inj.reported(truth, t)
        if start is not None and t >= start:
            reported = max(0.0, reported - offset)
        reading = camera(truth, t)
        if blind and blind[0] <= t < blind[1]:
            if blind[2] == "CAMERA_BLIND":
                reading = Reading(None, 0.0, "CAMERA_BLIND", None, False, False)
            else:
                reading = Reading(reading.level_pct, 0.3, "LOW_CONFIDENCE", None, False, False)
        plc_level = round(reported, 1)
        state, actions = interlock.step(state, reading, plc_level, 0.0, t, cfg)
        out.apply(actions)
        if state.tripped and res["trip_t"] is None:
            res.update(trip_t=state.trip_t, reason=state.trip_reason, truth_at_trip=truth)
        pump = plc.update(plc_level, out.permit)
        if trace is not None:
            trace.append((t, plc_level, reading.level_pct,
                          {"t": t, "status": reading.status, "pump_on": pump, "tripped": state.tripped,
                           "trip_reason": state.trip_reason, "actions": [a.value for a in actions]}))
        if res["trip_t"] is not None and stop:
            break
        tank.step(1.0, pump)
        if tank.overflowed:
            res["overflow_t"] = t
            if res["trip_t"] is None:
                break
    return res


# ---------------------------------------------------------------- 2. detection

def detect_one(kind, size, seeds, sigma, horizon_s, cfg=None):
    runs = [_run(kind, size, s, sigma, horizon_s, cfg=cfg) for s in seeds]
    caught = [r for r in runs if r["trip_t"] is not None and r["overflow_t"] is None]
    ttd = [r["trip_t"] - r["start"] for r in caught]
    n = len(runs)
    return {"kind": kind, "size": size, "unit": UNITS[kind], "n": n, "detected": len(caught), "caught": len(caught),
            "missed": n - len(caught), "overflow_reached": sum(r["overflow_t"] is not None for r in runs),
            "no_trip_no_overflow": sum(r["trip_t"] is None and r["overflow_t"] is None for r in runs),
            "ttd_median_s": _med(ttd), "ttd_p90_s": _pct(ttd, 0.9),
            "truth_at_trip_max": max((r["truth_at_trip"] for r in caught), default=None),
            "rules": dict(Counter(r["reason"] for r in caught)), "caught_frac": len(caught) / n}


def detection(cfg, sigma):
    seeds = range(1, cfg["detect_seeds"] + 1)
    rows = [detect_one(k, s, seeds, sigma, cfg["horizon_s"]) for k, sizes in DETECT_GRID for s in sizes]

    def floor(kinds):
        ok = sorted(r["size"] for r in rows if r["kind"] in kinds and r["caught_frac"] >= CATCH_TARGET)
        return ok[0] if ok else None

    groups = {"freeze": ("freeze",), "step": ("step",), "drift": ("drift", "slow_drift")}
    return {"sigma": sigma, "rows": rows, "floors": {k: floor(v) for k, v in groups.items()},
            "smallest_tested": {k: min(r["size"] for r in rows if r["kind"] in v) for k, v in groups.items()}}


# ---------------------------------------------------------------- 3. false alarms

def false_alarms(cfg, sigma):
    total_s, trips, exposure = cfg["false_minutes"] * 60.0, 0, 0.0
    reasons = Counter()
    for s in range(1, cfg["false_seeds"] + 1):
        r = _run("benign", 0.0, 10_000 + s, sigma, total_s)
        if r["trip_t"] is not None:
            trips += 1
            reasons[r["reason"]] += 1
        exposure += (r["trip_t"] if r["trip_t"] is not None else total_s) / 3600.0
    upper = 3.0 / exposure if trips == 0 else (trips + 1.96 * math.sqrt(trips) + 1) / exposure
    method = "rule of three (3 / hours)" if trips == 0 else "approximate: (n + 1.96 sqrt(n) + 1) / hours"
    return {"n_runs": cfg["false_seeds"], "minutes_each": cfg["false_minutes"], "n_false_trips": trips,
            "exposure_hours": exposure, "per_hour": trips / exposure, "upper95_per_hour": upper,
            "upper95_method": method, "reasons": dict(reasons), "sigma": sigma}


# ---------------------------------------------------------------- 4/5. real replay through Pipeline

def _with_blind(recs, span):
    out = [dict(r) for r in recs]
    if span:
        for r in out:
            if span[0] <= r["t"] < span[1]:
                r.update(level_pct=None, level=None, confidence=0.0, conf=0.0, status="CAMERA_BLIND",
                         alertable=False, top_row=None)
    return out


def _replay_run(recs, scenario, fault_start, blind=None, advice=None, **vision_kw):
    vision = ReplayVision(_with_blind(recs, blind), **vision_kw)
    with contextlib.redirect_stdout(io.StringIO()):
        pl = Pipeline(vision, scenario, float(fault_start), advice=advice)
    evs = [pl.tick((r["t"],)) for r in recs]
    pl.close()
    return pl, evs


def _replay_run_prompts_as_recorded(recs, scenario, fault_start, advice=None):
    """The recorded Spark calls were keyed to the prompts the pre-S6b readings produced, so section 6 replays with the
    S6b occluder check and plausibility guard off (the guard drops pour frame t = 2, which changes the prompt)."""
    return _replay_run(recs, scenario, fault_start, advice=advice, **OFF)


def _t95(recs):
    return next(r["t"] for r in recs if r["level_pct"] is not None and r["level_pct"] >= CAMERA_NEAR_FULL)


def replay_runs(recs):
    t95 = _t95(recs)
    by_t = {r["t"]: r for r in recs}
    rows = []
    for sc in ("freeze", "drift", "replay"):
        for fs in (12, 20, 30):
            pl, _ = _replay_run(recs, sc, fs)
            s = pl.summary
            rows.append({"scenario": sc, "fault_start": fs, "tripped": s.tripped, "trip_t": s.trip_t,
                         "trip_reason": s.trip_reason,
                         "camera_at_trip": by_t[s.trip_t]["level_pct"] if s.tripped else None,
                         "gap_at_trip": s.diff_at_trip,
                         "before_95": bool(s.tripped and s.trip_t < t95),
                         "margin_s": (t95 - s.trip_t) if s.tripped else None})
    pl, _ = _replay_run(recs, "benign", 0)
    return {"camera_first_ge_95_t": t95, "rows": rows, "n_before_95": sum(r["before_95"] for r in rows),
            "benign": {"tripped": pl.summary.tripped, "trip_t": pl.summary.trip_t,
                       "trip_reason": pl.summary.trip_reason}}


def _blind_case(recs, scenario, fault_start, span):
    pl, evs = _replay_run(recs, scenario, fault_start, blind=span)
    s = pl.summary
    unver = [e["t"] for e in evs if "ALARM_CAMERA_UNVERIFIED" in e["actions"] and span[0] <= e["t"] < span[1]]
    return {"scenario": scenario, "fault_start": fault_start, "blind_span": list(span), "tripped": s.tripped,
            "trip_t": s.trip_t, "trip_reason": s.trip_reason, "unverified_ticks": len(unver),
            "plc_end": evs[-1]["plc"], "hidden_camera_end": recs[-1]["level_pct"],
            "gap_end": recs[-1]["level_pct"] - evs[-1]["plc"]}


def blind_runs(recs):
    end = recs[-1]["t"] + 1
    return {"blind_benign": _blind_case(recs, "benign", 0, (10.0, end)),
            "blind_then_spoof": _blind_case(recs, "freeze", 12, (10.0, end)),
            "short_blind_then_spoof": _blind_case(recs, "freeze", 12, (10.0, 16.0)),
            "blind_recovers_late": _blind_case(recs, "freeze", 12, (10.0, 40.0))}


# ---------------------------------------------------------------- 6. diagnoser

# ---------------------------------------------------------------- 6. diagnoser: see evaluate_diag.py (S5 import rule)

# ---------------------------------------------------------------- 7. colour-threshold baseline

H_LO, H_HI, S_MIN, V_MIN = 14, 36, 110, 100  # PIL HSV scale 0..255: hue about 20-51 deg (orange to yellow-orange)
ROW_FRAC = 0.15  # a row counts as liquid when this share of the container width passes the colour test


def colour_top_row(img, box):
    """Naive reader: orange pixels inside the container box; rows with enough of them form runs; the run that
    reaches the lowest orange row gives the surface (top) row. None when nothing orange is found."""
    w, h = img.size
    left, top, right, bottom = (int(round(v)) for v in box)
    left, top, right, bottom = max(left, 0), max(top, 0), min(right, w), min(bottom, h)
    hsv = np.asarray(img.convert("RGB").convert("HSV"))[top:bottom, left:right].astype(int)
    mask = (hsv[..., 0] >= H_LO) & (hsv[..., 0] <= H_HI) & (hsv[..., 1] >= S_MIN) & (hsv[..., 2] >= V_MIN)
    rows = mask.mean(axis=1) >= ROW_FRAC if mask.size else np.zeros(0, dtype=bool)
    if not rows.any():
        return None
    r = int(np.flatnonzero(rows)[-1])
    while r > 0 and rows[r - 1]:
        r -= 1
    return top + r


def colour_level(img, box):
    row = colour_top_row(img, box)
    return None if row is None else auto_from_box(tuple(box)).level_at(row)[0]


def baseline(recs):
    sam, col = [r["level_pct"] for r in recs], []
    for r in recs:
        with Image.open(DEMO / "pour_frames" / r["photo"]) as img:
            col.append(colour_level(img, r["box"]))
    both = [abs(a - b) for a, b in zip(sam[FIRST_FRAME:], col[FIRST_FRAME:]) if b is not None]
    return {"n_frames": len(recs), "first_frame": FIRST_FRAME,
            "calibration": "Both readers use the SAM container box of the same frame (auto_from_box); the colour "
                           "reader has no container detector, so only the liquid-surface localisation differs",
            "thresholds": {"hue_pil_0_255": [H_LO, H_HI], "sat_min": S_MIN, "val_min": V_MIN, "row_frac": ROW_FRAC},
            "sam": _series_stats(sam[FIRST_FRAME:]), "colour": _series_stats(col[FIRST_FRAME:]),
            "mean_abs_diff_sam_vs_colour": statistics.mean(both) if both else None,
            "series": {"sam": sam, "colour": col}}


# ---------------------------------------------------------------- charts (hand-made SVG)

COLOURS = ("#2a6fdb", "#d45500", "#2e8b57", "#8a2be2", "#6b7280")


def _svg(path, title, w, h, body):
    t = html.escape(title)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                    f'role="img" aria-label="{t}" font-family="sans-serif" font-size="12">'
                    f'<rect width="{w}" height="{h}" fill="white"/>'
                    f'<text x="{w / 2}" y="18" text-anchor="middle" font-size="14">{t}</text>{body}</svg>',
                    encoding="utf-8")


def _lineplot(path, title, xlabel, ylabel, series, vlines=(), hlines=(), xmax=1.0, ymax=100.0):
    W, H, L, R, T, B = 640, 360, 56, 20, 34, 70
    px = lambda x: L + (W - L - R) * x / xmax  # noqa: E731
    py = lambda y: H - B - (H - B - T) * y / ymax  # noqa: E731
    g = [f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - B - T}" fill="none" stroke="#999"/>']
    for i in range(6):
        y, x = ymax * i / 5, xmax * i / 5
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{py(y)}" y2="{py(y)}" stroke="#eee"/>'
                 f'<text x="{L - 6}" y="{py(y) + 4}" text-anchor="end">{y:.0f}</text>'
                 f'<text x="{px(x)}" y="{H - B + 16}" text-anchor="middle">{x:.0f}</text>')
    for y, label in hlines:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{py(y)}" y2="{py(y)}" stroke="#c0392b" stroke-dasharray="5 3"/>'
                 f'<text x="{W - R - 4}" y="{py(y) - 4}" text-anchor="end" fill="#c0392b">{html.escape(label)}</text>')
    for i, (x, label, col) in enumerate(vlines):
        g.append(f'<line x1="{px(x)}" x2="{px(x)}" y1="{T}" y2="{H - B}" stroke="{col}" stroke-dasharray="3 3"/>'
                 f'<text x="{px(x) + 3}" y="{T + 12 + 12 * (i % 4)}" fill="{col}">{html.escape(label)}</text>')
    for i, (label, col, pts) in enumerate(series):
        runs, run = [], []
        for x, y in pts:  # a missing reading breaks the line
            if y is None:
                runs, run = runs + ([run] if run else []), []
            else:
                run.append(f"{px(x):.1f},{py(y):.1f}")
        runs += [run] if run else []
        g += [f'<polyline points="{" ".join(s)}" fill="none" stroke="{col}" stroke-width="2.5"/>' for s in runs]
        g.append(f'<text x="{L + 8 + 150 * i}" y="{H - 14}" fill="{col}">&#9632; {html.escape(label)}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{H - 34}" text-anchor="middle">{html.escape(xlabel)}</text>'
             f'<text transform="translate(14,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle">'
             f'{html.escape(ylabel)}</text>')
    _svg(path, title, W, H, "".join(g))


def _barplot(path, title, rows):
    """rows: (label, median, p90, caught_frac). Bar = median time to detect, tick = p90; share caught as text."""
    W, rowh, L = 640, 20, 190
    H = 60 + rowh * len(rows)
    top = max([r[2] or 0 for r in rows] + [1.0])
    k = (W - L - 110) / top
    g = []
    for i, (label, med, p90, frac) in enumerate(rows):
        y = 40 + i * rowh
        g.append(f'<text x="{L - 6}" y="{y + 13}" text-anchor="end">{html.escape(label)}</text>')
        if med is not None:
            g.append(f'<rect x="{L}" y="{y + 2}" width="{k * med:.1f}" height="14" fill="#2a6fdb"/>'
                     f'<line x1="{L + k * p90:.1f}" x2="{L + k * p90:.1f}" y1="{y}" y2="{y + 18}" '
                     f'stroke="#d45500" stroke-width="2.5"/>')
        g.append(f'<text x="{W - 104}" y="{y + 13}">{frac * 100:.0f}% caught</text>')
    g.append(f'<text x="{L}" y="{H - 8}">blue = median seconds from fault start to trip, orange tick = p90 '
             f'(axis max {top:.0f} s)</text>')
    _svg(path, title, W, H, "".join(g))


def write_charts(out, r):
    lv = r["level"]["levels"]
    _lineplot(out / "chart_level_sam_vs_colour.svg", "Real pour: SAM reading vs colour-threshold baseline",
              "frame (1 fps)", "level, %",
              [("SAM 3.1", COLOURS[0], list(enumerate(lv))),
               ("colour threshold", COLOURS[1], list(enumerate(r["baseline"]["series"]["colour"])))],
              vlines=[(FIRST_FRAME, "analysis starts", COLOURS[4])], xmax=len(lv) - 1)
    rows = [(f"{d['kind']} {d['size']:g} {d['unit']}", d["ttd_median_s"], d["ttd_p90_s"], d["caught_frac"])
            for d in r["detection"]["rows"]]
    _barplot(out / "chart_detection_time.svg", "Simulator: time to detect by fault and size", rows)
    col = {"freeze": COLOURS[0], "drift": COLOURS[1], "replay": COLOURS[2]}
    trips = [(x["trip_t"], f"{x['scenario'][0]}{x['fault_start']}", col[x["scenario"]])
             for x in r["replay"]["rows"] if x["tripped"]]
    _lineplot(out / "chart_replay_trips.svg", "Real replay: where each fault tripped (f=freeze d=drift r=replay)",
              "time, s", "camera level, %", [("camera (SAM)", COLOURS[4], list(enumerate(lv)))],
              vlines=trips, hlines=[(CAMERA_NEAR_FULL, "95 %")], xmax=len(lv) - 1)


# ---------------------------------------------------------------- RESULTS.md

def _table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _floor(de, key, unit):
    v = de["floors"][key]
    if v is None:
        return "none caught in the tested range"
    if v == de["smallest_tested"][key]:  # the grid ends here: the true floor may be lower
        return f"at or below {v:g} {unit} (the smallest setting tested, so the true floor may be lower)"
    return f"{v:g} {unit}"


def write_markdown(r, path):
    lv, st, de, fa, rp, bl, dg, bs = (r[k] for k in ("level", "stills", "detection", "false_alarms", "replay", "blind",
                                                     "diagnoser", "baseline"))
    c, sf, af = r["config"], lv["status_frac"], lv["status_frac_after_entry"]
    m, rs, bt = dg["model"], dg["rule_synthetic"], bl["blind_then_spoof"]
    ro = r["robustness"]
    hd, li, cb, pg = ro["hand"], ro["lights"], ro["combined"], ro["pour_guard"]
    pc = lambda x: _f(100 * x, 1)  # noqa: E731
    md = [
        "# SightGlass evaluation (offline, recorded data and simulator)\n",
        f"Generated by `python -m agentwarden.evaluate`. Seeds are fixed, so a rerun gives the same numbers. "
        f"Configuration: {c['detect_seeds']} seeds per detection setting, {c['false_seeds']} benign runs of "
        f"{_f(c['false_minutes'], 0)} simulated minutes, {c['diag_per_class']} synthetic incidents per class"
        f"{' (TINY smoke configuration)' if c.get('tiny') else ''}. Nothing here touched the network or a model.\n",
        "## 1. Level reading quality (real data, no tape)\n",
        f"The recorded pour has {lv['n_frames']} SAM readings at 1 fps from a hand-held bottle. There is no tape or "
        f"scale, so the true level is not measured. The honest proxy is that the level in a steady pour never goes "
        f"down. Frames 0-{lv['first_frame'] - 1} are a hand and bottle entering the shot and are left out of the "
        f"checks below ({lv['n_frames'] - lv['first_frame']} frames analysed).\n",
        f"- Monotonicity: {lv['mono_violations']} backward steps after frame {lv['first_frame'] - 1}, the largest "
        f"{_f(lv['max_backward_step'])} points ({lv['mono_violations_gt2']} larger than 2 points).",
        f"- Frame-to-frame noise: median |delta - local trend| = {_f(lv['noise_median'], 2)} points (local trend = "
        f"median of the 4 neighbouring deltas; pairs touching a clamped 100 are left out, {lv['n_noise_pairs']} pairs "
        f"used). As a single-reading sigma that is about {_f(lv['sigma_est'], 2)} points (median / (0.6745 x sqrt 2)); "
        f"rounded up with a minimum of 3, the simulator uses sigma = {_f(lv['sigma_used'], 0)} points.",
        f"- Status over all frames: OK {pc(sf['OK'])} %, EMPTY {pc(sf['EMPTY'])} %, BLIND {pc(sf['CAMERA_BLIND'])} %, "
        f"LOW_CONFIDENCE {pc(sf['LOW_CONFIDENCE'])} %. After the entry frames: OK {pc(af['OK'])} %, "
        f"EMPTY {pc(af['EMPTY'])} %, BLIND {pc(af['CAMERA_BLIND'])} %.\n",
        f"Still photos ({st['reference']}). The reference levels are eyeballed, not measured, so the differences below "
        f"are agreement with an approximate reference, not an accuracy figure.\n",
        _table(["photo", "SAM reading %", "status", f"{STILL_REF} %", "abs difference"],
               [(x["photo"], _f(x["reading"]), x["status"], _f(x["approx_ref"], 0), _f(x["abs_error"]))
                for x in st["rows"]]),
        f"\nMean absolute difference {_f(st['mean_abs_diff'])} points, largest {_f(st['max_abs_diff'])} points "
        f"({len(st['rows'])} photos).\n",
        "Chart: `chart_level_sam_vs_colour.svg`.\n",
        "## 2. Detection on the simulator\n",
        f"Simulated tank (20 % start, pump on below 30 %, off above 80 %), a fake camera with gaussian noise sigma = "
        f"{_f(de['sigma'], 0)} points, the default interlock (T = 15 points for 5 s, rate rule, high-high 95). The "
        f"fault starts at a random second between 20 and 80; a run ends at the first trip, at the first overflow "
        f"(level above 100), or {_f(c['horizon_s'], 0)} s after the fault starts. Freeze: the PLC value stops, and "
        f"the offset is how far below the true level it was frozen. Step: the PLC reads a constant amount low. Drift: "
        f"the PLC falls behind at the given rate. Replay: the PLC repeats the last 30 s. \"Caught\" means tripped "
        f"before any overflow. TTD is seconds from fault start to trip, over caught seeds.\n",
        _table(["fault", "size", "caught", "missed", "overflow reached", "no trip, no overflow", "TTD median s",
                "TTD p90 s", "max true level at trip %", "rule that fired"],
               [(d["kind"], f"{d['size']:g} {d['unit']}", f"{d['detected']}/{d['n']}", d["missed"],
                 d["overflow_reached"], d["no_trip_no_overflow"], _f(d["ttd_median_s"]), _f(d["ttd_p90_s"]), _f(d["truth_at_trip_max"]),
                 ", ".join(f"{k} {v}" for k, v in sorted(d["rules"].items())) or "-") for d in de["rows"]]),
        "\n\"Overflow reached\" counts seeds where the level got to 100 before any trip. \"No trip, no overflow\" "
        "seeds were missed by the interlock but did no harm inside the horizon (for example a step smaller than the "
        "divergence threshold); \"missed\" is the sum of the two.\n",
        f"Detection floor (smallest setting caught before overflow in at least {CATCH_TARGET * 100:.0f} % of seeds, "
        f"among the settings tested): freeze offset {_floor(de, 'freeze', 'pts')}; step "
        f"{_floor(de, 'step', 'pts')}; drift rate {_floor(de, 'drift', 'pts/s')}. A floor is only "
        f"as fine as the grid of settings that was tried.\n",
        "Chart: `chart_detection_time.svg`.\n",
        "## 3. False alarms\n",
        f"{fa['n_runs']} benign runs of {_f(fa['minutes_each'], 0)} simulated minutes at sigma = "
        f"{_f(fa['sigma'], 0)} points ({_f(fa['exposure_hours'], 1)} hours of exposure): {fa['n_false_trips']} false "
        f"trips, {_f(fa['per_hour'], 3)} per hour. Upper 95 % bound {_f(fa['upper95_per_hour'], 3)} per hour "
        f"({fa['upper95_method']}). The noise here is gaussian; real readings have outliers (see the entry frames "
        f"above), so this is not a promise about a real camera.\n",
        "## 4. Real replay through the pipeline\n",
        f"The recorded readings are played through `Pipeline` with a fault injected on the PLC side. In replay the "
        f"plant follows the recording, so \"no overflow\" is true by construction and is not claimed. The claim is: "
        f"tripped before the camera reached 95 % (the first reading at or above 95 is at t = "
        f"{_f(rp['camera_first_ge_95_t'], 0)} s). {rp['n_before_95']} of {len(rp['rows'])} runs satisfy it.\n",
        _table(["fault", "starts t", "trip t s", "rule", "camera at trip %", "gap at trip pts",
                "before camera 95 %", "margin s"],
               [(x["scenario"], x["fault_start"], _f(x["trip_t"], 0), x["trip_reason"], _f(x["camera_at_trip"]),
                 _f(x["gap_at_trip"]), "yes" if x["before_95"] else "NO", _f(x["margin_s"], 0)) for x in rp["rows"]]),
        f"\nNo fault injected: tripped = {rp['benign']['tripped']} at t = {_f(rp['benign']['trip_t'], 0)} s "
        f"({rp['benign']['trip_reason']}), because the real bottle itself fills to the top.\n",
        "Chart: `chart_replay_trips.svg`.\n",
        "## 5. Camera blind behaviour\n",
        "A blind camera never trips the interlock (it only raises CAMERA_UNVERIFIED), by design: no reading, no "
        "verdict.\n",
        _table(["case", "blind span s", "fault", "tripped", "trip t s", "CAMERA_UNVERIFIED ticks", "PLC at end %",
                "camera at end % (hidden)"],
               [(k, f"{x['blind_span'][0]:.0f}-{x['blind_span'][1]:.0f}", f"{x['scenario']} @ {x['fault_start']}",
                 x["tripped"], _f(x["trip_t"], 0), x["unverified_ticks"], _f(x["plc_end"]), _f(x["hidden_camera_end"]))
                for k, x in bl.items()]),
        f"\nThe blind-then-spoof gap, measured: with the camera blind from t = 10 s to the end and the PLC frozen from "
        f"t = 12 s, nothing trips ({bt['unverified_ticks']} blind ticks raised the alarm). At the end the PLC says "
        f"{_f(bt['plc_end'])} % while the hidden camera level in the recording is {_f(bt['hidden_camera_end'])} %, a "
        f"gap of {_f(bt['gap_end'])} points that stays undetected. If the camera comes back (blind until t = 40 s) "
        f"the trip follows at t = {_f(bl['blind_recovers_late']['trip_t'], 0)} s.\n",
        "## 6. Diagnoser\n",
        "Two different things, kept apart.\n",
        "### model (recorded Muse Spark calls)\n",
        f"The recorded cassette is replayed keylessly (no network) for freeze, drift and replay at fault start 12 and "
        f"20, on the readings as they were when the calls were recorded (S6b occluder check and plausibility guard off: a "
        f"cassette is keyed to the exact prompt and the guard drops pour frame t = 2): {m['n_incidents']} incidents (the cassette folder holds {m['cassette_files']} files; each incident "
        f"re-derives its own prompt, so a file no incident asks for is not used). Ground truth for all of them is "
        f"{m['ground_truth']}.\n",
        _table(["fault", "starts t", "trip t s", "state", "model label", "confidence", "rule label",
                "agrees with rule", "reason"],
               [(x["scenario"], x["fault_start"], _f(x["trip_t"], 0), x["state"], x["label"], _f(x["confidence"], 2),
                 x["rule_label"], x["agrees_with_rule"], x["reason"] or "-") for x in m["rows"]]),
        f"\nGrounding passed (every quoted number found in the data): {m['grounding_passed']}/{m['n_incidents']}; "
        f"failed grounding {m['grounding_failed']}; unavailable (cassette miss or other) {m['unavailable']}. Label "
        f"SPOOF: {m['label_correct']}/{m['n_incidents']}. Agrees with the rule label: "
        f"{m['agrees_with_rule']}/{m['n_incidents']}. Misses are counted here, not dropped. The advisory model never "
        f"touches the pump.\n",
        "### rule baseline (synthetic)\n",
        f"`rule_label` (no model) on {rs['n_per_class']} synthetic incidents per class from the simulator. SPOOF "
        f"incidents are taken when the interlock trips ({rs['spoof_untripped_skipped']} spoof runs never tripped and "
        f"were skipped). CAMERA_FAULT mixes sustained blind spells, flickering blindness and a jumpy but \"OK\" "
        f"camera. These incidents were written by the same author as the rule, so treat the accuracy as an "
        f"optimistic ceiling, not as field performance.\n",
        _table(["true class", "n", "rule correct", "accuracy %", "predicted"],
               [(k, v["n"], v["correct"], "n/a" if v["accuracy"] is None else pc(v["accuracy"]),
                 ", ".join(f"{a} {b}" for a, b in sorted(v["predicted"].items()))) for k, v in rs["per_class"].items()]),
        "\nBy subtype (correct/n): " + "; ".join(
            f"{k}: " + ", ".join(f"{s} {x['correct']}/{x['n']}" for s, x in v["by_subtype"].items())
            for k, v in rs["per_class"].items()) + f". Overall {pc(rs['overall_accuracy'])} %.\n",
        "## 7. Why SAM: colour-threshold baseline\n",
        f"A naive reader (HSV orange threshold inside the container box, lowest continuous run of rows gives the "
        f"surface) on the same {bs['n_frames']} raw frames. {bs['calibration']}. The thresholds were set by looking "
        f"at one frame, not tuned per frame. Frames {bs['first_frame']} onwards:\n",
        _table(["reader", "backward steps", "of them > 2 pts", "largest backward step pts", "median noise pts",
                "frames with no reading"],
               [(n, bs[k]["mono_violations"], bs[k]["mono_violations_gt2"], _f(bs[k]["max_backward_step"]),
                 _f(bs[k]["noise_median"], 2), bs[k]["no_reading_frames"])
                for n, k in (("SAM 3.1", "sam"), ("colour threshold", "colour"))]),
        f"\nMean absolute difference between the two readers: {_f(bs['mean_abs_diff_sam_vs_colour'])} points. The "
        f"comparison holds for this one clip only and says nothing about other bottles, liquids or lighting.\n",
        "## 8. Camera robustness (real clips)\n",
        f"Two recordings with live SAM 3.1 at 2 fps, a steady phone and a full orange bottle (so the true level is "
        f"{_f(CLIP_TRUTH, 0)} % in both; no tape, this is the owner's statement, not a measurement). "
        f"**hand**: a hand sweeps in front of the bottle twice ({hd['frames']} frames, {_f(hd['duration_s'])} s); the "
        f"liquid mask gets cut and SAM reports 11-82 % with status OK. **lights**: the room light goes off twice "
        f"({li['frames']} frames, {_f(li['duration_s'])} s); SAM finds nothing. The fix has two parts. (1) OCCLUDED: a "
        f"\"hand\" box covering at least {_f(100 * ro['overlap_threshold'], 0)} % of the container box makes the frame "
        f"OCCLUDED: no level, never alertable, alarm CAMERA_UNVERIFIED, mismatch timers reset, never a trip. In the "
        f"recordings that is the stored `hand_overlap` (SAM concept \"hand\", confidence 0.5 or more); in live mode it "
        f"is a third SAM image call per frame (+50 % image cost over the two-call path; `occluder_concept=None` drops "
        f"it). (2) Plausibility guard: a change faster than {_f(ro['max_rate_pts_s'], 0)} points/s between alertable "
        f"readings is LOW_CONFIDENCE (\"implausible jump\") for that frame.\n",
        _table(["clip", "frames", "OK", "OCCLUDED", "BLIND", "LOW_CONF", "OK before the fix",
                "false trips (default N_s 5 s)", "false trips before", f"false trips at N_s {N_S_SENSITIVITY:g} s: "
                "before / after", "max abs(reading - 100) among OK: before / after"],
               [(n, x["frames"], x["ok"], x["occluded"], x["blind"], x["low_conf"], x["frames_ok_before"],
                 x["false_trips"], x["false_trips_before"],
                 f"{x['false_trips_before_n2']} / {x['false_trips_n2']}",
                 f"{_f(x['max_abs_err_ok_before'])} / {_f(x['max_abs_err_ok_after'])}")
                for n, x in (("hand", hd), ("lights", li))]),
        f"\nFalse trips are counted with the PLC reading the true {_f(CLIP_TRUTH, 0)} % and HIGH_HIGH switched off "
        f"(the bottle really is full; a trip on a full bottle would be correct, not false). Before the fix, a hand "
        f"left the camera at \"OK\" and as far as {_f(hd['max_abs_err_ok_before'])} points from the truth. It did not "
        f"trip at the default N_s = 5 s only because each sweep lasted 4 s or less and one low-confidence frame "
        f"(0.43) cut the first one; with N_s = {N_S_SENSITIVITY:g} s the old behaviour trips "
        f"({hd['false_trips_before_n2']} of 1 run) and the fixed one does not ({hd['false_trips_n2']}). The "
        f"plausibility guard alone is not enough: with it but without the hand check, {hd['ok_guard_only']} frames "
        f"stay OK and the worst is still {_f(hd['max_abs_err_ok_guard_only'])} points off, because a mask that stays "
        f"wrong for two frames looks like a real change. After the fix the {hd['ok']} frames still marked OK read "
        f"within {_f(hd['max_abs_err_ok_after'])} points of 100. The {hd['occluded']} OCCLUDED frames are exactly "
        f"the frames the recorder marked with a hand overlap of 0.20 or more.\n",
        f"Lights: the {li['blind']} dark frames are CAMERA_BLIND (alarm only, no trip, the HMI says CAMERA BLOCKED - NO "
        f"TRIP), the readings return to 100 % when the light returns ({li['ok']} OK frames, worst "
        f"{_f(li['max_abs_err_ok_after'])} points off). Lights needed no new rule; the number shown is the existing "
        f"blind behaviour on a real recording.\n",
        f"Combined: the hand clip plus a frozen PLC (fault from t = 0, PLC stuck at the tank's 20 % while the bottle "
        f"is full), with {_f(cb['appended_clean_s'], 0)} s of clean frames appended (copies of the last real frame, "
        f"so these are not new data). HIGH_HIGH is off. Fixed: no trip during the occlusion; "
        f"trip = {cb['tripped']} at t = {_f(cb['trip_t'], 1)} s ({cb['trip_reason']}, on an {cb['trip_status']} "
        f"frame reading {_f(cb['camera_at_trip'], 0)} %), the last OCCLUDED frame was t = "
        f"{_f(cb['last_occluded_t'], 1)} s, so the {_f(interlock.Config().N_s, 0)} s divergence timer ran on clean "
        f"frames only (clean from t = {_f(cb['last_occluded_t'] + 0.5, 1)} s). Before the fix the same run tripped at t = {_f(cb['before']['trip_t'], 1)} s "
        f"({cb['before']['trip_reason']}), "
        f"{_f(cb['trip_t'] - cb['before']['trip_t'], 1)} s earlier, with the timer fed by hand-corrupted readings: "
        f"right fault, but the verdict leaned on frames that were not trustworthy.\n",
        f"Pour clip check (the fix must not break a real fill): {pg['frames']} frames, the plausibility guard flagged "
        f"{pg['flagged']} (t = {', '.join(_f(t, 0) for t in pg['flagged_t'])}, the hand entering the shot), the hand "
        f"check flagged {pg['occluded']}. After the entry frames {pg['frames_ok_after_entry']} of "
        f"{pg['frames_after_entry']} frames stay OK and the real pour peaks at {_f(pg['max_rate_after_entry'])} "
        f"points/s, under the {_f(ro['max_rate_pts_s'], 0)} points/s limit (the entry frames alone jump "
        f"{_f(pg['max_rate_incl_entry'], 0)} points/s).\n",
        "## Limits\n",
        "- Camera robustness (section 8) rests on one bottle, one hand, one indoor light and two short clips (22 and "
        "27 frames). Other occluders (a sleeve, a cloth), a hand SAM does not call \"hand\", or a partial shadow are "
        "untested. The truth of 100 % is the owner's statement, not a measurement.",
        "- The occluder concept adds a third SAM image per frame (+50 % image cost). A hand that covers less than 20 % "
        f"of the bottle is not OCCLUDED; the plausibility guard (8 points/s, from a pour that peaked at "
        f"{_f(pg['max_rate_after_entry'])}) is the only net there, and a wrong mask that persists for two frames "
        f"passes it.",
        "- The combined test appends 10 s of copied clean frames; the hand clip alone is too short for the 5 s "
        "divergence timer to run on clean frames.",
        "- No tape ground truth: level quality uses a monotonic-pour proxy and eyeballed approximate levels, so no "
        "accuracy figure is claimed.",
        "- One hand-held clip and one bottle with one orange liquid; the first frames show a hand entering the shot.",
        "- The PLC, the tank and the camera noise are simulated. Detection and false-alarm numbers describe the rules "
        "against a gaussian fake camera.",
        "- The T = 15 point threshold is the provisional default, not derived from a measured error distribution.",
        "- The blind-then-spoof gap (section 5) is a documented limit: a camera that goes blind and then a spoof "
        "begins is not detected.",
        "- The model results come from a handful of recorded Spark calls (section 6 gives the count), far too few for "
        "rates.",
        "- The synthetic incidents for the rule baseline are self-authored.",
        "- In replay the plant follows the recording, so overflow is not tested there; the claim is limited to "
        "tripping before the camera reached 95 %.\n",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------- 8. camera robustness (real clips)

CLIP_TRUTH = 100.0  # the bottle in the hand and lights clips is full the whole time (no tape: stated, not measured)
N_S_SENSITIVITY = 2.0  # the recorded sweeps last <= 4 s, so the default N_s = 5 never fires; 2 s shows the hazard
OFF = dict(occluder=False, max_rate_pts_s=None)  # ReplayVision flags that give back the pre-S6b behaviour


def _clip(name):
    return json.loads((DEMO / f"{name}_replay.json").read_text(encoding="utf-8"))


def _read_clip(recs, **kw):
    rv = ReplayVision(recs, **kw)
    return [rv.read((r["t"],)).reading for r in recs]


def _plc_at_truth_trips(recs, readings, n_s):
    """Interlock fed the clip with the PLC reading the true 100 %. HIGH_HIGH is off (the bottle really is full)."""
    cfg = interlock.Config(N_s=n_s, hh_limit=101.0)
    st = interlock.initial_state()
    for r, rd in zip(recs, readings):
        st, _ = interlock.step(st, rd, CLIP_TRUTH, 0.0, r["t"], cfg)
    return bool(st.tripped)


def _clip_stats(name):
    recs = _clip(name)
    after, before = _read_clip(recs), _read_clip(recs, **OFF)
    guard_only = _read_clip(recs, occluder=False)
    cnt = Counter(x.status for x in after)

    def worst(rs):
        errs = [abs(x.level_pct - CLIP_TRUTH) for x in rs if x.alertable and x.status == "OK"]
        return max(errs) if errs else None

    return {"frames": len(recs), "duration_s": recs[-1]["t"] - recs[0]["t"], "ok": cnt["OK"],
            "occluded": cnt["OCCLUDED"], "blind": cnt["CAMERA_BLIND"], "low_conf": cnt["LOW_CONFIDENCE"],
            "empty": cnt["EMPTY"],
            "frames_ok_before": sum(x.status == "OK" and x.alertable for x in before),
            "false_trips": int(_plc_at_truth_trips(recs, after, interlock.Config().N_s)),
            "false_trips_before": int(_plc_at_truth_trips(recs, before, interlock.Config().N_s)),
            "false_trips_n2": int(_plc_at_truth_trips(recs, after, N_S_SENSITIVITY)),
            "false_trips_before_n2": int(_plc_at_truth_trips(recs, before, N_S_SENSITIVITY)),
            "max_abs_err_ok_before": worst(before), "max_abs_err_ok_after": worst(after),
            "max_abs_err_ok_guard_only": worst(guard_only),
            "ok_guard_only": sum(x.status == "OK" and x.alertable for x in guard_only)}


def _combined():
    """Hand clip + PLC frozen (freeze fault from t = 0, PLC stuck at the tank's 20 %), 10 s of clean frames appended
    (copies of the last real frame). HIGH_HIGH is off. Trip is expected only on clean frames."""
    recs = _clip("hand")
    last = recs[-1]
    recs += [dict(last, t=last["t"] + 0.5 * (i + 1)) for i in range(20)]

    def run(**kw):
        cfg = interlock.Config(hh_limit=101.0)
        with contextlib.redirect_stdout(io.StringIO()):
            pl = Pipeline(ReplayVision(recs, **kw), "freeze", 0.0, cfg=cfg)
        evs = [pl.tick((r["t"],)) for r in recs]
        pl.close()
        s = pl.summary
        occ = [e["t"] for e in evs if e["status"] == "OCCLUDED"]
        hit = next((e for e in evs if e["t"] == s.trip_t), None) if s.tripped else None
        return {"tripped": s.tripped, "trip_t": s.trip_t, "trip_reason": s.trip_reason,
                "last_occluded_t": max(occ) if occ else None,
                "trip_status": hit["status"] if hit else None,
                "trip_during_occlusion": bool(hit and hit["status"] == "OCCLUDED"),
                "camera_at_trip": hit["camera"] if hit else None}

    return {**run(), "before": run(**OFF), "clip_s": recs[-1]["t"], "appended_clean_s": 10.0}


def _pour_guard():
    recs = _clip("pour")
    after, raw = _read_clip(recs), _read_clip(recs, **OFF)
    ok_ = [(r["t"], r["level_pct"]) for r in recs if r["t"] >= FIRST_FRAME and r["status"] == "OK"]
    rates = [abs(y - x) / (b - a) for (a, x), (b, y) in zip(ok_, ok_[1:])]
    allok = [(r["t"], r["level_pct"]) for r in recs if r["status"] == "OK"]
    flagged = [r["t"] for r, a, b in zip(recs, after, raw) if b.alertable and not a.alertable]
    return {"frames": len(recs), "flagged": len(flagged), "flagged_t": flagged,
            "occluded": sum(a.status == "OCCLUDED" for a in after),
            "max_rate_after_entry": max(rates), "max_rate_incl_entry": max(
                abs(y - x) / (b - a) for (a, x), (b, y) in zip(allok, allok[1:])),
            "frames_ok_after_entry": sum(a.status == "OK" and a.alertable for r, a in zip(recs, after)
                                         if r["t"] >= FIRST_FRAME),
            "frames_after_entry": sum(r["t"] >= FIRST_FRAME for r in recs)}


def robustness():
    return {"hand": _clip_stats("hand"), "lights": _clip_stats("lights"), "combined": _combined(),
            "pour_guard": _pour_guard(), "max_rate_pts_s": 8.0, "overlap_threshold": 0.20}


# ---------------------------------------------------------------- driver

def run_all(cfg):
    recs = _recs()
    lv = level_quality(recs)
    sigma = lv["sigma_used"]
    return {"config": cfg, "level": lv, "stills": stills_quality(), "detection": detection(cfg, sigma),
            "false_alarms": false_alarms(cfg, sigma), "replay": replay_runs(recs), "blind": blind_runs(recs),
            "diagnoser": {"model": evaluate_diag.diagnoser_model(recs, DEMO / "spark_cassette", _replay_run_prompts_as_recorded),
                          "rule_synthetic": evaluate_diag.diagnoser_rule_synthetic(cfg, sigma, _run)},
            "baseline": baseline(recs), "robustness": robustness()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SightGlass offline evaluation (recorded data + simulator)")
    p.add_argument("--out", default="eval", help="output directory")
    p.add_argument("--tiny", action="store_true", help="smoke configuration (seconds, not minutes)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = run_all(dict(TINY if a.tiny else FULL, tiny=a.tiny))
    (out / "results.json").write_text(json.dumps(r, indent=1, sort_keys=True), encoding="utf-8")
    write_markdown(r, out / "RESULTS.md")
    write_charts(out, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
