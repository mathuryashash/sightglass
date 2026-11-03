"""Diagnoser part of the offline evaluation (S6). Kept apart from evaluate.py because S5 forbids any module that imports
the diagnoser from importing the pump permit or the interlock: this module gets the simulator as a callable argument
and never touches either. Offline: recorded cassette only, no network, no keys.

Two different things are measured and never mixed:
  model (recorded Muse Spark calls): the cassette replayed through the advisor for the recorded incidents;
  rule baseline (synthetic): rule_label on incidents generated from the simulator.
"""
import random
from collections import Counter
from pathlib import Path

from agentwarden.advisor import build_advisor
from agentwarden.diagnoser import build_incident, rule_label

CLASSES = ("SPOOF", "LEAK", "CAMERA_FAULT", "BENIGN")


def diagnoser_model(recs, cassette_dir, replay_run):
    """replay_run(recs, scenario, fault_start, advice=) -> (pipeline, events): the replay driver from evaluate.py."""
    rows = []
    for sc in ("freeze", "drift", "replay"):
        for fs in (12, 20):
            adv = build_advisor(f"cassette:{cassette_dir}", readings=recs)
            pl, _ = replay_run(recs, sc, fs, advice=adv)
            adv.wait(30)
            snap = adv.snapshot()
            rows.append({"scenario": sc, "fault_start": fs, "trip_t": pl.summary.trip_t, "state": snap["state"],
                         "label": snap.get("label"), "confidence": snap.get("confidence"),
                         "rule_label": snap.get("rule_label"), "agrees_with_rule": bool(snap.get("agrees")),
                         "reason": snap.get("reason")})
    ready = [r for r in rows if r["state"] == "ready"]
    return {"ground_truth": "SPOOF", "n_incidents": len(rows), "rows": rows,
            "cassette_files": len(list(Path(cassette_dir).glob("*.json"))),
            "grounding_passed": len(ready),
            "grounding_failed": sum(r["reason"] == "ungrounded" for r in rows),
            "unavailable": sum(r["state"] != "ready" and r["reason"] != "ungrounded" for r in rows),
            "unavailable_reasons": dict(Counter(r["reason"] for r in rows if r["state"] != "ready")),
            "label_correct": sum(r["label"] == "SPOOF" for r in ready),
            "agrees_with_rule": sum(r["agrees_with_rule"] for r in ready),
            "rule_label_correct_same_incidents": sum(r["rule_label"] == "SPOOF" for r in rows)}


def _incident(trace):
    return build_incident([x[3] for x in trace], [(x[0], x[1]) for x in trace], [(x[0], x[2]) for x in trace])


def _synthetic(cls, k, sigma, run):
    """(subtype, incident) of class `cls`; incident None for a SPOOF the interlock never tripped on.
    run(kind, size, seed, sigma, horizon_s, trace=, stop=, ...) is the closed-loop simulator from evaluate.py."""
    rng = random.Random(f"{cls}:{k}")
    tr = []
    if cls == "SPOOF":
        kind, size = rng.choice((("freeze", rng.choice((0.0, 5.0, 10.0))), ("drift", rng.uniform(0.4, 1.0)),
                                 ("replay", 0.0), ("step", rng.uniform(20.0, 30.0))))
        r = run(kind, size, 20_000 + k, sigma, 300.0, trace=tr)
        return kind, (_incident(tr) if r["trip_t"] is not None else None)
    if cls == "LEAK":
        run("benign", 0.0, 30_000 + k, sigma, 100.0, trace=tr, stop=False, drain_lps=rng.uniform(10.5, 15.0),
            level0=rng.uniform(65.0, 80.0), pump0=True)
        return "leak", _incident(tr)
    if cls == "CAMERA_FAULT":
        sub = ("blind", "flicker", "jumpy")[k % 3]
        if sub == "blind":
            length = rng.uniform(11, 50)
            run("benign", 0.0, 40_000 + k, sigma, 110.0, trace=tr, stop=False,
                blind=(110 - length, 110, rng.choice(("CAMERA_BLIND", "LOW_CONFIDENCE"))))
        elif sub == "flicker":  # each second blind with probability p: bursts, no long unverified spell
            p = rng.uniform(0.3, 0.8)
            run("benign", 0.0, 40_000 + k, sigma, 110.0, trace=tr, stop=False)
            for i, (t, plc, _cam, e) in enumerate(tr):
                if t >= 40 and rng.random() < p:
                    tr[i] = (t, plc, None, {**e, "status": "CAMERA_BLIND",
                                            "actions": e["actions"] + ["ALARM_CAMERA_UNVERIFIED"]})
        else:  # jumpy but reported OK: looks like a spoof in the summary numbers
            run("benign", 0.0, 40_000 + k, 12.0, 110.0, trace=tr, stop=False)
        return sub, _incident(tr)
    run("benign", 0.0, 50_000 + k, sigma, 70.0 + (k * 37) % 530, trace=tr, stop=False)
    return "benign", _incident(tr)


def diagnoser_rule_synthetic(cfg, sigma, run):
    n = cfg["diag_per_class"]
    per_class, skipped = {}, 0
    for cls in CLASSES:
        labels, subs, k, done = Counter(), {}, 0, 0
        while done < n and k < 5 * n:
            sub, inc = _synthetic(cls, k, sigma, run)
            k += 1
            if inc is None:
                skipped += 1
                continue
            lab = rule_label(inc)
            labels[lab] += 1
            hit = subs.setdefault(sub, [0, 0])
            hit[0] += lab == cls
            hit[1] += 1
            done += 1
        per_class[cls] = {"n": done, "correct": labels[cls], "accuracy": labels[cls] / done if done else None,
                          "predicted": dict(labels),
                          "by_subtype": {s: {"correct": a, "n": b} for s, (a, b) in subs.items()}}
    total = sum(c["n"] for c in per_class.values())
    return {"n_per_class": n, "per_class": per_class, "spoof_untripped_skipped": skipped,
            "overall_accuracy": sum(c["correct"] for c in per_class.values()) / total}
