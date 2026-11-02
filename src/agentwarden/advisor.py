"""Runs one advisory diagnosis at a time on a background thread and holds the result for the dashboard.

SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM. The pipeline hands over copies of recent data through
submit() and reads snapshot(); it never waits for the model. Nothing here can touch the pump signal: this
module only imports the diagnoser and the model client plumbing.

    --diagnoser off | fake | cassette:<dir> | live      (live: a client passed in, or --key-file + --base-url)
"""
import json
import tempfile
import threading
from pathlib import Path

from agentwarden.diagnoser import Diagnosis, Tools, build_incident, diagnose, rule_label

MODES = "off | fake | cassette:<dir> | live"


class FakeSpark:
    """Offline stand-in for the model (`--diagnoser fake`): answers with the rule baseline, quoting the numbers
    from the incident it was shown. It makes no network call and needs no key."""

    def spark_json(self, prompt, schema, run_id):
        line = next(x for x in prompt.splitlines() if x.startswith("INCIDENT_JSON: "))
        inc = json.loads(line[len("INCIDENT_JSON: "):])
        s = inc["summary"]
        ev = []
        if s["plc_max"] is not None:
            ev.append(f"PLC level stayed within {s['plc_min']} to {s['plc_max']}")
        if s["cam_max"] is not None:
            ev.append(f"camera level ranged {s['cam_min']} to {s['cam_max']}")
        ev.append(f"largest PLC-camera gap {s['max_abs_diff']}" if s["max_abs_diff"] is not None
                  else f"window of {inc['window_s']} s")
        return {"label": rule_label(inc), "confidence": 0.5, "evidence": ev[:4],
                "operator_note": "Offline demo diagnosis from fixed rules, not from the model. Compare the PLC and "
                                "camera levels and check the tank by eye.",
                "suggested_check": "Look at the tank and compare it with the PLC level."}


class Advisor:
    def __init__(self, client, run_id, readings=None):
        self.client, self.run_id, self.readings = client, run_id, readings
        self._lock = threading.Lock()
        self._thread = None
        self._snap = {"state": "idle"}

    def submit(self, events, plc_series, cam_series, trigger="trip", limits=None):
        """Queue one diagnosis. Returns False if one is already running. Copies its inputs and returns at once."""
        events = [dict(e) for e in events]
        plc_series, cam_series = list(plc_series), list(cam_series)
        with self._lock:
            if self._snap["state"] == "pending":
                return False
            self._snap = {"state": "pending", "trigger": trigger}
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            args=(events, plc_series, cam_series, trigger, limits))
            self._thread.start()
        return True

    def _run(self, events, plc, cam, trigger, limits):
        snap = {"state": "unavailable", "trigger": trigger, "reason": "unavailable"}
        try:
            inc = build_incident(events, plc, cam, limits=limits)
            rule = rule_label(inc)
            snap["rule_label"] = rule
            out = diagnose(inc, self.client, self.run_id, Tools(inc, self.readings))
            if isinstance(out, Diagnosis):
                snap = {"state": "ready", "trigger": trigger, "label": out.label, "confidence": out.confidence,
                        "evidence": list(out.evidence), "operator_note": out.operator_note,
                        "suggested_check": out.suggested_check, "rule_label": rule, "agrees": out.label == rule,
                        "reason": None}
            else:
                snap["reason"] = out.reason
        except Exception as e:  # a diagnosis problem must never be anyone else's problem
            snap["reason"] = type(e).__name__
        with self._lock:
            self._snap = snap

    def wait(self, timeout=None):
        """Block until the running job (if any) is done; True when none is running. For tests and shutdown."""
        t = self._thread
        if t is not None:
            t.join(timeout)
            return not t.is_alive()
        return True

    def snapshot(self):
        with self._lock:
            return {**self._snap, "advisory": True}


def build_advisor(spec, run_id="sightglass", client=None, key_file=None, base_url=None, readings=None):
    """Advisor for a `--diagnoser` value, or None for "off". Raises ValueError for anything unusable."""
    if spec == "off":
        return None
    if spec == "fake":
        return Advisor(FakeSpark(), run_id, readings)
    from agentwarden.cassette import Cassette
    from agentwarden.cost import Ledger
    from agentwarden.meta import MetaClient, httpx_transport

    ledger = Ledger(Path(tempfile.mkdtemp(prefix="sg_ledger_")) / "ledger.jsonl", 1.0)
    if spec.startswith("cassette:"):
        def no_network(*_a, **_k):
            raise RuntimeError("cassette replay mode makes no network calls")

        client = MetaClient(no_network, ledger, Cassette(spec[len("cassette:"):]), run_id, 1.0, "replay")
    elif spec == "live":
        if client is None:
            if not key_file or not base_url:
                raise ValueError("live needs an injected client, or --key-file and --base-url")
            try:
                key = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError:
                raise ValueError("cannot read the key file") from None
            if not key:
                raise ValueError("the key file is empty")
            client = MetaClient(httpx_transport(key, base_url), ledger, Cassette(ledger.path.parent / "cassette"),
                                run_id, 1.0, "live")
    else:
        raise ValueError(f"--diagnoser must be one of: {MODES}")
    return Advisor(client, run_id, readings)


def add_diagnoser_args(p):
    p.add_argument("--diagnoser", default="off", help=MODES + " (advisory only; never controls the pump)")
    p.add_argument("--key-file", help="live only: file holding the model API key (the key is only ever read from this file)")
    p.add_argument("--base-url", help="live only: model API base URL")
