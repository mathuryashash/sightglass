# Recording script (run once by the author with a local key file; not needed for replay). Kept for provenance.
# Live open-weights run: the same 3 real-pour incidents (freeze/drift/replay, fault start 20) diagnosed by
# Muse Spark Open on Baseten. Records a cassette for keyless replay. Never prints the key.
import sys, json, pathlib, time
sys.path.insert(0, r"D:/hackathons/meta-ai-dev/agentwarden/src")
from agentwarden.pipeline import Pipeline
from agentwarden.vision import ReplayVision
from agentwarden import diagnoser as D
from agentwarden.open_spark import OpenSparkClient, httpx_transport

ROOT = pathlib.Path(r"D:/hackathons/meta-ai-dev")
key = next(l.split("=", 1)[1].strip() for l in open(ROOT / "secrets/.env", encoding="utf-8")
           if l.startswith("BASETEN_API_KEY="))
client = OpenSparkClient(httpx_transport(key), ROOT / "agentwarden/demo/real/open_spark_cassette", "record")
REPLAY = ROOT / "agentwarden/demo/real/pour_replay.json"


class Capture:
    def __init__(self): self.args = None
    def submit(self, events, plc, cam, trigger="trip", limits=None):
        if self.args is None: self.args = (events, plc, cam, trigger, limits)
        return True
    def snapshot(self): return {}


out = []
for scen in ["freeze", "drift", "replay"]:
    cap = Capture()
    pl = Pipeline(ReplayVision(str(REPLAY)), scenario=scen, fault_start_s=20, advice=cap)
    for i in range(54):
        pl.tick((float(i), b"", 478, 850))
        if cap.args: break
    ev, plc, cam, trig, lim = cap.args
    inc = D.build_incident(ev, plc, cam, limits=lim)
    t0 = time.time()
    res = D.diagnose(inc, client, "sightglass-open")
    row = {"model": "meta-models/muse-spark-open (Baseten)", "scenario": scen, "trip": inc.get("trip"),
           "rule": D.rule_label(inc), "secs": round(time.time() - t0, 1)}
    if isinstance(res, D.DiagnosisError):
        row["error"] = f"{res.reason}: {res.detail[:200]}"
    else:
        row.update(label=res.label, conf=res.confidence, evidence=list(res.evidence), note=res.operator_note,
                   check=res.suggested_check, tool_calls=len(res.tool_calls))
    print(json.dumps(row, indent=1), flush=True); out.append(row)
tok = sum((u.get("completion_tokens") or 0) for u in client.usage)
print("calls", client.calls, "completion_tokens", tok)
(ROOT / "agentwarden/demo/real/open_spark_runs.json").write_text(json.dumps(out, indent=2))
