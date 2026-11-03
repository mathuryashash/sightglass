# Recording script (run once by the author with a local key file; not needed for replay). Kept for provenance.
# Live check: real Muse Spark diagnoser on incidents built by the real pipeline over the real pour replay.
# Scenarios: freeze, drift, replay (SPOOF expected) + benign. Records into a cassette for keyless demo replay.
# Never prints the key.
import sys, json, pathlib, time
sys.path.insert(0, r"D:/hackathons/meta-ai-dev/agentwarden/src")
from agentwarden.cost import Ledger
from agentwarden.cassette import Cassette
from agentwarden.meta import MetaClient, httpx_transport
from agentwarden.pipeline import Pipeline
from agentwarden.vision import ReplayVision
from agentwarden import diagnoser as D

ROOT = pathlib.Path(r"D:/hackathons/meta-ai-dev")
key = next(l.split("=", 1)[1].strip() for l in open(ROOT / "secrets/.env", encoding="utf-8")
           if l.startswith("MODEL_API_KEY="))
cas = ROOT / "agentwarden/demo/real/spark_cassette"
client = MetaClient(httpx_transport(key, "https://api.meta.ai/v1"), Ledger(ROOT / "runs/ledger.jsonl", 100.0),
                    Cassette(cas), "sightglass", 0.40, "record")
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
    if not cap.args:
        print(scen, "no trip"); continue
    ev, plc, cam, trig, lim = cap.args
    inc = D.build_incident(ev, plc, cam, limits=lim)
    rule = D.rule_label(inc)
    t0 = time.time()
    res = D.diagnose(inc, client, "sightglass")
    row = {"scenario": scen, "trip": inc.get("trip"), "rule": rule, "secs": round(time.time() - t0, 1)}
    if isinstance(res, D.DiagnosisError):
        row["error"] = f"{res.reason}: {str(res)[:200]}"
    else:
        row.update(label=res.label, conf=res.confidence, evidence=res.evidence, note=res.operator_note,
                   check=res.suggested_check)
    print(json.dumps(row, indent=1), flush=True); out.append(row)
(ROOT / "runs/spark_live.json").write_text(json.dumps(out, indent=2))
