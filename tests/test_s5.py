"""S5: Muse Spark diagnoser (advisory only) + relay driver. Fake transport only; imports are lazy."""
import ast
import importlib
import json
import re
import threading
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "agentwarden"
SCENARIO_WORDS = ("freeze", "drift", "replay")


def mod(name):
    return importlib.import_module(f"agentwarden.{name}")


# ------------------------------------------------------------------ synthetic incidents


def events(n=61, status="OK", trip_at=None, reason="DIVERGENCE", pump=True, blind_from=None, extra=None):
    out = []
    for t in range(n):
        st = "CAMERA_BLIND" if blind_from is not None and t >= blind_from else status
        tripped = trip_at is not None and t >= trip_at
        acts = (["ALARM_CAMERA_UNVERIFIED"] if st == "CAMERA_BLIND" else []) + (
            ["WRITE_PERMIT_FALSE"] if tripped else [])
        ev = {"t": float(t), "status": st, "pump_on": pump, "tripped": tripped,
              "trip_reason": reason if tripped else None, "actions": acts}
        ev.update(extra or {})
        out.append(ev)
    return out


def ser(f, n=61, step=1.0):
    return [(i * step, f(i * step)) for i in range(n)]


def incident(kind):
    d = mod("diagnoser")
    cam = ser(lambda t: 20 + t)  # 20 -> 80
    ev = events()
    if kind == "freeze":
        plc = ser(lambda t: 40.0)
    elif kind == "drift":
        plc = ser(lambda t: 20 + t - 0.5 * t)
    elif kind == "replay":
        plc = ser(lambda t: 30 + 5 * (t % 10 > 4))
    elif kind == "leak":
        cam = ser(lambda t: 80 - t * 0.6)
        plc = ser(lambda t: 80 - t * 0.6 + 0.3)
    elif kind == "blind":
        cam = ser(lambda t: 20 + t if t < 15 else None)
        plc = ser(lambda t: 20 + t)
        ev = events(blind_from=15)
    else:  # benign
        plc = ser(lambda t: 20 + t + 0.4)
    return d.build_incident(ev, plc, cam)


def good_answer(inc, **over):
    s = inc["summary"]
    a = {"label": "SPOOF", "confidence": 0.8,
         "evidence": [f"PLC stayed near {s['plc_max']} for {s['plc_flat_s']} s", f"camera reached {s['cam_max']}"],
         "operator_note": "The PLC level looks stuck while the camera sees the tank filling. Check the sensor.",
         "suggested_check": "Look at the tank by eye and compare it with the PLC level."}
    a.update(over)
    return a


# ------------------------------------------------------------------ fake transport + client


def spark_body(text):
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 800, "output_tokens": 150}}


class Script:
    """Transport that replays scripted replies (last one repeats). str/dict -> 200 body; Exception -> raised."""

    def __init__(self, *replies, delay=0.0):
        self.replies, self.calls, self.delay = list(replies), [], delay
        self.lock = threading.Lock()

    def __call__(self, url, headers, body):
        with self.lock:
            self.calls.append(body)
            r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        time.sleep(self.delay)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, tuple):
            return r
        return 200, spark_body(r if isinstance(r, str) else json.dumps(r))


def mk_client(tmp_path, transport, mode="live", cas="cas", run_id="r1"):
    from agentwarden.cassette import Cassette
    from agentwarden.cost import Ledger
    from agentwarden.meta import MetaClient

    return MetaClient(transport, Ledger(tmp_path / "l.jsonl", 100.0), Cassette(tmp_path / cas), run_id, 1.0, mode)


def prompt_of(call):
    return call["input"]


# ------------------------------------------------------------------ build_incident


def test_build_incident_window_rounding_and_cap():
    d = mod("diagnoser")
    plc = ser(lambda t: 30.123456, n=400, step=0.5)  # 200 s at 2 Hz
    cam = ser(lambda t: 31.987654, n=400, step=0.5)
    inc = d.build_incident(events(n=200), plc, cam, window_s=60)
    json.dumps(inc)
    for key in ("plc", "camera"):
        pts = inc[key]
        assert 0 < len(pts) <= 120
        assert min(p[0] for p in pts) >= 199.5 - 60 - 0.5
        assert pts[-1][0] == 199.5  # newest point always kept
        assert all(p[0] == round(p[0], 1) and p[1] == round(p[1], 1) for p in pts)
    assert inc["window_s"] == 60


def test_build_incident_carries_trip_alarms_statuses():
    d = mod("diagnoser")
    inc = d.build_incident(events(trip_at=40, reason="RATE", blind_from=50), ser(lambda t: 40.0),
                           ser(lambda t: 20 + t if t < 50 else None))
    assert inc["trip"] == {"reason": "RATE", "t": 40.0}
    assert inc["alarms"]["ALARM_CAMERA_UNVERIFIED"] >= 10
    assert inc["camera_status"]["CAMERA_BLIND"] == 11 and inc["camera_status"]["OK"] > 0
    assert inc["summary"]["camera_unverified_s"] >= 10
    assert d.build_incident(events(), ser(lambda t: 1.0), ser(lambda t: 1.0))["trip"] is None


def test_build_incident_never_contains_injected_scenario_or_truth():
    d = mod("diagnoser")
    ev = events(extra={"scenario": "freeze", "truth": 55.5, "reported": 40.0, "injected": "drift"})
    text = json.dumps(d.build_incident(ev, ser(lambda t: 40.0), ser(lambda t: 20 + t)))
    for w in SCENARIO_WORDS + ("scenario", "truth", "reported", "injected"):
        assert w not in text.lower(), w
    assert "55.5" not in text


# ------------------------------------------------------------------ rule_label


@pytest.mark.parametrize("kind,label", [("freeze", "SPOOF"), ("drift", "SPOOF"), ("replay", "SPOOF"),
                                        ("leak", "LEAK"), ("blind", "CAMERA_FAULT"), ("benign", "BENIGN")])
def test_rule_label(kind, label):
    assert mod("diagnoser").rule_label(incident(kind)) == label


def test_rule_label_tripped_but_sources_agree_is_unsure():
    d = mod("diagnoser")
    inc = d.build_incident(events(trip_at=50, reason="HIGH_HIGH"), ser(lambda t: 20 + t), ser(lambda t: 20 + t))
    assert d.rule_label(inc) == "UNSURE"


# ------------------------------------------------------------------ diagnose: happy path + prompt


def test_diagnose_valid_json(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    tr = Script(good_answer(inc))
    out = d.diagnose(inc, mk_client(tmp_path, tr), "r1")
    assert isinstance(out, d.Diagnosis)
    assert (out.label, out.confidence) == ("SPOOF", 0.8)
    assert len(out.evidence) == 2 and out.tool_calls == ()
    assert len(tr.calls) == 1
    p = prompt_of(tr.calls[0]).lower()
    assert "you advise a person; you cannot control anything" in p
    for w in SCENARIO_WORDS:
        assert w not in json.dumps(inc).lower()
    schema = tr.calls[0]["text"]["format"]["schema"]
    assert set(schema["properties"]["label"]["enum"]) == {"SPOOF", "LEAK", "CAMERA_FAULT", "BENIGN", "UNSURE"}


def test_evidence_within_rounding_is_grounded(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")  # plc stays at 40.0, camera reaches 80.0
    ans = good_answer(inc, evidence=["PLC about 40 the whole time", "camera hit 80.04 at t=60"])
    assert isinstance(d.diagnose(inc, mk_client(tmp_path, Script(ans)), "r1"), d.Diagnosis)


# ------------------------------------------------------------------ strict validation


@pytest.mark.parametrize("over", [
    {"label": "OVERFLOW"}, {"label": "spoof"}, {"label": None},
    {"confidence": 1.5}, {"confidence": -0.1}, {"confidence": "0.8"}, {"confidence": True},
    {"evidence": []}, {"evidence": "PLC 40"}, {"evidence": [f"value {i} 40" for i in range(5)]},
    {"evidence": ["PLC stuck at 40 " + "x" * 200]}, {"evidence": [40]},
    {"operator_note": "n" * 281}, {"operator_note": ""}, {"operator_note": 5},
    {"suggested_check": "c" * 161}, {"suggested_check": ""},
    {"extra_field": "x"},
])
def test_schema_invalid_is_rejected(tmp_path, over):
    d = mod("diagnoser")
    inc = incident("freeze")
    out = d.diagnose(inc, mk_client(tmp_path, Script(good_answer(inc, **over))), "r1")
    assert isinstance(out, d.DiagnosisError) and out.reason == "schema"


@pytest.mark.parametrize("raw", ['["a"]', '"text"', "42", "null"])
def test_non_object_is_rejected(tmp_path, raw):
    d = mod("diagnoser")
    out = d.diagnose(incident("freeze"), mk_client(tmp_path, Script(raw)), "r1")
    assert isinstance(out, d.DiagnosisError) and out.reason == "schema"


def test_missing_key_is_rejected(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    ans = good_answer(inc)
    del ans["suggested_check"]
    out = d.diagnose(inc, mk_client(tmp_path, Script(ans)), "r1")
    assert out.reason == "schema"


def test_ungrounded_number_is_rejected_never_shown(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    out = d.diagnose(inc, mk_client(tmp_path, Script(good_answer(inc, evidence=["PLC read 73.4 at t=12"]))), "r1")
    assert isinstance(out, d.DiagnosisError) and out.reason == "ungrounded"
    assert not hasattr(out, "label")


def test_evidence_without_any_number_is_rejected(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    out = d.diagnose(inc, mk_client(tmp_path, Script(good_answer(inc, evidence=["the PLC looks stuck"]))), "r1")
    assert out.reason == "ungrounded"


# ------------------------------------------------------------------ refusal / timeout / parse / api


def test_refusal_is_unavailable_no_retry(tmp_path):
    d = mod("diagnoser")
    tr = Script((200, {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}))
    out = d.diagnose(incident("freeze"), mk_client(tmp_path, tr), "r1")
    assert isinstance(out, d.DiagnosisError) and out.reason == "refusal"
    assert len(tr.calls) == 1


@pytest.mark.parametrize("reply,reason", [
    (TimeoutError("slow"), "timeout"), ("not json at all", "parse"), ((500, {}), "api"),
    (RuntimeError("boom"), "unavailable"),
])
def test_failures_become_diagnosis_error(tmp_path, reply, reason):
    d = mod("diagnoser")
    out = d.diagnose(incident("freeze"), mk_client(tmp_path, Script(reply)), "r1")
    assert isinstance(out, d.DiagnosisError) and out.reason == reason
    assert "diagnosis unavailable" in str(out)


def test_cassette_miss_is_unavailable(tmp_path):
    d = mod("diagnoser")
    out = d.diagnose(incident("freeze"), mk_client(tmp_path, Script("x"), mode="replay"), "r1")
    assert isinstance(out, d.DiagnosisError)


# ------------------------------------------------------------------ tools (single round, capped at 2)


READINGS = [{"t": float(i), "level_pct": 20.0 + i, "confidence": 0.37 if i == 30 else 0.9, "status": "OK"}
            for i in range(61)]


def test_tool_round_works(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    req = {"tool_request": {"name": "recheck_frame", "args": {"t": 30}}}
    final = good_answer(inc, evidence=["recorded camera confidence at t=30 was 0.37", "PLC stuck at 40.0"])
    tr = Script(req, final)
    out = d.diagnose(inc, mk_client(tmp_path, tr), "r1", tools=d.Tools(inc, READINGS))
    assert isinstance(out, d.Diagnosis)
    assert len(tr.calls) == 2
    assert "0.37" in prompt_of(tr.calls[1]) and prompt_of(tr.calls[1]) != prompt_of(tr.calls[0])
    assert [c["name"] for c in out.tool_calls] == ["recheck_frame"]


def test_number_only_in_tool_result_is_grounded_but_not_otherwise(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    ans = good_answer(inc, evidence=["recorded confidence 0.37 at t=30"])
    out = d.diagnose(inc, mk_client(tmp_path, Script(ans)), "r1", tools=d.Tools(inc, READINGS))
    assert out.reason == "ungrounded"  # no tool round happened, so 0.37 is not in the data


def test_tool_requests_capped_at_two(tmp_path):
    d = mod("diagnoser")
    inc = incident("freeze")
    reqs = [{"name": "get_series", "args": {"name": "plc", "from_s": 0, "to_s": 10}} for _ in range(3)]
    tr = Script({"tool_request": reqs}, good_answer(inc))
    out = d.diagnose(inc, mk_client(tmp_path, tr), "r1", tools=d.Tools(inc, READINGS))
    assert isinstance(out, d.Diagnosis)
    assert len(out.tool_calls) == 2 and len(tr.calls) == 2


def test_second_tool_request_is_rejected(tmp_path):
    d = mod("diagnoser")
    req = {"tool_request": {"name": "get_series", "args": {"name": "plc", "from_s": 0, "to_s": 10}}}
    tr = Script(req, req)
    out = d.diagnose(incident("freeze"), mk_client(tmp_path, tr), "r1")
    assert isinstance(out, d.DiagnosisError) and out.reason == "tool"
    assert len(tr.calls) == 2  # never a third


def test_tools_are_read_only_and_validate_args():
    d = mod("diagnoser")
    inc = incident("freeze")
    tools = d.Tools(inc, READINGS)
    assert tools.NAMES == ("get_series", "recheck_frame")
    r = tools.run("get_series", {"name": "camera", "from_s": 10, "to_s": 20})
    assert [p[0] for p in r["points"]] == [float(t) for t in range(10, 21)]
    assert "error" in tools.run("get_series", {"name": "pump", "from_s": 0, "to_s": 5})
    assert "error" in tools.run("get_series", {"name": "plc", "from_s": "a", "to_s": 5})
    assert "error" in tools.run("set_pump", {"on": True})
    assert "error" in tools.run("recheck_frame", {"t": "x"})
    assert tools.run("recheck_frame", {"t": 30.4})["status"] == "OK"
    assert "error" in d.Tools(inc).run("recheck_frame", {"t": 3})  # no recorded readings


# ------------------------------------------------------------------ advisor (background job)


def test_advisor_ready_and_rule_cross_check(tmp_path):
    a = mod("advisor")
    inc = incident("freeze")
    adv = a.Advisor(mk_client(tmp_path, Script(good_answer(inc))), "r1")
    assert adv.snapshot()["state"] == "idle"
    ev, plc, cam = events(), ser(lambda t: 40.0), ser(lambda t: 20 + t)
    assert adv.submit(ev, plc, cam, trigger="trip") is True
    assert adv.wait(5)
    s = adv.snapshot()
    assert s["state"] == "ready" and s["label"] == "SPOOF" and s["rule_label"] == "SPOOF" and s["agrees"] is True
    assert s["trigger"] == "trip" and s["evidence"]
    json.dumps(s)


def test_advisor_flags_disagreement(tmp_path):
    a = mod("advisor")
    inc = incident("freeze")
    adv = a.Advisor(mk_client(tmp_path, Script(good_answer(inc, label="CAMERA_FAULT"))), "r1")
    adv.submit(events(), ser(lambda t: 40.0), ser(lambda t: 20 + t), trigger="trip")
    adv.wait(5)
    s = adv.snapshot()
    assert s["label"] == "CAMERA_FAULT" and s["rule_label"] == "SPOOF" and s["agrees"] is False


def test_advisor_unavailable_keeps_rule_label(tmp_path):
    a = mod("advisor")
    adv = a.Advisor(mk_client(tmp_path, Script(TimeoutError("t"))), "r1")
    adv.submit(events(), ser(lambda t: 40.0), ser(lambda t: 20 + t), trigger="trip")
    adv.wait(5)
    s = adv.snapshot()
    assert s["state"] == "unavailable" and s["reason"] == "timeout" and s["rule_label"] == "SPOOF"
    assert s.get("label") is None


def test_advisor_runs_one_job_at_a_time(tmp_path):
    a = mod("advisor")
    tr = Script("{}", delay=0.4)
    adv = a.Advisor(mk_client(tmp_path, tr), "r1")
    args = (events(), ser(lambda t: 40.0), ser(lambda t: 20 + t))
    assert adv.submit(*args, trigger="trip") is True
    assert adv.snapshot()["state"] == "pending"
    assert adv.submit(*args, trigger="trip") is False
    adv.wait(5)
    assert len(tr.calls) == 1


# ------------------------------------------------------------------ pipeline wiring


class StubAdvice:
    def __init__(self):
        self.calls = []

    def submit(self, events, plc_series, cam_series, trigger="trip", limits=None):
        self.calls.append({"events": events, "plc": plc_series, "cam": cam_series, "trigger": trigger,
                           "limits": limits})
        return True

    def snapshot(self):
        return {"state": "idle"}


def run_demo(advice, scenario="freeze", start=20.0, readings=None):
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")
    rv = v.ReplayVision(readings) if readings else v.ReplayVision.demo()
    pl = p.Pipeline(rv, scenario=scenario, fault_start_s=start, advice=advice)
    for f in cs.FakeSource(times=rv.times()):
        pl.tick(f)
    return pl


def test_trip_queues_exactly_one_job_with_clean_data():
    adv = StubAdvice()
    pl = run_demo(adv)
    assert pl.state.tripped and len(adv.calls) == 1
    c = adv.calls[0]
    assert c["trigger"] == "trip" and c["limits"]["high_high_pct"] == 95.0
    allowed = {"t", "status", "pump_on", "tripped", "trip_reason", "actions"}
    assert all(set(e) <= allowed for e in c["events"])
    assert all(len(p) == 2 for p in c["plc"] + c["cam"])
    assert "freeze" not in json.dumps(c).lower() and "truth" not in json.dumps(c).lower()


def test_benign_run_queues_nothing():
    adv = StubAdvice()  # the demo pour reaches 100 % and legitimately trips HIGH_HIGH, so use a calm recording
    calm = [{"t": float(i), "level_pct": 20.0 + i * 0.5, "confidence": 0.9, "status": "OK"} for i in range(40)]
    pl = run_demo(adv, scenario="benign", readings=calm)
    assert not pl.state.tripped and adv.calls == []


def test_camera_unverified_over_10s_queues_one_job():
    adv = StubAdvice()
    recs = [{"t": float(i), "level_pct": 30.0 + i * 0.1, "confidence": 0.9, "status": "OK"} if i < 5 else
            {"t": float(i), "level_pct": None, "confidence": 0.0, "status": "CAMERA_BLIND"} for i in range(30)]
    run_demo(adv, scenario="benign", readings=recs)
    assert len(adv.calls) == 1 and adv.calls[0]["trigger"] == "camera_unverified"
    assert adv.calls[0]["plc"][-1][0] == 16.0  # first tick with unverified time > 10 s (blind from t=5)


def test_pipeline_without_advice_snapshot_has_diagnosis_off():
    pl = run_demo(None, scenario="benign")
    assert pl.snapshot()["diagnosis"]["state"] == "off"


def test_slow_diagnosis_does_not_block_ticks(tmp_path):
    a, p, v, cs = mod("advisor"), mod("pipeline"), mod("vision"), mod("camera_source")
    tr = Script("{}", delay=2.0)
    adv = a.Advisor(mk_client(tmp_path, tr), "r1")
    rv = v.ReplayVision.demo()
    pl = p.Pipeline(rv, scenario="freeze", fault_start_s=20.0, advice=adv)
    durations, pending_seen = [], False
    t0 = time.monotonic()
    for f in cs.FakeSource(times=rv.times()):
        s = time.monotonic()
        pl.tick(f)
        durations.append(time.monotonic() - s)
        pending_seen = pending_seen or pl.snapshot()["diagnosis"]["state"] == "pending"
    loop_s = time.monotonic() - t0
    assert pl.state.tripped and pending_seen
    assert max(durations) < 0.5 and loop_s < 1.8  # the 2 s model call never held the tick loop
    assert adv.wait(10)
    assert pl.snapshot()["diagnosis"]["state"] in ("ready", "unavailable")
    assert pl.state.tripped  # the diagnosis result changed nothing


def test_diagnosis_result_cannot_clear_a_trip(tmp_path):
    a = mod("advisor")
    inc = incident("freeze")
    adv = a.Advisor(mk_client(tmp_path, Script(good_answer(inc, label="BENIGN"))), "r1")
    pl = run_demo(adv)
    adv.wait(5)
    assert adv.snapshot()["state"] in ("ready", "unavailable")
    assert pl.state.tripped and pl.permit is False


# ------------------------------------------------------------------ dashboard card


def app_with(snap):
    from fastapi.testclient import TestClient

    p, v, cs, w = mod("pipeline"), mod("vision"), mod("camera_source"), mod("web")

    class Adv:
        def submit(self, *a, **k):
            return True

        def snapshot(self):
            return snap

    rv = v.ReplayVision.demo()
    pl = p.Pipeline(rv, scenario="benign", advice=Adv())
    for f in list(cs.FakeSource(times=rv.times()))[:5]:
        pl.tick(f)
    return pl, TestClient(w.create_app(pl), base_url="http://127.0.0.1")


READY = {"state": "ready", "trigger": "trip", "label": "SPOOF", "confidence": 0.82,
         "evidence": ["PLC flat at 40.0 for 60.0 s <b>x</b>", "camera reached 80.0"],
         "operator_note": "<script>alert(1)</script> PLC looks stuck.", "suggested_check": "Look at the tank.",
         "rule_label": "SPOOF", "agrees": True, "reason": None}


def test_card_ready_is_escaped_and_marked_advisory():
    _, c = app_with(READY)
    html = c.get("/").text
    assert "Diagnosis (advisory - Muse Spark)" in html
    assert "SPOOF" in html and "82" in html and "camera reached 80.0" in html
    assert "advisory only - does not control the pump" in html
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    frag = c.get("/diagnosis").text
    assert "&lt;script&gt;" in frag and "<script>alert(1)" not in frag


def test_card_flags_disagreement_with_rule_baseline():
    _, c = app_with({**READY, "label": "CAMERA_FAULT", "agrees": False})
    html = c.get("/").text
    assert "DISAGREES" in html and "SPOOF" in html


def test_card_unavailable():
    _, c = app_with({"state": "unavailable", "reason": "refusal", "rule_label": "SPOOF", "agrees": None})
    html = c.get("/").text
    assert "diagnosis unavailable" in html and "advisory only - does not control the pump" in html
    assert "SPOOF" in html  # rule baseline still shown


def test_card_pending_and_state_json():
    _, c = app_with({"state": "pending", "trigger": "trip"})
    assert "diagnosing" in c.get("/").text.lower()
    assert c.get("/api/state").json()["diagnosis"]["state"] == "pending"


def test_diagnosis_route_is_get_only_no_posts():
    _, c = app_with(READY)
    assert c.post("/diagnosis").status_code in (404, 405)
    assert c.get("/diagnosis", headers={"host": "evil.example"}).status_code == 400


# ------------------------------------------------------------------ advisor construction / CLI flag


def test_build_advisor_modes(tmp_path):
    a = mod("advisor")
    assert a.build_advisor("off", run_id="r") is None
    with pytest.raises(ValueError):
        a.build_advisor("bogus", run_id="r")
    with pytest.raises(ValueError):  # live: no client and no key file
        a.build_advisor("live", run_id="r")
    fake = a.build_advisor("fake", run_id="r")
    fake.submit(events(), ser(lambda t: 40.0), ser(lambda t: 20 + t), trigger="trip")
    assert fake.wait(5)
    s = fake.snapshot()
    assert s["state"] == "ready" and s["label"] == s["rule_label"] == "SPOOF"  # offline demo model


def test_live_uses_injected_client_or_key_file_and_never_leaks_key(tmp_path, caplog):
    a = mod("advisor")
    inc = incident("freeze")
    injected = mk_client(tmp_path, Script(good_answer(inc)))
    adv = a.build_advisor("live", run_id="r", client=injected)
    assert adv.client is injected
    key = tmp_path / "k.txt"
    key.write_text("sk-SECRET-123\n", encoding="utf-8")
    with caplog.at_level("DEBUG"):
        adv2 = a.build_advisor("live", run_id="r", key_file=str(key), base_url="http://127.0.0.1:9/v1")
    assert adv2.client.mode == "live"  # built, but no request was made
    assert "sk-SECRET-123" not in caplog.text and "sk-SECRET-123" not in repr(adv2) + repr(adv2.client.__dict__)
    with pytest.raises(ValueError):
        a.build_advisor("live", run_id="r", key_file=str(tmp_path / "missing.txt"), base_url="http://127.0.0.1:9")
    with pytest.raises(ValueError):  # a key file needs a base url; none is invented
        a.build_advisor("live", run_id="r", key_file=str(key))


def test_cassette_mode_roundtrip(tmp_path):
    a = mod("advisor")
    inc = incident("freeze")
    rec = mk_client(tmp_path, Script(good_answer(inc)), mode="record", cas="c1", run_id="r9")
    adv = a.Advisor(rec, "r9")
    adv.submit(events(), ser(lambda t: 40.0), ser(lambda t: 20 + t), trigger="trip")
    adv.wait(5)
    assert adv.snapshot()["state"] == "ready"
    replay = a.build_advisor(f"cassette:{tmp_path / 'c1'}", run_id="r9")
    replay.submit(events(), ser(lambda t: 40.0), ser(lambda t: 20 + t), trigger="trip")
    replay.wait(5)
    assert replay.snapshot()["state"] == "ready"
    miss = a.build_advisor(f"cassette:{tmp_path / 'nothing'}", run_id="r9")
    miss.submit(events(), ser(lambda t: 40.0), ser(lambda t: 20 + t), trigger="trip")
    miss.wait(5)
    assert miss.snapshot()["state"] == "unavailable"


def test_cli_flags_parse():
    import argparse

    a = mod("advisor")
    p = argparse.ArgumentParser()
    a.add_diagnoser_args(p)
    ns = p.parse_args([])
    assert ns.diagnoser == "off" and ns.key_file is None
    ns = p.parse_args(["--diagnoser", "cassette:/x", "--key-file", "k.txt", "--base-url", "http://127.0.0.1:1"])
    assert (ns.diagnoser, ns.key_file) == ("cassette:/x", "k.txt")


# ------------------------------------------------------------------ relay


def test_default_relay_is_screen_and_follows_permit_output():
    pm, rl, il = mod("permit"), mod("relay"), mod("interlock")
    out = pm.PermitOutput()
    assert isinstance(out.relay, rl.ScreenRelay) and out.relay.snapshot()["pump_enabled"] is True
    out.apply([il.Action.ALARM_CAMERA_UNVERIFIED])
    assert out.relay.snapshot()["pump_enabled"] is True
    out.apply([il.Action.WRITE_PERMIT_FALSE])
    assert out.relay.snapshot()["pump_enabled"] is False and out.permit is False
    assert out.relay.snapshot()["driver"] == "screen"


def _grant():
    il = mod("interlock")
    from dataclasses import replace

    st = replace(il.initial_state(), tripped=True, trip_reason="RATE", trip_t=1.0)
    st, actions, grant = il.reset_with_grant(st, by_human=True)
    assert grant is not None
    return grant


def test_relay_restores_only_after_valid_grant():
    pm, il = mod("permit"), mod("interlock")
    out = pm.PermitOutput()
    out.apply([il.Action.WRITE_PERMIT_FALSE])
    for bad in (None, True, "grant", object()):
        with pytest.raises(PermissionError):
            out.rearm(bad)
        assert out.relay.snapshot()["pump_enabled"] is False
    g = _grant()
    out.rearm(g)
    assert out.relay.snapshot()["pump_enabled"] is True and out.permit is True
    with pytest.raises(PermissionError):  # single use
        out.rearm(g)


class FakeSerial:
    def __init__(self):
        self.sent, self.closed = [], False

    def write(self, b):
        self.sent.append(bytes(b))
        return len(b)

    def close(self):
        self.closed = True


def test_usb_serial_relay_sends_off_on_trip_and_on_only_after_rearm():
    pm, rl, il = mod("permit"), mod("relay"), mod("interlock")
    ser_ = FakeSerial()
    relay = rl.UsbSerialRelay("COM9", opener=lambda port: ser_)
    out = pm.PermitOutput(relay=relay)
    assert ser_.sent == [] and relay.UNTESTED_HARDWARE is True
    out.apply([il.Action.ALARM_PLC_STALE])
    assert ser_.sent == []
    out.apply([il.Action.WRITE_PERMIT_FALSE])
    assert ser_.sent == [rl.UsbSerialRelay.OFF_FRAME]
    with pytest.raises(PermissionError):
        out.rearm(None)
    assert ser_.sent == [rl.UsbSerialRelay.OFF_FRAME]
    out.rearm(_grant())
    assert ser_.sent == [rl.UsbSerialRelay.OFF_FRAME, rl.UsbSerialRelay.ON_FRAME]


def test_relay_failure_never_undoes_the_software_cut():
    pm, rl, il = mod("permit"), mod("relay"), mod("interlock")

    class Broken:
        def write(self, b):
            raise OSError("unplugged")

        def close(self):
            pass

    relay = rl.UsbSerialRelay("COM9", opener=lambda port: Broken())
    out = pm.PermitOutput(relay=relay)
    out.apply([il.Action.WRITE_PERMIT_FALSE])  # must not raise
    assert out.permit is False and out.relay_fault is True


def test_pipeline_snapshot_shows_relay_state():
    pl = run_demo(None)
    s = pl.snapshot()
    assert s["relay"]["driver"] == "screen" and s["relay"]["pump_enabled"] is False  # freeze run tripped
    assert pl.permit is False


def test_dashboard_shows_relay_line():
    _, c = app_with({"state": "idle"})
    assert "Relay" in c.get("/").text


# ------------------------------------------------------------------ static safety


FORBIDDEN_CALLS = {"reset", "reset_with_grant", "rearm", "rearm_from_reset", "clear_permit", "redeem", "apply",
                   "write_coil", "write_coils", "write_register", "write_registers", "publish_plc",
                   "publish_camera", "cut", "restore"}
FORBIDDEN_MODS = {"permit", "interlock", "relay", "modbus_io", "fault_injector", "pipeline", "web", "runner"}


def _tree(name):
    return ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8"))


def _imports(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            if n.module == "agentwarden":
                out.update(a.name for a in n.names)
            elif n.module.startswith("agentwarden."):
                out.add(n.module.split(".")[1])
        elif isinstance(n, ast.Import):
            out.update(a.name.split(".")[1] for a in n.names if a.name.startswith("agentwarden."))
    return out


def _call_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
    return out


def _closure(start):
    seen, todo = set(), [start]
    while todo:
        m = todo.pop()
        if m in seen or not (SRC / f"{m}.py").exists():
            continue
        seen.add(m)
        todo.extend(_imports(_tree(m)))
    return seen - {start}


@pytest.mark.parametrize("name", ["diagnoser", "advisor"])
def test_static_diagnoser_side_cannot_reach_the_safety_path(name):
    tree = _tree(name)
    assert _closure(name).isdisjoint(FORBIDDEN_MODS), _closure(name) & FORBIDDEN_MODS
    assert _call_names(tree).isdisjoint(FORBIDDEN_CALLS), _call_names(tree) & FORBIDDEN_CALLS
    text = (SRC / f"{name}.py").read_text(encoding="utf-8")
    for bad in ("WRITE_PERMIT", "ResetGrant", "COIL_PERMIT", "write_permit", "permit.", "interlock"):
        assert bad not in text, bad


def test_static_importers_of_diagnoser_are_clean():
    importers = {p.stem for p in SRC.glob("*.py")
                 if p.stem not in ("diagnoser", "advisor") and {"diagnoser", "advisor"} & _imports(_tree(p.stem))}
    assert "web" in importers and "pipeline" not in importers
    for m in importers:
        assert not ({"permit", "interlock", "relay"} & _imports(_tree(m))), m
        assert _call_names(_tree(m)).isdisjoint(FORBIDDEN_CALLS - {"apply"}), m


def test_static_pipeline_does_not_import_the_diagnoser():
    assert not ({"diagnoser", "advisor"} & _imports(_tree("pipeline")))


def test_static_only_permit_py_drives_the_relay():
    hits = {p.name for p in SRC.glob("*.py") if p.name not in ("permit.py", "relay.py")
            and re.search(r"\.(cut|restore)\(", p.read_text(encoding="utf-8"))}
    assert hits == set()
    assert re.search(r"\.cut\(", (SRC / "permit.py").read_text(encoding="utf-8"))
    assert re.search(r"\.restore\(", (SRC / "permit.py").read_text(encoding="utf-8"))


def test_static_new_modules_do_not_touch_env_or_open_sockets():
    for name in ("diagnoser.py", "advisor.py", "relay.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert "environ" not in text and "getenv" not in text, name
        assert "socket" not in text, name


def test_relay_marks_hardware_path_untested():
    text = (SRC / "relay.py").read_text(encoding="utf-8")
    assert "UNTESTED" in text
