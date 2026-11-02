"""S4fix: dashboard hardening (review M1, m1-m9) + demo polish. Imports are lazy so the red commit collects."""
import copy
import importlib
import inspect
import io
import json
import pickle
import random
import re
import shutil
import threading
import time
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_s3fix import raw_client, tripped_cleared_state
from test_s4 import FakeClient, jpeg, liquid, rising, sam_text
from test_s4 import frame as fframe

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "demo" / "real"
REPLAY = REAL / "pour_replay.json"
FRAMES = REAL / "pour_frames"


def mod(name):
    return importlib.import_module(f"agentwarden.{name}")


def text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def make_app(readings=None, ticks=12, scenario="benign", fault_start=20.0, **kw):
    from fastapi.testclient import TestClient

    p, v, cs, w = mod("pipeline"), mod("vision"), mod("camera_source"), mod("web")
    rv = v.ReplayVision(readings or rising())
    pl = p.Pipeline(rv, scenario=scenario, fault_start_s=fault_start, **kw)
    for f in list(cs.FakeSource(times=rv.times()))[:ticks]:
        pl.tick(f)
    app = w.create_app(pl)
    return pl, app, TestClient(app, base_url="http://127.0.0.1")


GETS = ("/api/state", "/frame.png", "/trend.svg", "/diagnosis")


# ------------------------------------------------------------------ A1: web hardening


@pytest.mark.parametrize("name,path", [("render_frame", "/frame.png"), ("render_trend", "/trend.svg")])
def test_rendering_never_holds_the_tick_lock(monkeypatch, name, path):
    w = mod("web")
    real, entered = getattr(w, name), threading.Event()

    def slow(*a, **k):
        entered.set()
        time.sleep(1.0)
        return real(*a, **k)

    monkeypatch.setattr(w, name, slow)
    pl, app, c = make_app()
    th = threading.Thread(target=lambda: c.get(path))
    th.start()
    assert entered.wait(5)
    t0 = time.perf_counter()
    pl.tick((50.0, b"", 1, 1))
    dt = time.perf_counter() - t0
    th.join(10)
    assert dt < 0.3, dt


@pytest.mark.parametrize("path", GETS)
def test_cross_site_and_foreign_origin_gets_rejected(path):
    pl, app, c = make_app()
    for site in ("cross-site", "same-site"):
        assert c.get(path, headers={"Sec-Fetch-Site": site}).status_code == 403, (path, site)
    assert c.get(path, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.get(path, headers={"Origin": "null"}).status_code == 403
    assert c.get(path, headers={"Referer": "http://evil.example/page"}).status_code == 403
    for ok in ({"Sec-Fetch-Site": "same-origin"}, {"Sec-Fetch-Site": "none"},
               {"Referer": "http://127.0.0.1:8765/"}, {"Origin": "http://localhost:8765"}, {}):
        assert c.get(path, headers=ok).status_code == 200, (path, ok)


def test_security_headers_on_every_response():
    pl, app, c = make_app()
    big = b"token=" + b"a" * 5000
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    cases = [
        (c.get("/"), 200), (c.get("/api/state"), 200), (c.get("/frame.png"), 200), (c.get("/trend.svg"), 200),
        (c.get("/diagnosis"), 200), (c.get("/nope"), 404), (c.get("/reset"), 405),
        (c.post("/reset", data={"token": "x"}), 403), (c.post("/reset", content=big, headers=form), 413),
        (c.get("/api/state", headers={"Host": "evil.example"}), 400),
        (c.get("/frame.png", headers={"Sec-Fetch-Site": "cross-site"}), 403),
        (c.post("/reset", data={"token": "x"}, headers={"Origin": "http://evil.example"}), 403),
    ]
    for r, status in cases:
        assert r.status_code == status, (r.request.url, r.status_code)
        assert r.headers["cross-origin-resource-policy"] == "same-origin", r.request.url
        assert r.headers["x-content-type-options"] == "nosniff", r.request.url
        assert "default-src 'none'" in r.headers["content-security-policy"], r.request.url
    assert "script-src 'nonce-" in cases[0][0].headers["content-security-policy"]


def test_post_body_is_capped_at_4kb():
    pl, app, c = make_app()
    calls = []
    pl.human_reset = lambda: calls.append(1) or {}
    tok = app.state.token
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    assert c.post("/reset", content=f"token={tok}&pad={'a' * 5000}".encode(), headers=form).status_code == 413
    assert c.post("/reset", content=(b"a" * 3000 for _ in range(2)), headers=form).status_code == 413  # chunked
    assert c.post("/inject/freeze", content=b"a" * 5000, headers={"X-Token": tok}).status_code == 413
    assert c.post("/recalibrate", content=b"a" * 5000, headers={"X-Token": tok}).status_code == 413
    assert calls == [] and pl.injector.scenario == "benign"
    assert c.post("/reset", data={"token": tok}, follow_redirects=False).status_code == 303
    assert calls == [1]


def test_renders_are_cached_per_tick(monkeypatch):
    w = mod("web")
    pl, app, c = make_app()
    n = {"frame": 0, "trend": 0}
    rf, rt = w.render_frame, w.render_trend
    monkeypatch.setattr(w, "render_frame", lambda *a, **k: n.__setitem__("frame", n["frame"] + 1) or rf(*a, **k))
    monkeypatch.setattr(w, "render_trend", lambda *a, **k: n.__setitem__("trend", n["trend"] + 1) or rt(*a, **k))
    for _ in range(20):
        assert c.get("/frame.png").status_code == 200
        assert c.get("/trend.svg").status_code == 200
    assert n == {"frame": 1, "trend": 1}
    pl.tick((50.0, b"", 1, 1))
    c.get("/frame.png")
    c.get("/trend.svg")
    assert n == {"frame": 2, "trend": 2}


# ------------------------------------------------------------------ A2: ResetGrant


def _grant():
    return mod("interlock").reset_with_grant(tripped_cleared_state(), by_human=True)[2]


def _hand_copy(g):
    il = mod("interlock")
    clone = object.__new__(il.ResetGrant)
    for slot in il.ResetGrant.__slots__:
        if hasattr(g, slot):
            setattr(clone, slot, getattr(g, slot))
    return clone


def test_grant_cannot_be_copied_or_pickled():
    g = _grant()
    for fn in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises((PermissionError, TypeError)):
            fn(g)


def test_hand_made_copy_of_a_grant_cannot_be_redeemed_twice():
    il = mod("interlock")
    g = _grant()
    clone = _hand_copy(g)
    il.redeem(g)
    with pytest.raises(PermissionError):
        il.redeem(clone)
    g2 = _grant()
    clone2 = _hand_copy(g2)
    il.redeem(clone2)
    with pytest.raises(PermissionError):
        il.redeem(g2)
    with pytest.raises(PermissionError):
        il.redeem(object.__new__(il.ResetGrant))  # no identity at all


def test_grant_docs_are_honest_about_the_limit():
    doc = (mod("interlock").ResetGrant.__doc__ or "").lower()
    assert "in-process" in doc and "not a security boundary" in doc


def test_permit_output_fails_safe_when_the_wire_is_dead():
    il, pm = mod("interlock"), mod("permit")

    class Dead:
        def read_permit(self):
            raise OSError("wire down")

        def clear_permit(self):
            raise OSError("wire down")

    out = pm.PermitOutput(client=Dead())
    assert out.permit is False  # unreadable = not permitted
    out.apply([il.Action.WRITE_PERMIT_FALSE])  # must not raise
    assert out.permit is False


# ------------------------------------------------------------------ A3: HttpShotSource


def serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data", "http://169.254.1.1/shot.jpg",
                                 "http://[fe80::1]/shot.jpg"])
def test_http_shot_refuses_link_local(url):
    with pytest.raises(ValueError):
        mod("camera_source").HttpShotSource(url)


@pytest.mark.parametrize("url", ["http://[fd12:3456::1]/shot.jpg", "http://[::1]:8080/shot.jpg",
                                 "http://192.168.0.7/shot.jpg", "http://127.0.0.1/x"])
def test_http_shot_still_allows_loopback_rfc1918_ula(url):
    mod("camera_source").HttpShotSource(url)


class _Handler(BaseHTTPRequestHandler):
    body, status, with_length = b"", 200, True

    def do_GET(self):
        self.send_response(self.status)
        if self.with_length:
            self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


def _handler(**kw):
    return type("H", (_Handler,), kw)


def test_http_shot_ignores_proxy_settings(monkeypatch):
    import httpx

    cs = mod("camera_source")
    seen, real = {}, httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: seen.update(k) or real(*a, **k))
    srv = serve(_handler(body=jpeg(100, 80)))
    try:
        assert len(list(cs.HttpShotSource(f"http://127.0.0.1:{srv.server_port}/s.jpg", n=1))) == 1
    finally:
        srv.shutdown()
    assert seen.get("trust_env") is False


@pytest.mark.parametrize("with_length", [True, False])
def test_http_shot_caps_the_response_size(with_length):
    cs = mod("camera_source")
    srv = serve(_handler(body=b"\xff\xd8\xff" + b"0" * 200_000, with_length=with_length))
    try:
        src = cs.HttpShotSource(f"http://127.0.0.1:{srv.server_port}/s.jpg", n=1, max_bytes=50_000)
        with pytest.raises((ValueError, RuntimeError), match="too large"):
            list(src)
    finally:
        srv.shutdown()
    assert inspect.signature(cs.HttpShotSource.__init__).parameters["max_bytes"].default == 5_000_000


def test_http_shot_errors_never_carry_the_url_or_credentials():
    cs = mod("camera_source")
    srv = serve(_handler(status=500, body=b"nope"))
    try:
        src = cs.HttpShotSource(f"http://127.0.0.1:{srv.server_port}/secret-path-xyz/shot.jpg", user="cam",
                                password="s3cret-pw", n=1)
        with pytest.raises(Exception) as ei:
            list(src)
    finally:
        srv.shutdown()
    msg = str(ei.value) + repr(ei.value)
    assert "secret-path-xyz" not in msg and "s3cret-pw" not in msg and "500" in msg
    assert ei.value.__cause__ is None and (ei.value.__suppress_context__ or ei.value.__context__ is None)


# ------------------------------------------------------------------ A4/A5: watchdog, PLC age


def test_vision_exception_becomes_camera_blind_and_the_loop_survives():
    p, cs = mod("pipeline"), mod("camera_source")

    class Boom:
        def read(self, f, index=None):
            raise RuntimeError("sam down")

    pl = p.Pipeline(Boom(), "benign")
    evs = [pl.tick(f) for f in cs.FakeSource(n=15)]
    assert all(e["status"] == "CAMERA_BLIND" for e in evs)
    assert "ALARM_CAMERA_UNVERIFIED" in evs[-1]["actions"] and not pl.state.tripped
    assert "CAMERA_READ_ERROR" in pl.snapshot()["alarms"]
    pl2 = p.Pipeline(Boom(), "benign")
    p.drive(pl2, cs.FakeSource(n=15), latest_wins=True)  # the worker path must still tick
    assert pl2.n_ticks >= 1 and pl2.last["status"] == "CAMERA_BLIND"


def test_drive_survives_a_tick_that_raises():
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")
    pl = p.Pipeline(v.ReplayVision(rising()), "benign")
    orig = pl.tick

    def flaky(frame, analysis=None):
        if frame[0] == 3.0:
            raise RuntimeError("boom")
        return orig(frame, analysis)

    pl.tick = flaky
    p.drive(pl, cs.FakeSource(n=10))
    assert pl.loop_errors == 1 and pl.last_t == 9.0 and pl.finished


class _Flaky:
    """A Modbus client whose PLC read can be broken on demand."""

    def __init__(self, real):
        self.real, self.broken = real, False

    def __getattr__(self, name):
        return getattr(self.real, name)

    def read_plc_level(self):
        if self.broken:
            raise OSError("wire down")
        return self.real.read_plc_level()


def test_plc_read_failure_counts_toward_staleness_and_the_loop_keeps_running():
    p, v, cs, mb = mod("pipeline"), mod("vision"), mod("camera_source"), mod("modbus_io")
    flat = [{"t": float(i), "level_pct": 50.0, "confidence": 0.9, "status": "OK"} for i in range(60)]
    rv = v.ReplayVision(flat)
    with mb.ModbusPlcServer(port=0) as srv:
        real = mb.ModbusClient("127.0.0.1", srv.port)
        client = _Flaky(real)
        try:
            pl = p.Pipeline(rv, "benign", server=srv, client=client)
            evs = []
            for f in cs.FakeSource(times=rv.times()):
                client.broken = f[0] >= 20.0  # last good read at t=19
                evs.append(pl.tick(f))
        finally:
            real.close()
    assert len(evs) == 60 and not pl.state.tripped
    stale = [e["t"] for e in evs if "ALARM_PLC_STALE" in e["actions"]]
    assert stale == [float(t) for t in range(30, 60)]  # age 11 > 10 s from t=30 on, none before


def test_loop_stalled_alarm_and_age_on_the_dashboard():
    from fastapi.testclient import TestClient

    p, v, w = mod("pipeline"), mod("vision"), mod("web")
    now = [100.0]
    pl = p.Pipeline(v.ReplayVision(rising()), "benign", clock=lambda: now[0])
    pl.tick((0.0, b"", 1, 1))
    s = pl.snapshot()
    assert s["loop_age_s"] == 0 and "LOOP_STALLED" not in s["alarms"]
    now[0] += 2.9
    assert "LOOP_STALLED" not in pl.snapshot()["alarms"]
    now[0] = 104.0
    s = pl.snapshot()
    assert s["loop_age_s"] == pytest.approx(4.0) and "LOOP_STALLED" in s["alarms"]
    c = TestClient(w.create_app(pl), base_url="http://127.0.0.1")
    html = c.get("/").text
    assert "LOOP_STALLED" in html and "loop last tick 4.0 s ago" in text(html)
    assert "LOOP_STALLED" in c.get("/api/state").json()["alarms"]
    pl.finished = True  # a finished replay is not a stalled loop
    assert "LOOP_STALLED" not in pl.snapshot()["alarms"]


# ------------------------------------------------------------------ A6: recalibrate


def test_camera_moved_stays_blocked_until_a_human_recalibrates():
    from fastapi.testclient import TestClient

    p, v, w = mod("pipeline"), mod("vision"), mod("web")
    lv = v.LiveVision(FakeClient(liquid(30)))
    pl = p.Pipeline(lv, "benign")
    app = w.create_app(pl)
    c = TestClient(app, base_url="http://127.0.0.1")
    assert pl.tick(fframe(0.0))["status"] == "OK"
    lv.client.texts["plastic bottle"] = sam_text([[1] * 20 for _ in range(50)], (60, 5, 79, 54))
    lv.client.texts["orange liquid"] = sam_text([[1] * 18 for _ in range(25)], (61, 30, 78, 54))
    for t in (1.0, 2.0, 3.0):  # permanent until a human acts
        ev = pl.tick(fframe(t))
        assert ev["camera_moved"] and ev["status"] == "LOW_CONFIDENCE"
    assert pl.snapshot()["camera_notice"] == "camera moved - recalibrate"
    assert "camera moved - recalibrate" in c.get("/").text
    assert c.get("/recalibrate").status_code == 405
    assert c.post("/recalibrate").status_code == 403
    assert c.post("/recalibrate", data={"token": "nope"}).status_code == 403
    assert c.post("/recalibrate", data={"token": app.state.token},
                  headers={"Origin": "http://evil.example"}).status_code == 403
    assert pl.tick(fframe(4.0))["camera_moved"]
    r = c.post("/recalibrate", data={"token": app.state.token}, follow_redirects=False)
    assert r.status_code == 303
    ev = pl.tick(fframe(5.0))
    assert not ev["camera_moved"] and ev["status"] != "LOW_CONFIDENCE"
    assert pl.snapshot()["camera_notice"] != "camera moved - recalibrate"
    assert any("recalibrat" in e["text"] for e in pl.snapshot()["events"])


def test_recalibrate_without_a_live_camera_changes_nothing():
    pl, app, c = make_app()
    assert pl.recalibrate() is False
    assert c.post("/recalibrate", data={"token": app.state.token}, follow_redirects=False).status_code == 303


# ------------------------------------------------------------------ A7: closed loop through Pipeline + Modbus


def test_closed_loop_through_pipeline_with_modbus_under_per_tick_attack():
    il, p, cs, mb = mod("interlock"), mod("pipeline"), mod("camera_source"), mod("modbus_io")
    trips = rearms = attacks = 0
    for seed in range(12):
        rng = random.Random(seed)
        scenario = rng.choice(["freeze", "drift", "replay", "benign"])
        used = _grant()
        il.redeem(used)
        stale_copy = _hand_copy(used)  # a clone of a spent grant
        with mb.ModbusPlcServer(port=0) as srv:
            client = mb.ModbusClient("127.0.0.1", srv.port)
            raw = raw_client(srv.port)
            try:
                pl = p.Pipeline(None, scenario, fault_start_s=rng.uniform(2, 15), seed=seed, server=srv,
                                client=client)
                for f in cs.FakeSource(n=90, fps=1.0):
                    before = srv.permit
                    assert raw.write_coil(1, True, device_id=1).isError()
                    assert raw.write_coils(1, [True], device_id=1).isError()
                    for bad in (None, True, "grant", stale_copy):
                        with pytest.raises(PermissionError):
                            srv.rearm_from_reset(bad)
                    attacks += 6
                    pl.tick(f)
                    accepted = False
                    if pl.state.tripped and rng.random() < 0.3:
                        if rng.random() < 0.5:
                            pl.inject("benign")  # the human fixes the fault, then resets
                        pl.human_reset()
                        accepted = not pl.state.tripped
                    after = srv.permit
                    assert before or not after or accepted, (seed, f[0])  # permit never returns without a grant
                    assert after == pl.permit
                    rearms += accepted and not before
                    trips += pl.state.tripped and before
            finally:
                raw.close()
                client.close()
    assert trips >= 3 and rearms >= 2 and attacks > 5000  # exercised, not vacuous


# ------------------------------------------------------------------ B: hero, pill, view model


def snap(plc=50.0, cam=50.0, tripped=False, status="OK", moved=False, T=15.0, pump=True, permit=True, alarms=()):
    return {"plc_level": plc, "pump_on": pump, "permit": permit, "tripped": tripped,
            "trip_reason": "DIVERGENCE" if tripped else None, "alarms": list(alarms),
            "camera": {"level_pct": cam, "status": status, "camera_moved": moved, "confidence": 0.9,
                       "alertable": status == "OK"},
            "camera_blind_notice": status in ("CAMERA_BLIND", "LOW_CONFIDENCE") or moved,
            "limits": {"T_pts": T}}


@pytest.mark.parametrize("kw,pill,label,gap_class", [
    (dict(plc=50, cam=52), "normal", "NORMAL", "ok"),
    (dict(plc=50, cam=57), "normal", "NORMAL", "ok"),  # gap 7 <= T/2
    (dict(plc=50, cam=58), "watching", "WATCHING", "amber"),  # gap 8 > T/2 = 7.5
    (dict(plc=50, cam=50, alarms=["ALARM_PLC_STALE"]), "watching", "WATCHING", "ok"),
    (dict(plc=50, cam=50, alarms=["LOOP_STALLED"]), "watching", "WATCHING", "ok"),
    (dict(plc=24, cam=70, tripped=True, pump=False, permit=False), "tripped", "TRIPPED - PUMP LOCKED OFF", "red"),
    (dict(plc=24, cam=None, status="CAMERA_BLIND"), "blocked", "CAMERA BLOCKED - NO TRIP", "ok"),
    (dict(plc=24, cam=70, status="LOW_CONFIDENCE"), "blocked", "CAMERA BLOCKED - NO TRIP", "ok"),
    (dict(plc=24, cam=70, status="LOW_CONFIDENCE", moved=True), "blocked", "CAMERA BLOCKED - NO TRIP", "ok"),
    (dict(plc=24, cam=None, status="CAMERA_BLIND", tripped=True), "tripped", "TRIPPED - PUMP LOCKED OFF", "red"),
])
def test_hmi_view_pill_and_gap_colour(kw, pill, label, gap_class):
    v = mod("web").hmi_view(snap(**kw))
    assert (v["pill"], v["pill_text"], v["gap_class"]) == (pill, label, gap_class)


def test_hmi_view_numbers_and_pump():
    web = mod("web")
    v = web.hmi_view(snap(plc=24.4, cam=69.6, tripped=True, pump=False, permit=False))
    assert (v["plc_txt"], v["cam_txt"], v["gap_txt"]) == ("24%", "70%", "GAP 45 pts")
    assert v["pump_txt"] == "PUMP OFF" and "OFF" in v["permit_txt"]
    v = web.hmi_view(snap(plc=50, cam=None, status="CAMERA_BLIND"))
    assert (v["cam_txt"], v["gap_txt"]) == ("--", "GAP -")
    assert web.hmi_view(snap(pump=True))["pump_txt"] == "PUMP ON"
    assert web.hmi_view(snap(T=40, plc=50, cam=65))["pill"] == "normal"  # T/2 = 20 comes from the snapshot


def _with_snapshot(pl, fake):
    real = pl.snapshot()
    pl.snapshot = lambda: {**real, **fake, "camera": {**real["camera"], **fake["camera"]}}


def test_page_shows_hero_gap_pill_and_pump_from_the_snapshot():
    pl, app, c = make_app()
    _with_snapshot(pl, snap(plc=24.0, cam=70.0, tripped=True, pump=False, permit=False))
    html = c.get("/").text
    t = text(html)
    assert "CONTROLLER SAYS 24%" in t and "CAMERA SEES 70%" in t and "GAP 46 pts" in t
    assert 'id="h-gap" class="hgap red"' in html and 'id="pill" class="pill tripped"' in html
    assert "TRIPPED - PUMP LOCKED OFF" in t and "PUMP OFF" in t
    assert "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM" in html
    _with_snapshot(pl, snap(plc=24.0, cam=70.0, status="CAMERA_BLIND"))
    html = c.get("/").text
    assert 'id="pill" class="pill blocked"' in html and "CAMERA BLOCKED - NO TRIP" in text(html)
    _with_snapshot(pl, snap(plc=50.0, cam=60.0))
    html = c.get("/").text
    assert 'id="h-gap" class="hgap amber"' in html and 'id="pill" class="pill watching"' in html


def test_api_state_carries_the_view_model():
    pl, app, c = make_app()
    s = c.get("/api/state").json()
    assert {"pill", "pill_text", "plc_txt", "cam_txt", "gap_txt", "gap_class", "pump_txt"} <= set(s["view"])
    assert {"limits", "events", "loop_age_s", "camera_notice", "fault_t"} <= set(s)


def test_big_fonts_and_no_scroll_layout():
    css = mod("web").CSS
    size = lambda sel: int(re.search(re.escape(sel) + r"\{[^}]*?font-size:(\d+)px", css).group(1))  # noqa: E731
    assert size(".big") >= 56 and size(".hgap") >= 56 and size(".pill") >= 40
    assert "100vh" in css and "overflow:hidden" in css.replace(" ", "")
    assert not re.search(r"@import|https?://|font-face", css)


def test_hostile_strings_are_escaped_in_the_new_panels():
    evil = "<img src=x onerror=alert(1)>"
    recs = [{"t": float(i), "level_pct": 40.0, "confidence": 0.9, "status": evil} for i in range(20)]
    pl, app, c = make_app(recs)
    pl.note(1.0, "x", "<script>alert(2)</script>")
    html = c.get("/").text
    assert "<img src=x" not in html and "<script>alert(2)" not in html
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html


# ------------------------------------------------------------------ B: timeline + trend


def test_timeline_keeps_the_last_8_newest_first():
    pl, app, c = make_app()
    for i in range(20):
        pl.note(float(i), "x", f"event {i}")
    ev = pl.snapshot()["events"]
    assert len(ev) == 8 and ev[0]["text"] == "event 19" and ev[-1]["text"] == "event 12"
    html = c.get("/").text
    assert 'id="timeline"' in html and "t=19" in html and "event 19" in html and "event 11" not in html


def test_timeline_records_fault_and_trip_on_a_freeze_run():
    pl, app, c = make_app(ticks=54, scenario="freeze", fault_start=20.0)
    assert pl.state.tripped
    ev = pl.snapshot()["events"]
    kinds = [e["kind"] for e in ev]
    assert "fault" in kinds and kinds[0] == "trip"
    assert [e["t"] for e in ev] == sorted((e["t"] for e in ev), reverse=True)
    t = text(c.get("/").text)
    assert "fault injected: freeze" in t and "TRIPPED: " in t and "t=20" in t
    assert pl.snapshot()["fault_t"] == 20.0


def test_trend_is_dark_thick_and_marks_trip_and_fault():
    w = mod("web")
    hist = [(float(t), 30.0 + t * 0.5, 30.0 + t * 1.2, 0.0) for t in range(60)]
    svg = w.render_trend(hist, trip_t=40.0, fault_t=20.0)
    assert svg.startswith("<svg") and w.CHART_BG in svg
    assert len(re.findall(r'<polyline[^>]*stroke-width="3"', svg)) >= 2
    trip = re.search(r'<line[^>]*data-marker="trip"[^>]*>', svg).group(0)
    assert "#ff" in trip.lower() and "stroke-dasharray" not in trip
    fault = re.search(r'<line[^>]*data-marker="fault"[^>]*>', svg).group(0)
    assert "stroke-dasharray" in fault
    assert "TRIPPED" in svg and "fault injected" in svg
    plain = w.render_trend(hist)
    assert 'data-marker="trip"' not in plain and 'data-marker="fault"' not in plain
    assert "polyline" in w.render_trend([])  # empty history still renders


def test_trend_route_marks_the_events_of_the_run():
    pl, app, c = make_app(ticks=54, scenario="freeze", fault_start=20.0)
    svg = c.get("/trend.svg").text
    assert 'data-marker="trip"' in svg and 'data-marker="fault"' in svg


def test_advisory_card_is_visibly_secondary():
    pl, app, c = make_app()
    html = c.get("/").text
    assert 'class="advisory"' in html and "Muse Spark" in html
    assert "cannot control the pump" in text(html)


# ------------------------------------------------------------------ B: real replay frames


def test_replay_frames_yield_real_frames_and_overlays():
    cs = mod("camera_source")
    rf = cs.ReplayFrames(REPLAY)
    frames = list(rf)
    assert len(frames) == 54 and frames[0].t == 0.0 and frames[53].t == 53.0 and frames[0].w == 478
    assert frames[0].jpeg == (FRAMES / "t001.jpg").read_bytes()
    assert frames[30].jpeg == (FRAMES / "t031.jpg").read_bytes()
    ov = (FRAMES / "ov_t031.jpg").read_bytes()
    assert rf.overlay(30.0) == ov and rf.overlay(30.4) == ov
    assert rf.times() == [float(i) for i in range(54)]


def _real_pipeline(rf, n=31):
    p, v = mod("pipeline"), mod("vision")
    pl = p.Pipeline(v.ReplayVision(REPLAY), "benign")
    pl.overlay_for = rf.overlay
    for f in list(rf)[:n]:
        pl.tick(f)
    return pl


def _png(resp):
    assert resp.status_code == 200 and resp.content.startswith(b"\x89PNG")
    return np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB")).astype(int)


def test_frame_panel_shows_the_real_prerendered_overlay():
    from fastapi.testclient import TestClient

    cs, w = mod("camera_source"), mod("web")
    pl = _real_pipeline(cs.ReplayFrames(REPLAY))
    c = TestClient(w.create_app(pl), base_url="http://127.0.0.1")
    got = _png(c.get("/frame.png"))
    want = np.asarray(Image.open(FRAMES / "ov_t031.jpg").convert("RGB")).astype(int)
    assert got.shape == want.shape == (850, 478, 3) and np.array_equal(got, want)
    assert got.std() > 20  # a real picture, not a grey box


def test_frame_panel_renders_tint_box_and_surface_line_when_no_overlay_exists(tmp_path):
    from fastapi.testclient import TestClient

    cs, w = mod("camera_source"), mod("web")
    for f in FRAMES.glob("t0*.jpg"):
        shutil.copy(f, tmp_path / f.name)
    rf = cs.ReplayFrames(REPLAY, root=tmp_path)
    assert rf.overlay(30.0) is None
    pl = _real_pipeline(rf)
    c = TestClient(w.create_app(pl), base_url="http://127.0.0.1")
    got = _png(c.get("/frame.png"))
    raw = np.asarray(Image.open(FRAMES / "t031.jpg").convert("RGB")).astype(int)
    rec = next(r for r in json.loads(REPLAY.read_text()) if r["t"] == 30.0)
    x0, y0, x1, y1 = rec["box"]
    top, mx = int(rec["top_row"]), (x0 + x1) // 2
    assert not np.array_equal(got[(top + y1) // 2, mx], raw[(top + y1) // 2, mx])  # liquid tinted
    assert np.array_equal(got[(y0 + top) // 2, mx], raw[(y0 + top) // 2, mx])  # air above the surface untouched
    r, g, b = got[top, mx]
    assert r > 200 and g < 80 and b < 80  # red surface line


def test_cli_replay_with_real_data_trips_before_the_camera_reaches_95(capsys):
    p = mod("pipeline")
    recs = json.loads(REPLAY.read_text())
    t95 = next(r["t"] for r in recs if r["level_pct"] >= 95)
    assert p.main(["--replay", str(REPLAY), "--scenario", "freeze", "--fault-start", "20"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["tripped"] and out["trip_t"] < t95 and not out["overflowed"]


def test_resolve_uses_real_frames_for_a_recording_that_has_them():
    p, cs = mod("pipeline"), mod("camera_source")
    ap = ArgumentParser()
    p.add_common_args(ap)
    source, _ = p.resolve(ap.parse_args(["--replay", str(REPLAY)]))
    assert isinstance(source, cs.ReplayFrames)
    source, _ = p.resolve(ap.parse_args(["--replay", "demo"]))
    assert isinstance(source, cs.FakeSource)


# ------------------------------------------------------------------ B: --start-at / --loop


def test_cli_has_start_at_and_loop():
    p = mod("pipeline")
    ap = ArgumentParser()
    p.add_common_args(ap)
    a = ap.parse_args(["--start-at", "12", "--loop"])
    assert a.start_at == 12.0 and a.loop is True
    a = ap.parse_args([])
    assert a.start_at == 0.0 and a.loop is False


def test_drive_start_at_skips_early_frames():
    p, cs = mod("pipeline"), mod("camera_source")
    pl = p.Pipeline(None, "benign")
    seen, orig = [], pl.tick
    pl.tick = lambda f, a=None: seen.append(f[0]) or orig(f, a)
    p.drive(pl, cs.FakeSource(n=10), start_at=4.0)
    assert seen == [float(t) for t in range(4, 10)]


def test_drive_loop_restarts_a_fresh_run():
    p, cs = mod("pipeline"), mod("camera_source")
    pl = p.Pipeline(None, "freeze", fault_start_s=2.0)
    stop, seen, orig = threading.Event(), {}, pl.restart

    def restart():
        seen["tripped_before"] = pl.state.tripped
        orig()
        seen.update(tripped=pl.state.tripped, permit=pl.permit, hist=len(pl.history), loops=pl.loop_count,
                    events=pl.snapshot()["events"])
        stop.set()

    pl.restart = restart
    th = threading.Thread(target=p.drive, args=(pl, cs.FakeSource(n=60, fps=1.0)),
                          kwargs={"loop": True, "stop": stop})
    th.start()
    th.join(60)
    assert not th.is_alive()
    assert seen["tripped_before"] is True
    assert seen["tripped"] is False and seen["permit"] is True and seen["hist"] == 0 and seen["loops"] == 1
    assert seen["events"][0]["text"].startswith("replay restarted")


def test_restart_is_refused_with_modbus_and_is_not_a_web_route():
    p, mb, w = mod("pipeline"), mod("modbus_io"), mod("web")
    with mb.ModbusPlcServer(port=0) as srv:
        client = mb.ModbusClient("127.0.0.1", srv.port)
        try:
            with pytest.raises(RuntimeError):
                p.Pipeline(None, "benign", server=srv, client=client).restart()
        finally:
            client.close()
    pl, app, c = make_app()
    assert c.post("/restart", data={"token": app.state.token}).status_code in (404, 405)
    assert "restart" not in Path(w.__file__).read_text(encoding="utf-8")


def test_web_source_stays_clean_of_safety_words():
    src = Path(mod("web").__file__).read_text(encoding="utf-8")
    for bad in ("interlock", "write_coil", "write_permit", "COIL_PERMIT", "WRITE_PERMIT_TRUE"):
        assert bad not in src, bad
    assert not re.search(r"permit\s*=\s*True", src)
