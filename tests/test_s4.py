"""S4: camera sources, vision, pipeline, dashboard. Imports are lazy so a missing module fails one test."""
import importlib
import io
import json
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src" / "agentwarden"


def mod(name):
    return importlib.import_module(f"agentwarden.{name}")


def jpeg(w=1280, h=720, color=(200, 120, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------ camera sources


def test_stills_source_downscales_and_spaces_time(tmp_path):
    cs = mod("camera_source")
    paths = []
    for i in range(3):
        p = tmp_path / f"a{i}.jpg"
        p.write_bytes(jpeg(1280, 720))
        paths.append(p)
    frames = list(cs.StillsSource(paths, period_s=2.0))
    assert [f[0] for f in frames] == [0.0, 2.0, 4.0]
    for t, data, w, h in frames:
        assert (w, h) == (640, 360)
        assert Image.open(io.BytesIO(data)).size == (640, 360)


def test_small_frames_are_not_upscaled(tmp_path):
    cs = mod("camera_source")
    p = tmp_path / "s.jpg"
    p.write_bytes(jpeg(320, 200))
    [(t, data, w, h)] = list(cs.StillsSource([p], 1.0))
    assert (w, h) == (320, 200)


def test_fake_source_yields_tuples():
    cs = mod("camera_source")
    frames = list(cs.FakeSource(n=4, fps=2.0))
    assert [f[0] for f in frames] == [0.0, 0.5, 1.0, 1.5]
    t, data, w, h = frames[0]
    assert Image.open(io.BytesIO(data)).size == (w, h)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_clip_source_samples_at_fps(tmp_path):
    cs = mod("camera_source")
    clip = tmp_path / "c.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=10:duration=3",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    frames = list(cs.ClipSource(clip, fps=2))
    assert 5 <= len(frames) <= 7
    assert frames[0][0] == 0.0 and frames[1][0] == pytest.approx(0.5)
    assert all(f[2] <= 640 for f in frames)
    assert Image.open(io.BytesIO(frames[0][1])).size == (frames[0][2], frames[0][3])


@pytest.mark.parametrize(
    "url", ["http://8.8.8.8/shot.jpg", "http://203.0.113.9:8080/shot.jpg", "http://example.com/shot.jpg",
            "ftp://127.0.0.1/shot.jpg", "http://[2001:4860:4860::8888]/shot.jpg"]
)
def test_http_shot_rejects_public_hosts(url):
    cs = mod("camera_source")
    with pytest.raises(ValueError):
        cs.HttpShotSource(url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8080/shot.jpg", "http://localhost/shot.jpg", "http://192.168.1.20:8080/shot.jpg",
            "http://10.0.0.5/shot.jpg", "http://172.16.3.4/shot.jpg"]
)
def test_http_shot_accepts_private_hosts(url):
    mod("camera_source").HttpShotSource(url)


def test_http_shot_fetches_with_basic_auth_and_hides_credentials():
    cs = mod("camera_source")
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = jpeg(800, 600)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ticks = iter(range(100))
        src = cs.HttpShotSource(
            f"http://127.0.0.1:{srv.server_port}/shot.jpg", user="cam", password="s3cret-pw",
            period_s=1.0, n=2, clock=lambda: float(next(ticks)), sleep=lambda s: None,
        )
        assert "s3cret-pw" not in repr(src) and "s3cret-pw" not in str(src)
        frames = list(src)
    finally:
        srv.shutdown()
    assert len(frames) == 2 and frames[0][2] == 640
    assert all(h and h.startswith("Basic ") for h in seen)


def test_http_shot_does_not_follow_redirects_off_host():
    cs = mod("camera_source")

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://8.8.8.8/x.jpg")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        src = cs.HttpShotSource(f"http://127.0.0.1:{srv.server_port}/shot.jpg", n=1, sleep=lambda s: None)
        with pytest.raises(Exception):
            list(src)
    finally:
        srv.shutdown()


# ------------------------------------------------------------------ vision


FW, FH = 80, 60


def sam_text(raster, box, conf="0.9"):
    h, w = len(raster), len(raster[0])
    flat = [int(v) for row in raster for v in row]
    x1, y1, x2, y2 = box
    return (
        f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={FW};h={FH};c={conf}|>"
        f"<|mask;x=0;y=0;data={h},{w},{_encode_raster(flat, w, h)}|>"
    )


BOTTLE_BOX = (20, 5, 59, 54)  # inclusive -> left 20 top 5 right 60 bottom 55
BOTTLE = sam_text([[1] * 40 for _ in range(50)], BOTTLE_BOX)


def liquid(top):
    rows = 54 - top + 1
    return sam_text([[1] * 36 for _ in range(rows)], (22, top, 57, 54))


class FakeClient:
    def __init__(self, liquid_text, bottle_text=BOTTLE):
        self.texts = {"orange liquid": liquid_text, "plastic bottle": bottle_text}
        self.calls = []

    def sam_image(self, image_bytes, concept, run_id):
        self.calls.append(concept)
        text = self.texts.get(concept, "")  # "hand": nothing found
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def frame(t=0.0):
    return (t, jpeg(FW, FH), FW, FH)


def calib():
    from agentwarden.level import Calibration

    return Calibration.from_points([(55, 0), (12, 100)])


def test_read_level_makes_two_sam_calls_and_estimates():
    v = mod("vision")
    c = FakeClient(liquid(30))
    r = v.read_level(frame(), c, calib(), occluder_concept=None)  # the 3-call path is tested in test_s6b
    assert sorted(c.calls) == ["orange liquid", "plastic bottle"]
    assert r.status == "OK" and r.alertable
    assert r.level_pct == pytest.approx((55 - 30) / 43 * 100, abs=1.5)


def test_read_level_no_bottle_is_camera_blind():
    v = mod("vision")
    r = v.read_level(frame(), FakeClient("", bottle_text=""), calib())
    assert r.status == "CAMERA_BLIND" and not r.alertable and r.level_pct is None


def test_read_level_no_liquid_is_empty():
    v = mod("vision")
    r = v.read_level(frame(), FakeClient(""), calib())
    assert r.status == "EMPTY" and r.level_pct == 0.0


def test_camera_moved_is_low_confidence_and_flagged():
    v = mod("vision")
    c = FakeClient(liquid(30))
    shifted = (20.0 + 40, 5.0, 60.0 + 40, 55.0)
    a = v.analyze(frame(), c, calib(), ref_box=shifted, tol_px=20)
    assert a.camera_moved and a.reading.status == "LOW_CONFIDENCE" and not a.reading.alertable
    ok = v.analyze(frame(), c, calib(), ref_box=(20.0, 5.0, 60.0, 55.0), tol_px=20)
    assert not ok.camera_moved and ok.reading.status == "OK"
    assert v.read_level(frame(), c, calib(), ref_box=shifted, tol_px=20).status == "LOW_CONFIDENCE"


def test_auto_calibration_maps_bottom_and_shoulder():
    v = mod("vision")
    cal = v.auto_from_box((20.0, 5.0, 60.0, 55.0), shoulder_frac=0.22)
    assert cal.level_at(55.0)[0] == pytest.approx(0.0)
    assert cal.level_at(5.0 + 0.22 * 50)[0] == pytest.approx(100.0)
    assert cal.level_at(30.0)[0] == pytest.approx((55 - 30) / (55 - 16) * 100)
    assert "approximate" in (v.auto_from_box.__doc__ or "").lower()


def test_live_vision_auto_calibrates_on_first_frame_and_flags_a_moved_box():
    v = mod("vision")
    lv = v.LiveVision(FakeClient(liquid(30)))
    a = lv.read(frame(0))
    assert a.reading.status == "OK" and not a.camera_moved
    assert lv.ref_box == (20.0, 5.0, 60.0, 55.0)
    lv.client.texts["plastic bottle"] = sam_text([[1] * 20 for _ in range(50)], (60, 5, 79, 54))
    lv.client.texts["orange liquid"] = sam_text([[1] * 18 for _ in range(25)], (61, 30, 78, 54))
    b = lv.read(frame(1))
    assert b.camera_moved and b.reading.status == "LOW_CONFIDENCE"


def test_newest_frame_wins_drops_stale_frames():
    v = mod("vision")
    gate, started, done = threading.Event(), threading.Event(), []

    def slow(f):
        started.set()
        gate.wait(5)
        return f

    w = v.LatestWorker(slow, done.append)
    w.submit("f1")
    assert started.wait(5)
    w.submit("f2")
    w.submit("f3")  # f2 must be replaced, never processed
    gate.set()
    w.close()
    assert done == ["f1", "f3"]
    assert w.dropped == 1


def test_replay_vision_by_time_and_index(tmp_path):
    v = mod("vision")
    data = {"readings": [
        {"t": 0.0, "level_pct": 10.0, "confidence": 0.9, "status": "OK"},
        {"t": 1.0, "level_pct": 20.0, "confidence": 0.8, "status": "OK"},
        {"t": 2.0, "level_pct": None, "confidence": 0.0, "status": "CAMERA_BLIND"},
    ]}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    rv = v.ReplayVision(p)
    assert rv.read((1.0, b"", 1, 1)).reading.level_pct == 20.0
    assert rv.read((0.4, b"", 1, 1)).reading.level_pct == 10.0  # last record at or before t
    blind = rv.read((2.0, b"", 1, 1)).reading
    assert blind.status == "CAMERA_BLIND" and not blind.alertable
    assert rv.read((1.0, b"", 1, 1)).reading.alertable
    assert rv.times() == [0.0, 1.0, 2.0]


def test_replay_vision_accepts_bare_list_and_demo():
    v = mod("vision")
    rv = v.ReplayVision([{"level_pct": 5.0, "confidence": 0.9, "status": "OK"},
                         {"level_pct": 9.0, "confidence": 0.9, "status": "OK"}])
    assert rv.read((7.0, b"", 1, 1), index=1).reading.level_pct == 9.0
    demo = v.ReplayVision.demo()
    lv = [demo.read((t, b"", 1, 1)).reading.level_pct for t in demo.times()]
    assert lv[0] < 10 and lv[-1] > 90


# ------------------------------------------------------------------ pipeline


def rising(n=54, lo=0.0, hi=100.0):
    return [{"t": float(i), "level_pct": lo + (hi - lo) * i / (n - 1), "confidence": 0.9, "status": "OK"}
            for i in range(n)]


def run_pipeline(scenario, readings=None, fault_start=20.0, **kw):
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")
    rv = v.ReplayVision(readings or rising())
    return p.run(cs.FakeSource(times=rv.times()), rv, scenario=scenario, fault_start_s=fault_start, **kw)


def test_replay_pipeline_camera_levels_rise():
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")
    rv = v.ReplayVision(rising())
    pl = p.Pipeline(rv, scenario="benign")
    levels = [pl.tick(f)["camera"] for f in cs.FakeSource(times=rv.times())]
    assert levels[0] < 5 and levels[-1] > 95
    assert all(b >= a - 1e-9 for a, b in zip(levels, levels[1:]))


def test_replay_benign_never_trips():
    s = run_pipeline("benign", rising(hi=90.0))  # below high-high (95); a pour to 100 trips HIGH_HIGH by design
    assert not s.tripped and not s.overflowed
    assert run_pipeline("benign").trip_reason == "HIGH_HIGH"


def test_replay_freeze_trips_before_overflow():
    s = run_pipeline("freeze")
    assert s.tripped and s.trip_reason in ("DIVERGENCE", "RATE")
    assert s.trip_t < 45
    assert not s.overflowed and s.max_level < 95
    assert s.permit_coil_final is False


def test_pipeline_camera_moved_never_trips_and_alarms():
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")

    class Moved:
        def read(self, f, index=None):
            r = mod("level").Reading(50.0, 0.3, "LOW_CONFIDENCE", None, False, False)
            return v.Analysis(r, None, None, True)

    pl = p.Pipeline(Moved(), scenario="freeze", fault_start_s=5.0)
    for f in cs.FakeSource(n=40, fps=1.0):
        ev = pl.tick(f)
    assert not pl.state.tripped
    assert ev["camera_moved"] is True
    assert "CAMERA_MOVED" in pl.snapshot()["alarms"]
    assert pl.snapshot()["camera_blind_notice"] is True


def test_pipeline_writes_modbus_camera_tag():
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")
    mb = mod("modbus_io")
    server = mb.ModbusPlcServer(port=0).start()
    client = mb.ModbusClient("127.0.0.1", server.port)
    try:
        rv = v.ReplayVision(rising())
        pl = p.Pipeline(rv, scenario="benign", server=server, client=client)
        for f in list(cs.FakeSource(times=rv.times()))[:10]:
            ev = pl.tick(f)
        lvl, conf, status = client.read_camera()
        assert lvl == pytest.approx(ev["camera"], abs=0.1) and status == "OK"
    finally:
        client.close()
        server.stop()


def test_pipeline_fake_camera_option_and_event_log(tmp_path):
    p, cs = mod("pipeline"), mod("camera_source")
    log = tmp_path / "ev.jsonl"
    s = p.run(cs.FakeSource(n=60, fps=1.0), None, scenario="freeze", log_path=log)
    assert s.tripped and not s.overflowed
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 60 and {"t", "camera", "plc", "status", "tripped"} <= set(lines[0])


def test_human_reset_refused_while_condition_holds_then_accepted():
    p, v, cs = mod("pipeline"), mod("vision"), mod("camera_source")
    rv = v.ReplayVision(rising())
    pl = p.Pipeline(rv, scenario="freeze", fault_start_s=20.0)
    for f in cs.FakeSource(times=rv.times()):
        pl.tick(f)
    assert pl.state.tripped and pl.permit is False
    pl.human_reset()
    assert pl.state.tripped and pl.permit is False  # camera still far from the frozen PLC
    # fault cleared and the camera agrees with the PLC again: reset is accepted
    pl.inject("benign")
    pl.vision = v.ReplayVision([{"level_pct": pl.truth, "confidence": 0.9, "status": "OK"}] * 5)
    for i in range(3):
        pl.tick((100.0 + i, b"", 1, 1))
    pl.human_reset()
    assert not pl.state.tripped and pl.permit is True


def test_cli_replay_demo_freeze(capsys):
    p = mod("pipeline")
    assert p.main(["--replay", "demo", "--scenario", "freeze"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["tripped"] is True and out["overflowed"] is False


def test_cli_fake_source(capsys):
    p = mod("pipeline")
    assert p.main(["--source", "fake:90", "--fps", "1", "--scenario", "benign"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["tripped"] is False


# ------------------------------------------------------------------ web


def make_app(readings=None, ticks=12):
    from fastapi.testclient import TestClient

    p, v, cs, w = mod("pipeline"), mod("vision"), mod("camera_source"), mod("web")
    rv = v.ReplayVision(readings or rising())
    pl = p.Pipeline(rv, scenario="benign")
    for f in list(cs.FakeSource(times=rv.times()))[:ticks]:
        pl.tick(f)
    app = w.create_app(pl)
    return pl, app, TestClient(app, base_url="http://127.0.0.1")


def test_web_refuses_non_local_bind():
    w = mod("web")
    for host in ("0.0.0.0", "192.168.1.5", "::", "example.com"):
        with pytest.raises(ValueError):
            w.make_server(object(), host, 0)


def test_web_server_binds_only_loopback():
    import socket
    import time

    import httpx

    w = mod("web")
    pl, app, _ = make_app()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = w.make_server(app, "127.0.0.1", port)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        hosts = {sock.getsockname()[0] for srv in server.servers for sock in srv.sockets}
        assert hosts == {"127.0.0.1"}
        r = httpx.get(f"http://127.0.0.1:{port}/api/state")
        assert r.status_code == 200
    finally:
        server.should_exit = True
        t.join(10)


def test_web_page_has_banner_panel_and_no_external_refs():
    pl, app, c = make_app()
    r = c.get("/")
    assert r.status_code == 200
    html = r.text
    assert "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM" in html
    for needle in ("PLC level", "Camera level", "Pump", "Permit", "Trip", "Alarms"):
        assert needle in html
    assert "<svg" in html
    assert not re.search(r'(src|href)\s*=\s*"https?://', html)
    assert "cdn" not in html.lower()
    assert app.state.token in html
    assert "script-src 'nonce-" in r.headers["content-security-policy"]


def test_web_api_state_schema_and_no_token_leak():
    pl, app, c = make_app()
    r = c.get("/api/state")
    assert r.status_code == 200
    s = r.json()
    assert {"t", "clock", "banner", "plc_level", "camera", "pump_on", "permit", "tripped", "trip_reason",
            "alarms", "scenario", "camera_blind_notice"} <= set(s)
    assert {"level_pct", "confidence", "status", "alertable", "camera_moved"} <= set(s["camera"])
    assert isinstance(s["alarms"], list) and isinstance(s["permit"], bool)
    assert app.state.token not in r.text


def test_web_frame_png_and_trend_svg():
    pl, app, c = make_app()
    r = c.get("/frame.png")
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    t = c.get("/trend.svg")
    assert t.status_code == 200 and t.text.startswith("<svg") and "polyline" in t.text


def test_web_frame_overlay_draws_mask_and_surface_line():
    import numpy as np

    from agentwarden.web import render_frame

    mask = np.zeros((60, 80), dtype=bool)
    mask[30:55, 22:58] = True
    png = render_frame(jpeg(80, 60, (0, 0, 0)), mask, (20.0, 5.0, 60.0, 55.0), 30.0)
    im = Image.open(io.BytesIO(png)).convert("RGB")
    assert im.getpixel((40, 45)) != (0, 0, 0)  # mask tint


def test_post_without_or_with_wrong_token_is_403():
    pl, app, c = make_app()
    assert c.post("/reset").status_code == 403
    assert c.post("/reset", data={"token": "nope"}).status_code == 403
    assert c.post("/inject/freeze", data={"token": "nope"}).status_code == 403
    assert pl.injector.scenario == "benign"


def test_post_with_token_works_via_form_or_header():
    pl, app, c = make_app()
    tok = app.state.token
    r = c.post("/inject/freeze", data={"token": tok}, follow_redirects=False)
    assert r.status_code == 303 and pl.injector.scenario == "freeze"
    r = c.post("/inject/drift", headers={"X-Token": tok}, follow_redirects=False)
    assert r.status_code in (200, 303) and pl.injector.scenario == "drift"
    assert c.post("/inject/nonsense", data={"token": tok}).status_code == 404
    assert c.post("/reset", data={"token": tok}, follow_redirects=False).status_code in (200, 303)


def test_post_cross_origin_rejected_even_with_token():
    pl, app, c = make_app()
    r = c.post("/reset", data={"token": app.state.token}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_get_on_mutating_routes_is_405():
    pl, app, c = make_app()
    assert c.get("/reset").status_code == 405
    assert c.get("/inject/freeze").status_code == 405
    assert pl.injector.scenario == "benign"


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "192.168.1.5", "0.0.0.0"])
def test_wrong_host_header_rejected(host):
    pl, app, c = make_app()
    for path in ("/", "/api/state", "/frame.png"):
        assert c.get(path, headers={"Host": host}).status_code in (400, 403)
    r = c.post("/reset", data={"token": app.state.token}, headers={"Host": host})
    assert r.status_code in (400, 403)


def test_host_with_port_and_localhost_allowed():
    pl, app, c = make_app()
    assert c.get("/", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    assert c.get("/", headers={"Host": "localhost:8765"}).status_code == 200


def test_hostile_alarm_string_is_escaped():
    pl, app, c = make_app()
    evil = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    pl.extra_alarms.append(evil)
    html = c.get("/").text
    assert evil not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x" not in html
    assert evil in c.get("/api/state").json()["alarms"]  # JSON carries it as inert data


def test_blind_camera_shows_no_trip_notice():
    blind = [{"t": float(i), "level_pct": None, "confidence": 0.0, "status": "CAMERA_BLIND"} for i in range(20)]
    pl, app, c = make_app(blind)
    assert "camera blinded - no trip" in c.get("/").text
    assert c.get("/api/state").json()["camera_blind_notice"] is True
    _, _, ok_c = make_app()
    assert c.get("/api/state").json()["camera_blind_notice"] is True
    assert ok_c.get("/api/state").json()["camera_blind_notice"] is False


def test_reset_through_web_goes_via_pipeline_human_reset():
    pl, app, c = make_app()
    calls = []
    pl.human_reset = lambda: calls.append("reset") or {}
    c.post("/reset", data={"token": app.state.token}, follow_redirects=False)
    assert calls == ["reset"]


# ------------------------------------------------------------------ static safety (S3 rules still hold)


def _src():
    return {p.name: p.read_text(encoding="utf-8") for p in SRC.glob("*.py")}


def test_web_cannot_set_permit_true():
    text = _src()["web.py"]
    for bad in ("WRITE_PERMIT_TRUE", "write_permit", "write_coil", "COIL_PERMIT", "interlock"):
        assert bad not in text, bad
    assert not re.search(r"permit\s*=\s*True", text)


def test_pipeline_permit_only_derived_and_reset_only_by_human():
    text = _src()["pipeline.py"]
    assert not re.search(r"permit\s*=\s*True", text)
    assert "write_permit" not in text and "WRITE_PERMIT_TRUE" not in text
    resets = re.findall(r"interlock\.reset(?:_with_grant)?\(([^)]*)\)", text)
    assert resets and all("by_human=True" in r for r in resets), resets


def test_new_modules_do_not_touch_env():
    for name in ("camera_source.py", "vision.py", "pipeline.py", "web.py"):
        text = _src()[name]
        assert "environ" not in text and "getenv" not in text, name
