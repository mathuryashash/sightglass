"""S6b: occlusion (hand) + lighting (lights off) robustness. Real clips: demo/real/{hand,lights}_replay.json."""
import argparse
import contextlib
import importlib
import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from test_s4 import FakeClient, calib, frame, liquid, sam_text

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "demo" / "real"
HAND, LIGHTS, POUR = (REAL / f"{n}_replay.json" for n in ("hand", "lights", "pour"))
FULL_ONLY = dict(hh_limit=101.0)  # the bottle in the clips really is full: isolate the occlusion effect


def mod(name):
    return importlib.import_module(f"agentwarden.{name}")


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def run_pipeline(recs, scenario="benign", start=0.0, **vkw):
    cfg = mod("interlock").Config(**FULL_ONLY)
    with contextlib.redirect_stdout(io.StringIO()):
        pl = mod("pipeline").Pipeline(mod("vision").ReplayVision(recs, **vkw), scenario, start, cfg=cfg)
    evs = [pl.tick((r["t"],)) for r in recs]
    pl.close()
    return pl, evs


def ok(level, conf=0.9):
    return mod("level").Reading(level, conf, "OK", 10.0, False, conf >= 0.5)


# ---------------------------------------------------------------- level: status, overlap, plausibility guard


def test_occluded_is_a_non_alertable_status():
    lv = mod("level")
    assert "OCCLUDED" in lv.Status.__args__
    r = lv.Reading(None, 0.0, "OCCLUDED", None, False, False, "hand/occluded")
    assert not r.alertable and r.reason == "hand/occluded"
    assert ok(50.0).reason == ""  # reason is optional, old call sites unchanged
    assert lv.estimate(None, None, calib()).reason == "dark/no detection"


def test_box_overlap_is_share_of_container_covered_by_best_box():
    lv = mod("level")
    box = (0.0, 0.0, 10.0, 10.0)
    assert lv.OCCLUDER_OVERLAP == 0.20
    assert lv.box_overlap(box, []) == 0.0
    assert lv.box_overlap(box, [(20, 20, 30, 30)]) == 0.0
    assert lv.box_overlap(box, [(0, 0, 10, 5)]) == pytest.approx(0.5)
    assert lv.box_overlap(box, [(0, 0, 10, 2), (0, 0, 10, 4)]) == pytest.approx(0.4)  # best single box, not a sum
    assert lv.box_overlap(box, [(-5, -5, 50, 50)]) == pytest.approx(1.0)


def test_guard_flags_a_jump_faster_than_max_rate_then_accepts_a_persisting_level():
    g = mod("level").RateGuard()
    assert g.max_rate_pts_s == 8.0
    assert g.check(ok(50.0), 0.0).status == "OK"
    assert g.check(ok(54.0), 1.0).status == "OK"  # 4 pts/s
    r = g.check(ok(20.0), 2.0)  # 34 pts/s
    assert r.status == "LOW_CONFIDENCE" and not r.alertable and r.reason == "implausible jump"
    assert g.check(ok(19.0), 3.0).status == "OK"  # it stayed: one frame of debounce, then trusted
    assert g.check(ok(18.0), 4.0).status == "OK"


def test_guard_ramp_just_under_the_limit_passes_and_just_over_fails():
    g = mod("level").RateGuard(max_rate_pts_s=8.0)
    for i in range(10):
        assert g.check(ok(10.0 + 7.9 * i), float(i)).status == "OK"
    g2 = mod("level").RateGuard(max_rate_pts_s=8.0)
    g2.check(ok(10.0), 0.0)
    assert g2.check(ok(18.5), 1.0).status == "LOW_CONFIDENCE"


def test_guard_ignores_non_alertable_readings_and_resets_when_time_goes_backwards():
    lv = mod("level")
    g = lv.RateGuard()
    g.check(ok(50.0), 0.0)
    blind = lv.Reading(None, 0.0, "CAMERA_BLIND", None)
    assert g.check(blind, 1.0) is blind
    low = lv.Reading(99.0, 0.3, "LOW_CONFIDENCE", None, False, False)
    assert g.check(low, 2.0) is low  # a guess never becomes the reference
    assert g.check(ok(52.0), 3.0).status == "OK"
    assert g.check(ok(5.0), 0.0).status == "OK"  # clock went backwards (loop restart): no stale reference


def test_guard_is_configurable_and_can_be_off():
    g = mod("level").RateGuard(max_rate_pts_s=100.0)
    g.check(ok(50.0), 0.0)
    assert g.check(ok(10.0), 1.0).status == "OK"
    recs = [{"t": 0.0, "level_pct": 90.0, "confidence": 0.9}, {"t": 1.0, "level_pct": 10.0, "confidence": 0.9}]
    rv = mod("vision").ReplayVision(recs, max_rate_pts_s=None)
    assert rv.read((0.0,)).reading.alertable and rv.read((1.0,)).reading.alertable
    rv = mod("vision").ReplayVision(recs)  # default: guard on
    rv.read((0.0,))
    assert not rv.read((1.0,)).reading.alertable


# ---------------------------------------------------------------- vision: third SAM concept


def hand_text(box_incl, conf="0.9"):
    x1, y1, x2, y2 = box_incl
    return sam_text([[1] * (x2 - x1 + 1) for _ in range(y2 - y1 + 1)], box_incl, conf)


def client_with_hand(text=""):
    c = FakeClient(liquid(30))
    c.texts["hand"] = text
    return c


def test_live_read_makes_a_third_call_for_the_hand_by_default():
    vis = mod("vision")
    c = client_with_hand()
    r = vis.read_level(frame(), c, calib())
    assert sorted(c.calls) == ["hand", "orange liquid", "plastic bottle"] and r.status == "OK"
    off = client_with_hand()
    vis.read_level(frame(), off, calib(), occluder_concept=None)
    assert sorted(off.calls) == ["orange liquid", "plastic bottle"]
    assert "cost" in vis.__doc__.lower()  # +1 image per frame is documented


def test_live_hand_over_the_bottle_is_occluded_not_a_level():
    a = mod("vision").analyze(frame(), client_with_hand(hand_text((20, 5, 59, 29))), calib())  # 40x25 of 40x50 = 0.5
    r = a.reading
    assert r.status == "OCCLUDED" and not r.alertable and r.level_pct is None and r.reason == "hand/occluded"
    assert len(a.hand_boxes) == 1


def test_live_small_or_unsure_hand_does_not_occlude():
    vis = mod("vision")
    small = vis.analyze(frame(), client_with_hand(hand_text((20, 5, 29, 14))), calib())  # 100/2000 = 0.05
    assert small.reading.status == "OK" and len(small.hand_boxes) == 1  # still drawn
    unsure = vis.analyze(frame(), client_with_hand(hand_text((20, 5, 59, 29), conf="0.4")), calib())
    assert unsure.reading.status == "OK" and unsure.hand_boxes == ()


def test_live_blind_skips_the_hand_call():
    c = FakeClient("", bottle_text="")
    c.texts["hand"] = ""
    assert mod("vision").read_level(frame(), c, calib()).status == "CAMERA_BLIND"
    assert "hand" not in c.calls


# ---------------------------------------------------------------- replay: the real hand clip


def test_hand_clip_every_occluded_frame_is_non_alertable_and_clean_frames_stay_ok():
    recs = load(HAND)
    rv = mod("vision").ReplayVision(recs)
    bad = clean = 0
    for r in recs:
        a = rv.read((r["t"],))
        if r["hand_overlap"] >= 0.20:
            bad += 1
            assert a.reading.status == "OCCLUDED" and not a.reading.alertable, r["t"]
            assert a.reading.level_pct is None and a.hand_boxes
        else:
            clean += 1
            assert a.reading.status == "OK" and a.reading.alertable, r["t"]
            assert a.reading.level_pct == pytest.approx(100.0)
    assert bad == 14 and clean == 8  # t=1.5-5.5 and 6.5-9.5


def test_hand_clip_without_the_fix_reproduces_the_problem():
    rv = mod("vision").ReplayVision(load(HAND), occluder=False, max_rate_pts_s=None)
    rs = [rv.read((r["t"],)).reading for r in load(HAND)]
    assert min(r.level_pct for r in rs) < 15 and all(r.alertable for r in rs)


def test_hand_clip_guard_alone_still_catches_the_first_jump():
    rv = mod("vision").ReplayVision(load(HAND), occluder=False)
    rs = [rv.read((r["t"],)).reading for r in load(HAND)]
    assert rs[3].status == "LOW_CONFIDENCE" and rs[3].reason == "implausible jump"  # 100 -> 37 in 0.5 s


@pytest.mark.parametrize("scenario", ["benign", "freeze"])
def test_hand_clip_never_trips_inside_the_occlusion(scenario):
    _, evs = run_pipeline(load(HAND), scenario, 0.0)
    for e in evs:
        if 1.5 <= e["t"] <= 5.5 or 6.5 <= e["t"] <= 9.5:
            assert not e["tripped"], e["t"]
            assert e["status"] == "OCCLUDED" and "ALARM_CAMERA_UNVERIFIED" in e["actions"]
            assert e["camera"] is None


def test_hand_clip_direct_interlock_plc_at_truth_never_trips_with_the_fix_but_does_without():
    il, vis = mod("interlock"), mod("vision")
    cfg = il.Config(**FULL_ONLY)

    def trips(**kw):
        rv, st, out = vis.ReplayVision(load(HAND), **kw), il.initial_state(), False
        for r in load(HAND):
            st, _ = il.step(st, rv.read((r["t"],)).reading, 100.0, 0.0, r["t"], cfg)
            out = out or st.tripped
        return out

    assert trips() is False
    assert trips(occluder=False, max_rate_pts_s=None) is True  # the old behaviour: a hand makes a false DIVERGENCE trip


def test_combined_hand_plus_freeze_trips_only_on_clean_frames():
    recs = load(HAND)
    last = recs[-1]
    recs += [dict(last, t=last["t"] + 0.5 * (i + 1)) for i in range(20)]  # 10 s of clean frames appended
    pl, evs = run_pipeline(recs, "freeze", 0.0)
    s = pl.summary
    assert s.tripped and s.trip_reason == "DIVERGENCE"
    assert s.trip_t > max(e["t"] for e in evs if e["status"] == "OCCLUDED")
    assert all(e["status"] == "OK" for e in evs if e["t"] >= 10.0)
    assert s.trip_t - 10.0 >= 5.0 - 1e-9  # needs N_s of CLEAN divergence after the last occlusion


# ---------------------------------------------------------------- replay: lights off


def test_lights_clip_blind_frames_never_trip_and_are_shown_blind():
    recs = load(LIGHTS)
    pl, evs = run_pipeline(recs)
    blind = [e for e in evs if e["status"] == "CAMERA_BLIND"]
    assert len(blind) == sum(r["status"] == "CAMERA_BLIND" for r in recs) == 11
    assert not any(e["tripped"] for e in evs)
    assert all("ALARM_CAMERA_UNVERIFIED" in e["actions"] for e in blind)
    assert any("blind" in e["text"] for e in pl.snapshot()["events"])


def test_lights_clip_readings_return_to_100_when_light_returns():
    rv = mod("vision").ReplayVision(load(LIGHTS))
    for r in load(LIGHTS):
        a = rv.read((r["t"],)).reading
        if r["status"] == "OK":
            assert a.status == "OK" and a.level_pct == pytest.approx(100.0)


# ---------------------------------------------------------------- the pour must still read OK


def test_pour_clip_guard_flags_at_most_two_frames_and_occluder_flags_none():
    vis = mod("vision")
    recs = load(POUR)
    raw = vis.ReplayVision(recs, occluder=False, max_rate_pts_s=None)
    rv = vis.ReplayVision(recs)
    flagged = occ = 0
    for r in recs:
        a, b = rv.read((r["t"],)).reading, raw.read((r["t"],)).reading
        flagged += b.alertable and not a.alertable
        occ += a.status == "OCCLUDED"
    assert flagged <= 2 and occ == 0
    ok_ = [(r["t"], r["level_pct"]) for r in recs if r["t"] >= 9 and r["status"] == "OK"]
    assert max(abs(y - x) / (b - a) for (a, x), (b, y) in zip(ok_, ok_[1:])) < 8.0  # real pour peaks near 5


def test_pour_clip_still_reads_ok_through_the_fill():
    recs = load(POUR)
    rv = mod("vision").ReplayVision(recs)
    got = [rv.read((r["t"],)).reading for r in recs if r["t"] >= 9]
    assert all(g.status == "OK" and g.alertable for g in got)


# ---------------------------------------------------------------- interlock / modbus


def test_interlock_treats_occluded_as_unverified_and_resets_timers():
    il, lv = mod("interlock"), mod("level")
    cfg = il.Config()
    st, _ = il.step(il.initial_state(), lv.Reading(60.0, 0.9, "OK", None, False, True), 20.0, 0.0, 0.0, cfg)
    assert st.mismatch_since == 0.0
    occ = lv.Reading(None, 0.0, "OCCLUDED", None, False, False, "hand/occluded")
    st, acts = il.step(st, occ, 20.0, 0.0, 3.0, cfg)
    assert il.Action.ALARM_CAMERA_UNVERIFIED in acts and st.mismatch_since is None and st.cam_hist == ()
    assert not st.tripped and il.Action.WRITE_PERMIT_FALSE not in acts
    assert il.unverified_reason(occ) == "occluded" and il.unverified_reason(ok(50.0)) == ""
    for i in range(10):  # a level riding on an OCCLUDED reading never counts
        st, _ = il.step(st, lv.Reading(99.0, 0.9, "OCCLUDED", None, False, False), 20.0, 0.0, 4.0 + i, cfg)
    assert not st.tripped


def test_modbus_carries_occluded():
    m = mod("modbus_io")
    assert m.STATUS_CODES["OCCLUDED"] == 4 and m.STATUS_NAMES[4] == "OCCLUDED"
    with m.ModbusPlcServer(port=0) as srv:
        srv.publish_camera(None, 0.0, "OCCLUDED")
        c = m.ModbusClient("127.0.0.1", srv.port)
        try:
            assert c.read_camera()[2] == "OCCLUDED"
        finally:
            c.close()


# ---------------------------------------------------------------- dashboard


def snap(status, reason=""):
    return {"plc_level": 100.0, "pump_on": False, "permit": True, "tripped": False, "trip_reason": None, "alarms": [],
            "camera": {"level_pct": None, "status": status, "camera_moved": False, "confidence": 0.0,
                       "alertable": False, "reason": reason},
            "camera_blind_notice": True, "camera_notice": "x", "limits": {"T_pts": 15.0}}


@pytest.mark.parametrize("status,reason", [("OCCLUDED", "hand/occluded"), ("CAMERA_BLIND", "dark/no detection"),
                                           ("LOW_CONFIDENCE", "implausible jump")])
def test_pill_says_camera_blocked_no_trip_with_a_sub_reason(status, reason):
    v = mod("web").hmi_view(snap(status, reason))
    assert v["pill"] == "blocked" and v["pill_text"] == "CAMERA BLOCKED - NO TRIP"
    assert v["sub_reason"] == reason


def test_pill_has_no_sub_reason_when_normal():
    s = snap("OK")
    s["camera_blind_notice"] = False
    s["camera"]["level_pct"] = 100.0
    assert mod("web").hmi_view(s)["sub_reason"] == ""


def test_snapshot_carries_reason_and_the_page_shows_it():
    pl, _ = run_pipeline(load(HAND)[:5])  # last tick is t=2.0, a hand frame
    s = pl.snapshot()
    assert s["camera"]["status"] == "OCCLUDED" and s["camera"]["reason"] == "hand/occluded"
    assert "occluded" in s["camera_notice"].lower()
    page = mod("web").render_page(s, "tok", "nonce", "<svg/>")
    assert "CAMERA BLOCKED - NO TRIP" in page and "hand/occluded" in page


def test_render_frame_draws_hand_boxes_in_amber():
    web = mod("web")
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (10, 10, 10)).save(buf, "JPEG")
    png = web.render_frame(buf.getvalue(), None, None, None, hand_boxes=[(10, 10, 60, 60)])
    assert Image.open(io.BytesIO(png)).convert("RGB").getpixel((10, 30)) == web.AMBER
    plain = web.render_frame(buf.getvalue(), None, None, None)
    assert Image.open(io.BytesIO(plain)).convert("RGB").getpixel((10, 30)) != web.AMBER


def test_replay_cli_loads_the_hand_clip_with_its_own_frames():
    p = mod("pipeline")
    ap = argparse.ArgumentParser()
    p.add_common_args(ap)
    src, vis = p.resolve(ap.parse_args(["--replay", str(HAND), "--frames", str(REAL / "hand_frames")]))
    assert len(list(src)) == 22 and vis.read((4.0,)).reading.status == "OCCLUDED"


# ---------------------------------------------------------------- evaluation section


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    ev = importlib.import_module("agentwarden.evaluate")
    out = tmp_path_factory.mktemp("eval6b")
    t0 = time.time()
    assert ev.main(["--out", str(out), "--tiny"]) == 0
    return out, time.time() - t0


def test_eval_robustness_numbers(tiny_run):
    r = json.loads((tiny_run[0] / "results.json").read_text(encoding="utf-8"))["robustness"]
    h, li = r["hand"], r["lights"]
    assert h["frames"] == 22 and h["occluded"] == 14 and h["false_trips"] == 0 and h["false_trips_before"] >= 1
    assert h["ok"] + h["occluded"] + h["blind"] + h["low_conf"] == h["frames"]
    assert h["max_abs_err_ok_before"] > 80 and h["max_abs_err_ok_after"] < 1.0
    assert li["frames"] == 27 and li["blind"] == 11 and li["false_trips"] == 0 and li["max_abs_err_ok_after"] < 1.0
    c = r["combined"]
    assert c["tripped"] and c["trip_during_occlusion"] is False and c["trip_t"] > c["last_occluded_t"]
    p = r["pour_guard"]
    assert p["flagged"] <= 2 and p["occluded"] == 0 and p["max_rate_after_entry"] < 8.0


def test_eval_markdown_has_the_section_and_limits(tiny_run):
    md = (tiny_run[0] / "RESULTS.md").read_text(encoding="utf-8")
    assert "Camera robustness (real clips)" in md
    assert "one hand" in md.lower() and "occluder" in md.lower()
