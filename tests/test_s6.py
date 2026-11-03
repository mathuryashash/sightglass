"""S6: offline evaluation on recorded data. Tiny config; the full run is `python -m agentwarden.evaluate`."""
import json
import time

import numpy as np
import pytest
from PIL import Image

from agentwarden import evaluate as ev

TINY = dict(detect_seeds=4, false_seeds=3, false_minutes=3.0, diag_per_class=6, horizon_s=300.0)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    t0 = time.time()
    rc = ev.main(["--out", str(out), "--tiny"])
    return out, rc, time.time() - t0


def _json(out):
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_tiny_run_is_fast_and_writes_files(tiny_run):
    out, rc, dt = tiny_run
    assert rc == 0 and dt < 20
    assert (out / "RESULTS.md").exists() and (out / "results.json").exists()
    charts = [p for p in out.iterdir() if p.suffix in (".svg", ".png")]
    assert len(charts) == 3
    for c in charts:
        assert c.stat().st_size > 200


def test_results_json_has_every_section(tiny_run):
    r = _json(tiny_run[0])
    for k in ("level", "stills", "detection", "false_alarms", "replay", "blind", "diagnoser", "baseline", "config"):
        assert k in r
    assert r["level"]["n_frames"] == 54
    assert len(r["stills"]["rows"]) == 7 and r["stills"]["reference"].startswith("approximate")
    assert len(r["replay"]["rows"]) == 9  # 3 scenarios x fault starts 12, 20, 30
    assert r["diagnoser"]["model"]["n_incidents"] == 6
    assert set(r["diagnoser"]["rule_synthetic"]["per_class"]) == {"SPOOF", "LEAK", "CAMERA_FAULT", "BENIGN"}


def test_level_numbers_match_the_recording(tiny_run):
    r = _json(tiny_run[0])["level"]
    # t=40 reads 82.1 after 83.9: the one backward step after frame 9
    assert r["mono_violations"] == 1 and r["max_backward_step"] == pytest.approx(1.8, abs=0.05)
    assert 0 < r["noise_median"] < 3
    assert r["sigma_used"] >= 3.0 and float(r["sigma_used"]).is_integer()
    assert r["status_frac"]["OK"] + r["status_frac"]["EMPTY"] == pytest.approx(1.0)


def test_stills_are_labelled_approximate(tiny_run):
    out = tiny_run[0]
    r = _json(out)["stills"]
    assert [round(x["reading"]) for x in r["rows"]] == [100, 77, 59, 44, 34, 24, 0]
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "approximate reference, no tape" in md
    assert "accuracy +-" not in md and "accuracy ±" not in md


def test_detection_covers_all_scenarios(tiny_run):
    d = _json(tiny_run[0])["detection"]
    assert {row["kind"] for row in d["rows"]} == {"freeze", "drift", "slow_drift", "replay", "step"}
    for row in d["rows"]:
        assert row["n"] == TINY["detect_seeds"]
        assert row["detected"] + row["missed"] == row["n"]
        assert 0 <= row["overflow_reached"] <= row["n"]
    assert set(d["floors"]) == {"freeze", "step", "drift"}


def test_detection_is_deterministic():
    cfg = dict(TINY, detect_seeds=3)
    assert ev.detection(cfg, sigma=3.0) == ev.detection(cfg, sigma=3.0)


def test_big_freeze_is_caught_before_overflow():
    row = ev.detect_one("freeze", 10.0, seeds=range(6), sigma=3.0, horizon_s=300.0)
    assert row["caught"] == 6 and row["overflow_reached"] == 0


def test_false_alarm_rule_of_three():
    r = ev.false_alarms(dict(TINY), sigma=3.0)
    assert r["exposure_hours"] > 0
    if r["n_false_trips"] == 0:
        assert r["upper95_per_hour"] == pytest.approx(3.0 / r["exposure_hours"])


def test_replay_claim_is_strict_and_measured(tiny_run):
    r = _json(tiny_run[0])["replay"]
    assert r["camera_first_ge_95_t"] == 45.0  # 94.9 at t=44, 97.7 at t=45
    for row in r["rows"]:
        assert row["tripped"]
        assert row["before_95"] == (row["trip_t"] < 45.0)
    freeze12 = next(x for x in r["rows"] if x["scenario"] == "freeze" and x["fault_start"] == 12)
    assert freeze12["trip_t"] == 23.0 and freeze12["before_95"]


def test_blind_camera_never_trips_but_alarms(tiny_run):
    b = _json(tiny_run[0])["blind"]
    assert b["blind_benign"]["tripped"] is False and b["blind_benign"]["unverified_ticks"] > 0
    lim = b["blind_then_spoof"]
    assert lim["tripped"] is False and lim["unverified_ticks"] > 0
    assert lim["hidden_camera_end"] - lim["plc_end"] > 15  # the documented gap, measured
    assert b["short_blind_then_spoof"]["tripped"] is True


def test_diagnoser_separates_model_from_rule_baseline(tiny_run):
    out = tiny_run[0]
    d = _json(out)["diagnoser"]
    m = d["model"]
    assert m["n_incidents"] == 6 and m["ground_truth"] == "SPOOF"
    assert m["grounding_passed"] + m["grounding_failed"] + m["unavailable"] == 6
    assert m["label_correct"] <= m["n_incidents"] and m["agrees_with_rule"] <= m["n_incidents"]
    assert d["rule_synthetic"]["n_per_class"] == TINY["diag_per_class"]
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "model (recorded Muse Spark calls)" in md and "rule baseline (synthetic)" in md


def test_colour_baseline_on_synthetic_image():
    arr = np.full((200, 100, 3), 200, dtype=np.uint8)
    arr[120:190, 20:80] = (240, 150, 10)  # orange liquid, rows 120..189
    arr[30:40, 20:80] = (240, 150, 10)  # detached orange blob above (glare / label): must be ignored
    img = Image.fromarray(arr)
    assert ev.colour_top_row(img, (10, 0, 90, 200)) == 120
    plain = Image.fromarray(np.full((200, 100, 3), 200, dtype=np.uint8))
    assert ev.colour_top_row(plain, (10, 0, 90, 200)) is None
    assert 0 < ev.colour_level(img, (10, 0, 90, 200)) < 100


def test_baseline_section_compares_on_the_same_frames(tiny_run):
    b = _json(tiny_run[0])["baseline"]
    assert b["n_frames"] == 54
    for k in ("sam", "colour"):
        assert {"mono_violations", "max_backward_step", "noise_median", "no_reading_frames"} <= set(b[k])


def test_numbers_in_markdown_match_json(tiny_run):
    out = tiny_run[0]
    r = _json(out)
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert f"{r['level']['mono_violations']} backward step" in md
    assert f"{r['level']['max_backward_step']:.1f}" in md
    assert f"{r['false_alarms']['n_false_trips']} false trip" in md
    assert f"{r['false_alarms']['exposure_hours']:.1f}" in md
    for row in r["detection"]["rows"]:
        assert f"{row['detected']}/{row['n']}" in md
    for row in r["replay"]["rows"]:
        assert f"{row['trip_t']:.0f}" in md
    assert "Limits" in md and "no tape" in md.lower() and "blind-then-spoof" in md


def test_no_env_or_network_in_module():
    import inspect

    src = inspect.getsource(ev)
    for bad in ("environ", "getenv", "socket", "urllib", "httpx", "requests"):
        assert bad not in src
