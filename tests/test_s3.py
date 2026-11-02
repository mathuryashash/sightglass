"""S3: simulated PLC, modbus IO, fault injector, interlock, runner. All localhost, all simulated."""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agentwarden import fault_injector, interlock, modbus_io, plc_sim, runner
from agentwarden.level import Reading

SRC = Path(__file__).resolve().parents[1] / "src" / "agentwarden"
BANNER = "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM"


def cam(level, conf=0.9, status="OK", alertable=True):
    return Reading(level, conf, status, None, False, alertable)


def blind():
    return Reading(None, 0.0, "CAMERA_BLIND", None, False, False)


def lowconf(level=99.0):
    return Reading(level, 0.2, "LOW_CONFIDENCE", None, False, False)


# ---------------------------------------------------------------- plc_sim


def test_tank_fills_and_drains():
    t = plc_sim.TankSim(capacity_l=1000, fill_rate_lps=10, drain_rate_lps=5, level_pct=50)
    assert t.step(10, True) == pytest.approx(55.0)  # (10-5)*10 L = 50 L = 5 pts
    assert t.step(10, False) == pytest.approx(50.0)  # -5*10 L = -50 L = -5 pts
    assert not t.overflowed


def test_tank_clamps_low_and_flags_overflow():
    t = plc_sim.TankSim(capacity_l=100, fill_rate_lps=1, drain_rate_lps=50, level_pct=10)
    assert t.step(1, False) == 0.0
    t = plc_sim.TankSim(capacity_l=100, fill_rate_lps=100, drain_rate_lps=0, level_pct=95)
    assert t.step(1, True) == 100.0
    assert t.overflowed


def test_controller_hysteresis_and_permit():
    c = plc_sim.PlcController(low=30, high=80)
    assert c.update(25, True) is True
    assert c.update(50, True) is True  # stays on inside band
    assert c.update(85, True) is False
    assert c.update(50, True) is False  # stays off inside band
    assert c.update(25, True) is True
    assert c.update(25, False) is False  # permit False forces OFF
    assert c.update(25, False) is False


# ---------------------------------------------------------------- fault_injector


def test_injector_refuses_non_localhost():
    for host in ("10.0.0.5", "0.0.0.0", "192.168.1.9", "plc.example.com"):
        with pytest.raises(ValueError):
            fault_injector.FaultInjector("freeze", host=host)
    fault_injector.FaultInjector("freeze", host="localhost")
    fault_injector.FaultInjector("freeze", host="127.0.0.1")


def test_injector_prints_banner(capsys):
    fault_injector.FaultInjector("benign")
    assert BANNER in capsys.readouterr().out


def test_injector_unknown_scenario():
    with pytest.raises(ValueError):
        fault_injector.FaultInjector("explode")


def test_freeze_sticks_at_value_when_it_starts():
    inj = fault_injector.FaultInjector("freeze", start_t=5)
    got = [inj.reported(10.0 + t, float(t)) for t in range(10)]
    assert got == [10, 11, 12, 13, 14, 15, 15, 15, 15, 15]


def test_drift_subtracts_k_times_elapsed_and_clamps():
    inj = fault_injector.FaultInjector("drift", start_t=0, drift_k=0.5)
    assert inj.reported(40.0, 0.0) == 40.0
    assert inj.reported(40.0, 10.0) == 35.0
    assert inj.reported(3.0, 10.0) == 0.0


def test_replay_loops_recorded_window():
    inj = fault_injector.FaultInjector("replay", start_t=6, replay_window_s=4)
    got = [inj.reported(10.0 * t, float(t)) for t in range(12)]
    # recorded window t=2..5 -> 20,30,40,50 replayed from t=6, looping
    assert got == [0, 10, 20, 30, 40, 50, 20, 30, 40, 50, 20, 30]


def test_benign_passes_truth():
    inj = fault_injector.FaultInjector("benign")
    assert inj.reported(42.0, 7.0) == 42.0


# ---------------------------------------------------------------- modbus_io


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_modbus_server_refuses_non_localhost(host):
    with pytest.raises(ValueError):
        modbus_io.ModbusPlcServer(host=host)


def test_modbus_roundtrip_typed_client():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        try:
            assert c.read_permit() is True  # starts permitted
            srv.publish_plc(41.3, True)
            assert c.read_plc_level() == pytest.approx(41.3)
            assert c.read_pump_running() is True
            srv.publish_camera(55.5, 0.87, "OK")
            assert c.read_camera() == (pytest.approx(55.5), pytest.approx(0.87), "OK")
            srv.publish_camera(None, 0.0, "CAMERA_BLIND")
            lvl, conf, st = c.read_camera()
            assert lvl is None and st == "CAMERA_BLIND"
            c.write_permit(False)
            assert srv.permit is False
            assert c.read_permit() is False
        finally:
            c.close()


def test_modbus_register_map_raw():
    from pymodbus.client import ModbusTcpClient

    with modbus_io.ModbusPlcServer(port=0) as srv:
        srv.publish_plc(41.3, True)
        srv.publish_camera(55.5, 0.87, "OK")
        raw = ModbusTcpClient("127.0.0.1", port=srv.port)
        raw.connect()
        try:
            assert raw.read_holding_registers(0, count=1, device_id=1).registers == [413]
            assert raw.read_coils(0, count=2, device_id=1).bits[:2] == [True, True]
            assert raw.read_holding_registers(0, count=3, device_id=2).registers == [555, 87, 0]
        finally:
            raw.close()


# ---------------------------------------------------------------- interlock


def drive(samples, cfg=None, state=None):
    """samples: iterable of (t, camera, plc_level, plc_age). Returns (state, per-step actions)."""
    cfg = cfg or interlock.Config()
    state = state or interlock.initial_state()
    out = []
    for t, c, plc, age in samples:
        state, acts = interlock.step(state, c, plc, age, float(t), cfg)
        out.append(acts)
    return state, out


def test_defaults():
    c = interlock.Config()
    assert (c.T_pts, c.N_s, c.W_s, c.plc_stale_s, c.hh_limit) == (15, 5, 20, 10, 95)


def test_divergence_trips_after_N_s():
    cfg = interlock.Config()
    s = interlock.initial_state()
    for t in range(5):
        s, a = interlock.step(s, cam(60), 40, 0, float(t), cfg)
        assert not s.tripped, t
    s, a = interlock.step(s, cam(60), 40, 0, 5.0, cfg)
    assert s.tripped and s.trip_reason == "DIVERGENCE"
    assert interlock.Action.WRITE_PERMIT_FALSE in a


def test_divergence_at_exactly_T_does_not_count():
    s, _ = drive((t, cam(55), 40, 0) for t in range(30))
    assert not s.tripped


def test_dip_restarts_timer():
    seq = [(t, cam(60), 40, 0) for t in range(4)]  # t=0..3 diverged
    seq += [(4, cam(40), 40, 0)]  # agree -> reset
    seq += [(t, cam(60), 40, 0) for t in range(5, 10)]  # t=5..9: only 4 s elapsed
    s, _ = drive(seq)
    assert not s.tripped
    s, _ = drive(seq + [(10, cam(60), 40, 0)])
    assert s.tripped


def test_hysteresis_prevents_chatter():
    # diff alternates 20 / 14: 14 is above T - hyst(3)=12, so the timer is NOT reset.
    seq = [(t, cam(60 if t % 2 == 0 else 54), 40, 0) for t in range(6)]
    s, _ = drive(seq)
    assert s.tripped


def test_high_high_trips_regardless_of_plc():
    s, a = drive([(0, cam(95), 94, 0)])
    assert s.tripped and s.trip_reason == "HIGH_HIGH"
    s, _ = drive([(0, cam(94.9), 94, 0)])
    assert not s.tripped


def test_rate_rule_catches_slow_lie_before_divergence_rule():
    # camera rises 0.7/s, PLC flat at 20 (frozen): diff only 14 at t=20 (< T=15)
    seq = [(t, cam(20 + 0.7 * t), 20, 0) for t in range(0, 21)]
    s, acts = drive(seq)
    assert s.tripped and s.trip_reason == "RATE"
    assert s.trip_t <= 20.0


def test_rate_rule_quiet_when_plc_follows():
    seq = [(t, cam(20 + 0.7 * t), 20 + 0.7 * t, 0) for t in range(0, 60)]
    s, _ = drive(seq)
    assert not s.tripped


def test_blind_camera_never_trips_and_alarms():
    for r in (blind(), lowconf(99.0)):
        s, acts = drive([(t, r, 10, 0) for t in range(60)])
        assert not s.tripped
        assert all(interlock.Action.ALARM_CAMERA_UNVERIFIED in a for a in acts)


def test_blind_resets_mismatch_timer():
    seq = [(t, cam(60), 40, 0) for t in range(4)]
    seq += [(4, blind(), 40, 0)]
    seq += [(t, cam(60), 40, 0) for t in range(5, 10)]
    s, _ = drive(seq)
    assert not s.tripped


def test_stale_plc_alarms_but_does_not_trip():
    s, acts = drive([(t, cam(50), 50, 11) for t in range(30)])
    assert not s.tripped
    assert all(interlock.Action.ALARM_PLC_STALE in a for a in acts)
    # exactly at the timeout is not stale
    s, acts = drive([(0, cam(50), 50, 10)])
    assert interlock.Action.ALARM_PLC_STALE not in acts[0]


def test_stale_plc_does_not_feed_mismatch_rule():
    s, _ = drive([(t, cam(90), 10, 11) for t in range(30)])
    assert not s.tripped  # stale PLC value is not trusted, no trip from stale alone


def tripped_state():
    s, _ = drive([(t, cam(60), 40, 0) for t in range(6)])
    assert s.tripped
    return s


def test_latch_holds_and_reset_rules():
    s = tripped_state()
    cfg = interlock.Config()
    # benign inputs forever: still tripped, permit False reasserted
    for t in range(6, 120):
        s, acts = interlock.step(s, cam(50), 50, 0, float(t), cfg)
        assert s.tripped
        assert interlock.Action.WRITE_PERMIT_FALSE in acts
    # not by a human: refused
    s2, acts = interlock.reset(s, by_human=False)
    assert s2.tripped and interlock.Action.WRITE_PERMIT_TRUE not in acts
    # by human with condition cleared: accepted
    s3, acts = interlock.reset(s, by_human=True)
    assert not s3.tripped
    assert acts == [interlock.Action.WRITE_PERMIT_TRUE]


def test_reset_refused_while_condition_holds():
    s = tripped_state()
    cfg = interlock.Config()
    s, _ = interlock.step(s, cam(60), 40, 0, 6.0, cfg)  # still diverged
    s2, acts = interlock.reset(s, by_human=True)
    assert s2.tripped
    assert interlock.Action.WRITE_PERMIT_TRUE not in acts
    s, _ = drive([(7, cam(99), 99, 0)], state=s)  # high-high
    s2, _ = interlock.reset(s, by_human=True)
    assert s2.tripped


def test_reset_refused_while_camera_unverifiable():
    s = tripped_state()
    s, _ = interlock.step(s, blind(), 50, 0, 6.0, interlock.Config())
    s2, _ = interlock.reset(s, by_human=True)
    assert s2.tripped


def test_after_reset_new_trip_needs_fresh_timer():
    cfg = interlock.Config()
    s = tripped_state()
    s, _ = interlock.step(s, cam(50), 50, 0, 6.0, cfg)
    s, _ = interlock.reset(s, by_human=True)
    assert not s.tripped
    s, _ = interlock.step(s, cam(60), 40, 0, 7.0, cfg)
    assert not s.tripped  # timer only just started


def random_reading(rng):
    kind = rng.choice(["ok", "ok", "ok", "low", "blind", "empty"])
    if kind == "blind":
        return blind()
    lvl = rng.uniform(-10, 120)
    if kind == "low":
        return Reading(lvl, rng.uniform(0, 0.5), "LOW_CONFIDENCE", None, False, False)
    st = "EMPTY" if kind == "empty" else "OK"
    conf = rng.uniform(0, 1)
    return Reading(lvl, conf, st, None, False, conf >= 0.5)


def test_safety_property_never_permit_true_never_pump_on():
    rng = random.Random(12345)
    allowed = set(interlock.Action) - {interlock.Action.WRITE_PERMIT_TRUE}
    cfg = interlock.Config()
    for _ in range(2500):
        s = interlock.initial_state()
        t = rng.uniform(0, 100)
        tripped_once = False
        for _ in range(rng.randint(1, 40)):
            t += rng.choice([rng.uniform(0, 3), rng.uniform(0, 3), -rng.uniform(0, 5)])
            s, acts = interlock.step(
                s, random_reading(rng), rng.uniform(-5, 105), rng.choice([0, 0, 5, 30]), t, cfg
            )
            assert set(acts) <= allowed
            assert not any("PUMP_ON" in str(a) for a in acts)
            if tripped_once:
                assert s.tripped  # latch: never un-trips inside step()
            tripped_once = tripped_once or s.tripped
            if s.tripped:
                assert interlock.Action.WRITE_PERMIT_FALSE in acts


# ---------------------------------------------------------------- runner


def test_benign_15_minutes_sigma_3_zero_trips():
    r = runner.run("benign", seconds=900, seed=1, camera_sigma=3.0)
    assert not r.tripped
    assert not r.overflowed


def test_freeze_trips_fast_and_tank_never_overflows():
    r = runner.run("freeze", seconds=300, seed=1, camera_sigma=0.0)
    assert r.tripped and not r.overflowed
    cfg = interlock.Config()
    assert r.trip_t <= r.t_diverge_exceeds_T + cfg.N_s + 3
    assert r.max_level < 100
    # fault starts at t=30; rate window is 20 s, so the trip lands by t=50 (well before overflow)
    assert 30 <= r.trip_t <= 50


def test_freeze_with_noise_still_caught():
    r = runner.run("freeze", seconds=300, seed=2, camera_sigma=3.0)
    assert r.tripped and not r.overflowed


def test_drift_caught_by_rate_rule_before_2T():
    r = runner.run("drift", seconds=300, seed=1, camera_sigma=0.0)
    assert r.tripped and r.trip_reason == "RATE"
    assert r.diff_at_trip < 2 * interlock.Config().T_pts


def test_replay_is_caught_and_never_overflows():
    r = runner.run("replay", seconds=400, seed=1, camera_sigma=0.0)
    assert r.tripped and not r.overflowed


def test_event_log_jsonl(tmp_path):
    log = tmp_path / "ev.jsonl"
    runner.run("benign", seconds=60, seed=1, log_path=log)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 60
    assert {"t", "truth", "reported", "camera", "actions"} <= set(lines[0])


def test_freeze_over_real_modbus_matches_inprocess():
    a = runner.run("freeze", seconds=150, seed=1, camera_sigma=0.0)
    b = runner.run("freeze", seconds=150, seed=1, camera_sigma=0.0, use_modbus=True)
    assert b.tripped and b.trip_t == a.trip_t
    assert b.permit_coil_final is False


def test_cli_prints_banner_and_summary():
    p = subprocess.run(
        [sys.executable, "-m", "agentwarden.runner", "--scenario", "freeze", "--seconds", "200",
         "--seed", "1"],
        capture_output=True, text=True, timeout=120,
    )
    assert p.returncode == 0, p.stderr
    assert BANNER in p.stdout
    summary = json.loads(p.stdout.strip().splitlines()[-1])
    assert summary["tripped"] is True


# ---------------------------------------------------------------- static: who writes the permit coil


def _src():
    return {p.name: p.read_text(encoding="utf-8") for p in SRC.glob("*.py")}


def test_only_modbus_wrapper_calls_write_coil():
    offenders = [n for n, t in _src().items() if n != "modbus_io.py" and "write_coil" in t]
    assert offenders == []


def test_only_runner_calls_write_permit_and_only_with_false():
    for name, text in _src().items():
        if name == "modbus_io.py":
            continue
        calls = re.findall(r"write_permit\(([^)]*)\)", text)
        if name == "runner.py":
            assert all(c.strip() == "False" for c in calls), calls
        else:
            assert calls == [], name


def test_permit_true_only_defined_in_interlock():
    offenders = [n for n, t in _src().items() if n != "interlock.py" and "WRITE_PERMIT_TRUE" in t]
    assert offenders == []


def test_permit_coil_constant_only_in_modbus_wrapper():
    offenders = [n for n, t in _src().items() if n != "modbus_io.py" and "COIL_PERMIT" in t]
    assert offenders == []
