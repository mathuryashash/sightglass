"""S3fix: permit hardening (wire + in-process), closed-loop safety property, reviewer MINORs."""
import copy
import inspect
import json
import random
import re
from pathlib import Path

import pytest

from agentwarden import fault_injector, interlock, modbus_io, permit, pipeline, runner
from agentwarden.level import Reading
from agentwarden.plc_sim import PlcController, TankSim

SRC = Path(__file__).resolve().parents[1] / "src" / "agentwarden"
BANNER = "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM"
CFG = interlock.Config()


def cam(level, conf=0.9, status="OK", alertable=True):
    return Reading(level, conf, status, None, False, alertable)


def blind():
    return Reading(None, 0.0, "CAMERA_BLIND", None, False, False)


def drive(samples, state=None, cfg=CFG):
    state = state or interlock.initial_state()
    for t, c, plc, age in samples:
        state, _ = interlock.step(state, c, plc, age, float(t), cfg)
    return state


def tripped_state():
    s = drive([(t, cam(60), 40, 0) for t in range(6)])
    assert s.tripped
    return s


def tripped_cleared_state():
    s = drive([(6, cam(50), 50, 0)], state=tripped_state())
    assert s.tripped and not s.condition_active
    return s


def raw_client(port):
    from pymodbus.client import ModbusTcpClient

    c = ModbusTcpClient("127.0.0.1", port=port)
    assert c.connect()
    return c


# ---------------------------------------------------------------- M1: permit cannot be set True from the wire


def test_raw_true_write_to_permit_coil_is_refused():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        raw = raw_client(srv.port)
        try:
            c.clear_permit()
            assert srv.permit is False
            assert raw.write_coil(1, True, device_id=1).isError()
            assert raw.write_coils(1, [True], device_id=1).isError()
            assert raw.write_coils(0, [True, True], device_id=1).isError()
            assert srv.permit is False and c.read_permit() is False
        finally:
            raw.close()
            c.close()


def test_true_write_refused_even_while_permitted_and_false_write_accepted():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        raw = raw_client(srv.port)
        try:
            assert raw.write_coil(1, True, device_id=1).isError()
            assert srv.permit is True
            assert not raw.write_coil(1, False, device_id=1).isError()
            assert srv.permit is False
        finally:
            raw.close()


def test_writes_to_camera_unit_refused():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        srv.publish_camera(55.5, 0.87, "OK")
        raw = raw_client(srv.port)
        try:
            assert raw.write_register(0, 999, device_id=2).isError()
            assert raw.write_registers(0, [1, 2, 3], device_id=2).isError()
            assert raw.write_coil(0, True, device_id=2).isError()
            assert raw.write_coil(1, False, device_id=2).isError()
            assert raw.read_holding_registers(0, count=3, device_id=2).registers == [555, 87, 0]
            assert srv.permit is True
        finally:
            raw.close()


def test_client_cannot_write_permit_true():
    assert not hasattr(modbus_io.ModbusClient, "write_permit")
    assert set(inspect.signature(modbus_io.ModbusClient.clear_permit).parameters) == {"self"}
    with modbus_io.ModbusPlcServer(port=0) as srv:
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        try:
            c.clear_permit()
            c.clear_permit()  # idempotent, always False
            assert srv.permit is False
        finally:
            c.close()


# ---------------------------------------------------------------- re-arm only through a ResetGrant


def test_grant_only_from_successful_human_reset():
    _, acts, g = interlock.reset_with_grant(tripped_cleared_state(), by_human=False)
    assert g is None and interlock.Action.ALARM_RESET_REFUSED in acts
    s = drive([(6, cam(60), 40, 0)], state=tripped_state())  # still diverged
    _, _, g = interlock.reset_with_grant(s, by_human=True)
    assert g is None
    s2, acts, g = interlock.reset_with_grant(tripped_cleared_state(), by_human=True)
    assert isinstance(g, interlock.ResetGrant) and not s2.tripped
    assert acts == [interlock.Action.WRITE_PERMIT_TRUE]
    _, _, g = interlock.reset_with_grant(interlock.initial_state(), by_human=True)
    assert g is None  # nothing to re-arm


def test_grant_cannot_be_constructed_directly():
    with pytest.raises((TypeError, PermissionError)):
        interlock.ResetGrant()
    with pytest.raises((TypeError, PermissionError)):
        interlock.ResetGrant(object())


def test_rearm_without_grant_refused():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        try:
            c.clear_permit()
            for bad in (None, True, "grant", object(), interlock.Action.WRITE_PERMIT_TRUE):
                with pytest.raises(PermissionError):
                    srv.rearm_from_reset(bad)
            assert srv.permit is False
        finally:
            c.close()


def test_rearm_with_grant_works_once():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        try:
            c.clear_permit()
            _, _, g = interlock.reset_with_grant(tripped_cleared_state(), by_human=True)
            srv.rearm_from_reset(g)
            assert srv.permit is True and c.read_permit() is True
            c.clear_permit()
            with pytest.raises(PermissionError):  # single use
                srv.rearm_from_reset(g)
            assert srv.permit is False
        finally:
            c.close()


def test_permit_output_local_and_wire():
    out = permit.PermitOutput()
    assert out.permit is True
    out.apply([interlock.Action.ALARM_PLC_STALE])
    assert out.permit is True
    out.apply([interlock.Action.WRITE_PERMIT_FALSE])
    assert out.permit is False
    with pytest.raises(PermissionError):
        out.rearm(None)
    assert out.permit is False
    _, _, g = interlock.reset_with_grant(tripped_cleared_state(), by_human=True)
    out.rearm(g)
    assert out.permit is True

    with modbus_io.ModbusPlcServer(port=0) as srv:
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        try:
            out = permit.PermitOutput(server=srv, client=c)
            out.apply([interlock.Action.WRITE_PERMIT_FALSE])
            assert srv.permit is False and out.permit is False
            with pytest.raises(PermissionError):
                out.rearm(None)
            _, _, g = interlock.reset_with_grant(tripped_cleared_state(), by_human=True)
            out.rearm(g)
            assert srv.permit is True and out.permit is True
        finally:
            c.close()


def _src():
    return {p.name: p.read_text(encoding="utf-8") for p in SRC.glob("*.py")}


def _calls(text, name):
    return re.findall(rf"(?<!def )\b{name}\(", text)


def test_static_only_permit_py_writes_the_coil():
    src = _src()
    assert [n for n, t in src.items() if n != "modbus_io.py" and "write_coil" in t] == []
    assert [n for n, t in src.items() if "write_permit" in t] == []
    assert [n for n, t in src.items() if n != "modbus_io.py" and "COIL_PERMIT" in t] == []
    assert [n for n, t in src.items() if n != "permit.py" and _calls(t, "clear_permit")] == []
    assert [n for n, t in src.items() if n != "permit.py" and _calls(t, "rearm_from_reset")] == []
    assert _calls(src["permit.py"], "clear_permit") and _calls(src["permit.py"], "rearm_from_reset")


def test_static_only_interlock_creates_grant_and_true_action():
    src = _src()
    assert [n for n, t in src.items() if n != "interlock.py" and "WRITE_PERMIT_TRUE" in t] == []
    assert [n for n, t in src.items() if n != "interlock.py" and "ResetGrant(" in t] == []
    assert [n for n, t in src.items() if n != "interlock.py" and "_MINT" in t] == []


# ---------------------------------------------------------------- M2: closed-loop property


def _plan(seed, n=120):
    rng = random.Random(seed)
    return {
        "mode": rng.choice(["adversarial", "noisy", "dropout"]),
        "scenario": rng.choice(fault_injector.SCENARIOS),
        "start": rng.uniform(0, 60),
        "drift_k": rng.uniform(0.1, 1.0),
        "steps": [
            {"u": rng.random(), "lvl": rng.uniform(-10, 120), "conf": rng.random(),
             "noise": rng.gauss(0, 5), "reset": rng.random() < 0.08, "human": rng.random() < 0.8}
            for _ in range(n)
        ],
    }


def _camera(mode, truth, s):
    if mode == "noisy":
        return cam(truth + s["noise"])
    if mode == "dropout":
        return blind() if s["u"] < 0.3 else cam(truth + s["noise"])
    if s["u"] < 0.15:
        return blind()
    if s["u"] < 0.3:
        return Reading(s["lvl"], s["conf"] * 0.5, "LOW_CONFIDENCE", None, False, False)
    status = "EMPTY" if s["u"] < 0.4 else "OK"
    return Reading(s["lvl"], s["conf"], status, None, False, s["conf"] >= 0.5)


def _sim(plan, with_interlock):
    tank = TankSim(capacity_l=1000, fill_rate_lps=9, drain_rate_lps=2, level_pct=20.0)
    plc = PlcController(low=30.0, high=80.0)
    inj = fault_injector.FaultInjector(plan["scenario"], start_t=plan["start"], drift_k=plan["drift_k"],
                                       replay_window_s=30.0)
    out, state, rec = permit.PermitOutput(), interlock.initial_state(), []
    for i, s in enumerate(plan["steps"]):
        t, truth = float(i), tank.level_pct
        plc_level = round(inj.reported(truth, t), 1)
        reading = _camera(plan["mode"], truth, s)
        before, reset_ok = out.permit, False
        if with_interlock:
            state, acts = interlock.step(state, reading, plc_level, 0.0, t, CFG)
            out.apply(acts)
            if s["reset"]:
                state, _, grant = interlock.reset_with_grant(state, by_human=s["human"])
                if grant:
                    out.rearm(grant)
                    reset_ok = True
        no_interlock = copy.copy(plc).update(plc_level, True)  # same plant + PLC state, interlock removed
        on = plc.update(plc_level, out.permit)
        tank.step(1.0, on)
        rec.append({"on": on, "cf": no_interlock, "before": before, "after": out.permit, "reset": reset_ok,
                    "tripped": state.tripped})
    return rec


def test_closed_loop_interlock_only_removes_pump_time():
    trips = resets = 0
    for seed in range(1000):
        plan = _plan(seed)
        with_i, base = _sim(plan, True), _sim(plan, False)
        first_reset = next((i for i, r in enumerate(with_i) if r["reset"]), len(with_i))
        for i, (a, b) in enumerate(zip(with_i, base)):
            assert not a["on"] or a["cf"], (seed, i)  # pump on => it would be on without the interlock
            if i < first_reset:  # identical history until a human re-arm: whole-run comparison
                assert not a["on"] or b["on"], (seed, i)
            assert not (not a["before"] and a["after"]) or a["reset"], (seed, i)  # no False->True w/o reset
            assert b["after"] is True  # baseline: nothing ever touches the permit
        trips += any(r["tripped"] for r in with_i)
        resets += first_reset < len(with_i)
    assert trips > 100 and resets > 20  # the property was actually exercised, not vacuous


# ---------------------------------------------------------------- MINORs


def test_hh_needs_two_consecutive_alertable_frames():
    one_glare = [(0, cam(50), 50, 0), (1, cam(97), 96, 0), (2, cam(50), 50, 0), (3, cam(97), 96, 0),
                 (4, cam(50), 50, 0)]
    assert not drive(one_glare).tripped
    assert not drive([(0, cam(97), 96, 0), (1, blind(), 96, 0), (2, cam(97), 96, 0)]).tripped
    s = drive([(0, cam(97), 96, 0), (1, cam(96), 96, 0)])
    assert s.tripped and s.trip_reason == "HIGH_HIGH" and s.trip_t == 1.0
    assert not drive([(0, cam(95), 94, 0), (1, cam(94.9), 94, 0)]).tripped


def test_reset_refused_while_divergence_timer_running_in_hysteresis_band():
    s = drive([(6, cam(54), 40, 0)], state=tripped_state())  # diff 14: not > T, above T - hyst
    assert s.mismatch_since is not None
    _, acts, g = interlock.reset_with_grant(s, by_human=True)
    assert g is None and interlock.Action.ALARM_RESET_REFUSED in acts
    assert interlock.reset(s, by_human=True)[0].tripped
    s = drive([(7, cam(50), 40, 0)], state=s)  # diff 10 <= T - hyst: timer cleared
    assert s.mismatch_since is None
    assert not interlock.reset(s, by_human=True)[0].tripped


def test_runner_stale_plc_end_to_end(tmp_path):
    log = tmp_path / "ev.jsonl"
    r = runner.run("benign", seconds=120, seed=1, camera_sigma=0.0, plc_stop_t=60.0, log_path=log)
    assert r.plc_stale_t == 70.0  # last update t=59, age 11 > 10 at t=70
    assert not r.tripped  # stale alone never trips
    ev = [json.loads(x) for x in log.read_text().splitlines()]
    assert all("ALARM_PLC_STALE" not in e["actions"] for e in ev[:70])
    assert all("ALARM_PLC_STALE" in e["actions"] for e in ev[70:])
    assert runner.run("benign", seconds=120, camera_sigma=0.0).plc_stale_t is None


def test_slow_drift_k0_3_caught_by_divergence_before_overflow():
    r = runner.run("drift", seconds=300, seed=1, camera_sigma=0.0, drift_k=0.3)
    assert r.tripped and r.trip_reason == "DIVERGENCE"  # k=0.3: PLC still moves, rate rule stays quiet
    assert 81 <= r.trip_t <= 95
    assert not r.overflowed and r.max_level < 95


def test_injector_host_is_where_it_talks_and_gates_before_anything_runs(capsys):
    for host in ("10.255.255.1", "0.0.0.0", "plc.example.com"):
        with pytest.raises(ValueError):
            fault_injector.FaultInjector("freeze", host=host, port=502)
    assert capsys.readouterr().out == ""  # no banner, nothing started
    with modbus_io.ModbusPlcServer(port=0) as srv:
        srv.publish_plc(40.0, False)
        inj = fault_injector.FaultInjector("freeze", host="localhost", port=srv.port, start_t=5)
        try:
            assert inj.host == "127.0.0.1"
            assert inj.reported_from_wire(0.0) == pytest.approx(40.0)
            srv.publish_plc(70.0, False)
            assert inj.reported_from_wire(2.0) == pytest.approx(70.0)
            srv.publish_plc(90.0, False)
            assert inj.reported_from_wire(6.0) == pytest.approx(90.0)  # freezes at first post-start truth
            srv.publish_plc(95.0, False)
            assert inj.reported_from_wire(7.0) == pytest.approx(90.0)
        finally:
            inj.close()


def test_banner_on_server_start_and_both_clis(capsys):
    with modbus_io.ModbusPlcServer(port=0):
        pass
    assert BANNER in capsys.readouterr().out
    runner.main(["--scenario", "benign", "--seconds", "5"])
    assert capsys.readouterr().out.splitlines()[0] == BANNER
    pipeline.main(["--replay", "demo", "--scenario", "benign"])
    assert capsys.readouterr().out.splitlines()[0] == BANNER


def test_unknown_camera_status_code_fails_safe_to_blind():
    with modbus_io.ModbusPlcServer(port=0) as srv:
        srv._store.camera = [555, 87, 99]
        c = modbus_io.ModbusClient("127.0.0.1", srv.port)
        try:
            assert c.read_camera() == (pytest.approx(55.5), pytest.approx(0.87), "CAMERA_BLIND")
        finally:
            c.close()


def test_threshold_from_error_stats():
    assert interlock.Config.from_error_stats(2).T_pts == 10  # 3*2 = 6 -> floor 10
    assert interlock.Config.from_error_stats(6).T_pts == 18
    assert interlock.Config.from_error_stats(3.4).T_pts == pytest.approx(10.2)
    assert interlock.Config.from_error_stats(6).N_s == interlock.Config().N_s
    assert interlock.Config().T_pts == 15
    assert "provisional" in interlock.Config.__doc__.lower()
