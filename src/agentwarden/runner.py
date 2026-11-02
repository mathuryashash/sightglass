"""Closed-loop run: tank + PLC control + fault injector + camera source + interlock, injected clock.

SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM. The only thing the interlock can do to the plant
is clear the pump permit; permit.PermitOutput applies that single action and nothing else.

    python -m agentwarden.runner --scenario freeze|drift|replay|benign --seconds 300 --seed 1
"""
import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from agentwarden import interlock
from agentwarden.fault_injector import SCENARIOS, FaultInjector
from agentwarden.level import Reading
from agentwarden.modbus_io import BANNER, ModbusClient, ModbusPlcServer
from agentwarden.permit import PermitOutput
from agentwarden.plc_sim import PlcController, TankSim

FAULT_START_S = 30.0
DRIFT_K = 0.6  # pct points per second of lie (truth rises 0.7/s while pumping)
REPLAY_WINDOW_S = 30.0


@dataclass
class Summary:
    scenario: str
    seconds: float
    tripped: bool = False
    trip_t: float | None = None
    trip_reason: str | None = None
    diff_at_trip: float | None = None  # |camera - plc| on the tripping tick
    t_diverge_exceeds_T: float | None = None  # first t where |truth - reported| > T_pts
    plc_stale_t: float | None = None  # first tick the interlock saw a stale PLC (ALARM_PLC_STALE)
    overflowed: bool = False
    max_level: float = 0.0
    permit_coil_final: bool = True


def fake_camera(sigma: float, rng: random.Random):
    """Truth +- gaussian noise, as an alertable Reading. Later chunks plug the SAM pipeline in here."""

    def read(truth: float, t: float) -> Reading:
        level = min(100.0, max(0.0, truth + rng.gauss(0.0, sigma)))
        return Reading(level, 0.9, "OK", None, False, True)

    return read


def run(
    scenario: str,
    seconds: float = 300,
    seed: int = 1,
    camera_sigma: float = 3.0,
    dt: float = 1.0,
    log_path: Path | str | None = None,
    use_modbus: bool = False,
    camera=None,
    cfg: interlock.Config | None = None,
    drift_k: float = DRIFT_K,
    plc_stop_t: float | None = None,
) -> Summary:
    """plc_stop_t: from this time on the PLC value stops refreshing for the interlock (stale-PLC scenario)."""
    cfg = cfg or interlock.Config()
    rng = random.Random(seed)
    camera = camera or fake_camera(camera_sigma, rng)
    tank = TankSim(capacity_l=1000, fill_rate_lps=9, drain_rate_lps=2, level_pct=20.0)
    plc = PlcController(low=30.0, high=80.0)
    injector = FaultInjector(
        scenario, start_t=FAULT_START_S, drift_k=drift_k, replay_window_s=REPLAY_WINDOW_S
    )
    summary = Summary(scenario, seconds)
    state = interlock.initial_state()
    server = client = None
    if use_modbus:
        server = ModbusPlcServer(port=0).start()
        client = ModbusClient("127.0.0.1", server.port)
    out = PermitOutput(server, client)
    seen_plc, seen_plc_t = None, 0.0  # last PLC value the interlock actually received, and when
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        for i in range(int(seconds / dt)):
            t = i * dt
            truth = tank.level_pct
            reported = injector.reported(truth, t)
            reading = camera(truth, t)
            plc_level = round(reported, 1)  # same x10 quantisation as the register; what the PLC itself uses
            if client:
                server.publish_plc(reported, plc.pump_on)
            if plc_stop_t is None or t < plc_stop_t:  # a read that fails counts as stale, not as a value
                try:
                    seen_plc, seen_plc_t = (client.read_plc_level() if client else plc_level), t
                except OSError:
                    pass
            if summary.t_diverge_exceeds_T is None and abs(truth - reported) > cfg.T_pts:
                summary.t_diverge_exceeds_T = t

            state, actions = interlock.step(
                state, reading, plc_level if seen_plc is None else seen_plc, t - seen_plc_t, t, cfg
            )
            if summary.plc_stale_t is None and interlock.Action.ALARM_PLC_STALE in actions:
                summary.plc_stale_t = t
            out.apply(actions)
            permit = out.permit
            if state.tripped and not summary.tripped:
                summary.tripped, summary.trip_t = True, state.trip_t
                summary.trip_reason = state.trip_reason
                if reading.level_pct is not None:
                    summary.diff_at_trip = abs(reading.level_pct - plc_level)

            pump_on = plc.update(plc_level, permit)
            tank.step(dt, pump_on)
            summary.max_level = max(summary.max_level, tank.level_pct)
            if log:
                log.write(json.dumps({
                    "t": t, "truth": round(truth, 3), "reported": round(reported, 3),
                    "camera": reading.level_pct, "pump_on": pump_on, "tripped": state.tripped,
                    "actions": [a.value for a in actions],
                }) + "\n")
        summary.overflowed = tank.overflowed
        summary.permit_coil_final = server.permit if server else out.permit
    finally:
        if log:
            log.close()
        if client:
            client.close()
        if server:
            server.stop()
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SightGlass closed-loop sim (SIMULATED, localhost only)")
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=3.0, help="fake camera noise, pct points")
    p.add_argument("--modbus", action="store_true", help="route tags through a localhost Modbus server")
    p.add_argument("--log", help="write JSONL event log here")
    a = p.parse_args(argv)
    print(BANNER)
    s = run(a.scenario, a.seconds, a.seed, a.sigma, log_path=a.log, use_modbus=a.modbus)
    print(json.dumps(asdict(s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
