"""Camera source + vision (live or replay) -> modbus camera tag -> interlock -> event log.

SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM. The interlock can only switch the pump permit off;
`permit` here is DERIVED from the interlock latch (permit == not tripped), so nothing in this module
sets it True. The one way a latch clears is `human_reset()` -> `interlock.reset(by_human=True)`.

    python -m agentwarden.pipeline --source clip:../spike/real/pour.mp4 --fps 1 --scenario freeze --replay <json>
    python -m agentwarden.pipeline --replay demo --scenario freeze
"""
import argparse
import glob
import json
import random
import threading
import time
from collections import deque
from pathlib import Path

from agentwarden import interlock
from agentwarden.camera_source import ClipSource, FakeSource, HttpShotSource, StillsSource
from agentwarden.fault_injector import SCENARIOS, FaultInjector
from agentwarden.level import Reading
from agentwarden.modbus_io import ModbusClient, ModbusPlcServer
from agentwarden.plc_sim import PlcController, TankSim
from agentwarden.runner import DRIFT_K, FAULT_START_S, REPLAY_WINDOW_S, Summary, fake_camera
from agentwarden.vision import Analysis, LatestWorker, LiveVision, ReplayVision

BANNER = "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM"
BLIND_STATUSES = ("CAMERA_BLIND", "LOW_CONFIDENCE")


class Pipeline:
    """One tick per frame. With a vision, the simulated plant is coupled to what the camera sees (the real
    bottle IS the process) until the interlock trips; with vision=None a TankSim plant and a noisy fake
    camera are used, as in runner.py."""

    def __init__(self, vision, scenario: str = "benign", fault_start_s: float = FAULT_START_S,
                 cfg: interlock.Config | None = None, seed: int = 1, camera_sigma: float = 3.0,
                 server: ModbusPlcServer | None = None, client: ModbusClient | None = None,
                 log_path: str | Path | None = None, history_s: float = 300.0):
        self.vision, self.cfg = vision, cfg or interlock.Config()
        self.server, self.client = server, client
        self.tank = TankSim(capacity_l=1000, fill_rate_lps=9, drain_rate_lps=2, level_pct=20.0)
        self.plc = PlcController(low=30.0, high=80.0)
        self.state = interlock.initial_state()
        self.injector = FaultInjector(scenario, start_t=fault_start_s, drift_k=DRIFT_K,
                                      replay_window_s=REPLAY_WINDOW_S)
        self._fake = fake_camera(camera_sigma, random.Random(seed)) if vision is None else None
        self.summary = Summary(scenario, 0.0)
        self.history_s = history_s
        self.history: deque = deque()  # (t, plc_level, camera_level, truth)
        self.extra_alarms: list[str] = []  # free-form notices; rendered escaped by the dashboard
        self.lock = threading.RLock()
        self.last_t: float | None = None
        self.last_jpeg: bytes | None = None
        self.last: dict | None = None
        self.last_analysis: Analysis | None = None
        self._alarms: list[str] = []
        self._refused = False
        self.finished = False
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None

    # -- state the HMI and tests read
    @property
    def truth(self) -> float:
        return self.tank.level_pct

    @property
    def permit(self) -> bool:
        return not self.state.tripped

    def close(self) -> None:
        if self._log:
            self._log.close()
            self._log = None

    # -- one frame
    def tick(self, frame, analysis: Analysis | None = None) -> dict:
        with self.lock:
            t = float(frame[0])
            dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
            truth = self.tank.level_pct
            reported = self.injector.reported(truth, t)
            if analysis is None:
                if self.vision is not None:
                    analysis = self.vision.read(frame)
                else:
                    analysis = Analysis(self._fake(truth, t))
            reading: Reading = analysis.reading
            if self.client:
                self.server.publish_plc(reported, self.plc.pump_on)
                plc_level = self.client.read_plc_level()
            else:
                plc_level = round(reported, 1)  # same x10 quantisation as the register
            if self.server:
                self.server.publish_camera(reading.level_pct, reading.confidence, reading.status)

            s = self.summary
            if s.t_diverge_exceeds_T is None and abs(truth - reported) > self.cfg.T_pts:
                s.t_diverge_exceeds_T = t
            self.state, actions = interlock.step(self.state, reading, plc_level, 0.0, t, self.cfg)
            if self.state.tripped and not s.tripped:
                s.tripped, s.trip_t, s.trip_reason = True, self.state.trip_t, self.state.trip_reason
                if reading.level_pct is not None:
                    s.diff_at_trip = abs(reading.level_pct - plc_level)

            pump_on = self.plc.update(plc_level, self.permit)
            if self.vision is None:
                self.tank.step(dt, pump_on)
            elif self.permit and reading.alertable:  # coupled plant; held once the pump permit is gone
                self.tank.level_pct = min(100.0, max(0.0, reading.level_pct))
            s.max_level = max(s.max_level, self.tank.level_pct)
            s.overflowed = self.tank.overflowed
            s.permit_coil_final = self.permit
            s.seconds = t

            self._alarms = [a.value for a in actions if a.value.startswith("ALARM_")]
            if analysis.camera_moved:
                self._alarms.append("CAMERA_MOVED")
            self.last_t, self.last_analysis = t, analysis
            if len(frame) > 1 and frame[1]:
                self.last_jpeg = frame[1]
            self.history.append((t, plc_level, reading.level_pct, truth))
            while self.history and self.history[0][0] < t - self.history_s:
                self.history.popleft()
            ev = {
                "t": t, "truth": round(truth, 3), "reported": round(reported, 3), "plc": plc_level,
                "camera": reading.level_pct, "confidence": reading.confidence, "status": reading.status,
                "camera_moved": analysis.camera_moved, "pump_on": pump_on, "tripped": self.state.tripped,
                "trip_reason": self.state.trip_reason, "actions": [a.value for a in actions],
            }
            self.last = ev
            if self._log:
                self._log.write(json.dumps(ev) + "\n")
            return ev

    # -- operator / demo controls
    def inject(self, scenario: str) -> None:
        """Start a fault scenario now (demo buttons). The PLC's reported level is the only thing it changes."""
        with self.lock:
            start = self.last_t if self.last_t is not None else 0.0
            inj = FaultInjector(scenario, start_t=start, drift_k=DRIFT_K, replay_window_s=REPLAY_WINDOW_S)
            for tt, _plc, _cam, truth in self.history:  # replay needs the pre-fault window
                if tt < start:
                    inj.reported(truth, tt)
            self.injector = inj
            self.summary.scenario = scenario

    def human_reset(self) -> dict:
        """A human operator's reset; the interlock refuses it while the trip condition still holds."""
        with self.lock:
            self.state, actions = interlock.reset(self.state, by_human=True)
            self._refused = interlock.Action.ALARM_RESET_REFUSED in actions
            return {"ok": not self.state.tripped, "refused": self._refused}

    def snapshot(self) -> dict:
        with self.lock:
            ev = self.last or {}
            a = self.last_analysis
            status = ev.get("status", "CAMERA_BLIND")
            moved = bool(a and a.camera_moved)
            alarms = list(self._alarms) + (["ALARM_RESET_REFUSED"] if self._refused else []) + list(self.extra_alarms)
            return {
                "t": self.last_t, "clock": time.strftime("%Y-%m-%d %H:%M:%S"), "banner": BANNER,
                "plc_level": ev.get("plc"), "pump_on": bool(ev.get("pump_on", False)),
                "camera": {
                    "level_pct": ev.get("camera"), "confidence": ev.get("confidence", 0.0), "status": status,
                    "alertable": bool(a and a.reading.alertable), "camera_moved": moved,
                },
                "permit": self.permit, "tripped": self.state.tripped, "trip_reason": self.state.trip_reason,
                "trip_t": self.state.trip_t, "alarms": alarms, "scenario": self.injector.scenario,
                "camera_blind_notice": status in BLIND_STATUSES or moved, "finished": self.finished,
            }


def drive(pl: Pipeline, source, latest_wins: bool = False, pace_s: float = 0.0, stop=None) -> Pipeline:
    """Feed every frame of `source` into the pipeline. latest_wins: slow vision never queues stale frames."""
    worker = None
    if latest_wins and pl.vision is not None:
        worker = LatestWorker(lambda f: (f, pl.vision.read(f)), lambda fa: pl.tick(fa[0], fa[1]))
    try:
        for frame in source:
            if stop is not None and stop.is_set():
                break
            if worker:
                worker.submit(frame)
            else:
                pl.tick(frame)
            if pace_s:
                time.sleep(pace_s)
    finally:
        if worker:
            worker.close()
        pl.finished = True
    return pl


def run(source, vision, scenario: str = "benign", fault_start_s: float = FAULT_START_S, seed: int = 1,
        camera_sigma: float = 3.0, log_path=None, use_modbus: bool = False,
        cfg: interlock.Config | None = None, latest_wins: bool = False) -> Summary:
    server = client = None
    if use_modbus:
        server = ModbusPlcServer(port=0).start()
        client = ModbusClient("127.0.0.1", server.port)
    pl = Pipeline(vision, scenario, fault_start_s, cfg, seed, camera_sigma, server, client, log_path)
    try:
        drive(pl, source, latest_wins)
    finally:
        pl.close()
        if client:
            client.close()
        if server:
            server.stop()
    return pl.summary


# -- CLI wiring (shared with web.py)

def build_source(spec: str, fps: float, cam_user=None, cam_password=None):
    kind, _, arg = spec.partition(":")
    if kind == "clip":
        return ClipSource(arg, fps)
    if kind == "stills":
        return StillsSource(sorted(glob.glob(arg)), 1.0 / fps)
    if kind == "http":  # spec is "http:http://host/shot.jpg"
        return HttpShotSource(arg, cam_user, cam_password, period_s=1.0 / fps)
    if kind == "fake":
        return FakeSource(n=int(arg), fps=fps)
    raise ValueError("source must be clip:PATH | stills:GLOB | http:URL | fake:N")


def build_vision(replay: str | None, cassette: str | None = None, run_id: str = "sightglass"):
    """Replay JSON (or 'demo') or a cassette-backed MetaClient: both keyless. Live Meta calls are not wired here."""
    if replay:
        return ReplayVision.demo() if replay == "demo" else ReplayVision(replay)
    if cassette:
        import tempfile

        from agentwarden.cassette import Cassette
        from agentwarden.cost import Ledger
        from agentwarden.meta import MetaClient

        def no_network(*_a, **_k):
            raise RuntimeError("cassette replay mode makes no network calls")

        ledger = Ledger(Path(tempfile.mkdtemp(prefix="sg_ledger_")) / "ledger.jsonl")
        return LiveVision(MetaClient(no_network, ledger, Cassette(cassette), run_id, 1.0, "replay"),
                          run_id=run_id)
    return None


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", help="clip:PATH | stills:GLOB | http:URL | fake:N (default: from --replay times)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--scenario", choices=SCENARIOS, default="benign")
    p.add_argument("--replay", help="recorded readings JSON, or 'demo' (keyless, no network)")
    p.add_argument("--cassette", help="cassette dir: replay recorded SAM responses through MetaClient (keyless)")
    p.add_argument("--run-id", default="sightglass")
    p.add_argument("--fault-start", type=float, default=FAULT_START_S)
    p.add_argument("--modbus", action="store_true", help="route tags through a localhost Modbus server")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=3.0, help="fake camera noise, pct points")
    p.add_argument("--cam-user")
    p.add_argument("--cam-password")


def resolve(a):
    """(source, vision) from parsed common args. No source + a replay = ticks at the recorded times."""
    vision = build_vision(a.replay, a.cassette, a.run_id)
    if a.source:
        return build_source(a.source, a.fps, a.cam_user, a.cam_password), vision
    if isinstance(vision, ReplayVision):
        return FakeSource(times=vision.times()), vision
    return FakeSource(n=300, fps=a.fps), vision


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SightGlass camera pipeline (SIMULATED, localhost only)")
    add_common_args(p)
    p.add_argument("--log", help="write JSONL event log here")
    a = p.parse_args(argv)
    source, vision = resolve(a)
    live_camera = a.source is not None and a.source.startswith("http:")
    s = run(source, vision, a.scenario, a.fault_start, a.seed, a.sigma, a.log, a.modbus,
            latest_wins=live_camera)
    print(json.dumps(s.__dict__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
