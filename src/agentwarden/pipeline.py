"""Camera source + vision (live or replay) -> modbus camera tag -> interlock -> event log.

SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM. The interlock can only switch the pump permit off;
`permit` here is held by permit.PermitOutput: this module can only pass it the interlock's actions, and the one
way it goes back on is `human_reset()` -> `interlock.reset_with_grant(by_human=True)` -> `PermitOutput.rearm(grant)`.

    python -m agentwarden.pipeline --source clip:../spike/real/pour.mp4 --fps 1 --scenario freeze --replay <json>
    python -m agentwarden.pipeline --replay demo --scenario freeze
"""
import argparse
import glob
import json
import logging
import random
import threading
import time
from collections import deque
from pathlib import Path

from agentwarden import interlock
from agentwarden.camera_source import ClipSource, FakeSource, HttpShotSource, ReplayFrames, StillsSource
from agentwarden.fault_injector import SCENARIOS, FaultInjector
from agentwarden.level import Reading
from agentwarden.modbus_io import ModbusClient, ModbusPlcServer
from agentwarden.permit import PermitOutput
from agentwarden.plc_sim import PlcController, TankSim
from agentwarden.runner import DRIFT_K, FAULT_START_S, REPLAY_WINDOW_S, Summary, fake_camera
from agentwarden.vision import Analysis, LatestWorker, LiveVision, ReplayVision

log = logging.getLogger(__name__)

BANNER = "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM"
BLIND_STATUSES = ("CAMERA_BLIND", "LOW_CONFIDENCE")
UNVERIFIED_ADVICE_S = 10.0  # camera unverified for longer than this queues one advisory diagnosis
ADVICE_WINDOW_S = 120.0  # recent events kept for the advisory diagnosis
STALL_S = 3.0  # no finished tick for this long (wall clock) raises LOOP_STALLED
EVENTS_SHOWN = 8  # the dashboard timeline
CAMERA_NOTICES = {"moved": "camera moved - recalibrate", "blind": "camera blinded - no trip"}


class Pipeline:
    """One tick per frame. With a vision, the simulated plant is coupled to what the camera sees (the real
    bottle IS the process) until the interlock trips; with vision=None a TankSim plant and a noisy fake
    camera are used, as in runner.py."""

    def __init__(self, vision, scenario: str = "benign", fault_start_s: float = FAULT_START_S,
                 cfg: interlock.Config | None = None, seed: int = 1, camera_sigma: float = 3.0,
                 server: ModbusPlcServer | None = None, client: ModbusClient | None = None,
                 log_path: str | Path | None = None, history_s: float = 300.0, advice=None, relay=None,
                 clock=time.monotonic, stall_s: float = STALL_S):
        self.vision, self.cfg = vision, cfg or interlock.Config()
        self.server, self.client = server, client
        # `advice`: anything with submit(events, plc_series, cam_series, trigger=, limits=) and snapshot(). It only
        # receives copies of recent readings and never sees or touches the permit; it is not imported here.
        self.advice = advice
        self._relay, self._scenario0, self._fault_start0 = relay, scenario, fault_start_s
        self._seed, self._sigma, self.history_s = seed, camera_sigma, history_s
        self.clock, self.stall_s = clock, stall_s
        self.extra_alarms: list[str] = []  # free-form notices; rendered escaped by the dashboard
        self.lock = threading.RLock()
        self.finished = False
        self.n_ticks = 0  # never reset: the dashboard caches its renders per tick
        self.loop_errors = self.loop_count = 0
        self.overlay_for = None  # optional t -> pre-drawn overlay picture (ReplayFrames.overlay)
        self._init_run(first=True)
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None

    def _init_run(self, first: bool = False) -> None:
        """All per-run state: used at construction and by restart()."""
        self._advice_events: deque = deque()
        self._advice_sent: set = set()
        self._unverified_since: float | None = None
        self.tank = TankSim(capacity_l=1000, fill_rate_lps=9, drain_rate_lps=2, level_pct=20.0)
        self.plc = PlcController(low=30.0, high=80.0)
        self.state = interlock.initial_state()
        self.out = PermitOutput(self.server, self.client, self._relay if first else None)
        self.injector = FaultInjector(self._scenario0, start_t=self._fault_start0, drift_k=DRIFT_K,
                                      replay_window_s=REPLAY_WINDOW_S)
        self._fake = fake_camera(self._sigma, random.Random(self._seed)) if self.vision is None else None
        self.summary = Summary(self._scenario0, 0.0)
        self.history: deque = deque()  # (t, plc_level, camera_level, truth)
        self.timeline: deque = deque(maxlen=60)  # {"t", "kind", "text"}, oldest first
        self.last_t: float | None = None
        self.last_jpeg: bytes | None = None
        self.last_overlay: bytes | None = None
        self.last: dict | None = None
        self.last_analysis: Analysis | None = None
        self.read_failed = False
        self._alarms: list[str] = []
        self._refused = False
        self._plc_ok_t: float | None = None  # frame time of the last successful PLC read
        self._plc_last: float | None = None
        self._fault_noted = self._trip_noted = False
        self._cam_state, self._plc_stale = "ok", False
        self._tick_mono = self.clock()

    # -- state the HMI and tests read
    @property
    def truth(self) -> float:
        return self.tank.level_pct

    @property
    def permit(self) -> bool:
        return self.out.permit

    def close(self) -> None:
        if self._log:
            self._log.close()
            self._log = None

    # -- one frame
    def read_frame(self, frame) -> Analysis:
        """Vision for one frame. Any exception means the camera is unverified (CAMERA_BLIND): never a crash, never a
        trip. Also the LatestWorker's function, so the worker path ticks on a failure too."""
        try:
            analysis = self.vision.read(frame)
            self.read_failed = False
            return analysis
        except Exception as e:  # ponytail: type name only, never the message
            self.read_failed = True
            log.warning("vision failed: %s", type(e).__name__)
            return Analysis(Reading(None, 0.0, "CAMERA_BLIND", None, False, False))

    def _read_plc(self, t: float, reported: float) -> tuple[float, float]:
        """(plc_level, age_s). A failed Modbus read keeps the last good value and the age keeps growing, so it
        counts toward PLC_STALE instead of killing the loop."""
        try:
            if self.client:
                self.server.publish_plc(reported, self.plc.pump_on)
                level = self.client.read_plc_level()
            else:
                level = round(reported, 1)  # same x10 quantisation as the register
            self._plc_ok_t, self._plc_last = t, level
        except Exception as e:
            log.warning("PLC read failed: %s", type(e).__name__)
            level = self._plc_last if self._plc_last is not None else round(reported, 1)
            if self._plc_ok_t is None:
                self._plc_ok_t = t
        return level, max(0.0, t - self._plc_ok_t)

    def tick(self, frame, analysis: Analysis | None = None) -> dict:
        with self.lock:
            t = float(frame[0])
            dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
            truth = self.tank.level_pct
            reported = self.injector.reported(truth, t)
            if analysis is None:
                analysis = self.read_frame(frame) if self.vision is not None else Analysis(self._fake(truth, t))
            reading: Reading = analysis.reading
            plc_level, plc_age = self._read_plc(t, reported)
            if self.server:
                try:
                    self.server.publish_camera(reading.level_pct, reading.confidence, reading.status)
                except Exception as e:
                    log.warning("camera tag publish failed: %s", type(e).__name__)

            s = self.summary
            if s.t_diverge_exceeds_T is None and abs(truth - reported) > self.cfg.T_pts:
                s.t_diverge_exceeds_T = t
            self.state, actions = interlock.step(self.state, reading, plc_level, plc_age, t, self.cfg)
            self.out.apply(actions)
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
            if self.read_failed:
                self._alarms.append("CAMERA_READ_ERROR")
            self.last_t, self.last_analysis = t, analysis
            if len(frame) > 1 and frame[1]:
                self.last_jpeg = frame[1]
            self.last_overlay = self._overlay(t)
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
            self._note_transitions(t, reading.status, analysis.camera_moved, plc_age)
            if self._log:
                self._log.write(json.dumps(ev) + "\n")
            if self.advice is not None:
                self._advise(t, reading.status, pump_on, actions)
            self.n_ticks += 1
            self._tick_mono = self.clock()
            return ev

    def _overlay(self, t: float) -> bytes | None:
        if self.overlay_for is None:
            return None
        try:
            return self.overlay_for(t)
        except Exception as e:  # a missing picture must never stop the loop
            log.warning("overlay lookup failed: %s", type(e).__name__)
            return None

    def note(self, t: float, kind: str, text: str) -> None:
        """One line for the dashboard timeline (rendered escaped)."""
        with self.lock:
            self.timeline.append({"t": float(t), "kind": kind, "text": str(text)})

    def _note_transitions(self, t: float, status: str, moved: bool, plc_age: float) -> None:
        inj = self.injector
        if not self._fault_noted and inj.scenario != "benign" and t >= inj.start_t:
            self._fault_noted = True
            self.note(inj.start_t, "fault", f"fault injected: {inj.scenario}")
        if self.state.tripped and not self._trip_noted:
            self._trip_noted = True
            self.note(t, "trip", f"TRIPPED: {self.state.trip_reason}")
        cam = "moved" if moved else "blind" if status in BLIND_STATUSES else "ok"
        if cam != self._cam_state:
            self._cam_state = cam
            self.note(t, "camera", CAMERA_NOTICES.get(cam, "camera verified again"))
        stale = plc_age > self.cfg.plc_stale_s
        if stale != self._plc_stale:
            self._plc_stale = stale
            self.note(t, "plc", "PLC data stale" if stale else "PLC data fresh again")

    def _advise(self, t: float, status: str, pump_on: bool, actions) -> None:
        """Queue ONE advisory diagnosis per trip (or per camera-unverified spell over 10 s). submit() only copies
        data and starts a thread, so the tick loop is never held. No scenario name and no simulated truth is
        passed on: only what a plant operator could see."""
        acts = [a.value for a in actions]
        self._advice_events.append({"t": t, "status": status, "pump_on": pump_on, "tripped": self.state.tripped,
                                    "trip_reason": self.state.trip_reason, "actions": acts})
        while self._advice_events and self._advice_events[0]["t"] < t - ADVICE_WINDOW_S:
            self._advice_events.popleft()
        unverified = interlock.Action.ALARM_CAMERA_UNVERIFIED.value in acts
        if unverified:
            self._unverified_since = t if self._unverified_since is None else self._unverified_since
        else:
            self._unverified_since = None
            self._advice_sent.discard("camera_unverified")
        trigger = ("trip" if self.state.tripped else
                   "camera_unverified" if unverified and t - self._unverified_since > UNVERIFIED_ADVICE_S else None)
        if trigger is None or trigger in self._advice_sent:
            return
        self._advice_sent.add(trigger)
        limits = {"divergence_pts": self.cfg.T_pts, "divergence_s": self.cfg.N_s, "high_high_pct": self.cfg.hh_limit}
        self.advice.submit(list(self._advice_events), [(h[0], h[1]) for h in self.history],
                           [(h[0], h[2]) for h in self.history], trigger=trigger, limits=limits)

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
            self._fault_noted = False
            self.summary.scenario = scenario

    def human_reset(self) -> dict:
        """A human operator's reset; the interlock refuses it while the trip condition still holds."""
        with self.lock:
            was_tripped = self.state.tripped
            self.state, actions, grant = interlock.reset_with_grant(self.state, by_human=True)
            if grant:
                self.out.rearm(grant)
                self._advice_sent.discard("trip")  # a later trip gets its own diagnosis
                self._trip_noted = False
            self._refused = interlock.Action.ALARM_RESET_REFUSED in actions
            if was_tripped:
                self.note(self.last_t or 0.0, "reset", "human reset refused - condition still active"
                          if self._refused else "human reset accepted - pump permit restored")
            return {"ok": not self.state.tripped, "refused": self._refused}

    def recalibrate(self) -> bool:
        """A human says the camera was moved on purpose (POST /recalibrate): the vision takes its reference box and
        calibration from the latest frame. False when this vision has nothing to recalibrate (replay, no camera)."""
        with self.lock:
            fn = getattr(self.vision, "recalibrate", None)
            ok = bool(fn and fn())
            if ok:
                self.note(self.last_t or 0.0, "camera", "camera recalibrated by human")
            return ok

    def restart(self) -> None:
        """Demo harness only (--loop): start the whole run again from t=0 with a fresh plant, interlock and permit.
        Not reachable over HTTP. Refused with Modbus (the wire would still hold the old cut permit) or a custom relay."""
        if self.server or self.client or self._relay is not None:
            raise RuntimeError("restart is only for the in-process replay demo (no Modbus, no custom relay)")
        with self.lock:
            self.loop_count += 1
            if hasattr(self.vision, "rewind"):
                self.vision.rewind()
            self._init_run()
            self.note(0.0, "restart", f"replay restarted (loop {self.loop_count})")

    def render_inputs(self) -> dict:
        """Everything the dashboard needs to draw the frame and the trend, copied under the lock so the drawing
        itself happens outside it. `key` changes on every tick."""
        with self.lock:
            return {"key": self.n_ticks, "jpeg": self.last_jpeg, "overlay": self.last_overlay,
                    "analysis": self.last_analysis, "history": list(self.history),
                    "trip_t": self.state.trip_t if self.state.tripped else None,
                    "fault_t": self.injector.start_t if self._fault_noted else None}

    def snapshot(self) -> dict:
        with self.lock:
            ev = self.last or {}
            a = self.last_analysis
            status = ev.get("status", "CAMERA_BLIND")
            moved = bool(a and a.camera_moved)
            loop_age = max(0.0, self.clock() - self._tick_mono)
            stalled = loop_age > self.stall_s and not self.finished
            alarms = (list(self._alarms) + (["ALARM_RESET_REFUSED"] if self._refused else [])
                      + list(self.extra_alarms) + (["LOOP_STALLED"] if stalled else []))
            return {
                "n_ticks": self.n_ticks, "loop_age_s": loop_age,
                "limits": {"T_pts": self.cfg.T_pts, "N_s": self.cfg.N_s, "hh": self.cfg.hh_limit},
                "events": [dict(e) for e in list(self.timeline)[::-1][:EVENTS_SHOWN]],
                "fault_t": self.injector.start_t if self._fault_noted else None,
                "camera_notice": CAMERA_NOTICES["moved"] if moved else
                CAMERA_NOTICES["blind"] if status in BLIND_STATUSES else "",
                "t": self.last_t, "clock": time.strftime("%Y-%m-%d %H:%M:%S"), "banner": BANNER,
                "plc_level": ev.get("plc"), "pump_on": bool(ev.get("pump_on", False)),
                "camera": {
                    "level_pct": ev.get("camera"), "confidence": ev.get("confidence", 0.0), "status": status,
                    "alertable": bool(a and a.reading.alertable), "camera_moved": moved,
                },
                "permit": self.permit, "tripped": self.state.tripped, "trip_reason": self.state.trip_reason,
                "trip_t": self.state.trip_t, "alarms": alarms, "scenario": self.injector.scenario,
                "camera_blind_notice": status in BLIND_STATUSES or moved, "finished": self.finished,
                "relay": {**self.out.relay.snapshot(), "relay_fault": self.out.relay_fault},
                "diagnosis": self.advice.snapshot() if self.advice is not None else {"state": "off"},
            }


def _safe_tick(pl: Pipeline, frame, analysis=None) -> None:
    """One tick that can never end the loop: an exception is counted and logged (type only); the dashboard then
    shows LOOP_STALLED because no tick finished."""
    try:
        pl.tick(frame) if analysis is None else pl.tick(frame, analysis)
    except Exception as e:
        pl.loop_errors += 1
        log.error("tick failed: %s", type(e).__name__)


def drive(pl: Pipeline, source, latest_wins: bool = False, pace_s: float = 0.0, stop=None, loop: bool = False,
          start_at: float = 0.0) -> Pipeline:
    """Feed every frame of `source` into the pipeline. latest_wins: slow vision never queues stale frames.
    start_at: skip frames before this time. loop: when the source ends, restart the whole run (replay demos)."""
    stopped = lambda: stop is not None and stop.is_set()  # noqa: E731
    worker = None
    if latest_wins and pl.vision is not None:
        if loop:
            raise ValueError("loop is for replays, not live cameras")
        worker = LatestWorker(lambda f: (f, pl.read_frame(f)), lambda fa: _safe_tick(pl, fa[0], fa[1]))
    try:
        while not stopped():
            fed = False
            for frame in source:
                if stopped():
                    break
                if frame[0] < start_at:
                    continue
                fed = True
                if worker:
                    worker.submit(frame)
                else:
                    _safe_tick(pl, frame)
                if pace_s:
                    time.sleep(pace_s)
            if not (loop and fed) or stopped():
                break
            pl.restart()
    finally:
        if worker:
            worker.close()
        pl.finished = True
    return pl


def run(source, vision, scenario: str = "benign", fault_start_s: float = FAULT_START_S, seed: int = 1,
        camera_sigma: float = 3.0, log_path=None, use_modbus: bool = False,
        cfg: interlock.Config | None = None, latest_wins: bool = False, start_at: float = 0.0) -> Summary:
    server = client = None
    if use_modbus:
        server = ModbusPlcServer(port=0).start()
        client = ModbusClient("127.0.0.1", server.port)
    pl = Pipeline(vision, scenario, fault_start_s, cfg, seed, camera_sigma, server, client, log_path)
    try:
        drive(pl, source, latest_wins, start_at=start_at)
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
    p.add_argument("--start-at", type=float, default=0.0, help="skip frames before this time (s), for recording takes")
    p.add_argument("--loop", action="store_true", help="restart the replay from the top when it ends")
    p.add_argument("--frames", help="dir with the recording's real frames (default: <replay json dir>/pour_frames)")
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
        if vision.path and ReplayFrames.available(vision.path, a.frames):
            return ReplayFrames(vision.path, a.frames), vision
        return FakeSource(times=vision.times()), vision
    return FakeSource(n=300, fps=a.fps), vision


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SightGlass camera pipeline (SIMULATED, localhost only)")
    add_common_args(p)
    p.add_argument("--log", help="write JSONL event log here")
    a = p.parse_args(argv)
    if a.loop:
        p.error("--loop is for the dashboard (python -m agentwarden.web)")
    print(BANNER)
    source, vision = resolve(a)
    live_camera = a.source is not None and a.source.startswith("http:")
    s = run(source, vision, a.scenario, a.fault_start, a.seed, a.sigma, a.log, a.modbus,
            latest_wins=live_camera, start_at=a.start_at)
    print(json.dumps(s.__dict__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
