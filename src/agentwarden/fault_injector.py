"""Fault scenarios applied to the PLC's REPORTED level only; the simulated truth keeps evolving."""
import bisect

BANNER = "SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM"
LOCAL_HOSTS = ("127.0.0.1", "localhost")
SCENARIOS = ("benign", "freeze", "drift", "replay")


class FaultInjector:
    def __init__(
        self,
        scenario: str,
        host: str = "127.0.0.1",
        start_t: float = 0.0,
        drift_k: float = 0.5,
        replay_window_s: float = 60.0,
    ):
        if host not in LOCAL_HOSTS:
            raise ValueError(f"fault injector only runs against localhost, got {host!r}")
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
        self.scenario = scenario
        self.start_t = start_t
        self.drift_k = drift_k
        self.replay_window_s = replay_window_s
        self._frozen: float | None = None
        self._ts: list[float] = []  # truth history, for replay
        self._vs: list[float] = []
        print(BANNER)

    def reported(self, truth: float, t: float) -> float:
        if t < self.start_t or self.scenario == "benign":
            self._record(truth, t)
            return truth
        if self.scenario == "freeze":
            if self._frozen is None:
                self._frozen = truth
            return self._frozen
        if self.scenario == "drift":
            return min(100.0, max(0.0, truth - self.drift_k * (t - self.start_t)))
        # replay: loop the recorded window [start_t - W, start_t)
        w = self.replay_window_s
        src_t = self.start_t - w + ((t - self.start_t) % w)
        i = bisect.bisect_right(self._ts, src_t) - 1
        return self._vs[max(i, 0)] if self._vs else truth

    def _record(self, truth: float, t: float) -> None:
        if self.scenario == "replay":
            self._ts.append(t)
            self._vs.append(truth)
