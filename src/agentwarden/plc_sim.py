"""Simulated tank + pump + PLC control. Pure: time is injected via dt, no wall clock, no I/O."""
from dataclasses import dataclass


@dataclass
class TankSim:
    capacity_l: float
    fill_rate_lps: float
    drain_rate_lps: float
    level_pct: float = 0.0
    overflowed: bool = False  # sticky: set once the unclamped level would exceed 100

    def step(self, dt: float, pump_on: bool) -> float:
        inflow = self.fill_rate_lps if pump_on else 0.0
        raw = self.level_pct + (inflow - self.drain_rate_lps) * dt / self.capacity_l * 100.0
        if raw > 100.0:
            self.overflowed = True
        self.level_pct = min(100.0, max(0.0, raw))
        return self.level_pct


@dataclass
class PlcController:
    """Pump ON below `low`, OFF above `high`, hold in between. permit=False forces OFF."""

    low: float = 30.0
    high: float = 80.0
    pump_on: bool = False

    def update(self, reported_level: float, permit: bool) -> bool:
        if not permit:
            self.pump_on = False
        elif reported_level < self.low:
            self.pump_on = True
        elif reported_level > self.high:
            self.pump_on = False
        return self.pump_on
