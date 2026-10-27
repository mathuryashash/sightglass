"""Spend ledger and guard: append-only JSONL with worst-case reservations."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# model id -> (usd per 1M input tokens, usd per 1M output tokens)
PRICES: dict[str, tuple[float, float]] = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20),
}
# model id -> usd per image
IMAGE_PRICES: dict[str, float] = {
    "sam-3.1": 0.0025,
    "muse-image-1.0": 0.01,
}
_EPS = 1e-9


class SpendRefused(Exception):
    """Raised when a reservation is refused."""


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost of the given token counts for a known model."""
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


class Ledger:
    """Append-only JSONL spend ledger guarded by a cross-platform file lock."""

    def __init__(self, path: str | Path, global_cap_usd: float = 100.0) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.global_cap_usd = global_cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self, timeout: float = 30.0) -> Iterator[None]:
        """Hold an O_EXCL lock file; retry until timeout."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError, PermissionError):  # Windows: PermissionError while unlinking
                if time.monotonic() > deadline:
                    raise TimeoutError(f"ledger lock busy: {self.lock_path}")
                time.sleep(0.002)
        try:
            yield
        finally:
            os.close(fd)
            os.unlink(self.lock_path)

    def _read(self) -> list[dict]:
        """Load all ledger rows (caller holds the lock when consistency matters)."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append(self, row: dict) -> None:
        """Append one row and flush it to disk."""
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _totals(rows: list[dict], run_id: str | None = None) -> tuple[float, float]:
        """Return (spent, reserved-and-unsettled) USD, optionally for one run."""
        settled = {r["id"] for r in rows if r["kind"] == "settle"}
        spent = reserved = 0.0
        for r in rows:
            if run_id is not None and r["run_id"] != run_id:
                continue
            if r["kind"] == "settle":
                spent += r["cost_usd"]
            elif r["id"] not in settled:
                reserved += r["cost_usd"]
        return spent, reserved

    def reserve(self, model: str, max_input_tokens: int, max_output_tokens: int | None,
                run_id: str, cap_usd: float) -> str:
        """Reserve worst-case cost; raise SpendRefused if any guard trips."""
        if model not in PRICES:
            raise SpendRefused(f"unknown model: {model!r}")
        if cap_usd is None or cap_usd <= 0:
            raise SpendRefused("cap_usd must be > 0")
        if not max_output_tokens or max_output_tokens <= 0:
            raise SpendRefused("max_output_tokens must be set and > 0")
        if max_input_tokens is None or max_input_tokens < 0:
            raise SpendRefused("max_input_tokens must be >= 0")
        worst = token_cost(model, max_input_tokens, max_output_tokens)
        with self._locked():
            rows = self._read()
            run_total = sum(self._totals(rows, run_id))
            if run_total + worst > cap_usd + _EPS:
                raise SpendRefused(f"run {run_id!r} would exceed cap {cap_usd}")
            if sum(self._totals(rows)) + worst > self.global_cap_usd + _EPS:
                raise SpendRefused(f"global cap {self.global_cap_usd} would be exceeded")
            rid = uuid.uuid4().hex
            self._append({"kind": "reserve", "id": rid, "model": model, "run_id": run_id,
                          "input_tokens": max_input_tokens, "output_tokens": max_output_tokens,
                          "cost_usd": worst, "cached": False, "ts": time.time()})
        return rid

    def settle(self, reservation_id: str, input_tokens: int, output_tokens: int,
               cached: bool = False) -> float:
        """Record actual cost (0 if cached) and release the reservation."""
        with self._locked():
            rows = self._read()
            res = next((r for r in rows if r["kind"] == "reserve" and r["id"] == reservation_id), None)
            if res is None:
                raise KeyError(f"unknown reservation: {reservation_id}")
            if any(r["kind"] == "settle" and r["id"] == reservation_id for r in rows):
                raise ValueError(f"reservation already settled: {reservation_id}")
            cost = 0.0 if cached else token_cost(res["model"], input_tokens, output_tokens)
            self._append({"kind": "settle", "id": reservation_id, "model": res["model"],
                          "run_id": res["run_id"], "input_tokens": input_tokens,
                          "output_tokens": output_tokens, "cost_usd": cost,
                          "cached": cached, "ts": time.time()})
        return cost

    def spent_total(self) -> float:
        """Total settled USD across all runs."""
        with self._locked():
            return self._totals(self._read())[0]

    def spent_for_run(self, run_id: str) -> float:
        """Total settled USD for one run."""
        with self._locked():
            return self._totals(self._read(), run_id)[0]
