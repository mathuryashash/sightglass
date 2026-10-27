"""Spend ledger and guard: append-only JSONL with worst-case reservations."""
from __future__ import annotations

import json
import math
import os
import time
import uuid
import warnings
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
STALE_LOCK_SECONDS = 10.0
_MICRO = 1_000_000  # micro-dollars per USD; USD-per-1M-tokens == micro-dollars per token
# (model, kind) -> (usd per unit, minimum units billed per request)
MEDIA_PRICES: dict[tuple[str, str], tuple[float, int]] = {
    **{(m, "image"): (p, 0) for m, p in IMAGE_PRICES.items()},
    ("sam-3.1", "video_frame"): (0.0002, 50),
}


class SpendRefused(Exception):
    """Raised when a reservation is refused."""


def _micro_cost(model: str, input_tokens: int, output_tokens: int) -> int:
    """Return integer micro-dollar cost of the given token counts."""
    p_in, p_out = PRICES[model]
    return round(input_tokens * p_in + output_tokens * p_out)


def _usd_to_micro(usd: float) -> int:
    return round(usd * _MICRO)


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost of the given token counts for a known model."""
    return _micro_cost(model, input_tokens, output_tokens) / _MICRO


def _media_cost(model: str, kind: str, count: int) -> int:
    """Return integer micro-dollar cost of `count` units, honouring the per-request minimum."""
    price, minimum = MEDIA_PRICES[(model, kind)]
    return _usd_to_micro(price) * max(count, minimum)


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _finite_positive(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


class Ledger:
    """Append-only JSONL spend ledger guarded by a cross-platform file lock."""

    def __init__(self, path: str | Path, global_cap_usd: float = 100.0) -> None:
        if not _finite_positive(global_cap_usd):
            raise ValueError(f"global_cap_usd must be a finite number > 0, got {global_cap_usd!r}")
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.global_cap_usd = global_cap_usd
        self._global_cap_micro = _usd_to_micro(global_cap_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _drop_lock(self) -> None:
        """Rename the lock to a unique dead name, then delete it (frees the name at once)."""
        dead = self.lock_path.with_name(f"{self.lock_path.name}.{uuid.uuid4().hex}.dead")
        os.replace(self.lock_path, dead)
        os.unlink(dead)

    @contextmanager
    def _locked(self, timeout: float = 30.0) -> Iterator[None]:
        """Hold an O_EXCL lock file (pid + timestamp inside); take over stale locks."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:  # any other OSError (e.g. PermissionError) propagates
                try:
                    stale = time.time() - os.stat(self.lock_path).st_mtime > STALE_LOCK_SECONDS
                    if stale:
                        # ponytail: stat->replace race can drop a fresh lock; use OS locks if it matters
                        self._drop_lock()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"ledger lock busy: {self.lock_path}")
                time.sleep(0.002)
        try:
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            yield
        finally:
            os.close(fd)
            self._drop_lock()

    def _read(self) -> list[dict]:
        """Load all ledger rows; a garbled LAST line is skipped with a warning."""
        if not self.path.exists():
            return []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = []
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except ValueError:
                if i != len(lines) - 1:
                    raise
                warnings.warn(f"ignoring truncated last ledger line in {self.path}")
        return rows

    def _repair_tail(self) -> None:
        """Cut an unparseable last line so the next append starts clean."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        stripped = data.rstrip(b"\r\n")
        cut = stripped.rfind(b"\n") + 1
        try:
            json.loads(stripped[cut:])
        except ValueError:
            if stripped[cut:].strip():
                self.path.write_bytes(data[:cut])
        else:
            if not data.endswith(b"\n"):
                self.path.write_bytes(data + b"\n")

    def _append(self, row: dict) -> None:
        """Append one row and flush it to disk."""
        self._repair_tail()
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _totals(rows: list[dict], run_id: str | None = None) -> tuple[int, int]:
        """Return (spent, reserved-and-open) micro-dollars, optionally for one run."""
        closed = {r["id"] for r in rows if r["kind"] in ("settle", "release")}
        spent = reserved = 0
        for r in rows:
            if run_id is not None and r["run_id"] != run_id:
                continue
            if r["kind"] == "settle":
                spent += r["cost_micro"]
            elif r["kind"] == "reserve" and r["id"] not in closed:
                reserved += r["cost_micro"]
        return spent, reserved

    def reserve(self, model: str, max_input_tokens: int, max_output_tokens: int | None,
                run_id: str, cap_usd: float) -> str:
        """Reserve worst-case cost; raise SpendRefused if any guard trips."""
        if model not in PRICES:
            raise SpendRefused(f"unknown model: {model!r}")
        if not _finite_positive(cap_usd):
            raise SpendRefused("cap_usd must be a finite number > 0")
        if not max_output_tokens or max_output_tokens <= 0:
            raise SpendRefused("max_output_tokens must be set and > 0")
        if max_input_tokens is None or max_input_tokens < 0:
            raise SpendRefused("max_input_tokens must be >= 0")
        worst = _micro_cost(model, max_input_tokens, max_output_tokens)
        return self._reserve(worst, run_id, cap_usd,
                             {"model": model, "input_tokens": max_input_tokens,
                              "output_tokens": max_output_tokens})

    def _reserve(self, worst: int, run_id: str, cap_usd: float, fields: dict) -> str:
        """Check run + global caps for `worst` micro-dollars and append the reserve row."""
        cap_micro = _usd_to_micro(cap_usd)
        with self._locked():
            rows = self._read()
            if sum(self._totals(rows, run_id)) + worst > cap_micro:
                raise SpendRefused(f"run {run_id!r} would exceed cap {cap_usd}")
            if sum(self._totals(rows)) + worst > self._global_cap_micro:
                raise SpendRefused(f"global cap {self.global_cap_usd} would be exceeded")
            rid = uuid.uuid4().hex
            self._append({"kind": "reserve", "id": rid, "run_id": run_id, "cost_micro": worst,
                          "cached": False, "ts": time.time(), **fields})
        return rid

    def reserve_media(self, model: str, kind: str, count: int, run_id: str, cap_usd: float) -> str:
        """Reserve worst-case cost of `count` images / video frames (minimum billing applies)."""
        if (model, kind) not in MEDIA_PRICES:
            raise SpendRefused(f"unknown media model/kind: {model!r}/{kind!r}")
        if not _finite_positive(cap_usd):
            raise SpendRefused("cap_usd must be a finite number > 0")
        if not _is_int(count) or count < 1:
            raise SpendRefused("count must be an int >= 1")
        worst = _media_cost(model, kind, count)
        return self._reserve(worst, run_id, cap_usd,
                             {"model": model, "media_kind": kind, "count": count})

    def _open_reservation(self, rows: list[dict], reservation_id: str) -> dict:
        """Return the reserve row; KeyError if unknown, ValueError if already closed."""
        res = next((r for r in rows if r["kind"] == "reserve" and r["id"] == reservation_id), None)
        if res is None:
            raise KeyError(f"unknown reservation: {reservation_id}")
        if any(r["kind"] in ("settle", "release") and r["id"] == reservation_id for r in rows):
            raise ValueError(f"reservation already closed: {reservation_id}")
        return res

    def settle(self, reservation_id: str, input_tokens: int, output_tokens: int,
               cached: bool = False) -> float:
        """Record actual cost (0 if cached) and close the reservation."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        with self._locked():
            res = self._open_reservation(self._read(), reservation_id)
            if "media_kind" in res:
                raise ValueError("media reservation: use settle_media")
            cost = 0 if cached else _micro_cost(res["model"], input_tokens, output_tokens)
            self._append({"kind": "settle", "id": reservation_id, "model": res["model"],
                          "run_id": res["run_id"], "input_tokens": input_tokens,
                          "output_tokens": output_tokens, "cost_micro": cost,
                          "cached": cached, "ts": time.time()})
        return cost / _MICRO

    def settle_media(self, reservation_id: str, count: int) -> float:
        """Record actual media cost (minimum billing applies) and close the reservation."""
        if not _is_int(count) or count < 0:
            raise ValueError("count must be an int >= 0")
        with self._locked():
            res = self._open_reservation(self._read(), reservation_id)
            if "media_kind" not in res:
                raise ValueError("token reservation: use settle")
            cost = _media_cost(res["model"], res["media_kind"], count)
            self._append({"kind": "settle", "id": reservation_id, "model": res["model"],
                          "run_id": res["run_id"], "media_kind": res["media_kind"],
                          "count": count, "cost_micro": cost, "cached": False, "ts": time.time()})
        return cost / _MICRO

    def release(self, reservation_id: str) -> None:
        """Free an unsettled reservation without charging anything."""
        with self._locked():
            res = self._open_reservation(self._read(), reservation_id)
            self._append({"kind": "release", "id": reservation_id, "model": res["model"],
                          "run_id": res["run_id"], "ts": time.time()})

    def spent_total(self) -> float:
        """Total settled USD across all runs."""
        with self._locked():
            return self._totals(self._read())[0] / _MICRO

    def spent_for_run(self, run_id: str) -> float:
        """Total settled USD for one run."""
        with self._locked():
            return self._totals(self._read(), run_id)[0] / _MICRO
