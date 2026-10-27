import json
import os
import threading
import time

import pytest

from agentwarden.cost import Ledger, SpendRefused


def test_guard_refuses(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", global_cap_usd=1.0)
    with pytest.raises(SpendRefused):  # unknown model
        led.reserve("nope", 10, 10, "r", 1.0)
    with pytest.raises(SpendRefused):  # zero cap
        led.reserve("muse-spark-1.3", 10, 10, "r", 0)
    with pytest.raises(SpendRefused):  # no output budget
        led.reserve("muse-spark-1.3", 10, 0, "r", 1.0)
    # 100k output tokens of muse-spark-1.3 = 0.425 USD worst case
    led.reserve("muse-spark-1.3", 0, 100_000, "r", 1.0)
    led.reserve("muse-spark-1.3", 0, 100_000, "r", 1.0)
    with pytest.raises(SpendRefused):  # 0.85 + 0.425 > 1.0 run cap
        led.reserve("muse-spark-1.3", 0, 100_000, "r", 1.0)


def test_guard_refuses_global_cap(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", global_cap_usd=1.0)
    led.reserve("muse-spark-1.3", 0, 150_000, "a", 10.0)  # 0.6375
    with pytest.raises(SpendRefused):  # other run, 0.6375 * 2 > 1.0 global
        led.reserve("muse-spark-1.3", 0, 150_000, "b", 10.0)


def test_cost_by_hand(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    rid = led.reserve("muse-spark-1.3", 1_000_000, 1_000_000, "r", 10.0)
    led.settle(rid, 1_000_000, 1_000_000)
    assert led.spent_total() == pytest.approx(5.50)  # 1.25 + 4.25
    assert led.spent_for_run("r") == pytest.approx(5.50)
    assert led.spent_for_run("other") == 0.0


def test_settle_releases_reservation(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", global_cap_usd=1.0)
    rid = led.reserve("muse-spark-1.3", 0, 200_000, "r", 1.0)  # 0.85 reserved
    led.settle(rid, 0, 1_000)  # actual 0.00425
    led.reserve("muse-spark-1.3", 0, 200_000, "r", 1.0)  # fits again


def test_settle_cached_costs_zero_keeps_tokens(tmp_path):
    p = tmp_path / "l.jsonl"
    led = Ledger(p)
    rid = led.reserve("muse-spark-1.3", 1000, 500, "r", 1.0)
    led.settle(rid, 1000, 500, cached=True)
    assert led.spent_total() == 0.0
    last = json.loads(p.read_text().splitlines()[-1])
    assert last["input_tokens"] == 1000 and last["output_tokens"] == 500
    assert last["cost_micro"] == 0 and last["cached"] is True


def test_ledger_concurrent_writes(tmp_path):
    p = tmp_path / "l.jsonl"
    led = Ledger(p)

    def work() -> None:
        for _ in range(25):
            rid = led.reserve("muse-spark-1.3-contributor", 1_000_000, 1_000_000, "r", 1000.0)
            led.settle(rid, 1_000_000, 1_000_000)

    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(p.read_text().splitlines()) == 400  # 200 reserve + 200 settle
    assert led.spent_total() == 60.0  # 200 * (0.10 + 0.20), exact


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0, -float("inf")])
def test_reserve_refuses_bad_cap(tmp_path, bad):
    led = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(SpendRefused):
        led.reserve("muse-spark-1.3", 10, 10, "r", bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0])
def test_ledger_refuses_bad_global_cap(tmp_path, bad):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "l.jsonl", global_cap_usd=bad)


def test_settle_refuses_negative_tokens(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    rid = led.reserve("muse-spark-1.3", 1000, 1000, "r", 1.0)
    with pytest.raises(ValueError):
        led.settle(rid, -1, 10)
    with pytest.raises(ValueError):
        led.settle(rid, 10, -1)
    assert led.spent_total() == 0.0
    led.settle(rid, 10, 10)  # reservation still open after refused settles


def test_stale_lock_taken_over(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.lock_path.write_text("999999 0")
    old = time.time() - 100
    os.utime(led.lock_path, (old, old))
    led.reserve("muse-spark-1.3", 10, 10, "r", 1.0)  # must not hang or time out
    assert not led.lock_path.exists()


def test_fresh_lock_is_busy(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.lock_path.write_text("999999 0")
    with pytest.raises(TimeoutError):
        with led._locked(timeout=0.1):
            pass


def test_lock_file_has_pid_and_timestamp(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    with led._locked():
        pid, ts = led.lock_path.read_text().split()
        assert pid == str(os.getpid())
        assert abs(float(ts) - time.time()) < 5


def test_permission_error_is_not_busy(tmp_path, monkeypatch):
    led = Ledger(tmp_path / "l.jsonl")
    real_open = os.open

    def fake_open(path, *a, **k):
        if str(path) == str(led.lock_path):
            raise PermissionError("denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr(os, "open", fake_open)
    t0 = time.monotonic()
    with pytest.raises(PermissionError):
        led.reserve("muse-spark-1.3", 10, 10, "r", 1.0)
    assert time.monotonic() - t0 < 5  # raised at once, not after the 30s timeout


def test_truncated_last_line_skipped_with_warning(tmp_path):
    p = tmp_path / "l.jsonl"
    led = Ledger(p)
    rid = led.reserve("muse-spark-1.3", 0, 1_000_000, "r", 10.0)
    led.settle(rid, 0, 1_000_000)  # 4.25
    with p.open("a") as f:
        f.write('{"kind": "reser')  # crash mid-write
    with pytest.warns(UserWarning):
        assert led.spent_total() == 4.25
    led.reserve("muse-spark-1.3", 0, 10, "r", 10.0)  # append after garbage stays readable
    assert led.spent_total() == 4.25


def test_garbled_earlier_line_raises(tmp_path):
    p = tmp_path / "l.jsonl"
    led = Ledger(p)
    led.reserve("muse-spark-1.3", 0, 10, "r", 10.0)
    with p.open("a") as f:
        f.write("garbage\n{}\n")
    with pytest.raises(ValueError):
        led.spent_total()


def test_release_frees_reservation(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", global_cap_usd=1.0)
    rid = led.reserve("muse-spark-1.3", 0, 200_000, "r", 1.0)  # 0.85
    led.release(rid)
    led.reserve("muse-spark-1.3", 0, 200_000, "r", 1.0)  # fits again
    assert led.spent_total() == 0.0


def test_double_release_settle_unknown_raise(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    a = led.reserve("muse-spark-1.3", 10, 10, "r", 1.0)
    led.release(a)
    with pytest.raises(ValueError):
        led.release(a)
    with pytest.raises(ValueError):
        led.settle(a, 1, 1)
    b = led.reserve("muse-spark-1.3", 10, 10, "r", 1.0)
    led.settle(b, 1, 1)
    with pytest.raises(ValueError):
        led.settle(b, 1, 1)
    with pytest.raises(ValueError):
        led.release(b)
    with pytest.raises(KeyError):
        led.release("nope")
    with pytest.raises(KeyError):
        led.settle("nope", 1, 1)
