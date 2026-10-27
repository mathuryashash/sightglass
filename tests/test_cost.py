import json
import threading

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
    assert last["cost_usd"] == 0.0 and last["cached"] is True


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
    assert led.spent_total() == pytest.approx(60.0)  # 200 * (0.10 + 0.20)
