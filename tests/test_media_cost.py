import pytest

from agentwarden.cost import Ledger, SpendRefused


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl", global_cap_usd=100.0)


def test_three_images_cost_0_0075(ledger):
    rid = ledger.reserve_media("sam-3.1", "image", 3, "r1", 1.0)
    assert ledger.settle_media(rid, 3) == 0.0075
    assert ledger.spent_for_run("r1") == 0.0075


def test_ten_video_frames_billed_fifty(ledger):
    rid = ledger.reserve_media("sam-3.1", "video_frame", 10, "r1", 1.0)
    assert ledger.settle_media(rid, 10) == 0.0100


def test_video_frames_over_minimum_billed_exact(ledger):
    rid = ledger.reserve_media("sam-3.1", "video_frame", 100, "r1", 1.0)
    assert ledger.settle_media(rid, 100) == 0.0200


def test_reserve_holds_worst_case_against_run_cap(ledger):
    ledger.reserve_media("sam-3.1", "video_frame", 10, "r1", 0.0100)  # reserves 50 frames = 0.0100
    with pytest.raises(SpendRefused):
        ledger.reserve_media("sam-3.1", "image", 1, "r1", 0.0100)  # 0.0100 + 0.0025 > cap


def test_run_cap_refuses_exact_overflow(ledger):
    with pytest.raises(SpendRefused):
        ledger.reserve_media("sam-3.1", "image", 4, "r1", 0.0099)  # 0.0100 > 0.0099


def test_global_cap_refuses(tmp_path):
    small = Ledger(tmp_path / "l.jsonl", global_cap_usd=0.005)
    small.reserve_media("sam-3.1", "image", 2, "r1", 1.0)  # 0.0050 fits exactly
    with pytest.raises(SpendRefused):
        small.reserve_media("sam-3.1", "image", 1, "r2", 1.0)


def test_settle_below_reserve_frees_headroom(ledger):
    rid = ledger.reserve_media("sam-3.1", "image", 4, "r1", 0.0100)
    ledger.settle_media(rid, 1)
    ledger.reserve_media("sam-3.1", "image", 3, "r1", 0.0100)  # 0.0025 spent + 0.0075 = cap


@pytest.mark.parametrize("model,kind", [("nope", "image"), ("sam-3.1", "audio"),
                                        ("muse-spark-1.3", "image")])
def test_unknown_model_or_kind_refused(ledger, model, kind):
    with pytest.raises(SpendRefused):
        ledger.reserve_media(model, kind, 1, "r1", 1.0)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, None, "3"])
def test_bad_count_refused(ledger, count):
    with pytest.raises(SpendRefused):
        ledger.reserve_media("sam-3.1", "image", count, "r1", 1.0)


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf"), None])
def test_bad_cap_refused(ledger, cap):
    with pytest.raises(SpendRefused):
        ledger.reserve_media("sam-3.1", "image", 1, "r1", cap)


@pytest.mark.parametrize("count", [-1, 1.5, True, None])
def test_bad_settle_count_rejected(ledger, count):
    rid = ledger.reserve_media("sam-3.1", "image", 1, "r1", 1.0)
    with pytest.raises(ValueError):
        ledger.settle_media(rid, count)


def test_settle_zero_images_costs_nothing(ledger):
    rid = ledger.reserve_media("sam-3.1", "image", 2, "r1", 1.0)
    assert ledger.settle_media(rid, 0) == 0.0


def test_settle_zero_video_frames_still_bills_minimum(ledger):
    rid = ledger.reserve_media("sam-3.1", "video_frame", 10, "r1", 1.0)
    assert ledger.settle_media(rid, 0) == 0.0100


def test_double_settle_and_release_after_settle_rejected(ledger):
    rid = ledger.reserve_media("sam-3.1", "image", 1, "r1", 1.0)
    ledger.settle_media(rid, 1)
    with pytest.raises(ValueError):
        ledger.settle_media(rid, 1)
    with pytest.raises(ValueError):
        ledger.release(rid)


def test_release_frees_media_reservation(ledger):
    rid = ledger.reserve_media("sam-3.1", "image", 4, "r1", 0.0100)
    ledger.release(rid)
    ledger.reserve_media("sam-3.1", "image", 4, "r1", 0.0100)
    assert ledger.spent_total() == 0.0


def test_unknown_reservation_keyerror(ledger):
    with pytest.raises(KeyError):
        ledger.settle_media("nope", 1)


def test_token_settle_on_media_reservation_rejected(ledger):
    rid = ledger.reserve_media("sam-3.1", "image", 1, "r1", 1.0)
    with pytest.raises(ValueError):
        ledger.settle(rid, 10, 10)


def test_media_settle_on_token_reservation_rejected(ledger):
    rid = ledger.reserve("muse-spark-1.3", 100, 100, "r1", 1.0)
    with pytest.raises(ValueError):
        ledger.settle_media(rid, 1)


def test_many_small_settles_sum_exactly(ledger):
    for _ in range(40):
        rid = ledger.reserve_media("sam-3.1", "image", 1, "r1", 1.0)
        ledger.settle_media(rid, 1)
    assert ledger.spent_total() == 0.1000  # 40 x 0.0025, no float drift
