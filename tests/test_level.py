"""Level estimator. Calibration used throughout: row 400 -> 0 %, row 100 -> 100 %
(so level = (400 - row) / 3). Frame is 500 rows x 100 cols; container box cols 10..50
(40 columns, middle 50% = cols 20..39 = 20 columns)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

BOX = (10.0, 0.0, 50.0, 500.0)
H, W = 500, 100


def calib():
    from agentwarden.level import Calibration

    return Calibration.from_points([(400, 0), (100, 100)])


def liquid(top_by_col, bottom=400):
    """Mask with liquid from top_by_col[c] down to `bottom` (exclusive) in each given column."""
    m = np.zeros((H, W), dtype=bool)
    for c, top in top_by_col.items():
        m[top:bottom, c] = True
    return m


def flat(top, cols=range(10, 50)):
    return liquid({c: top for c in cols})


# ---- Calibration ----

def test_calibration_interpolates_and_clamps():
    from agentwarden.level import Calibration

    cal = Calibration.from_points([(400, 0), (250, 40), (100, 100)])
    assert cal.level_at(325) == (20.0, False)  # halfway on first segment
    assert cal.level_at(250) == (40.0, False)
    assert cal.level_at(175) == (70.0, False)
    assert cal.level_at(400) == (0.0, False) and cal.level_at(100) == (100.0, False)
    assert cal.level_at(450) == (0.0, True)  # extrapolated, clamped to [0, 100]
    assert cal.level_at(50) == (100.0, True)


def test_calibration_accepts_ascending_rows():
    from agentwarden.level import Calibration

    cal = Calibration.from_points([(100, 100), (400, 0)])
    assert cal.level_at(250)[0] == pytest.approx(50.0)


@pytest.mark.parametrize(
    "pts",
    [
        [],
        [(400, 0)],
        [(400, 0), (400, 100)],  # duplicate row
        [(400, 0), (100, 100), (250, 40)],  # not monotonic
        [(400, 0), (100, 101)],  # level > 100
        [(400, -1), (100, 100)],  # level < 0
    ],
)
def test_calibration_rejects_bad_points(pts):
    from agentwarden.level import Calibration

    with pytest.raises(ValueError):
        Calibration.from_points(pts)


# ---- estimate ----

def test_flat_surface_hand_computed():
    from agentwarden.level import estimate

    r = estimate(flat(250), BOX, calib())
    assert r.status == "OK" and r.top_row == 250
    assert r.level_pct == pytest.approx(50.0)  # (400-250)/300*100
    assert r.confidence == pytest.approx(1.0)
    assert not r.extrapolated


def test_meniscus_shifts_by_at_most_hand_computed_amount():
    from agentwarden.level import estimate

    # every middle column 3 px higher than edges: shift = 3/300*100 = 1.0 %
    tops = {c: (247 if 20 <= c < 40 else 250) for c in range(10, 50)}
    r = estimate(liquid(tops), BOX, calib())
    assert r.level_pct == pytest.approx(51.0)
    assert abs(r.level_pct - 50.0) <= 1.0 + 1e-9
    # only the 6 centre columns bulge: median is unaffected
    tops = {c: (247 if 27 <= c < 33 else 250) for c in range(10, 50)}
    assert estimate(liquid(tops), BOX, calib()).level_pct == pytest.approx(50.0)


def test_reflections_ignored():
    from agentwarden.level import estimate

    m = flat(250)
    m[120:123, 25] = True  # stray pixels well above the surface, 2 of 40 columns
    m[120:123, 30] = True
    r = estimate(m, BOX, calib())
    assert r.status == "OK"
    assert r.level_pct == pytest.approx(50.0)


def test_liquid_outside_container_does_not_count():
    from agentwarden.level import estimate

    m = flat(250)
    m[:, 60:90] = True  # liquid elsewhere, above surface too
    assert estimate(m, BOX, calib()).level_pct == pytest.approx(50.0)


def test_camera_blind():
    from agentwarden.level import estimate

    r = estimate(flat(250), None, calib())
    assert r.status == "CAMERA_BLIND" and r.level_pct is None


def test_empty_container():
    from agentwarden.level import estimate

    r = estimate(np.zeros((H, W), dtype=bool), BOX, calib())
    assert r.status == "EMPTY" and r.level_pct == 0.0
    m = np.zeros((H, W), dtype=bool)
    m[:, 60:90] = True  # liquid only outside the box
    assert estimate(m, BOX, calib()).status == "EMPTY"


def test_empty_confidence_is_half_container_confidence():
    from agentwarden.level import estimate

    none = np.zeros((H, W), dtype=bool)
    assert estimate(none, BOX, calib()).confidence == pytest.approx(0.5)  # unknown container conf = 1.0
    assert estimate(none, BOX, calib(), container_confidence=0.8).confidence == pytest.approx(0.4)
    assert estimate(none, BOX, calib(), container_confidence=1.0).confidence == pytest.approx(0.5)
    assert estimate(None, BOX, calib(), container_confidence=0.6).confidence == pytest.approx(0.3)


def test_alertable():
    from agentwarden.level import estimate

    none = np.zeros((H, W), dtype=bool)
    # EMPTY: 1.0 x 0.5 = 0.5 -> alertable; 0.98 x 0.5 = 0.49 -> not
    assert estimate(none, BOX, calib(), container_confidence=1.0).alertable
    assert not estimate(none, BOX, calib(), container_confidence=0.98).alertable
    # OK: confidence 0.5 -> alertable, 0.4 -> not
    assert estimate(flat(250), BOX, calib(), sam_confidence=0.5).alertable
    assert not estimate(flat(250), BOX, calib(), sam_confidence=0.4).alertable
    # LOW_CONFIDENCE never alertable
    r = estimate(liquid({c: 250 for c in range(20, 25)}), BOX, calib())
    assert r.status == "LOW_CONFIDENCE" and not r.alertable
    assert not estimate(flat(250), None, calib()).alertable  # CAMERA_BLIND


def test_none_mask_is_empty_path():
    from agentwarden.level import estimate

    r = estimate(None, BOX, calib(), container_confidence=0.8)
    assert r.status == "EMPTY" and r.level_pct == 0.0 and not r.alertable


def test_glare_band_above_surface_ignored():
    from agentwarden.level import estimate

    m = flat(250)
    m[148:153, 10:50] = True  # detached glare/foam band across every column, centred at row 150
    r = estimate(m, BOX, calib())
    assert r.top_row == 250  # surface-only reading: (400-250)/3 = 50 %
    assert r.level_pct == pytest.approx(50.0)
    assert r.status == "OK" and r.confidence == pytest.approx(1.0)
    # a band that touches the liquid body is part of it: the surface really is at 148
    assert estimate(flat(148), BOX, calib()).top_row == 148


def test_low_confidence_threshold():
    from agentwarden.level import estimate

    # 5 of 20 middle columns have liquid = 25 % < 30 % -> LOW_CONFIDENCE
    r = estimate(liquid({c: 250 for c in range(20, 25)}), BOX, calib())
    assert r.status == "LOW_CONFIDENCE"
    assert r.confidence == pytest.approx(0.25)
    # exactly 6 of 20 = 30 % -> OK
    r = estimate(liquid({c: 250 for c in range(20, 26)}), BOX, calib())
    assert r.status == "OK" and r.confidence == pytest.approx(0.3)
    assert r.level_pct == pytest.approx(50.0)
    # liquid only in the outer columns: nothing in the middle -> LOW_CONFIDENCE, no level
    r = estimate(liquid({c: 250 for c in range(10, 15)}), BOX, calib())
    assert r.status == "LOW_CONFIDENCE" and r.level_pct is None and r.confidence == 0.0


def test_confidence_multiplies_sam_confidence():
    from agentwarden.level import estimate

    assert estimate(flat(250), BOX, calib(), sam_confidence=0.8).confidence == pytest.approx(0.8)
    r = estimate(liquid({c: 250 for c in range(20, 30)}), BOX, calib(), sam_confidence=0.5)
    assert r.confidence == pytest.approx(0.25)  # 10/20 columns x 0.5


def test_extrapolated_flagged_on_reading():
    from agentwarden.level import estimate

    r = estimate(liquid({c: 450 for c in range(10, 50)}, bottom=500), BOX, calib())  # surface below the 0 % calibration row
    assert r.level_pct == 0.0 and r.extrapolated


# ---- camera_moved ----

def test_camera_moved():
    from agentwarden.level import camera_moved

    ref = (10.0, 0.0, 50.0, 500.0)
    assert not camera_moved(ref, ref, 2)
    assert not camera_moved((12.0, 0.0, 50.0, 498.0), ref, 2)  # exactly at tolerance
    assert camera_moved((13.0, 0.0, 50.0, 500.0), ref, 2)
    assert camera_moved((10.0, 0.0, 50.0, 503.0), ref, 2)
    assert camera_moved(None, ref, 2)


# ---- property ----

@given(st.integers(0, 499), st.integers(0, 499))
def test_higher_top_row_means_lower_percent(a, b):
    from agentwarden.level import estimate

    lo, hi = sorted((a, b))
    r_hi_surface = estimate(flat(lo), BOX, calib())  # smaller row = surface higher
    r_lo_surface = estimate(flat(hi), BOX, calib())
    if lo == hi:
        assert r_hi_surface.level_pct == r_lo_surface.level_pct
    else:
        assert r_hi_surface.level_pct >= r_lo_surface.level_pct
        if 100 <= lo < hi <= 400:
            assert r_hi_surface.level_pct > r_lo_surface.level_pct


@given(
    st.integers(100, 380),
    st.lists(st.integers(0, 4), min_size=40, max_size=40),  # per-column meniscus, px
    st.integers(1, 15),  # surface drops by k px
)
def test_meniscus_surface_lowering_never_raises_level(base, mens, k):
    from agentwarden.level import estimate

    tops = {10 + i: base + m for i, m in enumerate(mens)}  # rows 100..384
    lowered = {c: t + k for c, t in tops.items()}  # rows <= 399
    r1 = estimate(liquid(tops), BOX, calib())
    r2 = estimate(liquid(lowered), BOX, calib())
    assert r1.status == r2.status == "OK"
    mid = sorted(tops[c] for c in range(20, 40))
    median = (mid[9] + mid[10]) / 2  # 20 middle columns
    assert r1.top_row == median and r2.top_row == median + k
    assert r1.level_pct == pytest.approx((400 - median) / 3)
    assert r2.level_pct < r1.level_pct
