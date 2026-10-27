"""Container/liquid selection. Frame 200 rows x 100 cols. Container box (10, 20, 50, 180):
cols 10..49, rows 20..179, height 160 -> the 95 % bottle-height limit is 152 rows."""

import numpy as np

H, W = 200, 100
CBOX = (10.0, 20.0, 50.0, 180.0)


def det(box, conf, rows=None, cols=None):
    from agentwarden.sam import Detection

    mask = None
    if rows is not None:
        mask = np.zeros((H, W), dtype=bool)
        mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    return Detection(box, conf, mask)


def container(conf=0.9):
    return det(CBOX, conf)


def pick(cont, liq):
    from agentwarden.select import pick

    return pick(cont, liq)


def test_no_container_is_none():
    assert pick([], [det((0, 0, 1, 1), 0.9, (0, 5), (0, 5))]) is None


def test_highest_confidence_container_wins():
    other = det((60.0, 20.0, 90.0, 180.0), 0.4)
    box, mask, conf = pick([other, container(0.9)], [])
    assert box == CBOX and mask is None and conf is None  # no liquid -> EMPTY path
    box, _, _ = pick([container(0.9), det((60.0, 20.0, 90.0, 180.0), 0.95)], [])
    assert box == (60.0, 20.0, 90.0, 180.0)


def test_liquid_accepted():
    liq = det((12.0, 100.0, 48.0, 180.0), 0.7, (100, 180), (12, 48))  # 80 rows < 152
    box, mask, conf = pick([container()], [liq])
    assert box == CBOX and conf == 0.7
    assert np.array_equal(mask, liq.mask)


def test_bottle_as_liquid_rejected():
    # bottle body tinted: mask spans rows 20..179 = 160 rows >= 152
    bottle = det(CBOX, 0.8, (20, 180), (10, 50))
    _, mask, conf = pick([container()], [bottle])
    assert mask is None and conf is None


def test_height_limit_boundary_is_95_percent_exclusive():
    # 151 rows < 152 accepted, 152 rows rejected
    ok = det((10.0, 29.0, 50.0, 180.0), 0.6, (29, 180), (10, 50))
    assert pick([container()], [ok])[1] is not None
    bad = det((10.0, 28.0, 50.0, 180.0), 0.6, (28, 180), (10, 50))
    assert pick([container()], [bad])[1] is None


def test_inside_fraction_boundary_is_80_percent_inclusive():
    # 10 x 10 mask = 100 px; cols 42..51 -> cols 42..49 inside = 80 px = 80 %
    ok = det((42.0, 100.0, 52.0, 110.0), 0.5, (100, 110), (42, 52))
    assert pick([container()], [ok])[1] is not None
    # cols 43..52 -> 7 cols inside = 70 %
    bad = det((43.0, 100.0, 53.0, 110.0), 0.5, (100, 110), (43, 53))
    assert pick([container()], [bad])[1] is None


def test_cap_and_label_outside_rejected_inside_accepted():
    cap = det((20.0, 5.0, 40.0, 15.0), 0.9, (5, 15), (20, 40))  # above the container
    table = det((0.0, 185.0, 99.0, 199.0), 0.9, (185, 199), (0, 99))  # reflection below
    assert pick([container()], [cap, table])[1] is None
    tiny_inside = det((20.0, 60.0, 23.0, 63.0), 0.55, (60, 63), (20, 23))
    _, mask, conf = pick([container()], [cap, table, tiny_inside])
    assert conf == 0.55 and np.array_equal(mask, tiny_inside.mask)


def test_split_liquid_unioned_with_max_confidence():
    top = det((12.0, 100.0, 48.0, 120.0), 0.6, (100, 120), (12, 48))
    bottom = det((12.0, 140.0, 48.0, 180.0), 0.8, (140, 180), (12, 48))  # label sits between
    _, mask, conf = pick([container()], [top, bottom])
    assert conf == 0.8
    assert np.array_equal(mask, top.mask | bottom.mask)
    assert mask.sum() == 20 * 36 + 40 * 36 and not mask[125, 20]


def test_union_excludes_rejected_candidates():
    liq = det((12.0, 100.0, 48.0, 180.0), 0.5, (100, 180), (12, 48))
    bottle = det(CBOX, 0.99, (20, 180), (10, 50))
    _, mask, conf = pick([container()], [bottle, liq])
    assert conf == 0.5 and np.array_equal(mask, liq.mask)


def test_candidate_without_mask_is_ignored():
    _, mask, conf = pick([container()], [det((12.0, 100.0, 48.0, 180.0), 0.9)])
    assert mask is None and conf is None
