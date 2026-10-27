"""SAM output parsing. Fixtures are encoded with meta_sam_parser's own one_bit encoder
(private `_encode_raster`, the exact inverse of the public `decode_mask_to_raster`)."""

import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster


def mask_text(raster, box, *, src=(60, 80), conf="0.9", obj=0):
    """One SAM record line. `box` = inclusive (x1, y1, x2, y2); `src` = (H, W) of source image."""
    h, w = len(raster), len(raster[0])
    flat = [int(v) for row in raster for v in row]
    payload = _encode_raster(flat, w, h)
    x1, y1, x2, y2 = box
    return (
        f"<0f>{obj}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={src[1]};h={src[0]};c={conf}|>"
        f"<|mask;x=0;y=0;data={h},{w},{payload}|>"
    )


FRAME = (60, 80)  # (H, W)


def test_mask_offset_lands_on_frame_pixels():
    from agentwarden.sam import parse_sam

    raster = [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]]  # 3 rows x 4 cols
    text = mask_text(raster, (10, 20, 13, 22))  # bounds left=10 top=20 right=14 bottom=23
    [det] = parse_sam(text, FRAME)
    assert det.box == (10.0, 20.0, 14.0, 23.0)
    assert det.confidence == pytest.approx(0.9)
    assert det.mask.shape == FRAME and det.mask.dtype == bool
    expected = np.zeros(FRAME, dtype=bool)
    for r, row in enumerate(raster):
        for c, v in enumerate(row):
            expected[20 + r, 10 + c] = bool(v)
    assert np.array_equal(det.mask, expected)
    assert det.mask.sum() == 6
    assert det.mask[20, 10] and not det.mask[0, 0]


def test_mask_raster_smaller_than_bounds_is_scaled():
    from agentwarden.sam import parse_sam

    # bounds: left=10 top=20 right=16 bottom=24 -> 6 wide x 4 tall; raster 2x2 -> scale_x=3, scale_y=2
    text = mask_text([[1, 0], [0, 1]], (10, 20, 15, 23))
    [det] = parse_sam(text, FRAME)
    expected = np.zeros(FRAME, dtype=bool)
    expected[20:22, 10:13] = True  # mask (0,0)
    expected[22:24, 13:16] = True  # mask (1,1)
    assert np.array_equal(det.mask, expected)
    assert det.mask.sum() == 12


def test_mask_clipped_to_frame():
    from agentwarden.sam import parse_sam

    text = mask_text([[1, 1], [1, 1]], (78, 58, 79, 59), src=(60, 80))
    [det] = parse_sam(text, (60, 80))
    assert det.mask.sum() == 4 and det.mask[59, 79]


def test_multiple_records_and_missing_confidence():
    from agentwarden.sam import parse_sam

    a = mask_text([[1]], (1, 2, 1, 2), conf="0.5")
    b = mask_text([[1]], (5, 6, 5, 6), obj=1).replace(";c=0.9", "")
    dets = parse_sam(a + "\n" + b + "\n", FRAME)
    assert [d.box for d in dets] == [(1.0, 2.0, 2.0, 3.0), (5.0, 6.0, 6.0, 7.0)]
    assert dets[0].confidence == pytest.approx(0.5)
    assert dets[1].confidence is None
    assert dets[1].mask[6, 5]


def test_without_frame_shape_masks_are_none():
    from agentwarden.sam import parse_sam

    [det] = parse_sam(mask_text([[1]], (1, 2, 1, 2)))
    assert det.mask is None and det.box == (1.0, 2.0, 2.0, 3.0)


@pytest.mark.parametrize("text", ["", "\n", "  \n\n"])
def test_empty_output(text):
    from agentwarden.sam import parse_sam

    assert parse_sam(text, FRAME) == []


def _bad_payload():
    good = mask_text([[1, 0], [0, 1]], (1, 1, 2, 2))
    return good[:-6] + "!!!!|>"  # corrupt the tail of the payload


@pytest.mark.parametrize(
    "text",
    [
        "I cannot find any liquid.",
        "<0f>0<|box;x1=1;y1=1|>",
        "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=80;h=60|>",  # no mask token
        "<3f>0<|box;x1=1;y1=1;x2=2;y2=2;w=80;h=60|>",  # image output must be frame 0
        mask_text([[1]], (5, 5, 1, 1)),  # x2 < x1
        mask_text([[1]], (1, 1, 99, 99)),  # box outside its declared source size
        _bad_payload(),
        "garbage line\n" + mask_text([[1]], (1, 2, 1, 2)),  # no partial results
        mask_text([[1]], (1, 2, 1, 2)) + "\n<0f>0<|box;oops",
    ],
)
def test_malformed_raises(text):
    from agentwarden.sam import SamParseError, parse_sam

    with pytest.raises(SamParseError):
        parse_sam(text, FRAME)
