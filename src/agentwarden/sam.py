"""Parse SAM 3.1 output_text into detections with frame-sized boolean masks."""

from dataclasses import dataclass

import meta_sam_parser as msp
import numpy as np


class SamParseError(ValueError):
    """SAM output was malformed; no partial results are returned."""


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # (left, top, right, bottom), half-open, frame pixels
    confidence: float | None
    mask: np.ndarray | None  # bool, frame shape (H, W); None if no frame_shape given


def _frame_mask(rec: "msp.SegmentationMaskRecord", frame_shape: tuple[int, int]) -> np.ndarray:
    """Place a box-local mask raster into a full-frame bool array.

    Frame pixel (y, x) inside the bounds samples mask pixel
    (floor((y - top) / scale_y), floor((x - left) / scale_x)), scale = bounds size / raster size.
    """
    h, w = frame_shape
    raster = np.frombuffer(msp.decode_mask_to_raster(rec.mask), dtype=np.uint8)
    raster = raster.reshape(rec.mask.height, rec.mask.width).astype(bool)
    b = rec.bounds
    sy = (b.bottom - b.top) / rec.mask.height
    sx = (b.right - b.left) / rec.mask.width
    ys = np.arange(max(int(b.top), 0), min(int(b.bottom), h))
    xs = np.arange(max(int(b.left), 0), min(int(b.right), w))
    out = np.zeros((h, w), dtype=bool)
    if ys.size and xs.size:
        r = np.minimum(((ys - b.top) / sy).astype(int), raster.shape[0] - 1)
        c = np.minimum(((xs - b.left) / sx).astype(int), raster.shape[1] - 1)
        out[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1] = raster[np.ix_(r, c)]
    return out


def parse_sam(
    output_text: str, frame_shape: tuple[int, int] | None = None
) -> list[Detection]:
    """Parse SAM output_text. `frame_shape` = (H, W); without it, masks are None.

    Raises SamParseError on any unparseable line, invalid record, or box/mask mismatch.
    """
    parser = msp.image_segmentation_format().create_parser()
    parser.push(output_text)
    result = parser.finish(msp.CompletedOutcome()).result

    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise SamParseError(f"line {errors[0].line}: {errors[0].code}: {errors[0].message}")

    boxes: dict[str, msp.SegmentationBoxRecord] = {}
    masks: dict[str, msp.SegmentationMaskRecord] = {}
    order: list[str] = []
    for rec in result.records:
        if rec.kind == "text":
            raise SamParseError(f"unexpected non-record text: {rec.text[:80]!r}")
        table = boxes if rec.kind == "box" else masks
        if rec.object_id in table:
            raise SamParseError(f"duplicate {rec.kind} for object {rec.object_id}")
        table[rec.object_id] = rec
        if rec.kind == "box":
            order.append(rec.object_id)
    if boxes.keys() != masks.keys():
        raise SamParseError("box/mask records do not pair up")

    dets = []
    for oid in order:
        box, mrec = boxes[oid], masks[oid]
        conf = mrec.confidence if mrec.confidence is not None else box.confidence
        try:
            mask = None if frame_shape is None else _frame_mask(mrec, frame_shape)
        except msp.InvalidSegmentationMaskError as e:
            raise SamParseError(str(e)) from e
        dets.append(
            Detection((float(box.left), float(box.top), float(box.right), float(box.bottom)), conf, mask)
        )
    return dets
