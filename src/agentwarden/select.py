"""Decide which detections are the container and the liquid."""

import math

import numpy as np

from agentwarden.level import Box
from agentwarden.sam import Detection

MIN_INSIDE = (4, 5)  # >= 4/5 = 80 % of mask pixels inside the container box
MAX_HEIGHT = (95, 100)  # mask height must be < 95 % of container box height, else it is the bottle


def _is_liquid(mask: np.ndarray, box: Box) -> bool:
    h, w = mask.shape
    l, t, r, b = box
    x0, x1 = max(0, math.floor(l)), min(w, math.ceil(r))
    y0, y1 = max(0, math.floor(t)), min(h, math.ceil(b))
    total = int(mask.sum())
    if total == 0 or x1 <= x0 or y1 <= y0:
        return False
    inside = int(mask[y0:y1, x0:x1].sum())
    if inside * MIN_INSIDE[1] < total * MIN_INSIDE[0]:
        return False
    rows = np.flatnonzero(mask.any(axis=1))
    height = int(rows[-1] - rows[0]) + 1
    return height * MAX_HEIGHT[1] < (b - t) * MAX_HEIGHT[0]


def pick(
    container_dets: list[Detection], liquid_dets: list[Detection]
) -> tuple[Box, np.ndarray | None, float | None] | None:
    """(container_box, liquid_mask, liquid_conf), or None if there is no container (CAMERA_BLIND).

    Container = highest-confidence detection of the separate "bottle" prompt. Liquid = union of
    the liquid-prompt masks that lie inside that box and are not bottle-tall (a split liquid
    around a label is one liquid), with the max confidence. No survivors -> (box, None, None):
    the EMPTY path.
    """
    if not container_dets:
        return None
    cont = max(container_dets, key=lambda d: d.confidence or 0.0)
    keep = [d for d in liquid_dets if d.mask is not None and _is_liquid(d.mask, cont.box)]
    if not keep:
        return cont.box, None, None
    mask = np.logical_or.reduce([d.mask for d in keep])
    return cont.box, mask, max(d.confidence or 0.0 for d in keep)
