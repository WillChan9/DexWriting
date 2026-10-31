"""Reference decoder: task map -> ordered strokes with a target force per point.

The benchmark hands a policy the task as an image (darkness = target press force),
never as a waypoint list. Something has to turn that picture into an order to draw
it in. This is a deliberately simple reference implementation so the interface is
usable out of the box - policies are free to ignore it and plan their own way,
and beating it is part of the point.

    strokes = map_to_strokes(obs["task"]["map"], tool="pen")
    for xy, force in zip(strokes[0].xy, strokes[0].force): ...

Coordinates come back in sheet metres relative to the sheet centre, the same frame
the task map is drawn in, so they can be compared directly with the tool tip.
"""
from dataclasses import dataclass

import numpy as np

from .scene import FORCE_MAX, PAPER_HALF, ink_width_mm


@dataclass
class Stroke:
    xy: np.ndarray        # (n, 2) sheet coordinates [m], relative to the sheet centre
    force: np.ndarray     # (n,) target press force [N] read off the map's darkness


def px_to_sheet(px, py, shape, half_m=None, center=(0.0, 0.0)):
    """Inverse of the env's sheet -> pixel mapping (row 0 is the sheet's +y edge).

    `shape` is the map's (rows, cols); `half_m` is its (half_x, half_y) in metres and
    `center` its centre, both arriving in the task vector as `map_half_m` /
    `map_center_m`. The map is the sheet's own rectangle, so rows and columns carry
    different extents even though the scale is one pixel per millimetre on both."""
    rows, cols = shape[0], shape[1]
    hx, hy = (PAPER_HALF[0], PAPER_HALF[1]) if half_m is None else (half_m[0], half_m[1])
    u = (px + 0.5) / cols
    v = 1.0 - (py + 0.5) / rows
    return np.array([center[0] + u * 2 * hx - hx,
                     center[1] + v * 2 * hy - hy])


def map_to_strokes(task_map, tool="pen", step_px=2.0, min_len=3, half_m=None, center=(0.0, 0.0)):
    """Greedy skeleton walk over the target image.

    Pixels are consumed by a disc whose radius is the ink width the pixel's own
    darkness implies, which is exactly how the map was drawn - so a thick stroke
    collapses to a single centre line instead of being traced as a zigzag.
    """
    img = np.asarray(task_map)
    shape = img.shape
    todo = img > 0
    if not todo.any():
        return []
    hx = PAPER_HALF[0] if half_m is None else half_m[0]
    px_per_m = shape[1] / (2 * hx)
    ys, xs = np.nonzero(todo)
    pts = np.stack([xs, ys], axis=1).astype(float)          # (n, 2) as (col, row)
    vals = img[ys, xs].astype(float)
    alive = np.ones(len(pts), bool)

    def consume(p, val):
        w_mm = ink_width_mm(tool, val / 255.0 * FORCE_MAX)
        r = max(1.0, 0.5 * w_mm / 1000.0 * px_per_m)
        alive[alive & (np.sum((pts - p) ** 2, axis=1) <= r * r)] = False

    strokes = []
    while alive.any():
        idx = np.flatnonzero(alive)
        # start where the remaining ink is most extreme along its principal axis:
        # that is an end of a stroke far more often than the middle of one is
        rem = pts[idx]
        c = rem.mean(axis=0)
        d = rem - c
        if len(rem) > 2:
            axis = np.linalg.svd(d - d.mean(0), full_matrices=False)[2][0]
            start = idx[int(np.argmin(d @ axis))]
        else:
            start = idx[0]
        cur = pts[start]
        chain, forces = [cur.copy()], [vals[start]]
        consume(cur, vals[start])
        reach = step_px * 2.2
        while True:
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            dist2 = np.sum((pts[idx] - cur) ** 2, axis=1)
            j = int(np.argmin(dist2))
            if dist2[j] > reach * reach:
                break
            k = idx[j]
            cur = pts[k]
            chain.append(cur.copy())
            forces.append(vals[k])
            consume(cur, vals[k])
        if len(chain) >= min_len:
            xy = np.array([px_to_sheet(p[0], p[1], shape, half_m, center) for p in chain])
            strokes.append(Stroke(xy=xy, force=np.array(forces) / 255.0 * FORCE_MAX))
    return order_strokes(strokes)


def order_strokes(strokes, start=None):
    """Nearest-neighbour ordering, flipping each stroke to the nearer end, so the
    pen-up hops between strokes stay short."""
    out, todo = [], list(strokes)
    cur = np.zeros(2) if start is None else np.asarray(start)
    while todo:
        best, flip, bd = 0, False, np.inf
        for i, s in enumerate(todo):
            for f, end in ((False, s.xy[0]), (True, s.xy[-1])):
                d = float(np.sum((end - cur) ** 2))
                if d < bd:
                    best, flip, bd = i, f, d
        s = todo.pop(best)
        if flip:
            s = Stroke(xy=s.xy[::-1].copy(), force=s.force[::-1].copy())
        out.append(s)
        cur = s.xy[-1]
    return out
