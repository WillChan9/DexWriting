"""Stroke sources for the tasks: a single-stroke block-capital font (A-Z,
0-9), a few simple Chinese characters, geometric shapes and the cursive
"daxo" word. Every glyph is a list of polylines in a unit frame (x right,
y up, cap height 1.4 for Latin, 1.0 for Chinese)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return json.load(f)

# Single-line faces only. `sans_bold`, `serif` and `serif_italic` are Hershey DUPLEX faces - a
# thick stem is two parallel contours ~1.2-1.5 font-units apart, which a 1 mm ballpoint cannot
# render as a stem and simply draws twice. They are excluded rather than reduced to a centre
# line: a merge is a guess about what the face meant, and the benchmark should ask for a path the
# tool can actually draw.
# `script` is Hershey Script SIMPLEX (scripts.jhf, single-line). The double-line Script COMPLEX
# face (scriptc.jhf) stays in the file as `script_complex`, unused.
HERSHEY = {f: g for f, g in _load("hershey.json").items() if f in ("sans", "script")}
HANZI = _load("hanzi_medians.json")        # Make Me a Hanzi stroke medians, unit em, y up
PRESSURE_PROFILES = _load("pressure_profiles.json")   # real Wacom pen-down pressure curves [N]
FONT_FAMILIES = ["block", "sans", "sans_round", "script"]      # every one single-line
_H_FRAME = {}


def _hershey_frame(font):
    """(scale, baseline) putting the face into the block frame: cap height 1.4, 'A' on y=0."""
    if font not in _H_FRAME:
        P = np.concatenate([np.asarray(st, float) for st in HERSHEY[font]["A"]["strokes"]])
        _H_FRAME[font] = (1.4 / max(P[:, 1].max() - P[:, 1].min(), 1e-9), P[:, 1].min())
    return _H_FRAME[font]


def _chaikin(P, iters=2):
    """Corner-cutting subdivision - rounds every corner, keeps the endpoints."""
    for _ in range(iters):
        if len(P) < 3:
            return P
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q.append(0.75 * a + 0.25 * b)
            Q.append(0.25 * a + 0.75 * b)
        Q.append(P[-1])
        P = np.asarray(Q)
    return P


def _refine(strokes, rounded=0.0, spacing=0.03):
    """Smooth + densify glyph polylines (unit frame): rounded > 0 applies Chaikin corner
    rounding (the "rounded" font look), then equal-arc resampling so the sharp polyline
    corners of the sources never survive into the task."""
    out = []
    for st in strokes:
        st = np.asarray(st, float)
        if rounded > 0 and len(st) > 2:
            st = _chaikin(st, 2)
        out.append(resample(st, spacing))
    return out


LATIN = {
    "A": [[(0, 0), (0.5, 1.4), (1, 0)], [(0.2, 0.56), (0.8, 0.56)]],
    "B": [[(0, 0), (0, 1.4), (0.65, 1.4), (0.85, 1.2), (0.85, 0.9), (0.65, 0.7), (0, 0.7)],
          [(0.65, 0.7), (0.9, 0.5), (0.9, 0.2), (0.7, 0), (0, 0)]],
    "C": [[(0.9, 1.2), (0.7, 1.4), (0.3, 1.4), (0, 1.1), (0, 0.3), (0.3, 0), (0.7, 0), (0.9, 0.2)]],
    "D": [[(0, 0), (0, 1.4), (0.6, 1.4), (0.9, 1.1), (0.9, 0.3), (0.6, 0), (0, 0)]],
    "E": [[(0.9, 1.4), (0, 1.4), (0, 0), (0.9, 0)], [(0, 0.7), (0.7, 0.7)]],
    "F": [[(0.9, 1.4), (0, 1.4), (0, 0)], [(0, 0.7), (0.7, 0.7)]],
    "G": [[(0.9, 1.2), (0.7, 1.4), (0.3, 1.4), (0, 1.1), (0, 0.3), (0.3, 0), (0.7, 0), (0.9, 0.3),
           (0.9, 0.6), (0.5, 0.6)]],
    "H": [[(0, 0), (0, 1.4)], [(0.9, 0), (0.9, 1.4)], [(0, 0.7), (0.9, 0.7)]],
    "I": [[(0.45, 0), (0.45, 1.4)], [(0.1, 1.4), (0.8, 1.4)], [(0.1, 0), (0.8, 0)]],
    "J": [[(0.9, 1.4), (0.9, 0.3), (0.6, 0), (0.3, 0), (0, 0.3)]],
    "K": [[(0, 0), (0, 1.4)], [(0.9, 1.4), (0, 0.6)], [(0.3, 0.85), (0.9, 0)]],
    "L": [[(0, 1.4), (0, 0), (0.9, 0)]],
    "M": [[(0, 0), (0, 1.4), (0.5, 0.6), (1, 1.4), (1, 0)]],
    "N": [[(0, 0), (0, 1.4), (0.9, 0), (0.9, 1.4)]],
    "O": [[(0.3, 0), (0, 0.3), (0, 1.1), (0.3, 1.4), (0.6, 1.4), (0.9, 1.1), (0.9, 0.3), (0.6, 0), (0.3, 0)]],
    "P": [[(0, 0), (0, 1.4), (0.65, 1.4), (0.9, 1.2), (0.9, 0.85), (0.65, 0.65), (0, 0.65)]],
    "Q": [[(0.3, 0), (0, 0.3), (0, 1.1), (0.3, 1.4), (0.6, 1.4), (0.9, 1.1), (0.9, 0.3), (0.6, 0), (0.3, 0)],
          [(0.55, 0.35), (0.95, -0.1)]],
    "R": [[(0, 0), (0, 1.4), (0.65, 1.4), (0.9, 1.2), (0.9, 0.85), (0.65, 0.65), (0, 0.65)],
          [(0.45, 0.65), (0.9, 0)]],
    "S": [[(0.9, 1.2), (0.7, 1.4), (0.2, 1.4), (0, 1.2), (0, 0.9), (0.2, 0.7), (0.7, 0.7), (0.9, 0.5),
           (0.9, 0.2), (0.7, 0), (0.2, 0), (0, 0.2)]],
    "T": [[(0, 1.4), (0.9, 1.4)], [(0.45, 1.4), (0.45, 0)]],
    "U": [[(0, 1.4), (0, 0.3), (0.3, 0), (0.6, 0), (0.9, 0.3), (0.9, 1.4)]],
    "V": [[(0, 1.4), (0.45, 0), (0.9, 1.4)]],
    "W": [[(0, 1.4), (0.25, 0), (0.5, 1.0), (0.75, 0), (1, 1.4)]],
    "X": [[(0, 0), (0.9, 1.4)], [(0, 1.4), (0.9, 0)]],
    "Y": [[(0, 1.4), (0.45, 0.7), (0.9, 1.4)], [(0.45, 0.7), (0.45, 0)]],
    "Z": [[(0, 1.4), (0.9, 1.4), (0, 0), (0.9, 0)]],
    "0": [[(0.3, 0), (0, 0.3), (0, 1.1), (0.3, 1.4), (0.6, 1.4), (0.9, 1.1), (0.9, 0.3), (0.6, 0), (0.3, 0)]],
    "1": [[(0.2, 1.1), (0.5, 1.4), (0.5, 0)]],
    "2": [[(0, 1.1), (0.3, 1.4), (0.6, 1.4), (0.9, 1.1), (0.9, 0.9), (0, 0), (0.9, 0)]],
    "3": [[(0, 1.4), (0.9, 1.4), (0.4, 0.8), (0.7, 0.8), (0.9, 0.6), (0.9, 0.2), (0.7, 0), (0.2, 0), (0, 0.2)]],
    "4": [[(0.7, 0), (0.7, 1.4), (0, 0.4), (0.9, 0.4)]],
    "5": [[(0.9, 1.4), (0, 1.4), (0, 0.8), (0.6, 0.8), (0.9, 0.6), (0.9, 0.2), (0.6, 0), (0.2, 0), (0, 0.2)]],
    "6": [[(0.8, 1.4), (0.3, 1.4), (0, 1.0), (0, 0.3), (0.3, 0), (0.6, 0), (0.9, 0.3), (0.9, 0.5),
           (0.6, 0.8), (0, 0.7)]],
    "7": [[(0, 1.4), (0.9, 1.4), (0.3, 0)]],
    "8": [[(0.45, 0.7), (0.15, 0.9), (0.15, 1.2), (0.45, 1.4), (0.75, 1.2), (0.75, 0.9), (0.45, 0.7),
           (0.1, 0.5), (0.1, 0.2), (0.45, 0), (0.8, 0.2), (0.8, 0.5), (0.45, 0.7)]],
    "9": [[(0.9, 0.7), (0.3, 0.6), (0, 0.9), (0, 1.1), (0.3, 1.4), (0.6, 1.4), (0.9, 1.1), (0.9, 0.4),
           (0.6, 0), (0.1, 0)]],
}
LATIN_ADVANCE = 1.25

CHINESE = {  # unit box, strokes in writing order
    "一": [[(0.1, 0.5), (0.9, 0.5)]],
    "二": [[(0.2, 0.7), (0.8, 0.7)], [(0.1, 0.3), (0.9, 0.3)]],
    "三": [[(0.2, 0.8), (0.8, 0.8)], [(0.25, 0.5), (0.75, 0.5)], [(0.1, 0.2), (0.9, 0.2)]],
    "十": [[(0.1, 0.5), (0.9, 0.5)], [(0.5, 0.9), (0.5, 0.1)]],
    "口": [[(0.2, 0.8), (0.2, 0.2)], [(0.2, 0.8), (0.8, 0.8), (0.8, 0.2)], [(0.2, 0.2), (0.8, 0.2)]],
    "日": [[(0.25, 0.9), (0.25, 0.1)], [(0.25, 0.9), (0.75, 0.9), (0.75, 0.1)], [(0.25, 0.5), (0.75, 0.5)],
          [(0.25, 0.1), (0.75, 0.1)]],
    "人": [[(0.5, 0.9), (0.15, 0.1)], [(0.5, 0.7), (0.85, 0.1)]],
    "大": [[(0.1, 0.6), (0.9, 0.6)], [(0.5, 0.95), (0.15, 0.05)], [(0.5, 0.6), (0.85, 0.05)]],
    "王": [[(0.2, 0.85), (0.8, 0.85)], [(0.25, 0.5), (0.75, 0.5)], [(0.5, 0.85), (0.5, 0.15)],
          [(0.1, 0.15), (0.9, 0.15)]],
    "中": [[(0.25, 0.75), (0.25, 0.3)], [(0.25, 0.75), (0.75, 0.75), (0.75, 0.3)], [(0.25, 0.3), (0.75, 0.3)],
          [(0.5, 0.95), (0.5, 0.05)]],
}


def _poly(pts):
    return [np.asarray(p, dtype=float) for p in pts]


def text_strokes(text, font="block", rounded=0.0):
    """Text -> centre-line strokes, cap height 1.4. `font` is "block" (hand-made
    capitals) or a Hershey single-line face: "sans" (the plotter Arial) or "script";
    "sans_round" = sans with every corner rounded. rounded > 0 rounds corners on any
    face (Chaikin)."""
    if font == "sans_round":
        font, rounded = "sans", max(rounded, 1.0)
    out, x = [], 0.0
    if font == "block":
        for ch in text.upper():
            if ch == " ":
                x += 0.7 * LATIN_ADVANCE
                continue
            g = LATIN.get(ch)
            if g is None:
                continue
            for stroke in g:
                out.append(np.array(stroke, dtype=float) + [x, 0.0])
            x += LATIN_ADVANCE
        return _refine(out, rounded)
    fnt = HERSHEY[font]
    sc, base = _hershey_frame(font)
    for ch in text:
        g = fnt.get(ch) or fnt.get(ch.upper())
        if ch == " " or g is None:
            x += 0.7 * LATIN_ADVANCE
            continue
        for stroke in g["strokes"]:
            P = np.asarray(stroke, float)
            out.append(np.stack([P[:, 0] * sc + x, (P[:, 1] - base) * sc], 1))
        x += g["adv"] * sc
    return _refine(out, rounded)


def chinese_strokes(text, font="medians", rounded=0.0):
    """"medians": Make Me a Hanzi stroke centre lines (240 characters, writing order
    preserved, smooth curves); "block": the built-in 10-character polyline dict."""
    out, x = [], 0.0
    for ch in text:
        g = CHINESE.get(ch) if font == "block" else (HANZI.get(ch) or CHINESE.get(ch))
        if g is None:
            continue
        for stroke in g:
            out.append(np.asarray(stroke, dtype=float) + [x, 0.0])
        x += 1.2
    return _refine(out, rounded, spacing=0.02)


def cursive_daxo():
    w = json.load(open(os.path.join(HERE, "assets", "cursive_daxo.json")))
    return [np.asarray(s, dtype=float) for s in w["strokes"]]


def shape_strokes(kind, n=120):
    t = np.linspace(0, 1, n)
    if kind == "line":
        return [np.stack([t, np.zeros(n)], 1)]
    if kind == "circle":
        a = 2 * np.pi * t
        return [np.stack([np.cos(a), np.sin(a)], 1) * 0.5]
    if kind == "square":
        return [np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], dtype=float)]
    if kind == "triangle":
        return [np.array([(0, 0), (1, 0), (0.5, 0.9), (0, 0)], dtype=float)]
    if kind == "spiral":
        a = 3 * 2 * np.pi * t
        r = 0.5 * t
        return [np.stack([r * np.cos(a), r * np.sin(a)], 1)]
    if kind == "figure8":
        a = 2 * np.pi * t
        return [np.stack([np.sin(a), np.sin(a) * np.cos(a)], 1) * 0.5]
    if kind == "sine":
        return [np.stack([t, 0.25 * np.sin(2 * np.pi * 2 * t)], 1)]
    if kind == "zigzag":
        return [np.array([(i / 6, (i % 2) * 0.5) for i in range(7)], dtype=float)]
    if kind == "hatching":      # parallel short strokes (sketch shading)
        return [np.array([(0.15 * i, 0.0), (0.15 * i + 0.25, 0.5)], dtype=float) for i in range(6)]
    raise KeyError(kind)


SHAPES = ["line", "circle", "square", "triangle", "spiral", "figure8", "sine", "zigzag"]


def normalize(strokes):
    """Scale + translate so the bounding box is centred with height 1 (or width 1
    if wider than tall); returns (strokes, width, height)."""
    P = np.concatenate(strokes)
    lo, hi = P.min(axis=0), P.max(axis=0)
    size = max(hi - lo)
    c = 0.5 * (lo + hi)
    out = [(s - c) / size for s in strokes]
    return out, (hi[0] - lo[0]) / size, (hi[1] - lo[1]) / size


def resample(stroke, spacing):
    """Equal-arc-length resampling of a polyline."""
    d = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(d)])
    if L[-1] < 1e-9:
        return stroke[:1]
    s = np.arange(0.0, L[-1], spacing)
    s = np.append(s, L[-1])
    return np.stack([np.interp(s, L, stroke[:, k]) for k in range(stroke.shape[1])], axis=1)
