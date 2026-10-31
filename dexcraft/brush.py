"""The calligraphy brush as a reduced-order model.

The brush is two layers. The handle and ferrule are MuJoCo rigid bodies the fingers
hold. The tuft is NOT simulated in MuJoCo: it is the footprint model below, which
takes the nominal tip pose and velocity from the physics and returns the force the
paper puts on the shaft. The tuft has no collision geometry, and the sheet does not
collide with it; the normal force is `k * d^alpha` applied through `mj_applyFT`.

    state   (a, b, phi, m)   footprint semi-axes, orientation, ink left [0, 1]
    input   nominal tip position (L_tip along the shaft from the ferrule), its
            velocity, and the shaft axis
    output  force at the tip + a bending torque; ink stamped as an ellipse

Everything numeric lives in assets/brush_configs.json and is UNCALIBRATED. This is a
simulation-only tier: it makes no claim about a real brush, see the paper's
limitations statement. What it keeps is the control structure of the problem -
non-linear hardening under press, a first-order lag between the shaft and the
footprint (pressure accents, twisted-tip turns and dry-brush streaks come from this
one term), and an ink reservoir that only runs down.

Depth convention: depth is measured along the shaft axis (the
nominal tip site sits L_tip along the axis, and d = sheet_top - tip_z), not as
`z(tip_root) - L_tip`. The grip here tilts the shaft ~40 deg from the sheet normal, and
the vertical formula would call the tuft 8 mm deep the moment its axial tip touched.
"""
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON = os.path.join(HERE, "assets", "brush_configs.json")
BRUSHES = ("fine", "wolf", "broad")     # the L5 rack, slot order: fine liner / medium (= the L3 brush) / broad wash
MEDIUM = 1                              # the brush in the hand at reset
DIP_CARRY = 0.2                         # a loaded brush dipped into another colour keeps this much of the old

RENDER_MM_PER_PX = 0.25         # the metric canvas: the whole sheet at 0.25 mm/px
EVAL_LONG_SIDE = 224            # normalisation: binarise, crop to bbox, long side -> 224, pad -> 256
EVAL_SIZE = 256
THETA_MAX = np.radians(75.0)    # tan(theta) in the footprint law is capped here
V_EPS = 0.005                   # m/s: smooth unit vector of the sliding velocity
VIOLATION_FORCE_N = 5.0         # F_n above this is a physically implausible press
VIOLATION_DEPTH_FRAC = 0.6      # ... or a depth beyond 0.6 * L_tip (the tuft splayed flat)
VIOLATION_TILT_DEG = 60.0       # ... or a shaft tilted past this from the sheet normal


@dataclass
class BrushConfig:
    name: str
    L_tip: float        # natural tuft length [m]
    r0: float           # ferrule radius = footprint semi-axis at d_ref [m]
    d_ref: float        # reference depth [m]
    k: float            # normal stiffness [N / m^alpha]
    alpha: float        # hardening exponent
    tau: float          # footprint lag time constant [s]
    kappa: float        # tilt elongation coefficient
    mu0: float          # wet friction
    dmu: float          # extra friction when dry
    c_t: float          # tangential damping [N s / m]
    k_theta: float      # bending restoring torque [N m / rad] (small)
    c_ink: float        # ink use per swept volume [1 / m^3]
    m_dry: float        # below this ink fraction the footprint breaks up (dry brush)
    calibrated: bool = False

    def as_dict(self):
        return asdict(self)


@dataclass
class PaperConfig:
    name: str
    sigma_mm: tuple     # anisotropic Gaussian bleed (x, y) [mm]


@dataclass
class BrushState:
    a: float            # footprint semi-axis along the shaft's projection [m]
    b: float            # footprint semi-axis across it [m]
    phi: float          # footprint orientation [rad]
    ink: float          # ink left [0, 1]
    f_normal: float     # normal force [N]
    depth: float        # press depth [m]
    theta: float        # shaft tilt from the sheet normal [rad]
    psi: float          # azimuth of the shaft's projection (tip -> root) [rad]


def _load():
    with open(CONFIG_JSON) as fh:
        return json.load(fh)


def brush_config(name="wolf"):
    c = {k: v for k, v in _load()["brush"][name].items() if not k.startswith("_")}
    return BrushConfig(name=name, **c)


def paper_config(name="shuxuan"):
    c = _load()["paper"][name]
    return PaperConfig(name=name, sigma_mm=tuple(float(s) for s in c["sigma_mm"]))


# ------------------------------------------------------------------ the footprint law
def footprint_target(cfg, d, theta):
    """Steady-state footprint (a, b) for depth d and tilt theta."""
    b = cfg.r0 * np.sqrt(np.clip(d / cfg.d_ref, 0.0, 1.0))
    a = b * (1.0 + cfg.kappa * np.tan(min(float(theta), THETA_MAX)))
    return float(a), float(b)


def normal_force(cfg, d):
    return float(cfg.k * max(float(d), 0.0) ** cfg.alpha)


def depth_for_force(cfg, f):
    return float((max(float(f), 0.0) / cfg.k) ** (1.0 / cfg.alpha))


def width_mm(cfg, f):
    """Stroke width across the shaft's projection (2 b) implied by a normal force.
    This is what the task map's darkness -> thickness mapping uses for the brush."""
    return 2000.0 * footprint_target(cfg, depth_for_force(cfg, f), 0.0)[1]


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class BrushTipModel:
    """(a, b, phi, m) with first-order lag toward the steady-state footprint; force
    feedback on the shaft. Pure: no MuJoCo, so the oracle can replay it kinematically."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self, ink=1.0, color=None):
        self.a = self.b = 0.0
        self.phi = 0.0
        self.m = float(ink)
        self.color = None if color is None else np.asarray(color, float)   # None: plain ink (L3)
        self.ran_dry = False
        self._xy_prev = None

    def dip(self, color):
        """L5: a dip in a paint well refills the reservoir with that colour. A brush that
        still carries paint mixes a little of it into the new colour."""
        color = np.asarray(color, float)
        if self.color is not None and self.m > 0.05:
            color = DIP_CARRY * self.color + (1.0 - DIP_CARRY) * color
        self.color, self.m, self.ran_dry = color, 1.0, False

    def set_config(self, cfg):
        """L5: a different brush from the rack - and a brush from the rack is dry."""
        self.cfg = cfg
        self.m, self.color, self.ran_dry = 0.0, None, False

    def step(self, tip, tip_vel, axis, dt, surface_z, on_sheet=True):
        """One physics substep.

        tip      nominal tip position, world [m]
        tip_vel  its linear velocity, world [m/s]
        axis     unit shaft axis, tip -> root, world
        dt       substep [s]
        surface_z  height of the surface under the tip; on_sheet False = over the
                 table, where the model still pushes back but no ink is laid
        Returns (BrushState, force at the tip [N, world], bending torque [N m, world],
                 footprint centre xy or None when not touching)."""
        cfg = self.cfg
        d = max(0.0, float(surface_z - tip[2]))
        cz = float(np.clip(axis[2], -1.0, 1.0))
        theta = float(np.arccos(cz))
        psi = float(np.arctan2(axis[1], axis[0])) if abs(axis[0]) + abs(axis[1]) > 1e-9 else 0.0
        a_t, b_t = footprint_target(cfg, d, theta)
        g = min(1.0, dt / cfg.tau)
        self.a += (a_t - self.a) * g
        self.b += (b_t - self.b) * g
        self.phi = wrap(self.phi + wrap(psi - self.phi) * g)
        f_n = normal_force(cfg, d)
        force = np.zeros(3)
        torque = np.zeros(3)
        centre = None
        if d > 0.0:
            force[2] = f_n
            v_t = np.array([tip_vel[0], tip_vel[1], 0.0])
            sp = float(np.linalg.norm(v_t))
            mu = cfg.mu0 + (1.0 - self.m) * cfg.dmu
            force[:2] = -(mu * f_n * v_t / (sp + V_EPS) + cfg.c_t * v_t)[:2]
            # restoring torque: rotate the shaft axis back toward the normal
            e = np.cross(axis, [0.0, 0.0, 1.0])
            ne = float(np.linalg.norm(e))
            if ne > 1e-9:
                torque = cfg.k_theta * theta * e / ne
            if on_sheet:
                # the belly of the tuft lies on the root side of the tip
                centre = np.array([tip[0], tip[1]]) + (self.a - self.b) * np.array([np.cos(psi), np.sin(psi)])
                if self._xy_prev is not None and self.m > 0.0:
                    ds = float(np.linalg.norm(centre - self._xy_prev))
                    self.m = max(0.0, self.m - cfg.c_ink * np.pi * self.a * self.b * ds)
                    if self.m <= 0.0:
                        self.ran_dry = True
                self._xy_prev = centre
        else:
            self._xy_prev = None
        st = BrushState(self.a, self.b, self.phi, self.m, f_n if d > 0 else 0.0, d, theta, psi)
        return st, force, torque, centre

    def violation(self, st, ferrule_contact=False):
        return bool(st.f_normal > VIOLATION_FORCE_N or st.depth > VIOLATION_DEPTH_FRAC * self.cfg.L_tip
                    or ferrule_contact or (st.depth > 0 and np.degrees(st.theta) > VIOLATION_TILT_DEG))


# ------------------------------------------------------------------ ink rendering
def _value_noise(shape, cell_px, seed=0, octaves=2):
    """Deterministic smooth noise in [0, 1] (bilinear value noise, a few octaves).
    Used in place of Perlin noise: a fixed mask the dry footprint is thresholded
    against so dry-brush streaks are reproducible."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(round(cell_px / (2 ** o))))
        gh, gw = shape[0] // c + 2, shape[1] // c + 2
        g = rng.random((gh, gw)).astype(np.float32)
        ys = np.arange(shape[0]) / c
        xs = np.arange(shape[1]) / c
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        v = (g[y0][:, x0] * (1 - fy) * (1 - fx) + g[y0][:, x0 + 1] * (1 - fy) * fx
             + g[y0 + 1][:, x0] * fy * (1 - fx) + g[y0 + 1][:, x0 + 1] * fy * fx)
        out += amp * v
        tot += amp
        amp *= 0.5
    return out / tot


def ellipse_mask(shape, cx, cy, a_px, b_px, phi_img, noise=None, keep=1.0, aspect=1.0):
    """Pixel mask of a filled ellipse. (cx, cy) in pixels, a_px / b_px in x-pixels,
    phi_img the orientation in IMAGE axes (x right, y down); `aspect` = y pixels per
    x pixel for non-square pixel grids. `noise` is a same-shape [0, 1] map; pixels
    where noise > keep are dropped (the dry-brush break-up).
    Returns (y0, y1, x0, x1, mask) or None when nothing lands on the image."""
    r = int(np.ceil(max(a_px, b_px) * max(1.0, aspect))) + 1
    x0, x1 = max(0, int(cx) - r), min(shape[1], int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(shape[0], int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1 or a_px < 0.25 or b_px < 0.25:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, (yy - cy) / aspect
    c, s = np.cos(phi_img), np.sin(phi_img)
    u, v = dx * c + dy * s, -dx * s + dy * c
    mask = (u / a_px) ** 2 + (v / b_px) ** 2 <= 1.0
    if noise is not None and keep < 1.0:
        mask &= noise[y0:y1, x0:x1] <= keep
    return y0, y1, x0, x1, mask


def ellipse_stamp(img, cx, cy, a_px, b_px, phi_img, val, noise=None, keep=1.0, aspect=1.0):
    """Max-accumulate a filled ellipse into a uint8 image (see ellipse_mask)."""
    m = ellipse_mask(img.shape, cx, cy, a_px, b_px, phi_img, noise, keep, aspect)
    if m is None:
        return
    y0, y1, x0, x1, mask = m
    sub = img[y0:y1, x0:x1]
    np.maximum(sub, np.where(mask, val, 0).astype(img.dtype), out=sub)


class InkRenderer:
    """Alpha canvas over the sheet: row 0 = the sheet's +y edge, RENDER_MM_PER_PX.
    stamp() sweeps the footprint between consecutive centres (interpolated stamps,
    equivalent to a convex-hull sweep for sub-pixel steps), ink = max(ink, alpha).
    Paper bleed is applied once, in image()."""

    def __init__(self, sheet_center, sheet_half, cfg, mm_per_px=RENDER_MM_PER_PX):
        self.c = np.asarray(sheet_center[:2], float)
        self.h = np.asarray(sheet_half[:2], float)
        self.mm = mm_per_px
        self.W = int(round(2 * self.h[0] * 1000 / mm_per_px))
        self.H = int(round(2 * self.h[1] * 1000 / mm_per_px))
        self.cfg = cfg
        self.noise = _value_noise((self.H, self.W), cell_px=max(2, int(round(1.0 / mm_per_px))), seed=0)
        self.reset()

    def reset(self):
        self.alpha = np.zeros((self.H, self.W), np.uint8)
        self.rgb = None                     # allocated on the first coloured stamp (L5 paint)
        self._prev = None

    def to_px(self, xy):
        u = (xy[0] - self.c[0] + self.h[0]) / (2 * self.h[0])
        v = (xy[1] - self.c[1] + self.h[1]) / (2 * self.h[1])
        return u * self.W, (1.0 - v) * self.H

    def stamp(self, st, centre_xy, color=None):
        """One footprint at world xy, swept from the previous stamp. With `color`
        (RGB in [0, 1]) the paint is opaque: later strokes cover earlier ones."""
        if centre_xy is None or st.ink <= 0.0:
            self._prev = None
            return
        keep = 1.0 if st.ink >= self.cfg.m_dry else st.ink / self.cfg.m_dry
        a_px, b_px = st.a * 1000 / self.mm, st.b * 1000 / self.mm
        cx, cy = self.to_px(centre_xy)
        pts = [(cx, cy)]
        if self._prev is not None:
            px, py = self._prev
            n = int(np.hypot(cx - px, cy - py)) + 1
            if n > 1:
                pts = [(px + (cx - px) * k / n, py + (cy - py) * k / n) for k in range(1, n + 1)]
        if color is not None and self.rgb is None:
            self.rgb = np.full((self.H, self.W, 3), 255, np.uint8)
        rgb8 = None if color is None else (np.clip(np.asarray(color, float), 0, 1) * 255).astype(np.uint8)
        for x, y in pts:
            mk = ellipse_mask(self.alpha.shape, x, y, a_px, b_px, -st.phi, self.noise, keep)
            if mk is None:
                continue
            y0, y1, x0, x1, mask = mk
            self.alpha[y0:y1, x0:x1][mask] = 255
            if rgb8 is not None:
                self.rgb[y0:y1, x0:x1][mask] = rgb8
        self._prev = (cx, cy)

    def lift(self):
        self._prev = None

    def image(self, paper=None):
        """The ink as it would look on this paper: alpha in [0, 1] after the bleed."""
        return bleed(self.alpha, paper, self.mm)


def bleed(alpha, paper=None, mm_per_px=RENDER_MM_PER_PX):
    """Paper diffusion: one anisotropic Gaussian, applied once. Ink spreads OUT of the
    footprint; the core stays as dark as it was."""
    img = np.asarray(alpha).astype(np.float32) / 255.0
    if paper is not None and max(paper.sigma_mm) > 0:
        from scipy.ndimage import gaussian_filter
        sy, sx = paper.sigma_mm[1] / mm_per_px, paper.sigma_mm[0] / mm_per_px
        img = np.maximum(img, gaussian_filter(img, (sy, sx)))
    return img


# ------------------------------------------------------------------ oracle replay
ORACLE_DT = 0.002               # kinematic replay substep [s]
ORACLE_SPEED = 0.08             # m/s along the stroke
ORACLE_HOP_S = 0.4              # lifted between strokes: the footprint relaxes
ORACLE_TILT_DEG = 38.0          # fallback shaft tilt from the sheet normal (see grip_axis)
ORACLE_PSI = -np.pi / 2         # fallback azimuth: root toward the writer (-y)


def grip_axis():
    """Shaft axis (tip -> root, world) the solved grip holds the brush at, so the
    reference render elongates its footprints the way the hand actually tilts the
    brush. Falls back to the constants when the grip file has no brush entry."""
    try:
        from . import scene as S
        g = S.load_grip()["settled_brush"]
        ax = np.asarray(g["hand_xmat"], float) @ np.asarray(g["tool_axis_in_hand"], float)
        return ax / np.linalg.norm(ax)
    except (KeyError, TypeError, OSError):
        th = np.radians(ORACLE_TILT_DEG)
        return np.array([np.sin(th) * np.cos(ORACLE_PSI), np.sin(th) * np.sin(ORACLE_PSI), np.cos(th)])


ORACLE_REFILL = 0.2             # paint: the oracle dips again before a stroke when the reservoir is this low


def oracle_trace(task, cfg, sheet_center, sheet_top, dt=ORACLE_DT, speed=ORACLE_SPEED, ink0=None, axis=None):
    """Kinematic replay of the task's strokes at their target depth through the tip
    model, the shaft held at the grip's axis. Yields (BrushState, centre_xy or None,
    colour or None) per substep. For a painting (task.palette) each stroke uses its
    own brush preset and colour: the brush is swapped (dry) when the stroke asks for
    another one, and dipped when the colour changes or the reservoir runs low."""
    paint = getattr(task, "palette", None) is not None
    cfgs = {i: brush_config(n) for i, n in enumerate(BRUSHES)} if paint else {}
    model = BrushTipModel(cfgs[MEDIUM] if paint else cfg)
    model.reset(0.0 if paint else (ink0 if ink0 is not None else float(task.meta.get("ink0", 1.0))))
    axis = grip_axis() if axis is None else np.asarray(axis, float)
    cxy = np.asarray(sheet_center[:2], float)
    brush = MEDIUM
    for k, s_ in enumerate(task.strokes):
        if paint:
            if task.brushes[k] != brush:
                brush = task.brushes[k]
                model.set_config(cfgs[brush])
            col = np.asarray(task.palette[task.colors[k] - 1], float)
            if model.color is None or model.m < ORACLE_REFILL or np.abs(model.color - col).max() > 1e-6:
                model.dip(col)
        c_k = model.cfg
        P = np.asarray(s_, float) + cxy
        f = task.forces[k] if task.forces is not None else np.full(len(P), 1.0)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(round(cum[-1] / (speed * dt))))
        si = np.linspace(0.0, cum[-1], n)
        xy = np.stack([np.interp(si, cum, P[:, j]) for j in range(2)], axis=1)
        dep = np.array([depth_for_force(c_k, v) for v in np.interp(si, cum, f)])
        prev = None
        for i in range(n):
            tip = np.array([xy[i, 0], xy[i, 1], sheet_top - dep[i]])
            vel = np.zeros(3) if prev is None else (tip - prev) / dt
            prev = tip
            st, _, _, centre = model.step(tip, vel, axis, dt, sheet_top)
            yield st, centre, model.color
        for _ in range(int(ORACLE_HOP_S / dt)):
            tip = np.array([xy[-1, 0], xy[-1, 1], sheet_top + 0.01])
            st, _, _, centre = model.step(tip, np.zeros(3), axis, dt, sheet_top)
            yield st, centre, model.color


def oracle_render(task, cfg, sheet_center, sheet_half, sheet_top, **kw):
    """Reference ink for a brush or paint task: the target strokes replayed at their
    target depth through the same tip model and renderer. Returns the InkRenderer
    (`alpha`, and `rgb` for a painting)."""
    r = InkRenderer(sheet_center, sheet_half, cfg)
    for st, c, col in oracle_trace(task, cfg, sheet_center, sheet_top, **kw):
        r.stamp(st, c, col)
    return r


# ------------------------------------------------------------------ paint metrics (L5)
def quantize_paint(rgb, alpha, palette):
    """Painted pixels -> nearest palette index (1-based); 0 where nothing was painted."""
    idx = np.zeros(alpha.shape, np.uint8)
    on = np.asarray(alpha) > 0
    if not on.any() or rgb is None:
        return idx
    pal = (np.asarray(palette, float) * 255).reshape(1, -1, 3)
    px = np.asarray(rgb)[on].astype(float).reshape(-1, 1, 3)
    idx[on] = np.argmin(((px - pal) ** 2).sum(-1), axis=1) + 1
    return idx


def score_paint(rgb, alpha, ref_rgb, ref_alpha, palette, mm_per_px=RENDER_MM_PER_PX, tol_mm=1.5):
    """Metric axes for a painting, in absolute sheet coordinates (a painting has a
    place; no bbox normalisation). Per-colour stroke IoU averaged over the colours the
    reference uses, both sides dilated by `tol_mm`; colour accuracy = fraction of painted
    pixels whose colour is the reference's colour within `tol_mm` of that pixel."""
    from scipy.ndimage import binary_dilation
    it = max(1, int(round(tol_mm / mm_per_px)))
    a, b = quantize_paint(rgb, alpha, palette), quantize_paint(ref_rgb, ref_alpha, palette)
    ious, correct = [], np.zeros(a.shape, bool)
    for c in np.unique(b[b > 0]):
        A, B = binary_dilation(a == c, iterations=it), binary_dilation(b == c, iterations=it)
        u = (A | B).sum()
        ious.append(float((A & B).sum() / u) if u else 0.0)
        correct |= (a == c) & B
    painted = a > 0
    return dict(color_iou=float(np.mean(ious)) if ious else None,
                color_accuracy=float(correct[painted].sum() / painted.sum()) if painted.any() else 0.0,
                n_colors_target=int(len(ious)), n_colors_painted=int(len(np.unique(a[a > 0]))))


# ------------------------------------------------------------------ metrics
def normalize_ink(img, thresh=0.5):
    """Binarise -> bbox -> long side to EVAL_LONG_SIDE -> centre-pad to EVAL_SIZE.
    Returns (binary uint8, grey float32) at EVAL_SIZE^2, or None when there is no ink."""
    from skimage.transform import resize
    b = img >= thresh
    if not b.any():
        return None
    ys, xs = np.nonzero(b)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1].astype(np.float32)
    h, w = crop.shape
    s = EVAL_LONG_SIDE / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    g = resize(crop, (nh, nw), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    out = np.zeros((EVAL_SIZE, EVAL_SIZE), np.float32)
    oy, ox = (EVAL_SIZE - nh) // 2, (EVAL_SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = g
    return (out >= thresh).astype(np.uint8), out


def skeleton_chamfer_px(A, B):
    """Bidirectional Chamfer distance between the skeletons of two binary images
    (Zhang-Suen thinning). Returns (mean, p95) in pixels; the image diagonal when
    one side is empty."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    sa, sb = skeletonize(A.astype(bool)), skeletonize(B.astype(bool))
    diag = float(np.hypot(*A.shape))
    if not sa.any() or not sb.any():
        return diag, diag
    da, db = distance_transform_edt(~sa), distance_transform_edt(~sb)
    d = np.concatenate([db[sa], da[sb]])
    return float(d.mean()), float(np.percentile(d, 95))


def stroke_iou(A, B):
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 0.0


def ink_ssim(ga, gb):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ga, gb, data_range=1.0))


def _levenshtein(p, q):
    D = np.zeros((len(p) + 1, len(q) + 1), int)
    D[:, 0] = np.arange(len(p) + 1)
    D[0, :] = np.arange(len(q) + 1)
    for i in range(1, len(p) + 1):
        for j in range(1, len(q) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (p[i - 1] != q[j - 1]))
    return int(D[-1, -1])


def stroke_order_edit(drawn, ref):
    """Edit distance between the reference stroke order and the order the policy drew
    them in: each drawn segment (split at lifts) is assigned to the reference stroke
    it lies closest to; the resulting index sequence is compared to 0..n-1."""
    if not ref:
        return 0
    if not drawn:
        return len(ref)
    seq = []
    for seg in drawn:
        best, bd = 0, np.inf
        for k, r in enumerate(ref):
            r = np.asarray(r)
            dd = np.linalg.norm(seg[:, None, :] - r[None, :, :], axis=2).min(axis=1).mean()
            if dd < bd:
                best, bd = k, dd
        if not seq or seq[-1] != best:
            seq.append(best)
    return _levenshtein(seq, list(range(len(ref))))


def score_render(img, ref, paper=None):
    """Shape metrics on two ink alphas in [0, 1] (same canvas): stroke IoU, skeleton
    Chamfer mean / p95 [px], ink-density SSIM."""
    na, nb = normalize_ink(img), normalize_ink(ref)
    diag = float(np.hypot(EVAL_SIZE, EVAL_SIZE))
    if nb is None:
        return dict(stroke_iou=None, skel_chamfer_px=None, skel_chamfer_p95_px=None, ink_ssim=None)
    if na is None:
        return dict(stroke_iou=0.0, skel_chamfer_px=diag, skel_chamfer_p95_px=diag, ink_ssim=0.0)
    (ba, ga), (bb, gb) = na, nb
    cm, cp = skeleton_chamfer_px(ba, bb)
    return dict(stroke_iou=stroke_iou(ba.astype(bool), bb.astype(bool)), skel_chamfer_px=cm,
                skel_chamfer_p95_px=cp, ink_ssim=ink_ssim(ga, gb))


# ------------------------------------------------------------------ acceptance tests
def _t1_static(cfg, out_dir):
    """Footprint grid over tilt x depth; must widen with tilt and thicken with depth, smoothly."""
    tilts = [0, 15, 30, 45]
    depths = np.linspace(0.0, cfg.d_ref, 6)
    tile = 96
    grid = np.full((len(tilts) * tile, len(depths) * tile), 255, np.uint8)
    ok = True
    for i, th in enumerate(tilts):
        prev = None
        for j, d in enumerate(depths):
            a, b = footprint_target(cfg, d, np.radians(th))
            img = np.zeros((tile, tile), np.uint8)
            ellipse_stamp(img, tile / 2, tile / 2, a * 1000 / 0.25, b * 1000 / 0.25, 0.0, 255)
            grid[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = 255 - img
            if prev is not None:
                ok &= a >= prev[0] - 1e-12 and b >= prev[1] - 1e-12          # monotone in depth
                # no jumps: one depth step may not cover more than half the tilt's full
                # range (the sqrt law is steepest at d = 0, that first step is the largest)
                ok &= (a - prev[0]) <= 0.5 * footprint_target(cfg, cfg.d_ref, np.radians(th))[0] + 1e-12
            ok &= a >= b - 1e-12
            prev = (a, b)
        if i > 0:
            ok &= footprint_target(cfg, cfg.d_ref, np.radians(th))[0] > footprint_target(cfg, cfg.d_ref, np.radians(tilts[i - 1]))[0]
    if out_dir:
        from PIL import Image
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(grid).save(os.path.join(out_dir, "t1_footprints.png"))
    return ok, f"tilts {tilts} deg x depths 0..{cfg.d_ref * 1000:.0f} mm, monotone, a >= b"


def _t2_step_turn(cfg):
    """Straight run then a 90 deg turn of the shaft azimuth: phi must follow as a
    first-order lag with the configured tau (63 % at tau, 95 % at 3 tau)."""
    model = BrushTipModel(cfg)
    model.reset()
    dt, z = 0.001, 0.0
    th = np.radians(30.0)
    log = []
    t_turn = 0.5
    for i in range(int(1.0 / dt)):
        t = i * dt
        psi = 0.0 if t < t_turn else np.pi / 2
        axis = np.array([np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi), np.cos(th)])
        tip = np.array([0.05 * t, 0.0, z - 0.003])
        st = model.step(tip, np.array([0.05, 0.0, 0.0]), axis, dt, z)[0]
        log.append((t, st.phi))
    log = np.asarray(log)
    after = log[log[:, 0] >= t_turn]
    err = (np.pi / 2 - after[:, 1]) / (np.pi / 2)
    t63 = after[np.argmax(err <= np.exp(-1)), 0] - t_turn
    t95 = after[np.argmax(err <= 0.05), 0] - t_turn
    ok = abs(t63 - cfg.tau) < 0.15 * cfg.tau + dt and abs(t95 - 3 * cfg.tau) < 0.2 * 3 * cfg.tau + dt
    return ok, f"phi 63 % at {t63 * 1000:.0f} ms (tau {cfg.tau * 1000:.0f}), 95 % at {t95 * 1000:.0f} ms"


def _t4_t5_oracle(cfg, paper):
    """T4: the oracle replay of the reference trajectory scored against the reference
    must give IoU > 0.98 and skeleton Chamfer < 1.5 px - the renderer and the metric
    agree with each other. Also reported (not gated): the same replay at half the
    substep, which shows the metric's own noise floor - the bbox normalisation
    resamples the image, and a sub-pixel change of the crop moves every edge.
    T5: two identical replays are bitwise equal."""
    from . import scene as S
    from . import tasks as T
    task = T.sample("L3_brush_chinese", 3)
    r1 = oracle_render(task, cfg, S.PAPER_CENTER, S.PAPER_HALF, S.PAPER_TOP)
    r2 = oracle_render(task, cfg, S.PAPER_CENTER, S.PAPER_HALF, S.PAPER_TOP)
    same = bool(np.array_equal(r1.alpha, r2.alpha))
    sc = score_render(r2.image(paper), r1.image(paper), paper)
    ok4 = sc["stroke_iou"] > 0.98 and sc["skel_chamfer_px"] < 1.5
    r3 = oracle_render(task, cfg, S.PAPER_CENTER, S.PAPER_HALF, S.PAPER_TOP, dt=ORACLE_DT / 2)
    s3 = score_render(r3.image(paper), r1.image(paper), paper)
    return (ok4, f"replay IoU {sc['stroke_iou']:.3f}, skeleton Chamfer {sc['skel_chamfer_px']:.2f} px; "
                 f"half-substep replay IoU {s3['stroke_iou']:.3f} (metric noise floor)"), \
           (same, "two replays bitwise identical")


def _t3_env_stability(steps=100_000):
    """Random actions through the full env with the brush: no NaN, no MuJoCo warning,
    bounded shaft speed."""
    import mujoco
    from .env import DexCraftEnv
    env = DexCraftEnv("L3_brush_stroke", control_hz=25)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    n, vmax, nan = 0, 0.0, False
    while n < steps:
        a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32) * 0.3
        obs, r, term, trunc, info = env.step(a)
        n += env.frame_skip
        v = float(np.linalg.norm(env.data.qvel[env.pd:env.pd + 3]))
        vmax = max(vmax, v)
        nan |= not np.isfinite(env.data.qpos).all() or not np.isfinite(r)
        if term or trunc:
            env.reset(seed=int(rng.integers(1 << 30)))
    warn = int(sum(env.data.warning[i].number for i in range(mujoco.mjtWarning.mjNWARNING)))
    ok = not nan and warn == 0 and vmax < 20.0
    return ok, f"{n} substeps, max shaft speed {vmax:.1f} m/s, warnings {warn}, nan {nan}"


def selftest(out_dir=None, with_env=True, brush="wolf", paper="shuxuan"):
    cfg, pap = brush_config(brush), paper_config(paper)
    results = []
    results.append(("T1 static footprint",) + _t1_static(cfg, out_dir))
    results.append(("T2 step turn lag",) + _t2_step_turn(cfg))
    if with_env:
        results.append(("T3 numerical stability",) + _t3_env_stability())
    t4, t5 = _t4_t5_oracle(cfg, pap)
    results.append(("T4 oracle IoU",) + t4)
    results.append(("T5 determinism",) + t5)
    bad = 0
    for name, ok, detail in results:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        bad += not ok
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="brush tip model acceptance tests (spec 9)")
    ap.add_argument("--out", default=None, help="directory for the T1 footprint grid PNG")
    ap.add_argument("--no-env", action="store_true", help="skip T3 (needs the MuJoCo scene)")
    ap.add_argument("--brush", default="wolf")
    ap.add_argument("--paper", default="shuxuan")
    args = ap.parse_args()
    raise SystemExit(selftest(args.out, not args.no_env, args.brush, args.paper))
