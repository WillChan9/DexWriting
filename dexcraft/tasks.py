"""Task suite: each task id maps to a sampler that, given a seed, returns a
TaskSpec (strokes on the paper, target ink widths, instruction, time limit).

Levels
  L0 hold     keep the pen while the arm is driven by the environment; pick the pen
              up off the table (the one task that starts with an empty hand)
  L1 trace    follow geometric shapes
  L2 write    block capitals / digits / Chinese / cursive, random size & slant
  L3 brush    strokes with a target line-width profile (pressure control)
  L4 sketch   hatching: many short parallel strokes
  L5 paint    coloured paintings: dip in the wells for colour, change brushes at the rack
  L6 chopsticks  a ladder: keep hold of the sticks -> soft cylinder -> cube -> heavy
              block -> a ball, which is close to impossible (no ink;
              the object / bowl / grasp state ride in the same `task` slots, see env.py)
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import fonts as F
from .scene import FORCE_MAX, PAPER_HALF, BOWL_R, TOOL_REST_X, TOOL_REST_Y, PAINT_PALETTE

# Content pools, listed as train/test; they are merged below into exactly one pool per kind
# of content.
SPLITS = {
    "letters": {"train": list("ABCDEFGHIJKLMNOPQRST"), "test": list("UVWXYZ")},
    "digits": {"train": list("01234567"), "test": list("89")},
    "words": {"train": ["HELLO", "ROBOT", "DEXTER", "WRITE", "DAXO", "CORL", "HAND", "PEN", "MUJOCO", "BENCH"],
              "test": ["VORTEX", "QUIZ", "JUMPY", "KEYS", "WAX"]},
    "chinese": {"train": ["一", "二", "三", "十", "口", "日", "人"], "test": ["大", "王", "中"]},
    "shapes": {"train": ["line", "circle", "square", "triangle", "spiral", "sine"], "test": ["figure8", "zigzag"]},
    "objects": {"train": ["sphere", "cube"], "test": ["cylinder", "capsule"]},
}
# Font families for the Latin/digit tasks - single-line faces only; `script` is Hershey
# Script Simplex (see fonts.py). The Chinese pool is every Make Me a Hanzi median shipped.
SPLITS["fonts"] = {"train": ["block", "sans", "sans_round"],
                   "test": ["script"]}
_CH_TEST = [c for c in "大王中学生天空海河花" if c in F.HANZI]
SPLITS["chinese"] = {"train": sorted(set(F.HANZI) - set(_CH_TEST)), "test": _CH_TEST}
# Easy-writing pool: single BIG glyphs; the Chinese pool keeps only characters with
# <= 6 stroke medians (118 chars).
_EASY = sorted(ch for ch in F.HANZI if len(F.HANZI[ch]) <= 6)
_EASY_TEST = [c for c in "大王中天口日三" if c in _EASY]
SPLITS["chinese_easy"] = {"train": [c for c in _EASY if c not in _EASY_TEST], "test": _EASY_TEST}
WORDS = SPLITS["words"]["train"]


def _brush_pools():
    """L3 brush pools. Characters by stroke count: 5 with <= 4
    strokes, 10 with 5-8, 5 with >= 9 as the SEEN set and a structurally similar UNSEEN
    set drawn the same way (2 / 4 / 2); 3-4 character columns from the <= 6-stroke pool
    for the layouts (5 seen, 2 unseen). Deterministic, disjoint from the unseen set."""
    rng = np.random.default_rng(20260905)
    by = {"A": [], "B": [], "C": []}
    for ch in sorted(F.HANZI):
        n = len(F.HANZI[ch])
        by["A" if n <= 4 else "B" if n <= 8 else "C"].append(ch)
    seen, unseen = [], []
    for tier, n_seen, n_unseen in (("A", 5, 2), ("B", 10, 4), ("C", 5, 2)):
        pool = [str(c) for c in rng.permutation(by[tier])]
        seen += pool[:n_seen]
        unseen += pool[n_seen:n_seen + n_unseen]
    short = iter(str(c) for c in rng.permutation([ch for ch in sorted(F.HANZI) if len(F.HANZI[ch]) <= 6 and ch not in unseen]))
    groups = ["".join(next(short) for _ in range(4 if i % 2 == 0 else 3)) for i in range(7)]
    return {"train": seen, "test": unseen}, {"train": groups[:5], "test": groups[5:]}


SPLITS["brush_chinese"], SPLITS["brush_layout"] = _brush_pools()
# ONE pool per kind of content: train and test are merged, so there is no content split.
# The per-pool lists above stay so a held-out split can be reinstated by changing this
# block alone.
SPLITS = {_k: list(_v["train"]) + [x for x in _v["test"] if x not in _v["train"]]
          for _k, _v in SPLITS.items()}

# Generalisation sets. EVALUATION ONLY: they live in EVAL_TASKS, never in TASKS, so no `--task all` / `L2` sweep can train on them.
#   words_unseen    English words never collected or trained on, 3-6 letters like the training pool,
#                   written in the same four faces -> compositional generalisation (seen letters)
#   idioms          four-character Chinese idioms whose characters are all in the trained pool, but
#                   training only ever wrote ONE character -> composition AND sequence-length shift
#   chinese_unseen  characters outside the trained pool. Every one of them has 7+ strokes (the pool
#                   is exactly the <= 6-stroke characters), so novelty is confounded with complexity.
SPLITS["words_unseen"] = ["CANDLE", "MARBLE", "PUZZLE", "FROST", "GLIDE", "ZEBRA", "SPARK", "TUNNEL",
                          "VOYAGE", "QUARTZ", "JUNGLE", "WIDGET", "FABRIC", "NOMAD", "OXIDE", "PRISM",
                          "EMBER", "SWIFT", "TOPAZ", "CLIFF"]
SPLITS["idioms"] = ["十分小心", "七上八下", "五光十色", "九牛一毛", "左右开弓",
                    "风风火火", "土生土长", "一五一十", "一目十行", "千山万水"]
SPLITS["chinese_unseen"] = sorted(set(SPLITS["chinese"]) - set(SPLITS["chinese_easy"]))


@dataclass
class TaskSpec:
    task_id: str
    level: str
    instruction: str
    strokes: List[np.ndarray]                    # paper coords [m], relative to sheet centre
    forces: Optional[List[np.ndarray]] = None    # target press force [N] per stroke point; this is
    #                                              what the task map encodes as darkness
    widths: Optional[List[np.ndarray]] = None    # target ink width [mm], derived from `forces`
    duration: float = 10.0                       # time limit [s]
    capture_radius: float = 0.0025               # waypoint capture radius [m]
    coverage_for_success: float = 0.8            # fraction of target within 3 mm of ink
    hold_motion: Optional[str] = None            # L0: "static" | "moving" | "scribble" - the ENV
    #                                              drives arm + wrist; the policy keeps the tool
    arm_frozen: bool = False                     # the arm cannot move AT ALL this episode -
    #                                              part of what the task asks, not a track option
    wrist_frozen: bool = False                   # ... and neither can the wrist
    place_jitter: float = 0.01                   # reset scatter of the tip over the target [m]
    min_lifts: int = 0                           # deliberate re-landings required for success
    place_at: Optional[np.ndarray] = None        # where to put the tip when there are no strokes
    hold_scale: float = 1.0                      # L0 hold: amplitude multiplier on the motion
    #                                              the env drives (both teleop and waypoint)
    hold_waypoints: int = 5                      # ... and how many direction changes it packs
    #                                              into the episode - speed and acceleration
    hold_box: Optional[np.ndarray] = None        # L0 hold_moving: half-extents [m] of the box the
    #                                              ARM carries the wrist through (x, y, z about the start)
    hold_jitter_deg: float = 0.0                 # L0 hold_moving: 4-9 Hz shake on the wrist, amplitude
    scribble_depth_m: float = 0.0                # L0 scribble: how far below the sheet top the env
    #                                              drives the tip - the press comes from this
    scribble_wrist_deg: float = 0.0              # L0 scribble: wrist wobble the env adds on top
    path_ref_mm: float = 0.0                     # >0: score how FAR the tip travels along the
    #                                              sheet, not what it draws
    impact_limit_N: float = 0.0                  # >0: a peak contact force above this fails the
    #                                              episode - slamming the tool down is not a way
    #                                              to move fast
    coverage_ref_mm2: float = 0.0                # >0: score inked AREA against this, not a target
    extent_ref_mm: float = 0.0                   # >0: score how FAR the ink spreads instead -
    #                                              area has no dynamic range at finger scale
    start: str = "held"                          # "held": the solved grip is restored at reset
    #                                              "table": the tool lies on the table, pick it up first
    tool: str = "pen"                            # "pen" | "brush" (L3, L5) | "chopsticks" (L6)
    palette: Optional[list] = None               # L5 paint: the well colours, RGB in [0, 1]; None = not a painting
    colors: Optional[list] = None                # ... per-stroke palette index (1-based)
    brushes: Optional[list] = None               # ... per-stroke brush, index into brush.BRUSHES
    meta: dict = field(default_factory=dict)


def _place(strokes, rng, height_m, margin=0.03, slant=0.0, rot=None):
    """Normalise, scale to the requested height, optional italic shear and a small
    random rotation, then a random position inside the sheet.

    Position, scale, slant AND rotation are redrawn every episode: the discrete
    content can be split into pools, but these continuous nuisances are what stop
    a policy from memorising a fixed set of trajectories."""
    S, w, h = F.normalize(strokes)
    scale = height_m / max(h, 1e-6)
    if w * scale > 2 * (PAPER_HALF[0] - margin):        # too wide: fit width
        scale = 2 * (PAPER_HALF[0] - margin) / w
    a = rng.uniform(-0.18, 0.18) if rot is None else rot
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    shear = np.array([[1.0, slant], [0.0, 1.0]])
    M = (shear @ R).T
    pts = np.concatenate([(s * scale) @ M for s in S]) if S else np.zeros((1, 2))
    size = pts.max(0) - pts.min(0)
    lo = -PAPER_HALF[:2] + margin + size / 2
    hi = np.maximum(PAPER_HALF[:2] - margin - size / 2, lo)
    c = rng.uniform(lo, hi) - (pts.max(0) + pts.min(0)) / 2
    return [(s * scale) @ M + c for s in S]


def _force_profile(strokes, rng, kind="flat", lo=0.9, hi=2.5, amp=1.6):
    """Target press force [N] per stroke point - the darkness of the task map.
    Defaults are a human writer's pressure: Wacom Intuos session, p10-p90 of
    the pen-down pressure = 0.9-2.5 N at the pen's 4 N full scale (median 1.6 N).

    "flat"  a constant pressure per stroke, redrawn each episode
    "swell" thin-thick-thin along each stroke
    "ramp"  monotonically lighter or heavier across the stroke
    "wacom" a real pen-down pressure curve sampled from a recorded Wacom session
            (dexcraft/data/pressure_profiles.json), rescaled into [lo, hi] - natural
            entry/exit tapering, so the map's darkness varies the way handwritten ink does
    """
    out = []
    for s in strokes:
        n = len(s)
        d = np.linalg.norm(np.diff(s, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(d)])
        u = u / max(u[-1], 1e-9)
        base = rng.uniform(lo, hi)
        if kind == "wacom":
            pr = np.asarray(F.PRESSURE_PROFILES[rng.integers(len(F.PRESSURE_PROFILES))], float)
            pr = np.interp(u, np.linspace(0.0, 1.0, len(pr)), pr)
            f = lo + (hi - lo) * (pr - pr.min()) / max(pr.max() - pr.min(), 1e-6)
        elif kind == "swell":
            f = base + amp * np.sin(np.pi * u)
        elif kind == "ramp":
            f = base * (1.0 + rng.choice([-1.0, 1.0]) * 0.5 * (u - 0.5))
        else:
            f = np.full(n, base)
        out.append(np.clip(f, 0.15, FORCE_MAX))
    return out


def widths_from_forces(forces, tool):
    """Ink width implied by the target force, so the map's darkness and its stroke
    thickness always describe the same press."""
    from .scene import ink_width_mm
    return [np.array([ink_width_mm(tool, f) for f in fs]) for fs in forces]


# ------------------------------------------------------------------ samplers
def hold_static(rng):
    return TaskSpec("L0_hold_static", "L0", "Hold the pen still for 6 seconds.", [], duration=6.0,
                    hold_motion="static")


def hold_moving(rng):
    """L0: the env throws the tool around and the hand has to keep it.

    The motion is large and fast enough that merely holding the grip pose does not pass:
    2.2x the base sweep with 11 waypoints in 8 s is ~2.4x the speed and ~5x the
    acceleration. The wrist sweep is FITTED inside the joint range (see
    env._teleop_motion) - clipped at the limits it would turn into a bang-bang between
    them - and the difficulty comes from the ARM carrying the wrist through `hold_box` at
    the same time (11 waypoints in 8 s: 0.3-0.5 m/s, 2-3 m/s^2) while a 4-9 Hz shake of
    `hold_jitter_deg` rides on the wrist."""
    return TaskSpec("L0_hold_moving", "L0",
                    "Keep hold of the pen while the arm swings it around, fast.", [], duration=8.0,
                    hold_motion="moving", hold_scale=2.2, hold_waypoints=11,
                    hold_box=np.array([0.12, 0.08, 0.085]), hold_jitter_deg=4.0)


SCRIBBLE_DEPTH_M = 0.003        # the env drives the tip this far below the sheet top (scripted
                                # sweep: 1.5 mm -> contact 0.73, 3 mm -> 0.91, peak 6 N; see env.DRIVE_Z_GAIN)
SCRIBBLE_IMPACT_N = 15.0        # a peak contact above this fails the episode
SCRIBBLE_WAYPOINTS = 14         # direction changes in the 10 s: ~0.3 m/s peak, ~2 m of path


def scribble(rng):
    """L0: KEEP THE TOOL WHILE THE ENV DRAGS IT FAST ACROSS THE SHEET.

    Environment-driven, like the two hold rungs. The env carries arm and wrist along a
    random fast path inside the sheet with the tip pressed `scribble_depth_m` into it (plus
    a wrist wobble); the policy controls the fingers only and has to keep the pen in the
    grip against the drag, without letting it ride up out of contact or jam and slam.

    Why not policy-driven: with the arm free and a per-metre travel reward, a policy lays
    its path in the first seconds and then wanders off the sheet. There is no target to
    draw at L0, so there is nothing for a policy to learn about WHERE, only about HOLDING. Driving the
    motion from the env makes every submission face the same drag, the same speeds and
    the same sheet, and removes the drift by construction.

    Score: the fraction of the driven path during which the tip was in contact with the
    sheet (scaled by the fraction of the episode survived). Success: pen kept, no peak
    above `impact_limit_N`, contact fraction >= `coverage_for_success`.
    """
    c = rng.uniform(-PAPER_HALF[:2] + 0.06, PAPER_HALF[:2] - 0.06)
    return TaskSpec("L0_scribble", "L0",
                    "Keep hold of the pen while the arm drags it fast across the paper - "
                    "keep it on the sheet, and do not let it slam into it.",
                    [], duration=10.0, place_jitter=0.003, place_at=c,
                    hold_motion="scribble", hold_waypoints=SCRIBBLE_WAYPOINTS,
                    scribble_depth_m=SCRIBBLE_DEPTH_M, scribble_wrist_deg=8.0,
                    impact_limit_N=SCRIBBLE_IMPACT_N,
                    meta={"scribble_depth_m": SCRIBBLE_DEPTH_M, "impact_limit_N": SCRIBBLE_IMPACT_N})


SCRIBBLE_FINGER_REF_MM = 9.0       # fingers alone, in millimetres of spread; see below


def scribble_fingers(rng):
    """L0: the arm AND the wrist are frozen. Only the fingers can move the tool.

    L0_scribble lets the wrist help, and every policy takes that offer: measured over
    eight compass directions with the arm frozen, the wrist reaches further than the
    fingers in EVERY one of them (35.1 mm against 0.0 at its best heading, 10.2 against
    14.3 at the fingers'), so the wrist's workspace strictly contains the fingers' and
    no target can require fingers while the wrist is available. Removing the wrist is
    the only way to make finger motion necessary rather than merely allowed.

    Scored on how far the ink SPREADS from where the pen landed, not on area. At finger
    scale area has no dynamic range: with both arm and wrist frozen an open-loop sweep
    inks 39 mm2 net of the landing blob and a do-nothing policy already inks 35, so area
    cannot tell them apart. Spread can - the same sweep carries the tip 9.3 mm while a
    resting pen drifts 1-2 - and 9 mm is the reference. The sweep drops the pen 0/10
    times, so the range is reachable without risking the grip.
    """
    c = rng.uniform(-PAPER_HALF[:2] + 0.06, PAPER_HALF[:2] - 0.06)
    return TaskSpec("L0_scribble_fingers", "L0",
                    "With the arm and wrist both still, scribble over as wide an area as "
                    "the fingers alone can reach, lifting the pen between marks.",
                    [], duration=10.0, arm_frozen=True, wrist_frozen=True, place_jitter=0.003,
                    place_at=c, extent_ref_mm=SCRIBBLE_FINGER_REF_MM, min_lifts=2,
                    meta={"extent_ref_mm": SCRIBBLE_FINGER_REF_MM})


def pick_up(rng):
    """The only task that starts with an empty hand.

    Deliberately its own L0 task rather than a prologue bolted onto every other
    one: picking a thin cylinder off a flat table is a different skill from
    writing with it, and a task that tests two skills at once tells you which one
    failed only by accident."""
    return TaskSpec("L0_pick_up", "L0",
                    "Pick the pen up off the table and hold it.", [], duration=12.0,
                    hold_motion="static", start="table",
                    meta={"rest_x": float(rng.uniform(*TOOL_REST_X)),
                          "rest_y": float(rng.uniform(*TOOL_REST_Y)),
                          "rest_yaw": float(rng.uniform(-np.pi, np.pi))})


def trace_line(rng):
    L = rng.uniform(0.06, 0.12)
    ang = rng.uniform(0, np.pi)
    s = F.shape_strokes("line")[0] @ np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    st = _place([s], rng, L * abs(np.sin(ang)) + 1e-3) if abs(np.sin(ang)) > 0.3 else _place([s], rng, 0.002)
    # _place scales by height; rescale to the requested length instead
    P = np.concatenate(st)
    cur = np.linalg.norm(P[-1] - P[0])
    st = [(s_ - P.mean(0)) * (L / max(cur, 1e-6)) + P.mean(0) for s_ in st]
    fo = _force_profile(st, rng, "flat")
    return TaskSpec("L1_trace_line", "L1", f"Draw a straight line {L * 1000:.0f} mm long.", st,
                    forces=fo, widths=widths_from_forces(fo, "pen"), duration=8.0)


def trace_circle(rng):
    D = rng.uniform(0.04, 0.09)
    st = _place(F.shape_strokes("circle"), rng, D)
    fo = _force_profile(st, rng, "flat")
    return TaskSpec("L1_trace_circle", "L1", f"Draw a circle {D * 1000:.0f} mm in diameter.", st,
                    forces=fo, widths=widths_from_forces(fo, "pen"), duration=10.0)


def trace_shape(rng):
    pool = SPLITS["shapes"]
    kind = pool[rng.integers(len(pool))]
    H = rng.uniform(0.04, 0.09)
    st = _place(F.shape_strokes(kind), rng, H)
    fo = _force_profile(st, rng, rng.choice(["flat", "ramp"]))
    return TaskSpec("L1_trace_shape", "L1", f"Trace a {kind} about {H * 1000:.0f} mm tall.", st,
                    forces=fo, widths=widths_from_forces(fo, "pen"), duration=12.0, meta={"shape": kind})


def write_letters(rng):
    n = int(rng.integers(1, 4))
    text = str(rng.choice(SPLITS["letters"]))
    H = rng.uniform(0.05, 0.065)
    font = SPLITS["fonts"][rng.integers(len(SPLITS["fonts"]))]
    st = _place(F.text_strokes(text, font), rng, H)
    fo = _force_profile(st, rng, "wacom")
    return TaskSpec("L2_write_letters", "L2", f"Write a big '{text}' in {font}, {H * 1000:.0f} mm tall.",
                    st, forces=fo, widths=widths_from_forces(fo, "pen"), duration=10.0,
                     meta={"text": text, "font": font})


def write_word(rng):
    # the joined cursive "daxo" is one more entry in the word pool
    pool = SPLITS["words"]
    i = int(rng.integers(len(pool) + 1))
    text = "daxo" if i == len(pool) else pool[i]
    H = rng.uniform(0.02, 0.035)
    slant = rng.choice([0.0, 0.0, 0.3])
    font = SPLITS["fonts"][rng.integers(len(SPLITS["fonts"]))]
    if i == len(pool):
        font, slant = "cursive", 0.0
        st = _place(F.cursive_daxo(), rng, H)
    else:
        st = _place(F.text_strokes(text, font), rng, H, slant=slant)
    style = "italic " if slant else ""
    how = "joined cursive" if font == "cursive" else f"{style}{font} capitals"
    fo = _force_profile(st, rng, "wacom")
    return TaskSpec("L2_write_word", "L2", f"Write the word '{text}' in {how}.",
                    st, forces=fo, widths=widths_from_forces(fo, "pen"), duration=8.0 + 3.0 * len(text),
                    meta={"text": text, "slant": float(slant), "font": font})


def write_digits(rng):
    text = str(rng.choice(SPLITS["digits"]))
    font = SPLITS["fonts"][rng.integers(len(SPLITS["fonts"]))]
    if font == "script":        # Script Simplex draws every digit double-line (its capitals are
        font = "sans"           # single-line); the instruction names no face, and the draw stays the same
    st = _place(F.text_strokes(text, font), rng, rng.uniform(0.05, 0.065))
    fo = _force_profile(st, rng, "wacom")
    return TaskSpec("L2_write_digits", "L2", f"Write a big {text} with the pen.", st,
                    forces=fo, widths=widths_from_forces(fo, "pen"), duration=10.0,
                     meta={"text": text, "font": font})


def write_word_unseen(rng):
    """L2_write_word with the held-out words (same lengths, faces, scale and jitter)."""
    pool = SPLITS["words_unseen"]
    text = pool[rng.integers(len(pool))]
    H = rng.uniform(0.02, 0.035)
    slant = rng.choice([0.0, 0.0, 0.3])
    font = SPLITS["fonts"][rng.integers(len(SPLITS["fonts"]))]
    st = _place(F.text_strokes(text, font), rng, H, slant=slant)
    fo = _force_profile(st, rng, "wacom")
    style = "italic " if slant else ""
    return TaskSpec("E2_write_word_unseen", "L2", f"Write the word '{text}' in {style}{font} capitals.",
                    st, forces=fo, widths=widths_from_forces(fo, "pen"), duration=8.0 + 3.0 * len(text),
                    meta={"text": text, "slant": float(slant), "font": font})


def write_idiom(rng):
    """Four trained characters in a row - training wrote one character per episode."""
    text = str(rng.choice(SPLITS["idioms"]))
    st = _place(F.chinese_strokes(text), rng, rng.uniform(0.05, 0.065))   # _place fits the width
    fo = _force_profile(st, rng, "wacom")
    return TaskSpec("E2_write_idiom", "L2", f"Write '{text}' with the pen.", st, forces=fo,
                    widths=widths_from_forces(fo, "pen"), duration=12.0 * len(text), meta={"text": text})


def write_chinese_unseen(rng):
    """L2_write_chinese with a character the training pool never contained (7+ strokes)."""
    text = str(rng.choice(SPLITS["chinese_unseen"]))
    st = _place(F.chinese_strokes(text), rng, rng.uniform(0.05, 0.065))
    fo = _force_profile(st, rng, "wacom")
    return TaskSpec("E2_write_chinese_unseen", "L2", f"Write a big '{text}' with the pen.", st,
                    forces=fo, widths=widths_from_forces(fo, "pen"), duration=12.0, meta={"text": text})


def write_chinese(rng):
    text = str(rng.choice(SPLITS["chinese_easy"]))
    st = _place(F.chinese_strokes(text), rng, rng.uniform(0.05, 0.065))
    fo = _force_profile(st, rng, "wacom")
    return TaskSpec("L2_write_chinese", "L2", f"Write a big '{text}' with the pen.",
                    st, forces=fo, widths=widths_from_forces(fo, "pen"), duration=12.0,
                     meta={"text": text})


# ------------------------------------------------------------------ L3 brush
# The brush presses to a DEPTH (brush.py): a task's force profile is the normal force
# its depth profile implies under the reduced-order tuft, so the map's darkness, the
# stroke width it draws and the timeline's z all describe the same press.
BRUSH_DEPTH = (0.0015, 0.0050)          # m: the thin and the thick of a stroke (~4 and ~8 mm wide)
BRUSH_DEPTH_TAPER = 0.0005              # m: where pie / na / gou / ti strokes run out to a point
BRUSH_DEPTH_LAYOUT = (0.0012, 0.0032)   # m: a column on one brushful has to be written lighter
BRUSH_STROKES = ["heng", "shu", "pie", "na", "dian", "zhe", "gou", "ti"]     # 横 竖 撇 捺 点 折 钩 提
BRUSH_PROFILE = {"heng": "press_release", "shu": "press_release", "pie": "taper_out", "na": "taper_in_out",
                 "dian": "dot", "zhe": "corner", "gou": "hook", "ti": "taper_out"}
BRUSH_NAMES = {"heng": "横 heng (horizontal)", "shu": "竖 shu (vertical)", "pie": "撇 pie (falling left)",
               "na": "捺 na (falling right)", "dian": "点 dian (dot)", "zhe": "折 zhe (turn)",
               "gou": "钩 gou (hook)", "ti": "提 ti (rising flick)"}


def _brush_primitive(kind, n=60):
    """The eight basic strokes as unit-box polylines, y up, in writing order."""
    t = np.linspace(0.0, 1.0, n)
    if kind == "heng":
        return np.stack([t, 0.5 + 0.06 * t], 1)
    if kind == "shu":
        return np.stack([np.full(n, 0.5), 1.0 - t], 1)
    if kind == "pie":
        return np.stack([0.72 - 0.62 * t ** 1.3, 1.0 - t], 1)
    if kind == "na":
        return np.stack([0.1 + 0.85 * t, 1.0 - t ** 0.8], 1)
    if kind == "dian":
        return np.stack([0.42 + 0.2 * t, 0.85 - 0.4 * t], 1)
    if kind == "zhe":
        return np.concatenate([np.stack([0.05 + 0.85 * t, np.full(n, 0.9)], 1)[:-1],
                               np.stack([0.9 - 0.04 * t, 0.9 - 0.85 * t], 1)])
    if kind == "gou":
        return np.concatenate([np.stack([np.full(n, 0.55), 1.0 - 0.85 * t], 1)[:-1],
                               np.stack([0.55 - 0.28 * t, 0.15 + 0.12 * t], 1)[:n // 3]])
    if kind == "ti":
        return np.stack([0.15 + 0.7 * t, 0.15 + 0.6 * t ** 0.9], 1)
    raise KeyError(kind)


def _place_long_side(strokes, rng, L):
    """_place, but scaling the LONG side to L (a heng is wide, not tall)."""
    _, _, h = F.normalize(strokes)
    return _place(strokes, rng, L * h)


def _brush_forces(strokes, rng, kinds, depth=BRUSH_DEPTH):
    """Target normal force per stroke point from a depth profile: a press (dun) at the
    ends and lighter through the run, or a taper out to a point, or one dot."""
    from .brush import brush_config, normal_force
    cfg = brush_config()
    lo, hi = depth
    out = []
    for s, kind in zip(strokes, kinds):
        d = np.linalg.norm(np.diff(s, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(d)])
        u = u / max(u[-1], 1e-9)
        if kind == "press_release":
            w = 0.45 + 0.55 * np.maximum(np.exp(-u / 0.1), np.exp(-(1 - u) / 0.1))
        elif kind == "taper_out":
            w = 1.0 - u ** 1.5
        elif kind == "taper_in_out":
            w = np.where(u < 0.7, 0.35 + 0.65 * (u / 0.7) ** 1.2, 1.0 - np.clip((u - 0.7) / 0.3, 0, 1) ** 0.7)
        elif kind == "dot":
            w = np.maximum(np.sin(np.pi * u) ** 0.5, 0.3)
        elif kind == "corner":
            w = 0.5 + 0.5 * np.maximum.reduce([np.exp(-u / 0.08), np.exp(-np.abs(u - 0.5) / 0.06), np.exp(-(1 - u) / 0.08)])
        elif kind == "hook":
            w = np.where(u < 0.72, 0.5 + 0.5 * np.exp(-u / 0.1), np.clip(1.0 - (u - 0.72) / 0.28, 0, 1))
        else:
            w = 0.3 + 0.7 * np.sin(np.pi * u)
        floor = BRUSH_DEPTH_TAPER if kind in ("taper_out", "taper_in_out", "hook") else lo
        dep = floor + (hi * rng.uniform(0.75, 1.0) - floor) * np.clip(w, 0.0, 1.0)
        out.append(np.array([normal_force(cfg, x) for x in dep]))
    return out


def brush_stroke(rng):
    """Brush rung 1: one of the eight basic strokes, with its own lift-and-press profile."""
    kind = str(rng.choice(BRUSH_STROKES))
    st = _place_long_side([_brush_primitive(kind)], rng, rng.uniform(0.05, 0.09))
    fo = _brush_forces(st, rng, [BRUSH_PROFILE[kind]])
    return TaskSpec("L3_brush_stroke", "L3",
                    f"Paint the stroke {BRUSH_NAMES[kind]} with a brush, pressing where the target is dark.",
                    st, forces=fo, widths=widths_from_forces(fo, "brush"), duration=8.0, tool="brush",
                    meta={"stroke": kind})


def brush_chinese(rng):
    """Brush rung 2: one character (seen / unseen pools by stroke-count tier)."""
    text = str(rng.choice(SPLITS["brush_chinese"]))
    st = _place(F.chinese_strokes(text), rng, rng.uniform(0.05, 0.08))
    fo = _brush_forces(st, rng, ["press_release"] * len(st))
    n = len(st)
    return TaskSpec("L3_brush_chinese", "L3",
                    f"Write '{text}' ({n} strokes) with a brush: press at the start and end of each stroke, lighter through it.",
                    st, forces=fo, widths=widths_from_forces(fo, "brush"), duration=6.0 + 1.2 * n, tool="brush",
                    meta={"text": text, "n_strokes": n, "tier": "A" if n <= 4 else "B" if n <= 8 else "C"})


def brush_layout(rng):
    """Brush rung 3: a column of 3-4 characters on ONE brushful of ink. There is no
    re-dipping: the reservoir (brush.py `ink`) only runs down, a heavy hand runs dry
    before the column ends, and a stroke laid with a dry brush is not ink."""
    text = str(rng.choice(SPLITS["brush_layout"]))
    strokes = []
    for k, ch in enumerate(text):
        for s in F.chinese_strokes(ch):
            strokes.append(np.asarray(s) + [0.0, -1.3 * k])           # top to bottom, 0.3 em apart
    st = _place(strokes, rng, rng.uniform(0.045, 0.055) * (1.3 * len(text) - 0.3))
    fo = _brush_forces(st, rng, ["press_release"] * len(st), depth=BRUSH_DEPTH_LAYOUT)
    n = len(st)
    return TaskSpec("L3_brush_layout", "L3",
                    f"Write the column '{text}' ({n} strokes) with one brushful of ink - no re-dipping.",
                    st, forces=fo, widths=widths_from_forces(fo, "brush"), duration=8.0 + 1.2 * n, tool="brush",
                    meta={"text": text, "n_strokes": n, "ink0": 1.0, "layout": "column"})


def sketch_hatching(rng):
    st = _place(F.shape_strokes("hatching"), rng, rng.uniform(0.04, 0.07))
    fo = _force_profile(st, rng, rng.choice(["flat", "ramp"]))
    return TaskSpec("L4_sketch_hatching", "L4", "Shade the area with six parallel hatching strokes.",
                    st, forces=fo, widths=widths_from_forces(fo, "pen"), duration=14.0)


# ------------------------------------------------------------------ L5 paint
# Colour comes from six paint wells beside the sheet (scene.WELL_XY, in scene.PAINT_PALETTE
# order); brushes from a three-slot rack (brush.BRUSHES: fine / medium / broad). The brush
# starts DRY - the first thing a painter does is dip - paint runs out and is refilled by
# dipping again, and a brush taken from the rack is dry. Every stroke carries its colour
# and its brush; a filled region is one boustrophedon (zigzag) stroke, cut into pieces a
# reservoir can cover.
RED, ORANGE, YELLOW, GREEN, BLUE, BROWN = range(1, 7)
COLOUR_NAMES = {RED: "red", ORANGE: "orange", YELLOW: "yellow", GREEN: "green", BLUE: "blue", BROWN: "brown"}
FINE, MEDIUM, BROAD = range(3)
PAINT_DEPTH = {FINE: (0.0008, 0.0025), MEDIUM: (0.0015, 0.0035), BROAD: (0.0020, 0.0045)}   # m, thin..thick
#                                                        (broad at 4.5 mm is ~2.1 N; deeper presses pull the shaft out of the grip)
PAINT_SPLIT_M = 0.30          # a stroke longer than this is cut into pieces (one brushful covers ~0.4-0.6 m)
PAINT_SPLITS = ["sun", "house", "tree", "flower", "boat", "fish", "mountains"]   # the rainbow is its own rung


def _poly(*pts):
    return np.asarray(pts, float)


def _rect(x0, y0, x1, y1):
    return _poly((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _ngon(c, rx, ry=None, n=32):
    ry = rx if ry is None else ry
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + rx * np.cos(a), c[1] + ry * np.sin(a)], 1)


def _arc(c, r, a0, a1, n=40):
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def _hatch(poly, spacing):
    """Fill a convex polygon with ONE zigzag stroke: horizontal scanlines `spacing`
    apart, alternating direction, joined at the ends."""
    P = np.asarray(poly, float)
    ys = np.arange(P[:, 1].min() + spacing / 2, P[:, 1].max(), spacing)
    out = []
    for i, y in enumerate(ys):
        xs = []
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            if (a[1] - y) * (b[1] - y) < 0:
                xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        if len(xs) >= 2:
            x0, x1 = min(xs), max(xs)
            out += [(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]
    return np.asarray(out, float)


def _dot(x, y, r=0.02):
    return _poly((x - r, y), (x + r, y))


def _E(strokes, color, brush, profile):
    return [(np.asarray(s, float), color, brush, profile) for s in strokes]


def _painting(name):
    """Compositions in the unit box; list order is painting order (backgrounds first)."""
    if name == "sun":
        rays = [_poly((0.5 + 0.30 * np.cos(a), 0.55 + 0.30 * np.sin(a)), (0.5 + 0.44 * np.cos(a), 0.55 + 0.44 * np.sin(a)))
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        return _E([_hatch(_ngon((0.5, 0.55), 0.22), 0.09)], YELLOW, BROAD, "flat") + _E(rays, ORANGE, MEDIUM, "press_release")
    if name == "house":
        # the two big fills are the longest drags in the library: sparse rows, or the shaft
        # creeps in the grip before the roof is done (measured: 3/3 house seeds dropped at 0.09)
        # ground first: the longest single line while the grip is fresh (as the last stroke,
        # pressed at both ends after 80 s of fills, it cost the shaft its grip in 3/3 seeds)
        return (_E([_poly((0.05, 0.08), (0.95, 0.08))], GREEN, BROAD, "flat")
                + _E([_hatch(_rect(0.25, 0.1, 0.75, 0.5), 0.12)], BROWN, BROAD, "flat")
                + _E([_hatch(_poly((0.2, 0.5), (0.8, 0.5), (0.5, 0.85)), 0.12)], RED, BROAD, "flat")
                + _E([_hatch(_rect(0.44, 0.1, 0.56, 0.34), 0.035)], BLUE, MEDIUM, "flat"))
    if name == "tree":
        return (_E([_poly((0.5, 0.05), (0.5, 0.45))], BROWN, BROAD, "press_release")
                + _E([_hatch(_ngon((0.5, 0.65), 0.26), 0.09)], GREEN, BROAD, "flat")
                + _E([_dot(0.40, 0.70), _dot(0.60, 0.62), _dot(0.50, 0.80)], RED, FINE, "dot"))
    if name == "flower":
        petals = [_dot(0.5 + 0.17 * np.cos(a), 0.68 + 0.17 * np.sin(a), 0.03) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        return (_E([_poly((0.5, 0.05), (0.5, 0.52))], GREEN, MEDIUM, "press_release")
                + _E([_arc((0.36, 0.2), 0.14, -0.3, 1.2), _arc((0.64, 0.28), 0.14, np.pi - 1.2, np.pi + 0.3)], GREEN, MEDIUM, "taper_out")
                + _E([_hatch(_ngon((0.5, 0.68), 0.08), 0.04)], YELLOW, MEDIUM, "flat")
                + _E(petals, RED, MEDIUM, "dot"))
    if name == "rainbow":
        # arcs 0.09 apart: at the task's 13-16 cm height that is 12-14 mm, just over the
        # broad brush's width, so the bands touch without the tip crossing a neighbour's line
        return sum((_E([_arc((0.5, 0.1), r, np.pi, 0.0)], c, BROAD, "flat")
                    for r, c in zip((0.46, 0.37, 0.28, 0.19, 0.10), (RED, ORANGE, YELLOW, GREEN, BLUE))), [])
    if name == "boat":
        waves = [_poly(*[(0.05 + 0.9 * t, y0 + 0.02 * np.sin(6 * np.pi * t)) for t in np.linspace(0, 1, 40)]) for y0 in (0.10, 0.04)]
        return (_E([_hatch(_poly((0.15, 0.3), (0.85, 0.3), (0.75, 0.16), (0.25, 0.16)), 0.09)], BLUE, BROAD, "flat")
                + _E([_poly((0.5, 0.3), (0.5, 0.85))], BROWN, MEDIUM, "press_release")
                + _E([_hatch(_poly((0.52, 0.82), (0.52, 0.4), (0.82, 0.4)), 0.09)], RED, BROAD, "flat")
                + _E(waves, BLUE, FINE, "press_release"))
    if name == "fish":
        return (_E([_hatch(_ngon((0.45, 0.5), 0.30, 0.17), 0.09)], ORANGE, BROAD, "flat")
                + _E([_hatch(_poly((0.74, 0.5), (0.95, 0.68), (0.95, 0.32)), 0.05)], ORANGE, MEDIUM, "flat")
                + _E([_dot(0.28, 0.55, 0.015)], BLUE, FINE, "dot")
                + _E([_poly((x, 0.40), (x, 0.60)) for x in (0.42, 0.50, 0.58)], RED, FINE, "press_release"))
    if name == "mountains":
        return (_E([_hatch(_poly((0.05, 0.1), (0.55, 0.1), (0.30, 0.75)), 0.09)], BLUE, BROAD, "flat")
                + _E([_hatch(_poly((0.45, 0.1), (0.95, 0.1), (0.70, 0.60)), 0.09)], BLUE, BROAD, "flat")
                + _E([_hatch(_ngon((0.84, 0.82), 0.10), 0.05)], YELLOW, MEDIUM, "flat"))
    raise KeyError(name)


def _paint_forces(strokes, rng, brushes, profiles):
    """Target normal force and width per stroke point from each stroke's own brush and
    depth band: fills ride at a steady depth, lines press at the ends, dots are one press."""
    from .brush import BRUSHES, brush_config, normal_force, width_mm
    cfgs = [brush_config(n) for n in BRUSHES]
    forces, widths = [], []
    for s, b, kind in zip(strokes, brushes, profiles):
        cfg, (lo, hi) = cfgs[b], PAINT_DEPTH[b]
        d = np.linalg.norm(np.diff(s, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(d)])
        u = u / max(u[-1], 1e-9)
        if kind == "flat":
            w = np.full(len(u), 0.75 + 0.1 * rng.uniform(-1.0, 1.0))
        elif kind == "dot":
            w = np.maximum(np.sin(np.pi * u) ** 0.5, 0.4)
        elif kind == "taper_out":
            w = 1.0 - u ** 1.5
        else:
            w = 0.45 + 0.55 * np.maximum(np.exp(-u / 0.1), np.exp(-(1 - u) / 0.1))
        f = np.array([normal_force(cfg, x) for x in lo + (hi - lo) * np.clip(w, 0.0, 1.0)])
        forces.append(f)
        widths.append(np.array([width_mm(cfg, v) for v in f]))
    return forces, widths


def _split_long(items, max_len=PAINT_SPLIT_M):
    """Cut strokes longer than max_len (placed, in metres) into equal pieces."""
    out = []
    for s, c, b, p in items:
        L = float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())
        n = int(np.ceil(L / max_len))
        if n <= 1:
            out.append((s, c, b, p))
            continue
        r = F.resample(s, 0.002)
        for i in range(n):
            piece = r[int(len(r) * i / n): int(len(r) * (i + 1) / n) + 1]
            if len(piece) >= 2:
                out.append((piece, c, b, p))
    return out


def _paint_task(tid, rng, elements, height, desc, meta):
    st = _place([e[0] for e in elements], rng, height)
    items = _split_long([(s, c, b, p) for s, (_, c, b, p) in zip(st, elements)])
    st, colors, brushes, profiles = (list(x) for x in zip(*items))
    fo, wd = _paint_forces(st, rng, brushes, profiles)
    n = len(st)
    changes = 1 + sum(colors[i] != colors[i - 1] or brushes[i] != brushes[i - 1] for i in range(1, n))
    swaps = int(brushes[0] != MEDIUM) + sum(brushes[i] != brushes[i - 1] for i in range(1, n))
    # a station visit is a round trip across the table (~0.3 m each way) plus a second inside:
    # ~7 s at a follower's pace, and refills add roughly one visit per three strokes
    return TaskSpec(tid, "L5", desc, st, forces=fo, widths=wd,
                    duration=14.0 + 2.0 * n + 8.0 * (changes + swaps + n // 3), tool="brush",
                    palette=[list(c) for c in PAINT_PALETTE], colors=[int(c) for c in colors],
                    brushes=[int(b) for b in brushes],
                    meta={**meta, "n_strokes": n, "colour_changes": int(changes), "brush_swaps": int(swaps)})


def paint_shape(rng):
    """Paint rung 1: one filled shape in one colour - dip, fill, keep the paint on the shape."""
    kind = str(rng.choice(["circle", "square", "triangle"]))
    poly = (_ngon((0.5, 0.5), 0.45) if kind == "circle" else _rect(0.1, 0.1, 0.9, 0.9) if kind == "square"
            else _poly((0.08, 0.1), (0.92, 0.1), (0.5, 0.88)))
    color, brush = int(rng.integers(1, 7)), int(rng.choice([MEDIUM, BROAD]))
    el = _E([_hatch(poly, 0.09 if brush == BROAD else 0.06)], color, brush, "flat")
    return _paint_task("L5_paint_shape", rng, el, rng.uniform(0.05, 0.08),
                       f"Fill a {kind} with {COLOUR_NAMES[color]} paint: dip the brush in the {COLOUR_NAMES[color]} well, then paint.",
                       {"shape": kind, "color": color, "brush": brush})


def paint_rainbow(rng):
    """Paint rung 2: five arcs in five colours with one broad brush - re-dipping and colour order."""
    return _paint_task("L5_paint_rainbow", rng, _painting("rainbow"), rng.uniform(0.13, 0.16),
                       "Paint a rainbow: five arcs, five colours, one broad brush.", {"painting": "rainbow"})


def paint_scene(rng):
    """Paint rung 3: a small picture with several colours and more than one brush."""
    name = str(rng.choice(PAINT_SPLITS))
    return _paint_task("L5_paint_scene", rng, _painting(name), rng.uniform(0.09, 0.12),
                       f"Paint a {name}: several colours, more than one brush, wells and rack as needed.",
                       {"painting": name})


def _pick_layout(rng, margin=0.05, min_sep=0.10):
    """Object and bowl positions on the sheet (sheet-centre coords), >= min_sep apart."""
    lo, hi = -PAPER_HALF[:2] + margin, PAPER_HALF[:2] - margin
    for _ in range(100):
        obj, bowl = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(obj - bowl) >= min_sep:
            break
    return obj, bowl


def _chop_task(tid, rng, shape, desc, *, size, mass, mu, solref=(0.005, 1.0), duration=15.0,
               aspect=1.0):
    """One rung of the L6 ladder. The object is always the same task - pinch it
    with two sticks and drop it in the bowl - and only its shape and physics move,
    so a failure says something specific about the contact it demands."""
    obj, bowl = _pick_layout(rng)
    return TaskSpec(tid, "L6", desc, [], duration=duration, tool="chopsticks",
                    meta={"shape": shape, "size": float(size), "mass": float(mass),
                          "friction": float(mu), "solref": list(solref), "aspect": float(aspect),
                          "object_xy": obj.tolist(), "bowl_xy": bowl.tolist(), "bowl_r": BOWL_R})


def chop_hold(rng):
    """Rung 1: no object at all - just do not drop the chopsticks."""
    return TaskSpec("L6_chop_hold", "L6",
                    "Hold the chopsticks while the arm moves them around.", [],
                    duration=10.0, tool="chopsticks", hold_motion="moving")


def chop_cylinder(rng):
    """Rung 2: a soft rubber cylinder - grippy, deformable, flat sides to pinch."""
    return _chop_task("L6_chop_cylinder", rng, "cylinder",
                      "Pick up the rubber cylinder with the chopsticks and put it in the bowl.",
                      size=rng.uniform(0.009, 0.012), mass=rng.uniform(0.004, 0.010),
                      mu=rng.uniform(1.2, 1.6), solref=(0.02, 1.0))


def chop_cube(rng):
    """Rung 3: a rigid cube - flat faces still, but no give."""
    return _chop_task("L6_chop_cube", rng, "cube",
                      "Pick up the cube with the chopsticks and put it in the bowl.",
                      size=rng.uniform(0.008, 0.011), mass=rng.uniform(0.008, 0.018),
                      mu=rng.uniform(0.8, 1.1))


def chop_block(rng):
    """Rung 4: a heavy oblong - the pinch has to hold real weight and a moment."""
    return _chop_task("L6_chop_block", rng, "cube",
                      "Pick up the heavy block with the chopsticks and put it in the bowl.",
                      size=rng.uniform(0.009, 0.012), mass=rng.uniform(0.030, 0.060),
                      mu=rng.uniform(0.7, 1.0), aspect=rng.uniform(1.8, 2.6), duration=18.0)


def chop_sphere(rng):
    """Rung 5, mission impossible: a smooth ball. Point contact on both sticks and
    it rolls out of any pinch that is not exactly on its equator."""
    return _chop_task("L6_chop_sphere", rng, "sphere",
                      "Pick up the ball with the chopsticks and put it in the bowl.",
                      size=rng.uniform(0.008, 0.011), mass=rng.uniform(0.004, 0.012),
                      mu=rng.uniform(0.4, 0.7), duration=18.0)


TASKS = {
    # L0 in order of difficulty: hold still -> hold under a driven arm -> keep the tool while it is
    # dragged -> scribble with the fingers alone -> pick the tool up off the table (empty hand)
    "L0_hold_static": hold_static, "L0_hold_moving": hold_moving, "L0_scribble": scribble,
    "L0_scribble_fingers": scribble_fingers, "L0_pick_up": pick_up,
    "L1_trace_line": trace_line, "L1_trace_circle": trace_circle, "L1_trace_shape": trace_shape,
    "L2_write_letters": write_letters, "L2_write_word": write_word, "L2_write_digits": write_digits,
    "L2_write_chinese": write_chinese,
    "L3_brush_stroke": brush_stroke, "L3_brush_chinese": brush_chinese, "L3_brush_layout": brush_layout,
    "L4_sketch_hatching": sketch_hatching,
    # L5 paint: one filled shape -> a rainbow (five colours) -> a picture (colours + brushes)
    "L5_paint_shape": paint_shape, "L5_paint_rainbow": paint_rainbow, "L5_paint_scene": paint_scene,
    # L6 ladder: hold -> soft cylinder -> cube -> heavy block -> ball
    "L6_chop_hold": chop_hold, "L6_chop_cylinder": chop_cylinder, "L6_chop_cube": chop_cube,
    "L6_chop_block": chop_block, "L6_chop_sphere": chop_sphere,
}
LEVELS = {lvl: [t for t in TASKS if t.startswith(lvl)] for lvl in ("L0", "L1", "L2", "L3", "L4", "L5", "L6")}
# Held-out content, evaluated with the same env, metrics and trajectory input as the trained tasks.
EVAL_TASKS = {"E2_write_word_unseen": write_word_unseen, "E2_write_idiom": write_idiom,
              "E2_write_chinese_unseen": write_chinese_unseen}


def sample(task_id, seed):
    """One pool, one sample: there is no content split (see SPLITS above)."""
    return (TASKS.get(task_id) or EVAL_TASKS[task_id])(np.random.default_rng(seed))
