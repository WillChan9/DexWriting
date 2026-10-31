"""Episode metrics and leaderboard aggregation.

Per episode (from DexCraftEnv.episode_record()):
  completion       how much of the task was done, in [0, 1]: target centre-line points
                   (1.5 mm apart) and ink points (resampled at 1.5 mm) are matched ONE TO
                   ONE (Hungarian); the fraction of target points whose exclusive partner
                   lies within 3 mm counts. Injective matching is the point: a blob of ink
                   parked near the line can only claim its own arc length of credit, where
                   an any-ink-within-3-mm rule would pay it for every nearby target point.
                   The fraction of the episode survived for L0 hold, phase/3 for L6.
  proximity        the orderless stat (target points with ANY ink within 3 mm), kept
                   because completion <= proximity and the gap measures "ink near the line
                   but not shaped like it".
  direction        of the ink strokes that match a target stroke (Frechet <= 6 mm), the
                   fraction drawn in the target's direction.
  stroke_frechet_mm  mean discrete-Frechet distance of matched ink/target stroke pairs. This is what the leaderboard ranks on -
                   a binary threshold throws away the difference between a policy that
                   draws 79 % of a word and one that draws nothing.
  success          the strict binary version, kept as a secondary stat: pen kept AND
                   >= coverage_for_success of the pen-down target within 3 mm of ink
  completion_time  seconds until the last waypoint was captured (None if not)
  chamfer_mm       0.5 * (mean ink->target distance + mean target->ink distance)
  ink_iou          IoU of rasterised ink vs target centre line (1 mm grid, both dilated 1 mm)
  force_mae_N      |actual press force - the force the task map asked for|, averaged
                   over the steps that were actually inking
  arm_travel_m     total hand translation
  finger_share     share of the distance the pen tip actually covered (while writing)
                   that the finger joints produced: each group's tip velocity is
                   projected onto the direction the tip really moved, over the
                   distance it really moved. A group fighting the motion scores
                   negative. The three shares sum to 1 only insofar as the rigid-
                   tool Jacobian explains the tip - a total well under 1 means the
                   tool is sliding in the grip and the split is not to be trusted.
  wrist_share      ... by the wrist joints
  arm_share        ... by the arm joints. MEASURED, not inferred as the remainder:
                   inferring it would bury the model error in the arm and force the
                   three to sum to 1 however wrong the split is.
  idle_rms_deg     ring + pinky (no part in the tripod): RMS deviation from the captured grip
  idle_vel_rps     pose, and RMS joint speed - they are allowed to rest, not to flail
  quality          the per-task quality term of the score: 1 - chamfer / 10 mm for ink tasks
                   (0 at 10 mm; ink_iou is kept alongside), placement accuracy
                   (1 - distance to the bowl centre / bowl radius) for L6
L6 chopsticks: success = object lifted with a two-stick pinch and released inside the bowl, stick kept;
  completion_time, grasp_time, lift_time, object_drops (let go in mid-air), object_lost (off the table), object_to_bowl_mm.
"""
import numpy as np

TOL = 0.003


def _nn_dist(A, B):
    """For each row of A, distance to the nearest row of B (chunked)."""
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.inf)
    out = np.empty(len(A))
    for i in range(0, len(A), 2048):
        a = A[i:i + 2048]
        d2 = np.sum(a[:, None, :] ** 2, -1) + np.sum(B[None, :, :] ** 2, -1) - 2 * a @ B.T
        out[i:i + 2048] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _raster(points, origin, size, res=0.001, dilate=0.002):
    g = np.zeros((int(size[1] / res) + 1, int(size[0] / res) + 1), dtype=bool)
    if len(points) == 0:
        return g
    r = int(round(dilate / res))
    ij = np.round((points[:, :2] - origin) / res).astype(int)
    for (i, j) in ij:
        y0, y1 = max(0, j - r), min(g.shape[0], j + r + 1)
        x0, x1 = max(0, i - r), min(g.shape[1], i + r + 1)
        g[y0:y1, x0:x1] = True
    return g


COMPLETION_TOL_MM = 3.0     # target centre line counts as traced when ink passes within this
COVERAGE_TOL_MM = 6.0       # the lenient variant the env's training log calls `progress`
QUALITY_CHAMFER_MM = 10.0   # quality = 1 - chamfer / this (0 at 10 mm, 0.65 at 3.5 mm, 0.9 at 1 mm)


def _vector_scores(rec):
    """Completion / coverage / precision / IoU from the vector record - the pen-down
    waypoints (1.5 mm apart along the target's centre lines) and the ink points -
    rather than the 128-px maps, whose 3.1 mm pixels would round a 1 mm ballpoint line
    up to 3 mm and make every tolerance a multiple of 3 mm.

    completion  fraction of centre-line points within COMPLETION_TOL_MM of any ink
    coverage    same with COVERAGE_TOL_MM (what the env reports as `progress`)
    precision   fraction of ink within COMPLETION_TOL_MM of the centre line
    ink_iou     IoU on a 1 mm grid, both sides dilated 1 mm - a shape overlap that does
                not depend on stroke width (the ballpoint line is 1 mm anyway)
    Order-free: a policy may draw the strokes in any order."""
    from .scene import PAPER_CENTER, PAPER_HALF
    down = np.asarray(rec["wp_down"], bool)
    if not down.any():
        return None
    target = np.asarray(rec["wp"])[down][:, :2]
    ink = np.asarray(rec["ink_points"]).reshape(-1, 4)[:, :2]
    if len(ink) == 0:
        return dict(completion=0.0, coverage=0.0, precision=0.0, precision_3mm=0.0, ink_iou=0.0)
    t2i, i2t = _nn_dist(target, ink), _nn_dist(ink, target)
    origin, size = PAPER_CENTER[:2] - PAPER_HALF[:2], 2 * PAPER_HALF[:2]
    gt, gi = _raster(target, origin, size, dilate=0.001), _raster(ink, origin, size, dilate=0.001)
    return dict(completion=float(np.mean(t2i <= COMPLETION_TOL_MM / 1000.0)),
                coverage=float(np.mean(t2i <= COVERAGE_TOL_MM / 1000.0)),
                precision=float(np.mean(i2t <= STRICT_TOL)),              # no ink further than 1 mm from the line
                precision_3mm=float(np.mean(i2t <= COMPLETION_TOL_MM / 1000.0)),
                ink_iou=float((gt & gi).sum() / max((gt | gi).sum(), 1)))


STROKE_SPACING = 0.0005        # 0.5 mm: a 1 mm tolerance must not be defeated by the resampling itself
STRICT_TOL = 0.001             # the strict pair: completion_1mm / precision_1mm
MATCH_TOL = 0.003              # a matched point pair within 3 mm earns completion
STROKE_MATCH_TOL = 0.006       # a stroke pair within 6 mm Frechet counts for direction


def _resample_arc(P, spacing=STROKE_SPACING):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(d)])
    if L[-1] < 1e-9:
        return P[:1]
    u = np.append(np.arange(0.0, L[-1], spacing), L[-1])
    return np.stack([np.interp(u, L, P[:, k]) for k in range(P.shape[1])], axis=1)


def _target_strokes(rec):
    """Pen-down runs of the ORDERED target. The timeline (what the task actually
    instructs, in time order) is the reference when the record carries it - the wp
    array's storage order can differ from the commanded stroke order/direction, which
    would make the order-aware completion score a correct trace as "reversed". Falls
    back to wp for records (and synthetic tests) without a timeline."""
    if rec.get("timeline") is not None:
        down = np.asarray(rec["tl_down"]) > 0.5
        wp = np.asarray(rec["timeline"])[:, :2]
    else:
        down = np.asarray(rec["wp_down"], bool)
        wp = np.asarray(rec["wp"])[:, :2]
    out, s = [], None
    for i, d in enumerate(down):
        if d and s is None:
            s = i
        if s is not None and (not d or i == len(down) - 1):
            e = i + (1 if d else 0)
            if e - s >= 2:
                out.append(wp[s:e])
            s = None
    return out


def _ink_strokes(rec):
    """Split the ink trail into strokes at pen lifts: a time gap or a position jump."""
    ink = np.asarray(rec["ink_points"]).reshape(-1, 4)
    if len(ink) < 2:
        return []
    cut = (np.diff(ink[:, 3]) > 0.35) | (np.linalg.norm(np.diff(ink[:, :2], axis=0), axis=1) > 0.004)
    out = []
    for seg in np.split(ink[:, :2], np.where(cut)[0] + 1):
        if len(seg) >= 3:
            out.append(seg)
    return out


def _frechet(P, Q, n=32):
    """Discrete Frechet distance on n-point arc resamplings."""
    P, Q = _resample_arc(P, max(1e-4, _arc(P) / n)), _resample_arc(Q, max(1e-4, _arc(Q) / n))
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    C = np.full_like(D, np.inf)
    C[0, 0] = D[0, 0]
    for i in range(len(P)):
        for j in range(len(Q)):
            if i or j:
                prev = min(C[i - 1, j] if i else np.inf, C[i, j - 1] if j else np.inf,
                           C[i - 1, j - 1] if i and j else np.inf)
                C[i, j] = max(prev, D[i, j])
    return float(C[-1, -1])


def _arc(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _stroke_scores(rec):
    """The shape-aware scores: 1:1 point matching for completion, stroke-level
    Frechet matching for direction and stroke quality."""
    from scipy.optimize import linear_sum_assignment
    tstrokes, istrokes = _target_strokes(rec), _ink_strokes(rec)
    if not tstrokes:
        return None
    target = np.concatenate([_resample_arc(t) for t in tstrokes])
    out = dict(completion=0.0, completion_3mm=0.0, completion_2mm=0.0, direction=None, stroke_frechet_mm=None)
    if not istrokes:
        return out
    # ORDER-AWARE completion via monotone (subsequence-DTW) alignment: legibility IS
    # in-order following. Ink points, in drawing order, are aligned monotonically to
    # the target centre line in stroke order: local speed is free, overtracing and
    # junk ink before/after/between strokes are absorbed, but the path can never go
    # BACKWARDS - out-of-order corridor scribble earns nothing. completion = fraction
    # of target points whose aligned ink lies within 3 mm.
    ink = np.concatenate([_resample_arc(st) for st in istrokes])
    D = np.minimum(np.linalg.norm(target[:, None, :] - ink[None, :, :], axis=2),
                   2 * MATCH_TOL).astype(np.float32)
    N = len(target)
    C = np.empty_like(D)                    # C[i,j] = best monotone cost ending with
    C[0] = D[0]                             # target i aligned at ink column j
    for i in range(1, N):
        C[i] = D[i] + np.minimum.accumulate(C[i - 1])   # predecessor at any j' <= j
    j = int(np.argmin(C[-1]))                            # free end
    dmatch = np.empty(N)
    for i in range(N - 1, -1, -1):
        dmatch[i] = D[i, j]
        if i:
            j = int(np.argmin(C[i - 1][:j + 1]))
    # completion is the 1 mm, in-order number; 3 mm kept for reference
    out["completion"] = float(np.mean(dmatch <= STRICT_TOL))
    out["completion_3mm"] = float(np.mean(dmatch <= MATCH_TOL))
    out["completion_2mm"] = float(np.mean(dmatch <= 0.002))
    F = np.zeros((len(tstrokes), len(istrokes)))
    R = np.zeros_like(F, dtype=bool)
    for i, t in enumerate(tstrokes):
        for j, s in enumerate(istrokes):
            f, b = _frechet(t, s), _frechet(t, s[::-1])
            F[i, j], R[i, j] = min(f, b), b < f
    ri, ci = linear_sum_assignment(np.minimum(F, 2 * STROKE_MATCH_TOL))
    ok = F[ri, ci] <= STROKE_MATCH_TOL
    if ok.any():
        out["direction"] = float(np.mean(~R[ri, ci][ok]))
        out["stroke_frechet_mm"] = float(1000 * np.mean(F[ri, ci][ok]))
    # strict stroke fidelity: every TARGET stroke counts, an unmatched or reversed one at the cap
    fs = np.full(len(tstrokes), STROKE_MATCH_TOL)
    good = ok & ~R[ri, ci]
    fs[ri[good]] = np.minimum(F[ri, ci][good], STROKE_MATCH_TOL)
    out["frechet_strict_mm"] = float(1000 * np.mean(fs))
    return out


def _inked_area_mm2(rec, dilate_mm=2):
    """Area the ink covers, dilated so a dot counts for little and a spread scribble for
    a lot. The map is 1 mm/px, so a pixel is a square millimetre. The caller subtracts
    the blob the pen makes just by landing - that is not scribbling."""
    ink = np.asarray(rec.get("ink_map"))
    if ink is None or ink.size == 0:
        return 0.0
    h = ink > 0
    for _ in range(int(dilate_mm)):
        h = (h | np.roll(h, 1, 0) | np.roll(h, -1, 0) | np.roll(h, 1, 1) | np.roll(h, -1, 1))
    return float(h.sum())


def _ink_spread_mm(rec):
    """How far the ink reaches from its own centre - the radius a scribble covers.
    Area is the right measure when the wrist can help; at finger scale it cannot
    separate a sweep from a resting pen, and this can."""
    ink = np.asarray(rec.get("ink_map"))
    if ink is None or ink.size == 0:
        return 0.0
    ys, xs = np.nonzero(ink > 0)
    if len(xs) < 2:
        return 0.0
    P = np.stack([xs, ys], 1).astype(float)      # 1 px = 1 mm
    return float(np.linalg.norm(P - P.mean(0), axis=1).max())


def episode_metrics(rec):
    out0 = _episode_metrics(rec)
    out0["time_s"] = float(rec["done_time"]) if rec.get("done_time") is not None else float(rec.get("duration") or 0.0)
    return out0


def _episode_metrics(rec):
    task = rec["task"]
    out = {"dropped": bool(rec["dropped"]), "duration": float(rec["duration"]),
           "progress": float(rec["progress"]), "completion_time": rec["done_time"],
           "force_mae_N": rec.get("force_mae")}
    if rec.get("pickup"):                        # L0_pick_up: empty hand -> tool in hand
        ph = int(rec.get("pick_phase", 0))
        out["completion"] = ph / 3.0
        out.update(success=ph >= 3 and not rec["dropped"], coverage=None, chamfer_mm=None,
                   ink_iou=None, width_mae_mm=None, quality=1.0 if ph >= 3 else 0.0,
                   pick_time=rec.get("pick_time"))
    elif rec.get("tool") == "chopsticks" and "placed" in rec:   # L5 pick and place
        placed = bool(rec["placed"]) and not rec["dropped"]
        acc = 1.0 - min(rec["object_to_bowl_mm"] / rec["bowl_r_mm"], 1.0)
        out["completion"] = float(rec.get("phase", 0)) / 3.0
        out.update(success=placed, coverage=None, chamfer_mm=None, ink_iou=None, width_mae_mm=None,
                   quality=acc if placed else 0.0, grasp_time=rec["grasp_time"], lift_time=rec["lift_time"],
                   object_drops=int(rec["object_drops"]), object_lost=bool(rec.get("object_lost", False)),
                   object_to_bowl_mm=float(rec["object_to_bowl_mm"]))
    else:
        sc = _vector_scores(rec)
        if getattr(task, "extent_ref_mm", 0.0) > 0:        # L0_scribble_fingers: spread
            sp = _ink_spread_mm(rec)
            out["completion"] = float(min(1.0, sp / task.extent_ref_mm))
            out["ink_spread_mm"] = sp
            out["n_lifts"] = int(rec.get("n_lifts", 0))
            out.update(success=bool(not rec["dropped"]
                                    and out["n_lifts"] >= int(getattr(task, "min_lifts", 0))
                                    and out["completion"] >= task.coverage_for_success),
                       coverage=None, chamfer_mm=None, ink_iou=None, width_mae_mm=None,
                       quality=out["completion"])
        elif getattr(task, "path_ref_mm", 0.0) > 0:       # L0_scribble: tip travel, not a picture
            tip = np.asarray(rec["tip"]).reshape(-1, 3)
            path = 0.0
            if len(tip) > 1:
                from .scene import PAPER_TOP
                step = np.linalg.norm(np.diff(tip, axis=0), axis=1)
                # ON the sheet means TOUCHING it: scored on proximity instead, a policy
                # can hover over the paper and accumulate path without ever touching it.
                touch = np.asarray(rec.get("touch", []), bool)
                on = touch[1:len(tip)] if len(touch) >= len(tip) else np.zeros(len(step), bool)
                path = 1000.0 * float(step[:len(on)][on].sum())
            peak = float(rec.get("peak_force", 0.0))
            lim = float(getattr(task, "impact_limit_N", 0.0))
            slam = bool(lim > 0 and peak > lim)
            out["completion"] = float(min(1.0, path / task.path_ref_mm))
            out["path_mm"] = path
            out["peak_force_N"] = peak
            out["slammed"] = slam
            out.update(success=bool(not rec["dropped"] and not slam
                                    and out["completion"] >= task.coverage_for_success),
                       coverage=None, chamfer_mm=None, ink_iou=None, width_mae_mm=None,
                       quality=out["completion"])
        elif getattr(task, "coverage_ref_mm2", 0.0) > 0:   # L0_scribble: area, not accuracy
            area = max(0.0, _inked_area_mm2(rec) - float(rec.get("cov_base", 0)))
            out["completion"] = float(min(1.0, area / task.coverage_ref_mm2))
            out["inked_area_mm2"] = area
            out["n_lifts"] = int(rec.get("n_lifts", 0))
            out.update(success=bool(not rec["dropped"]
                                    and out["n_lifts"] >= int(getattr(task, "min_lifts", 0))
                                    and out["completion"] >= task.coverage_for_success),
                       coverage=None, chamfer_mm=None, ink_iou=None, width_mae_mm=None,
                       quality=out["completion"])
        elif getattr(task, "hold_motion", None) == "scribble":   # L0_scribble: env-driven drag
            # The env drags the tip along a set path pressed into the sheet; what is scored
            # is whether the pen STAYED THERE - in the grip and on the paper - not where it
            # went. completion = fraction of the driven steps with tool-sheet contact, scaled
            # by the fraction of the episode survived; a slam (peak above the limit) fails.
            touch = np.asarray(rec.get("touch", []), bool)
            tip = np.asarray(rec["tip"]).reshape(-1, 3)
            path = 0.0
            if len(tip) > 1 and len(touch) >= len(tip):
                step = np.linalg.norm(np.diff(tip, axis=0), axis=1)
                path = 1000.0 * float(step[touch[1:len(tip)]].sum())
            frac = float(touch.mean()) if len(touch) else 0.0
            alive = float(min(rec["duration"] / max(task.duration, 1e-6), 1.0))
            peak = float(rec.get("peak_force", 0.0))
            lim = float(getattr(task, "impact_limit_N", 0.0))
            slam = bool(lim > 0 and peak > lim)
            out["completion"] = frac * alive
            out["contact_fraction"] = frac
            out["path_mm"] = path
            out["peak_force_N"] = peak
            out["slammed"] = slam
            out.update(success=bool(not rec["dropped"] and not slam
                                    and out["completion"] >= task.coverage_for_success),
                       coverage=None, chamfer_mm=None, ink_iou=None, width_mae_mm=None,
                       quality=out["completion"])
        elif sc is None:                         # L0 hold: no target to draw
            out["completion"] = float(min(rec["duration"] / max(task.duration, 1e-6), 1.0))
            out.update(success=not rec["dropped"], coverage=None, chamfer_mm=None, ink_iou=None,
                       width_mae_mm=None, quality=None)
        else:
            st = _stroke_scores(rec) or {}
            out["completion"] = st.get("completion", 0.0)          # 1 mm, in order
            out["completion_3mm"] = st.get("completion_3mm")
            out["completion_2mm"] = st.get("completion_2mm")
            out["proximity"] = sc["completion"]
            out["direction"] = st.get("direction")
            out["stroke_frechet_mm"] = st.get("stroke_frechet_mm")
            out["frechet_strict_mm"] = st.get("frechet_strict_mm")
            ink = rec["ink_points"][:, :2] if len(rec["ink_points"]) else np.zeros((0, 2))
            down = rec["wp_down"]
            target = rec["wp"][down][:, :2] if down.any() else np.zeros((0, 2))
            chamfer = 0.05
            if len(ink) and len(target):
                t2i, i2t = _nn_dist(target, ink), _nn_dist(ink, target)
                chamfer = 0.5 * (np.mean(np.minimum(t2i, 0.05)) + np.mean(np.minimum(i2t, 0.05)))
            # a task may also demand that the tool was lifted and re-landed (L0_scribble
            # asks for separate marks; one long smear is not the same skill)
            lifts_ok = int(rec.get("n_lifts", 0)) >= int(getattr(task, "min_lifts", 0))
            out["n_lifts"] = int(rec.get("n_lifts", 0))
            out.update(success=bool(not rec["dropped"] and lifts_ok
                                    and out["completion"] >= task.coverage_for_success),
                       coverage=sc["coverage"], precision=sc["precision"], ink_iou=sc["ink_iou"],
                       quality=float((1.0 - 1000 * float(chamfer) / QUALITY_CHAMFER_MM)),
                       chamfer_mm=1000 * float(chamfer), width_mae_mm=None)
            if rec.get("tool") == "brush" and rec.get("brush_canvas") is not None:
                out.update(_brush_scores(rec))
                if out["stroke_iou"] is not None:        # the brush tier's quality axis IS its IoU
                    out["quality"] = out["ink_iou"] = out["stroke_iou"]
                if rec.get("paint") and out.get("color_iou") is not None:
                    out["quality"] = out["color_iou"]    # ... and a painting's is its per-colour IoU
    arm = rec["arm"]
    out["arm_travel_m"] = float(np.sum(np.linalg.norm(np.diff(arm[:, :3], axis=0), axis=1))) if len(arm) > 1 else 0.0
    out["finger_share"] = rec.get("finger_share")
    out["wrist_share"] = rec.get("wrist_share")
    out["arm_share"] = rec.get("arm_share")
    out["idle_rms_deg"] = rec.get("idle_rms_deg")      # ring + pinky: RMS deviation from the grip pose
    out["idle_vel_rps"] = rec.get("idle_vel_rps")      # ... and RMS joint speed - "do not flail"
    return out


def _brush_scores(rec):
    """The brush tier's own axes (see brush.py): stroke IoU and skeleton Chamfer
    (mean / p95, px at the 256-px normalisation) against the ORACLE render of the target -
    the same tip model replaying the target at its depth - ink-density SSIM, the
    stroke-order edit distance, and the fraction of steps with a physical violation
    (press > 5 N or past 0.6 L_tip, ferrule on the paper, shaft past 60 deg). No
    weighted total; the leaderboard's quality column shows the IoU."""
    from . import brush as B
    from .scene import PAPER_CENTER, PAPER_HALF, PAPER_TOP
    cfg, paper = B.brush_config(rec.get("brush_cfg", "wolf")), B.paper_config(rec.get("paper_cfg", "shuxuan"))
    ref = B.oracle_render(rec["task"], cfg, PAPER_CENTER, PAPER_HALF, PAPER_TOP)
    out = B.score_render(B.bleed(rec["brush_canvas"], paper), ref.image(paper), paper)
    refs = [np.asarray(s) + PAPER_CENTER[:2] for s in rec["task"].strokes]
    out["stroke_order_edit"] = B.stroke_order_edit(_ink_strokes(rec), refs)
    out["violation_rate"] = float(rec.get("brush_violation_steps", 0)) / max(int(rec.get("brush_steps", 0)), 1)
    out["ran_dry"] = bool(rec.get("brush_ran_dry", False))
    out["ink_left"] = float(rec.get("brush_ink_left", 1.0))
    if rec.get("paint"):
        # L5: colour, in absolute sheet coordinates (per-colour IoU, colour accuracy), and how
        # the painter used the stations
        out.update(B.score_paint(rec.get("brush_rgb"), rec["brush_canvas"], ref.rgb, ref.alpha, rec["palette"]))
        out["dips"], out["brush_swaps"] = int(rec.get("dips", 0)), int(rec.get("brush_swaps", 0))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(episodes):
    """episodes: list of dicts with keys task_id, track, metrics."""
    by = {}
    for e in episodes:
        by.setdefault((e["track"], e["task_id"]), []).append(e["metrics"])
    per_task = {}
    for (track, tid), ms in by.items():
        per_task.setdefault(track, {})[tid] = {
            "n": len(ms),
            "completion": _mean([m.get("completion") for m in ms]),
            "completion_2mm": _mean([m.get("completion_2mm") for m in ms]),
            "proximity": _mean([m.get("proximity") for m in ms]),
            "direction": _mean([m.get("direction") for m in ms]),
            "stroke_frechet_mm": _mean([m.get("stroke_frechet_mm") for m in ms]),
            "completion_3mm": _mean([m.get("completion_3mm") for m in ms]),
            "precision": _mean([m.get("precision") for m in ms]),
            "time_s": _mean([m.get("time_s") for m in ms]),
            "frechet_strict_mm": _mean([m.get("frechet_strict_mm") for m in ms]),
            "success_rate": _mean([m["success"] for m in ms]),
            "completion_time": _mean([m["completion_time"] for m in ms]),
            "completion_rate": _mean([m["completion_time"] is not None for m in ms]),
            "chamfer_mm": _mean([m["chamfer_mm"] for m in ms]),
            "ink_iou": _mean([m["ink_iou"] for m in ms]),
            "quality": _mean([m.get("quality", m["ink_iou"]) for m in ms]),
            "width_mae_mm": _mean([m.get("width_mae_mm") for m in ms]),
            "force_mae_N": _mean([m.get("force_mae_N") for m in ms]),
            "object_drops": _mean([m.get("object_drops") for m in ms]),
            "arm_travel_m": _mean([m["arm_travel_m"] for m in ms]),
            "finger_share": _mean([m.get("finger_share") for m in ms]),
            "wrist_share": _mean([m.get("wrist_share") for m in ms]),
            "arm_share": _mean([m.get("arm_share") for m in ms]),
            "idle_rms_deg": _mean([m.get("idle_rms_deg") for m in ms]),
            "idle_vel_rps": _mean([m.get("idle_vel_rps") for m in ms]),
            # brush tier only (None elsewhere)
            "stroke_iou": _mean([m.get("stroke_iou") for m in ms]),
            "skel_chamfer_px": _mean([m.get("skel_chamfer_px") for m in ms]),
            "skel_chamfer_p95_px": _mean([m.get("skel_chamfer_p95_px") for m in ms]),
            "ink_ssim": _mean([m.get("ink_ssim") for m in ms]),
            "stroke_order_edit": _mean([m.get("stroke_order_edit") for m in ms]),
            "violation_rate": _mean([m.get("violation_rate") for m in ms]),
            "ran_dry_rate": _mean([m.get("ran_dry") for m in ms]),
            # paint tier only
            "color_iou": _mean([m.get("color_iou") for m in ms]),
            "color_accuracy": _mean([m.get("color_accuracy") for m in ms]),
            "dips": _mean([m.get("dips") for m in ms]),
            "brush_swaps": _mean([m.get("brush_swaps") for m in ms]),
        }
    summary = {}
    for track, tasks in per_task.items():
        summary[track] = {
            "completion": _mean([t["completion"] for t in tasks.values()]),
            "success_rate": _mean([t["success_rate"] for t in tasks.values()]),
            "completion_rate": _mean([t["completion_rate"] for t in tasks.values()]),
            "chamfer_mm": _mean([t["chamfer_mm"] for t in tasks.values()]),
            "ink_iou": _mean([t["ink_iou"] for t in tasks.values()]),
            "quality": _mean([t["quality"] for t in tasks.values()]),
            "width_mae_mm": _mean([t["width_mae_mm"] for t in tasks.values()]),
            "force_mae_N": _mean([t["force_mae_N"] for t in tasks.values()]),
            "finger_share": _mean([t["finger_share"] for t in tasks.values()]),
            "wrist_share": _mean([t["wrist_share"] for t in tasks.values()]),
            "arm_share": _mean([t["arm_share"] for t in tasks.values()]),
        }
    nom = summary.get("nominal", {})
    rob = summary.get("robust", {})
    con = summary.get("constrained", {})
    q_score = lambda s: (s.get("quality") if s.get("quality") is not None else s.get("ink_iou")) or 0.0
    score = {
        "completion": nom.get("completion") or 0.0,
        "success": nom.get("success_rate") or 0.0,          # secondary stat, not in `overall`
        "quality": q_score(nom),
        "time": _mean([1.0 - min((t["completion_time"] or 1e9) / max(_task_duration(tid), 1e-6), 1.0)
                       for tid, t in per_task.get("nominal", {}).items()]) or 0.0,
        "robustness": ((rob.get("completion") or 0.0) / max(nom.get("completion") or 1e-9, 1e-9)
                       if nom.get("completion") else 0.0),
        "constrained": con.get("completion") or 0.0,
    }
    score["robustness"] = min(score["robustness"], 1.0)
    # No weighted total. Any weighting over completion / quality / time / robustness /
    # finger track would be an unjustifiable claim about their relative worth, so the
    # leaderboard reports the axes and ranks on completion.
    return {"per_task": per_task, "summary": summary, "score": score}


def _task_duration(task_id):
    from . import tasks as T
    return T.sample(task_id, 0).duration
