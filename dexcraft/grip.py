"""Solve a tripod pen grip for dex_hand: thumb / index / middle joint
angles and the pen pose such that the three fingertip capsules touch the
shaft with in-plane force closure. Writes grip_pocket.json, then checks that
position servos actually hold the pen under gravity.

    python -m dexcraft.grip
"""
import json
import os

import mujoco
import numpy as np

from . import scene as S
from . import gen_pen as P

# starting guess (rad); ring / pinky stay curled out of the way
INIT_POSE = {
    "thumb_mcp_joint": 0.35, "thumb_mcp_abd_joint": 0.50, "thumb_pip_joint": 0.30,
    "thumb_dip_joint": 0.30,
    "index_mcp_joint": 0.55, "index_mcp_abd_joint": -0.05, "index_pip_joint": 0.60,
    "index_dip_joint": 0.40,
    "middle_mcp_joint": 0.60, "middle_pip_joint": 0.75, "middle_dip_joint": 0.45,
    "ring_cmc_joint": 0.10, "ring_mcp_joint": 0.90, "ring_pip_joint": 1.00, "ring_dip_joint": 0.60,
    "pinky_cmc_joint": 0.15, "pinky_mcp_joint": 0.95, "pinky_pip_joint": 1.00, "pinky_dip_joint": 0.60,
}
SHAFT_DIR0 = np.array([-0.433, -0.75, 0.5])     # tip -> back end, hand_base frame
# chopsticks: ring + pinky are less curled so the ring fingertip sits under the fixed lower stick
CHOP_SUPPORT_POSE = {
    "ring_cmc_joint": 0.10, "ring_mcp_joint": 0.55, "ring_mcp_abd_joint": -0.35, "ring_pip_joint": 0.95,
    "ring_dip_joint": 0.55,
    "pinky_cmc_joint": 0.15, "pinky_mcp_joint": 0.75, "pinky_mcp_abd_joint": -0.15, "pinky_pip_joint": 1.0,
    "pinky_dip_joint": 0.6,
}


HUMAN_GRIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_grip.json")


def human_pose():
    """The captured human tripod grip: median over 2915 teleop frames with the grip
    weight above 0.9. All 25 joints are covered - the 9 abduction DoF the model
    omits are exactly the ones a human does not use to hold a pen.

    This is the ANCHOR, not a starting guess. A free CEM search over contact geometry
    finds poses that close force but are not human, e.g. the pen held against the palm
    by the index and middle fingers with the thumb carrying no load.
    """
    with open(HUMAN_GRIP) as fh:
        return json.load(fh)["hand_pose"]


def solve(seed=0, iters=150, verbose=True, anchor_w=6.0):
    m = S.build_model(grip=None)
    d = mujoco.MjData(m)
    jn = S.names(m, mujoco.mjtObj.mjOBJ_JOINT)
    qadr = {n: m.jnt_qposadr[jn.index(n)] for n in S.HAND_JOINTS}
    ctl = [n for n in S.HAND_JOINTS if n.split("_")[0] in S.GRASP_FINGERS]
    NC = len(ctl)          # grasp-finger DoF: 13 once pip/dip are hinges
    lo = np.array([S.JOINT_RANGE[n][0] for n in ctl])
    hi = np.array([S.JOINT_RANGE[n][1] for n in ctl])
    hp = human_pose()
    q0 = np.array([hp.get(n, INIT_POSE.get(n, 0.0)) for n in ctl])
    q_anchor = q0.copy()                      # the captured human grip
    full0 = {n: hp.get(n, INIT_POSE.get(n, 0.0)) for n in S.HAND_JOINTS}
    # `tip_gids` is the core capsule: it carries the phalanx axis, so it is what the
    # IK and the gap geometry are written against. `tip_r` has to be the radius of
    # what actually touches the pen, which on a padded hand is the pad envelope.
    tip_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_dip_geom") for f in S.GRASP_FINGERS]
    tip_r = [S.fingertip_radius(m, f) for f in S.GRASP_FINGERS]
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")

    S.reset_home(m, d)
    home_q = d.qpos.copy()

    def set_hand(qc):
        d.qpos[:] = home_q
        for n, v in full0.items():
            d.qpos[qadr[n]] = v
        for n, v in zip(ctl, qc):
            d.qpos[qadr[n]] = v
        mujoco.mj_kinematics(m, d)

    set_hand(q0)
    # seed the thumb opposite the index pad: IK its fingertip to a point on the
    # radial-dorsal side of where the pen will run
    idx_tip = S.capsule_segment(m, d, tip_gids[1])[1]
    mid_tip = S.capsule_segment(m, d, tip_gids[2])[1]
    hb_pos = d.xpos[hb]
    thumb_target = hb_pos + (0.5 * (idx_tip + mid_tip) - hb_pos) + np.array([-0.028, -0.012, 0.012])
    th = [i for i, n in enumerate(ctl) if n.startswith("thumb_")]
    for _ in range(25):
        set_hand(q0)
        tip = S.capsule_segment(m, d, tip_gids[0])[1]
        r = tip - thumb_target
        if np.linalg.norm(r) < 1e-3:
            break
        J = np.zeros((3, len(th)))
        for k, i in enumerate(th):
            q1 = q0.copy(); q1[i] += 1e-4
            set_hand(q1)
            J[:, k] = (S.capsule_segment(m, d, tip_gids[0])[1] - tip) / 1e-4
        dq = -np.linalg.pinv(J, rcond=1e-3) @ r
        q0[th] = np.clip(q0[th] + 0.5 * dq, lo[th], hi[th])
    set_hand(q0)
    tips0 = np.array([0.5 * (a + b) for a, b in (S.capsule_segment(m, d, g) for g in tip_gids)])
    p0 = tips0.mean(axis=0)
    ax0 = SHAFT_DIR0 / np.linalg.norm(SHAFT_DIR0)
    e1 = np.cross(ax0, [1, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax0, e1)

    def pen_pose(x):
        u = ax0 + x[NC + 3] * e1 + x[NC + 4] * e2
        u /= np.linalg.norm(u)
        return p0 + x[NC:NC + 3], u

    pq, _ = S.pen_joint(m)
    pen_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pen_set = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pen_bid}
    tip_set = set().union(*(S.fingertip_geoms(m, f) for f in S.GRASP_FINGERS))
    # 4th contact: the shaft rests on the index proximal phalanx (the web side),
    # which gives the grip a long lever against the pen pivoting
    web_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "index_mcp_geom")
    web_r = m.geom_size[web_gid][0]

    def evaluate(x, show=False):
        set_hand(np.clip(x[:NC], lo, hi))
        c, u = pen_pose(x)
        b1, b2 = c - S.PEN_HALF * u, c + S.PEN_HALF * u
        cost, dirs, gaps = 0.0, [], []
        for gid, r in zip(tip_gids, tip_r):
            a1, a2 = S.capsule_segment(m, d, gid)
            gap = S.seg_seg_dist(a1, a2, b1, b2) - r - S.PEN_RADIUS
            gaps.append(gap)
            cost += ((gap + 0.0005) / 0.002) ** 2          # tips: 0.5 mm squeeze
            v = 0.5 * (a1 + a2) - c
            v -= (v @ u) * u
            dirs.append(v / (np.linalg.norm(v) + 1e-9))
        w1, w2 = S.capsule_segment(m, d, web_gid)
        gap_web = S.seg_seg_dist(w1, w2, b1, b2) - web_r - S.PEN_RADIUS
        cost += ((gap_web + 0.0005) / 0.002) ** 2            # shaft touches the web / index base
        # the pen must not penetrate anything else (other phalanges, palm)
        d.qpos[pq:pq + 3] = c
        d.qpos[pq + 3:pq + 7] = S.pen_quat(u)
        mujoco.mj_forward(m, d)
        pen_depth = 0.0
        for i in range(d.ncon):
            ct = d.contact[i]
            other = ct.geom1 if ct.geom2 in pen_set else (ct.geom2 if ct.geom1 in pen_set else None)
            if other is not None and other not in tip_set and other not in pen_set \
                    and other != web_gid and ct.dist < 0:
                pen_depth += (-ct.dist / 0.001) ** 2
        cost += pen_depth
        f1 = dirs[0]
        f2 = np.cross(u, f1)
        ang = np.sort([np.arctan2(v @ f2, v @ f1) for v in dirs])
        G = np.degrees(np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]])).max())
        cost += (max(0.0, G - 150.0) / 10.0) ** 2
        # stay near the captured human grip. This is what stops the search from
        # inventing a non-anthropomorphic grip that happens to close force.
        cost += anchor_w * np.sum(((x[:NC] - q_anchor) / 0.30) ** 2)
        cost += np.sum((x[NC:NC + 3] / 0.02) ** 2) + np.sum((x[NC + 3:NC + 5] / 0.35) ** 2)
        if show:
            print(f"  gaps mm {[round(g * 1000, 2) for g in gaps]}  web gap {gap_web * 1000:.2f} mm  "
                  f"max normal gap {G:.0f} deg  pen shift mm {np.round(x[NC:NC + 3] * 1000, 1)}  "
                  f"other-penetration cost {pen_depth:.1f}")
        return cost

    x = np.concatenate([q0, np.zeros(5)])
    if verbose:
        print("start:"); evaluate(x, True)
    rng = np.random.default_rng(seed)
    sig = np.concatenate([np.full(NC, 0.15), np.full(3, 0.01), np.full(2, 0.15)])
    best, best_c = x.copy(), evaluate(x)
    for it in range(iters):
        pop = best + rng.normal(size=(96, x.size)) * sig
        pop[:, :NC] = np.clip(pop[:, :NC], lo, hi)
        cs = np.array([evaluate(p) for p in pop])
        elite = pop[np.argsort(cs)[:12]]
        cand = elite.mean(axis=0)
        cc = evaluate(cand)
        if cc < best_c:
            best, best_c = cand, cc
        sig = 0.9 * sig + 0.1 * elite.std(axis=0) + 1e-4
        if verbose and it % 30 == 0:
            print(f"  iter {it}: cost {best_c:.3f}")
    if verbose:
        print("solved:"); evaluate(best, True)
    set_hand(np.clip(best[:NC], lo, hi))
    c, u = pen_pose(best)
    pose = dict(full0)
    pose.update({n: float(v) for n, v in zip(ctl, np.clip(best[:NC], lo, hi))})
    hb_pos = d.xpos[hb].copy()         # hand_base is identity-oriented at rest
    return {"hand_pose": pose,
            "pen_grip_point": (c - (S.PEN_HALF - S.GRIP_FROM_TIP) * u - hb_pos).tolist(),
            "pen_shaft_dir": u.tolist(),
            "note": f"solved by dexcraft.grip for PEN_RADIUS={S.PEN_RADIUS}; pen pose in hand_base frame"}


def _finger_tip_jacobian(m, d, qadr, joints, gid):
    """d(tip position)/d(joint angles) by finite differences (3 x n)."""
    base = S.capsule_segment(m, d, gid)[1].copy()
    J = np.zeros((3, len(joints)))
    for k, n in enumerate(joints):
        q = d.qpos[qadr[n]]
        d.qpos[qadr[n]] = q + 1e-4
        mujoco.mj_kinematics(m, d)
        J[:, k] = (S.capsule_segment(m, d, gid)[1] - base) / 1e-4
        d.qpos[qadr[n]] = q
    mujoco.mj_kinematics(m, d)
    return J


def squeeze_targets(grip, depth=0.004):
    """Servo targets that press each fingertip `depth` metres into the pen
    axis (joint-space least-squares through the fingertip Jacobian)."""
    m = S.build_model(grip=None)
    d = mujoco.MjData(m)
    jn = S.names(m, mujoco.mjtObj.mjOBJ_JOINT)
    qadr = {n: m.jnt_qposadr[jn.index(n)] for n in S.HAND_JOINTS}
    S.reset_home(m, d)
    for n, v in grip["hand_pose"].items():
        d.qpos[qadr[n]] = v
    mujoco.mj_kinematics(m, d)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
    u = np.asarray(grip["pen_shaft_dir"])
    c = d.xpos[hb] + np.asarray(grip["pen_grip_point"]) + (S.PEN_HALF - S.GRIP_FROM_TIP) * u
    servo = dict(grip["hand_pose"])
    for f in S.GRASP_FINGERS:
        joints = [n for n in S.HAND_JOINTS if n.startswith(f + "_")]
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_dip_geom")
        a1, a2 = S.capsule_segment(m, d, gid)
        tip = 0.5 * (a1 + a2)
        v = tip - c
        v -= (v @ u) * u                       # from the shaft axis to the fingertip
        push = -v / (np.linalg.norm(v) + 1e-9) * depth
        J = _finger_tip_jacobian(m, d, qadr, joints, gid)
        dq = np.linalg.pinv(J, rcond=1e-2) @ push
        for n, dqi in zip(joints, dq):
            lo, hi = S.JOINT_RANGE[n]
            servo[n] = float(np.clip(grip["hand_pose"][n] + dqi, lo, hi))
    return servo


def hold_test(grip, seconds=3.0, verbose=True, tool="pen"):
    """From the cached settled state: does the tool stay put for `seconds` at
    0.3 g / 1 g and with the arm shaking?"""
    results = {}
    key = "settled" if tool == "pen" else f"settled_{tool}"
    for label, g, shake in (("0.3g", 0.3, False), ("1g", 1.0, False), ("1g+arm", 1.0, True)):
        m = S.build_model(grip=None, tool=tool, chop_grip=grip)
        d = mujoco.MjData(m)
        m.opt.gravity[:] = [0, 0, -9.81 * g]
        an = S.names(m, mujoco.mjtObj.mjOBJ_ACTUATOR)
        d.qpos[:] = grip[key]["qpos"]
        d.ctrl[:] = grip[key]["ctrl"]
        mujoco.mj_forward(m, d)
        pen_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
        pen_set = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pen_bid}
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
        pq, _ = S.pen_joint(m)
        a0, a1 = an.index(S.ACTUATOR_OF[S.ARM_JOINTS[0]]), an.index(S.ACTUATOR_OF[S.ARM_JOINTS[1]])
        c0, c1 = d.ctrl[a0], d.ctrl[a1]
        rel = lambda: d.xmat[hb].reshape(3, 3).T @ (d.qpos[pq:pq + 3] - d.xpos[hb])
        ref = rel()
        for i in range(int(seconds / m.opt.timestep)):
            if shake:
                t = i * m.opt.timestep
                d.ctrl[a0] = c0 + 0.06 * np.sin(2 * np.pi * 0.7 * t)
                d.ctrl[a1] = c1 + 0.04 * np.sin(2 * np.pi * 1.1 * t)
            S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
        drift = np.linalg.norm(rel() - ref) * 1000
        ncon = sum(1 for i in range(d.ncon) if d.contact[i].geom1 in pen_set or d.contact[i].geom2 in pen_set)
        ok = drift < 5.0 and ncon >= 2
        results[label] = dict(drift_mm=float(drift), contacts=ncon, holds=bool(ok))
        if verbose:
            print(f"hold test [{tool}] {label:7s}: drift {drift:5.1f} mm over {seconds:.0f} s, {ncon} contacts "
                  f"-> {'HOLDS' if ok else 'DROPS'}")
    return results


def writing_wrist(grip):
    """Wrist flexion that pitches the held pen to PEN_PITCH_DEG above the paper."""
    m = S.build_model(grip=None)
    d = mujoco.MjData(m)
    jn = S.names(m, mujoco.mjtObj.mjOBJ_JOINT)
    wf = m.jnt_qposadr[jn.index("wrist_flex")]
    S.reset_home(m, d)
    for n, v in grip["hand_pose"].items():
        d.qpos[m.jnt_qposadr[jn.index(n)]] = v
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
    best, best_err = 0.0, 1e9
    for ang in np.linspace(-1.0, 1.0, 201):
        d.qpos[wf] = ang
        mujoco.mj_forward(m, d)
        u = d.xmat[hb].reshape(3, 3) @ np.asarray(grip["pen_shaft_dir"])   # tip -> back, world
        pitch = np.degrees(np.arcsin(np.clip(u[2], -1, 1)))                  # back end above tip
        err = abs(pitch - S.PEN_PITCH_DEG)
        if err < best_err:
            best, best_err = ang, err
    return float(best)


def settled_state(grip, settle=1.0, tool="pen", chop_lower=True):
    """Place, assist, settle (in the writing posture): the canonical initial
    state (qpos, ctrl) that env.reset() restores instantly.

    Always settled with the arm at ARM_BASE_XML: the stored qpos carries the pen's WORLD
    pose, and scene.settled_hand() reconstructs the hand at that base to recover the
    hand-frame pen pose. Settling at any other base would place the pen far from the
    hand at reset."""
    base_now = S.ARM_BASE_POS
    S.ARM_BASE_POS = S.ARM_BASE_XML.copy()
    try:
        m = S.build_model(grip=None, tool=tool, chop_lower=chop_lower, chop_grip=grip)
    finally:
        S.ARM_BASE_POS = base_now
    d = mujoco.MjData(m)
    jn = S.names(m, mujoco.mjtObj.mjOBJ_JOINT)
    an = S.names(m, mujoco.mjtObj.mjOBJ_ACTUATOR)
    S.reset_home(m, d)
    support = CHOP_SUPPORT_POSE if tool == "chopsticks" else {}
    servo = grip.get(f"servo_pose_{tool}", grip["servo_pose"])
    for n, v in grip["hand_pose"].items():
        d.qpos[m.jnt_qposadr[jn.index(n)]] = support.get(n, v)
        d.ctrl[an.index(n + "_act")] = support.get(n, servo[n])
    wrist = writing_wrist(grip)
    d.qpos[m.jnt_qposadr[jn.index("wrist_flex")]] = wrist
    d.ctrl[an.index("wrist_flex_act")] = wrist
    mujoco.mj_forward(m, d)
    p0 = S.place_pen(m, d, grip)
    pq, pd = S.pen_joint(m)
    q0 = d.qpos[pq + 3:pq + 7].copy()
    for _ in range(int(grip["assist_time"] / m.opt.timestep)):
        d.qpos[pq:pq + 3] = p0
        d.qpos[pq + 3:pq + 7] = q0
        d.qvel[pd:pd + 6] = 0.0
        S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
    for _ in range(int(settle / m.opt.timestep)):
        S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
    tip = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")]
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pitch = float(np.degrees(np.arcsin(d.xmat[pb].reshape(3, 3)[2, 2])))
    R = d.xmat[hb].reshape(3, 3)
    return {"qpos": d.qpos.tolist(), "ctrl": d.ctrl.tolist(),
            "joint_names": jn, "actuator_names": an, "wrist_flex": wrist, "pen_pitch_deg": pitch,
            "tip_in_hand": (R.T @ (tip - d.xpos[hb])).tolist(),
            "tool_axis_in_hand": (R.T @ d.xmat[pb].reshape(3, 3)[:, 2]).tolist(),    # tip -> back
            "hand_xmat": R.tolist()}


def chopstick_layout(grip):
    """Lay out the fixed lower stick from the settled upper stick (no lower stick
    present): same axis direction, offset horizontally (world, at the reset pose)
    toward the ring finger so both tips are level, converging toward the tips."""
    st = settled_state(grip, tool="chopsticks", chop_lower=False)
    R = np.asarray(st["hand_xmat"])
    tip_u = np.asarray(st["tip_in_hand"])
    u = np.asarray(st["tool_axis_in_hand"]); u /= np.linalg.norm(u)            # tip -> back, hand frame
    back_u = tip_u + P.CHOP_LEN * u
    e_w = np.cross([0, 0, 1.0], R @ u)                                         # horizontal, perpendicular to the stick
    e_w /= np.linalg.norm(e_w)
    e = R.T @ e_w
    if e @ np.array([1.0, 0, -0.3]) < 0:                                       # toward the ulnar / palmar side
        e = -e
    return {"back": (back_u + S.CHOP_GAP_BACK * e).tolist(), "tip": (tip_u + S.CHOP_GAP_TIP * e).tolist(),
            "upper_tip": tip_u.tolist(), "upper_axis": u.tolist()}


def refine_servo(grip, seed=0, iters=50, pop=40, verbose=True, tool="pen", load=0.7, settle_s=1.5):
    """Physics-in-the-loop search over the grasp-finger servo targets:
    after an assisted close the free tool must stay put, all three fingertips
    must touch it, and a `load` N lateral force at the tip must not pivot it.
    tool="chopsticks" re-solves for the tip-heavy stick (result stored as
    servo_pose_chopsticks; solved WITHOUT the fixed lower stick so the fingers
    alone must hold the upper one)."""
    m = S.build_model(grip=None, tool=tool, chop_lower=False)
    d = mujoco.MjData(m)
    jn = S.names(m, mujoco.mjtObj.mjOBJ_JOINT)
    an = S.names(m, mujoco.mjtObj.mjOBJ_ACTUATOR)
    ctl = [n for n in S.HAND_JOINTS if n.split("_")[0] in S.GRASP_FINGERS]
    lo = np.array([S.JOINT_RANGE[n][0] for n in ctl])
    hi = np.array([S.JOINT_RANGE[n][1] for n in ctl])
    support = CHOP_SUPPORT_POSE if tool == "chopsticks" else {}
    pose = np.array([grip["hand_pose"][n] for n in ctl])
    x0 = np.array([grip.get(f"servo_pose_{tool}", grip["servo_pose"])[n] for n in ctl])
    wrist = writing_wrist(grip)
    pq, pd = S.pen_joint(m)
    pen_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pen_set = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pen_bid}
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
    tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")
    # per finger, so a padded tip touching on 5 spheres still counts as one finger
    tip_sets = [S.fingertip_geoms(m, f) for f in S.GRASP_FINGERS]
    n_touching = lambda hit: sum(1 for gs in tip_sets if gs & hit)
    dt = m.opt.timestep

    def rel():
        R = d.xmat[hb].reshape(3, 3)
        return R.T @ (d.qpos[pq:pq + 3] - d.xpos[hb]), R.T @ d.xmat[pen_bid].reshape(3, 3)[:, 2]

    def rollout(x, show=False):
        S.reset_home(m, d)
        for n, v in grip["hand_pose"].items():
            d.qpos[m.jnt_qposadr[jn.index(n)]] = support.get(n, v)
            d.ctrl[an.index(n + "_act")] = support.get(n, v)
        for n, v in zip(ctl, np.clip(x, lo, hi)):
            d.ctrl[an.index(n + "_act")] = v
        d.qpos[m.jnt_qposadr[jn.index("wrist_flex")]] = wrist
        d.ctrl[an.index("wrist_flex_act")] = wrist
        mujoco.mj_forward(m, d)
        p0 = S.place_pen(m, d, grip)
        q0 = d.qpos[pq + 3:pq + 7].copy()
        for _ in range(int(0.15 / dt)):
            d.qpos[pq:pq + 3] = p0
            d.qpos[pq + 3:pq + 7] = q0
            d.qvel[pd:pd + 6] = 0.0
            S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
        for _ in range(int(0.3 / dt)):
            S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
        p_s, a_s = rel()
        for _ in range(int(settle_s / dt)):               # long static settle: no creep / roll
            S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
        p1, a1 = rel()
        settle_drift = np.linalg.norm(p1 - p_s) * 1000
        settle_tilt = np.degrees(np.arccos(np.clip(a1 @ a_s, -1, 1)))
        touching = {d.contact[i].geom1 for i in range(d.ncon) if d.contact[i].geom2 in pen_set} | \
                   {d.contact[i].geom2 for i in range(d.ncon) if d.contact[i].geom1 in pen_set}
        n_tip = n_touching(touching)
        # lateral load at the tip, two directions
        worst_tilt, worst_drift = 0.0, 0.0
        for f in (np.array([load, 0, 0]), np.array([0, load, 0]), np.array([-0.7 * load, -0.7 * load, 0])):
            for _ in range(int(0.3 / dt)):
                d.xfrc_applied[pen_bid] = 0.0
                d.qfrc_applied[:] = 0.0
                mujoco.mj_applyFT(m, d, f, np.zeros(3), d.site_xpos[tip_sid], pen_bid, d.qfrc_applied)
                S.compensate_arm_gravity(m, d, S.arm_dofs(m)); mujoco.mj_step(m, d)
            p2, a2 = rel()
            worst_tilt = max(worst_tilt, np.degrees(np.arccos(np.clip(a1 @ a2, -1, 1))))
            worst_drift = max(worst_drift, np.linalg.norm(p2 - p1) * 1000)
        still = {d.contact[i].geom1 for i in range(d.ncon) if d.contact[i].geom2 in pen_set} | \
                {d.contact[i].geom2 for i in range(d.ncon) if d.contact[i].geom1 in pen_set}
        n_tip2 = n_touching(still)
        cost = 3.0 * settle_drift + 3.0 * settle_tilt + worst_drift + 2.0 * worst_tilt \
            + 20.0 * (3 - n_tip) + 20.0 * (3 - n_tip2) + 0.5 * np.sum(((x - pose) / 0.3) ** 2)
        if show:
            print(f"  fingertips {n_tip}/3 (after load {n_tip2}/3)  static 1.5 s: drift {settle_drift:.1f} mm "
                  f"tilt {settle_tilt:.1f} deg  loaded: drift {worst_drift:.1f} mm tilt {worst_tilt:.1f} deg")
        return cost

    if verbose:
        print("servo refine start:"); rollout(x0, True)
    rng = np.random.default_rng(seed)
    sig = np.full(len(ctl), 0.08)
    best, best_c = x0.copy(), rollout(x0)
    for it in range(iters):
        P = np.clip(best + rng.normal(size=(pop, len(ctl))) * sig, lo, hi)
        cs = np.array([rollout(p) for p in P])
        elite = P[np.argsort(cs)[:6]]
        cand = elite.mean(axis=0)
        cc = rollout(cand)
        if cc < best_c:
            best, best_c = cand, cc
        sig = 0.85 * sig + 0.15 * elite.std(axis=0) + 1e-3
        if verbose and it % 10 == 0:
            print(f"  iter {it}: cost {best_c:.2f}")
    if verbose:
        print("servo refine solved:"); rollout(best, True)
    out = dict(grip)
    key = "servo_pose" if tool == "pen" else f"servo_pose_{tool}"
    out[key] = dict(grip["hand_pose"])
    out[key].update({n: float(v) for n, v in zip(ctl, np.clip(best, lo, hi))})
    return out


if __name__ == "__main__":
    grip = solve()
    grip["servo_pose"] = squeeze_targets(grip)
    grip["assist_time"] = 0.15
    grip = refine_servo(grip)
    grip["settled"] = settled_state(grip)
    grip["settled_brush"] = settled_state(grip, tool="brush")
    grip = refine_servo(grip, tool="chopsticks", load=0.3, settle_s=3.0)   # tip-heavy stick needs its own squeeze
    grip["chop_lower"] = chopstick_layout(grip)
    grip["settled_chopsticks"] = settled_state(grip, tool="chopsticks")
    json.dump(grip, open(S.GRIP_JSON, "w"), indent=1)
    print("wrote", S.GRIP_JSON, "| settled pen tip in hand frame (mm):",
          np.round(np.array(grip["settled"]["tip_in_hand"]) * 1000, 1),
          f"| wrist_flex {np.degrees(grip['settled']['wrist_flex']):.0f} deg, pen pitch {grip['settled']['pen_pitch_deg']:.0f} deg")
    hold_test(grip)
    hold_test(grip, tool="brush")
    hold_test(grip, tool="chopsticks")
