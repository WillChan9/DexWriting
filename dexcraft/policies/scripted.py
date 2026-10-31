"""Scripted baseline for the four-group / position+force interface.

It sees what a learned policy sees: the task arrives only as the map in
`obs["task"]["map"]`. It decodes that picture into an order to draw it
(`planning.map_to_strokes`), drives the tip along the result with damped
least-squares differential IK on the six arm joints, and presses with the FORCE
channel - a feed-forward joint torque `J^T f` for the press force the map asks
for. The action is joint torque, so it plans in joint positions and closes its
own PD loop around that plan, adding the press torque as a feed-forward term.

Apart from the contact force it reads in `observe()`, nothing here is privileged;
it is a reference controller, not a ceiling.
"""
import mujoco
import numpy as np

from .. import scene as S
from ..brush import BRUSHES, brush_config, depth_for_force
from ..env import LIFT_HEIGHT, NO_LOAD_SPEED, NU, unpack
from ..planning import map_to_strokes
from .base import Policy

# closing the upper chopstick (L6): extending index / middle and abducting the
# middle pad swings its tip toward the fixed lower stick in this pocket. pip is a
# 1-DoF hinge on every finger but the thumb, so the middle finger's abduction is
# on its mcp.
CLOSE_DQ = {"middle_mcp_joint": -0.2, "thumb_pip_abd_joint": 0.2,
            "index_mcp_joint": -0.15, "index_dip_joint": -0.15, "middle_mcp_abd_joint": 0.3}
SQUEEZE_MAX = 0.55
# The fingers write, the arm carries. Driver set for the finger block = the
# tripod's own joints; ring/pinky take no part (idle_lock servoes them to the grip
# pose anyway). Measured tip sensitivity at the settled grip over this set:
# 87/66/25 mm per radian, i.e. a 20 deg coordinated excursion walks the tip ~30 mm
# - one glyph - so the arm does not have to draw the strokes itself.
FINGER_DRIVERS = ("thumb_", "index_", "middle_")
# OFF by default: the reference controller writes with the arm. Turning the block
# on (finger_span > 0) halves the arm's share of the tip motion but costs grip -
# 28/48 drops against 8/48, measured over 8 writing tasks x 6 seeds - so it is an
# option, not the baseline.
FINGER_SPAN = 0.0         # rad the drivers may wander from the settled grip pose
ABD_COST = 3.0            # abduction spreads the tripod: expensive, not forbidden
# The wrist outguns the fingers 165 vs 87 mm/rad at the tip, and a least-norm
# solve hands the work to whichever group is cheapest per millimetre - price the
# wrist up or it takes the whole stroke and finger_share never moves (measured:
# wrist 0.61 / fingers 0.09 at WRIST_COST 1).
WRIST_COST = 8.0
# PD gains for the 25 Hz torque loop, tuned by sweep (position-servo values are
# far too stiff/damped when held for a whole control step)
KP_SCALE = 0.5
KV_SCALE = 0.10


RAMP_TICKS = 5                       # control ticks over which the arm's per-tick step opens up
                                     # from a fifth to full after a reset (0.2 s at 25 Hz)


class ScriptedTracker(Policy):
    control_mode = "hybrid"
    architecture = "generalist"
    demos = "none"
    compute = "none (hand-written)"
    control_hz = 25

    def __init__(self, gain=1.2, lookahead=0, rot_gain=0.5, damping=1e-3, max_step=0.006,
                 lead=4.0, force_gain=0.6, capture=0.0035, stall_limit=45,
                 hand_damping=3e-4, recentre=0.5, wrist_cost=WRIST_COST, finger_span=FINGER_SPAN,
                 axis_gain=0.0, leak=0.05, preload_scale=1.0):
        self.gain, self.lookahead, self.max_step, self.lead = gain, lookahead, max_step, lead
        self.rot_gain, self.damping, self.capture = rot_gain, damping, capture
        self.force_gain = force_gain
        self.hand_damping, self.recentre = hand_damping, recentre
        self.finger_span, self.axis_gain, self.leak = finger_span, axis_gain, leak
        self.preload_scale = preload_scale
        self.stall_limit = stall_limit
        self.m = S.build_model()
        self.d = mujoco.MjData(self.m)
        jn = S.names(self.m, mujoco.mjtObj.mjOBJ_JOINT)
        self.qadr = np.array([self.m.jnt_qposadr[jn.index(j)] for j in S.ALL_JOINTS])
        self.dadr = np.array([self.m.jnt_dofadr[jn.index(j)] for j in S.ALL_JOINTS])
        self.hb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
        self.max_vel = np.array([S.MAX_VEL[j] for j in S.ALL_JOINTS])
        self.tau = self.m.actuator_forcerange[:, 1].copy()
        # this teacher plans in joint positions; the action is torque, so it closes
        # its own PD around that plan using the same gains the env's servo used
        self.kp = self.m.actuator_gainprm[:, 0].copy() * KP_SCALE
        # A 25 Hz policy cannot damp like the 500 Hz servo does: a damping torque
        # held constant for a whole 40 ms control step injects energy instead of
        # removing it. Scale it down to what the control rate can actually support.
        self.kv = -self.m.actuator_biasprm[:, 2].copy() * KV_SCALE
        self.n_arm = len(S.ARM_JOINTS)
        self.n_wrist = len(S.WRIST_JOINTS)
        # wrist + tripod: the joints that write. Ordered as in ALL_JOINTS so the
        # slice indexes straight into the action / q_target vectors.
        self.hand_drv = np.array([i for i, n in enumerate(S.ALL_JOINTS)
                                  if self.n_arm <= i < self.n_arm + self.n_wrist
                                  or n.startswith(FINGER_DRIVERS)])
        self.hand_w = np.array([wrist_cost if i < self.n_arm + self.n_wrist else
                                ABD_COST if "_abd" in S.ALL_JOINTS[i] else 1.0
                                for i in self.hand_drv])
        self.lo_drv = np.array([S.JOINT_RANGE[S.ALL_JOINTS[i]][0] for i in self.hand_drv])
        self.hi_drv = np.array([S.JOINT_RANGE[S.ALL_JOINTS[i]][1] for i in self.hand_drv])
        self.grasp_bids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_dip")
                           for f in S.GRASP_FINGERS]
        self.close_dir = np.zeros(NU)
        for j, v in CLOSE_DQ.items():
            if j in S.ALL_JOINTS:            # tolerate a hand built with fewer DoF
                self.close_dir[S.ALL_JOINTS.index(j)] = v
        self.dt = 1.0 / self.control_hz
        self.reset({})

    def reset(self, task_info):
        self.level = task_info.get("level", "")
        self.tool = task_info.get("tool", "pen")
        self._brush_cfg = brush_config() if self.tool == "brush" else None
        self._brush_cfgs = [brush_config(n) for n in BRUSHES] if self.tool == "brush" else None
        self._paint = False                     # L5: set from the observation on the first act
        self.dt = task_info.get("dt", self.dt)
        self.R0 = None
        self.plan = None
        self.i = 0
        self.stall = 0
        self.ticks = 0                           # control ticks since reset, for the step ramp
        self.stage, self.stage_t, self.squeeze = 0, 0.0, 0.0
        self.q_hand0, self.z_lift0 = None, None
        # The joint plan this teacher servos to. It is seeded from the live qpos, plus
        # the solved grip's SQUEEZE PRELOAD on the finger joints: the offset between
        # that grip's servo target and its qpos is what actually holds the tool (up to
        # 2.3 N.m on the grasp fingers), and starting flat drops it. The arm is
        # excluded - reset puts it in a task pose, not the grip pose.
        self.q_target = None
        self.q_grip0 = None          # the settled grip pose the drivers box around
        self.axis0 = None            # the tool's axis in the hand when writing began
        g = S.load_grip()
        key = "settled" if self.tool == "pen" else f"settled_{self.tool}"
        self.preload = np.zeros(NU)
        if g and key in g:
            off = np.asarray(g[key]["ctrl"], dtype=float) - np.asarray(g[key]["qpos"], dtype=float)[:NU]
            # Finger-driven strokes put tangential load on the pads, and the pen
            # ROLLS between them - which the slip test reads as rotation in the hand
            # and ends the episode. Squeezing harder is the direct answer.
            # Fingers only, not the wrist: the settled grip's wrist ctrl/qpos gap is
            # -0.26 rad (-14.9 deg) of pitch, so a wrist preload makes the wrist PD chase
            # a pose 15 deg below where it stands, driving the tip into the sheet on the
            # first step and biasing the press for the whole episode.
            nf = self.n_arm + self.n_wrist
            self.preload[nf:] = off[nf:] * self.preload_scale
        self.f_meas = 0.0
        self._cursor = None                      # the schedule via info when it is not in the obs
        self.act_dim = int(task_info.get("act_dim", NU))

    def observe(self, info):
        # privileged: the true tool-paper force is not in the observation.
        # This teacher force-servos on it; a learned policy has to estimate it.
        if "tool_contact_force" in info:
            self.f_meas = float(info.get("tool_contact_force", 0.0))
        if info.get("cursor") is not None:
            self._cursor = info["cursor"]

    def act(self, obs):
        a = np.asarray(self._act34(obs), np.float32)
        self.ticks += 1
        if self.act_dim > NU:
            # policy-declared end: finished when the schedule is done and the tip has arrived
            cur = self._cursor
            done = bool(cur and cur.get("finished")) and self._tip_err < 0.004
            a = np.concatenate([a, [1.0 if done else -1.0]]).astype(np.float32)
        return a

    def _build_plan(self, task_map, tip_z=None, half_m=None, center=(0.0, 0.0)):
        """Decoded strokes -> a list of (xyz target, pen_down, target force).

        Pen-down targets sit ON the surface, not inside it: depth is the force
        channel's job. Commanding position into a rigid sheet with a kp-2000 arm is
        how you get 35 N instead of 1 N. The first move is a transit at the height
        the tool is already at - diving straight at the sheet lets the differential
        IK walk the arm into an over-extended posture it cannot climb out of (with
        the long brush the elbow ends up near its limit and the tip bottoms out
        30 mm above the sheet).
        """
        z_dn, z_up = S.PAPER_TOP, S.PAPER_TOP + LIFT_HEIGHT
        plan = []
        strokes = map_to_strokes(task_map, tool=self.tool, half_m=half_m, center=center)
        if strokes and tip_z is not None and tip_z > z_up:
            p0 = strokes[0].xy[0] + S.PAPER_CENTER[:2]
            plan.append((np.array([p0[0], p0[1], float(tip_z)]), False, 0.0))
        for st in strokes:
            P = st.xy + S.PAPER_CENTER[:2]
            plan.append((np.array([P[0, 0], P[0, 1], z_up]), False, 0.0))
            for xy, f in zip(P, st.force):
                plan.append((np.array([xy[0], xy[1], z_dn]), True, float(f)))
            plan.append((np.array([P[-1, 0], P[-1, 1], z_up]), False, 0.0))
        return plan

    def _arm_jacobian(self, qpos, tip_in_hand):
        self.d.qpos[self.qadr] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        R = self.d.xmat[self.hb].reshape(3, 3)
        tip = self.d.xpos[self.hb] + R @ tip_in_hand
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jacp, jacr, tip, self.hb)
        cols = self.dadr[:self.n_arm]
        return tip, R, jacp[:, cols], np.vstack([jacp[:, cols], jacr[:, cols]])

    def _arm_step(self, dp, R, J, pos, integrate=False):
        if self.R0 is None:
            self.R0 = R.copy()
        n = np.linalg.norm(dp)
        # Ramp the step over the first few ticks. The tracker knows its target from tick 0
        # (the reset info carries the cursor), and commanding a full 6 mm of tip motion while
        # the grip is still settling out of the reset pose shakes the pen loose.
        lim = self.max_step * min(1.0, (self.ticks + 1) / float(RAMP_TICKS))
        if n > lim:
            dp = dp * (lim / n)
        Rerr = self.R0 @ R.T
        e_rot = 0.5 * np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
        if J.shape[1] >= 6:
            e = np.concatenate([dp, self.rot_gain * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), e)
        else:
            # 4-DoF arm (pan, lift, elbow, forearm roll): it cannot hold orientation on its
            # own, and feeding it the rotation error only corrupts the position solve -
            # measured 5/6 drops inside 1.5 s. Position rows only; the wrist PD around
            # q_target (seeded from the placed pose) holds the hand orientation.
            Jp = J[:3]
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + self.damping * np.eye(3), dp)
        step = self.max_vel[:self.n_arm] * self.dt
        dq /= max(1.0, np.max(np.abs(dq) / step))
        if integrate:
            # While the normal axis is force-controlled the arm is loaded, so qpos
            # lags its servo target on purpose. The lead term below reads that lag
            # as wind-up and unwinds the target - which lifts the tool off the sheet
            # and kills the contact it was told to make. Under load, integrate.
            desired = pos["arm_servo_target"] + dq
        else:
            desired = pos["qpos"][:self.n_arm] + self.lead * dq
        return np.clip((desired - pos["arm_servo_target"]) / step, -1, 1)

    def _hand_jacobian(self, tip):
        """Composite tip Jacobian for the wrist + finger drivers.

        The tool is a FREE body held by three fingertips, so its motion follows
        theirs; a point anchored on hand_base has structurally zero columns for
        every finger joint (env._tip_velocity_split hit exactly that and read
        finger_share 0.00 for every policy). Average the per-fingertip Jacobians.
        """
        jac, tmp = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for b in self.grasp_bids:
            mujoco.mj_jac(self.m, self.d, tmp, None, tip, b)
            jac += tmp
        jac /= max(len(self.grasp_bids), 1)
        return jac[:, self.dadr[self.hand_drv]]

    def _grasp_frame(self):
        """Tripod geometry the writing motion has to respect, all in the HAND frame.

        Returns (N, Jw, axis_cols):
          N   driver directions that leave the three pad gaps unchanged - a
              least-norm solve without this extends the index while flexing the
              thumb, the gap opens and the pen is gone (4/4 drops at wrist_cost 20);
          Jw  how the triad ROTATES per driver radian. Constraining rotation away
              outright costs too much (it is how the fingers reach), so instead it
              is closed-loop: _hand_step spends it undoing the drift the drop test
              measures. Hand frame, because world-frame references go stale the
              moment the wrist turns.
        Only finger columns can change the relative geometry; the wrist is upstream
        of every finger, so its columns are zero and it stays free either way.
        """
        nd = len(self.hand_drv)
        cols = self.dadr[self.hand_drv]
        Rh = self.d.xmat[self.hb].reshape(3, 3)
        P, Jf = [], []
        for b in self.grasp_bids:
            tmp = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, tmp, None, self.d.xpos[b], b)
            P.append(Rh.T @ (self.d.xpos[b] - self.d.xpos[self.hb]))
            Jf.append(Rh.T @ tmp[:, cols])
        f = slice(self.n_wrist, nd)
        D = np.zeros((6, nd))                              # d(difference vectors)/dq
        D[0:3, f] = (Jf[1] - Jf[0])[:, f]
        D[3:6, f] = (Jf[2] - Jf[0])[:, f]
        gap = np.zeros((3, nd))                            # d(pad distances)/dq
        for r, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            u = P[i] - P[j]
            n = np.linalg.norm(u)
            if n > 1e-9:
                gap[r, f] = (u / n) @ (Jf[i] - Jf[j])[:, f]
        _, sv, Vt = np.linalg.svd(gap)
        rank = int((sv > 1e-6 * max(sv[0], 1e-12)).sum())
        N = Vt[rank:].T
        # d(difference) = omega x difference = -skew(difference) omega, so the
        # triad's angular velocity per driver radian is pinv(B) D.
        skew = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        B = np.vstack([-skew(P[1] - P[0]), -skew(P[2] - P[0])])
        return N, np.linalg.pinv(B) @ D

    def _hand_step(self, dp, tip, axis_err):
        """Wrist + tripod tracking. Returns (velocity fractions, tip motion delivered).

        IN-PLANE ONLY: the sheet normal is force-controlled (the press creep in
        act()), and fingers chasing z would fight it - they would lift the tool off
        the contact the press just made. Weighted damped least squares inside the
        gap-preserving null space, plus a term that spends the triad's rotation
        freedom holding the tool's axis where it started in the hand: pen rotation
        is what actually ends these episodes (dax 0.56 against 0.518 allowed, while
        the drivers were only 8 deg off their reference - it is the tool rolling
        between the pads, not the fingers overreaching). Then two clamps: one
        control step's travel, and a box finger_span wide around the grip pose.

        dp None = pen up: unwind onto the grip pose instead, so the next stroke
        starts with the full finger range and the transit stays the arm's job.
        """
        q = self.q_target[self.hand_drv]
        ref = self.q_grip0[self.hand_drv]
        step = self.max_vel[self.hand_drv] * self.dt
        if dp is None:
            return np.clip((ref - q) / step, -1, 1) * self.recentre, np.zeros(3)
        e = np.array([dp[0], dp[1], 0.0])
        n = np.linalg.norm(e)
        if n > self.max_step:
            e *= self.max_step / n
        N, Jw = self._grasp_frame()
        Jf = self._hand_jacobian(tip)
        J = Jf @ N
        # weighted least norm INSIDE the null space: min (Ny)'W(Ny) s.t. JNy = e.
        # A diagonal weight on the reduced coordinates does not work - the basis
        # mixes wrist and finger columns, so the group prices wash out and the
        # wrist keeps the whole stroke (measured: finger_share 0.13 either way).
        Minv = np.linalg.inv(N.T @ np.diag(self.hand_w) @ N + 1e-9 * np.eye(N.shape[1]))
        y = Minv @ J.T @ np.linalg.solve(J @ Minv @ J.T + self.hand_damping * np.eye(3), e)
        dq = N @ y
        if axis_err is not None:
            dq += self.axis_gain * (N @ np.linalg.lstsq(Jw @ N, axis_err, rcond=None)[0])
        # Leak back onto the grip pose every step. Without it the drivers walk to
        # the edge of their box in ~10 steps and pin there (measured: maxdev 20.1
        # of 20.0 deg allowed, from t=100 on), so the fingers stop being able to
        # write at all. Leaking lets them oscillate around the grip pose and leaves
        # the slow drift to the arm - which is the human split.
        dq += self.leak * (ref - q)
        dq = np.clip(dq, -step, step)
        dq = np.clip(q + dq, np.maximum(ref - self.finger_span, self.lo_drv),
                     np.minimum(ref + self.finger_span, self.hi_drv)) - q
        return np.clip(dq / step, -1, 1), Jf @ dq

    def _to_action(self, pos, dq):
        """Joint plan -> the hybrid action.

        Arm (6): passed straight through as the position-target step the env
        integrates. Hand + wrist (37): the plan is integrated here and a PD closed
        around it, returned as a duty cycle - a fraction of the torque available at
        the current joint speed."""
        n = self.n_arm
        q = pos["qpos"]
        qd = pos["qvel_x0.1"] * 10.0
        qd_target = dq * self.max_vel                 # the velocity the plan asks for
        self.q_target = self.q_target + qd_target * self.dt
        # damp the velocity ERROR, not the velocity. Damping absolute velocity fights
        # the motion the plan is asking for: measured at 70 N.m mean against a 28 N.m
        # limit, which saturated the command 69% of the time.
        tau = self.kp * (self.q_target - q) + self.kv * (qd_target - qd)
        avail = self.tau * np.clip(1.0 - np.abs(qd) / (NO_LOAD_SPEED * self.max_vel), 0.0, 1.0)
        a = np.clip(tau / np.maximum(avail, 1e-9), -1.0, 1.0)
        a[:n] = dq[:n]                                # arm: position step, not torque
        return a.astype(np.float32)

    def _act34(self, obs):
        self._tip_err = 1.0
        if self.tool == "chopsticks":
            return self._act_chopsticks(obs)
        pos = unpack(obs["position"], "position")
        ext = unpack(obs["external"], "external")
        if self._cursor is not None and not np.any(ext["write_target_in_hand"]):
            ext = dict(ext)                      # the cursor came through info, not the obs
            ext["write_target_in_hand"] = np.asarray(self._cursor["write_target_in_hand"], float)
            ext["write_ahead_near"] = np.asarray(self._cursor["ahead_near"], float)
            ext["write_ahead_far"] = np.asarray(self._cursor["ahead_far"], float)
            ext["write_pen_down"] = np.array([float(self._cursor["pen_down"])])
        if self.q_target is None:
            self.q_target = pos["qpos"] + self.preload
        if self.plan is None:
            r6 = ext["hand_rot6"]
            R0 = np.stack([r6[:3], r6[3:], np.cross(r6[:3], r6[3:])], axis=1)
            tip0 = ext["hand_pos"] + R0 @ ext["tip_pos_in_hand"]
            tv = unpack(obs["task"], "task")
            self.plan = self._build_plan(obs["task"]["map"], tip_z=float(tip0[2]),
                                         half_m=tv["map_half_m"], center=tv["map_center_m"])
            self.i = 0
            dn = [(q[:2], f) for q, d, f in self.plan if d]
            self._down_pts = np.array([q for q, _ in dn]) if dn else None
            self._down_f = np.array([f for _, f in dn]) if dn else None
        dq = np.zeros(NU)
        if not self.plan:
            return self._to_action(pos, dq)

        tip, R, jacp, J = self._arm_jacobian(pos["qpos"], ext["tip_pos_in_hand"])
        # STRICT-DIRECTION mode: the task's schedule is the instruction, and the
        # observation carries it every step (write_target_in_hand + write_pen_down).
        # Follow the commanded point directly - stroke order and direction are then
        # compliant by construction. The waypoints decoded from the map serve only as
        # a force-target lookup table.
        r6 = ext["hand_rot6"]
        R0 = np.stack([r6[:3], r6[3:], np.cross(r6[:3], r6[3:])], axis=1)
        cmd = ext["hand_pos"] + R0 @ ext["write_target_in_hand"]
        down = float(ext["write_pen_down"][0]) > 0.5
        z_dn, z_up = S.PAPER_TOP, S.PAPER_TOP + LIFT_HEIGHT
        if self.tool == "brush" and "paint" in obs["external"]:
            # L5: the timeline also visits the wells and the rack, so follow its height when the
            # pen is up, and read the depth law from whichever brush the rack handed over
            self._paint = bool(np.any(obs["task"]["palette"]))
            if self._paint:
                self._brush_cfg = self._brush_cfgs[int(np.argmax(obs["external"]["paint"][3:6]))]
        z_free = cmd[2] if self._paint else max(z_up, min(tip[2], cmd[2]))
        target = np.array([cmd[0], cmd[1], z_dn if down else z_free])
        self._tip_err = float(np.linalg.norm((cmd - tip)[:2]))
        f_target = 1.0
        if down and self._down_pts is not None and len(self._down_pts):
            k = int(np.argmin(np.linalg.norm(self._down_pts - target[:2], axis=1)))
            f_target = float(self._down_f[k])
        f_meas = self.f_meas

        # hybrid split: position control in the plane, force control along the
        # sheet normal. The only vertical position command is a creep to make
        # first contact, or a small retreat if the press runs away.
        dp = self.gain * (target - tip)
        h = float(ext["tip_above_sheet"][0])
        if down and self.tool == "brush":
            # The tuft is a reduced-order model with no contact to feel for: pressing
            # IS a depth. Servo the nominal tip to the depth the target force implies.
            z_t = S.PAPER_TOP - depth_for_force(self._brush_cfg, f_target)
            dp[2] = float(np.clip(self.gain * (z_t - tip[2]), -0.003, 0.003))
        elif down:
            if h >= 0.006:
                # taper the approach: diving at full step rate into a rigid sheet
                # with a kp-2000 arm spikes to 20+ N and throws the pen
                dp[2] = max(dp[2], -min(0.004, 0.25 * h))
            elif f_meas <= 0.02:
                dp[2] = -0.0012
            elif f_meas > 2.5 * max(f_target, 0.2):
                dp[2] = 0.0008
            else:
                dp[2] = 0.0
        # the wrist + tripod get first refusal on the in-plane error and the arm
        # is handed only what they could not reach.
        if self.q_grip0 is None:
            self.q_grip0 = self.q_target.copy()
        axis = ext["tool_axis_in_hand"]
        if self.axis0 is None:
            self.axis0 = axis.copy()
        # small-angle rotation that puts the tool's axis back where it started
        axis_err = np.cross(axis, self.axis0)
        dq_h, got = self._hand_step(dp if down else None, tip, axis_err)
        dq[self.hand_drv] = dq_h
        dq[:self.n_arm] = self._arm_step(dp - got, R, J, pos, integrate=down and h < 0.006)

        # FORCE channel: tau = J^T f with f pressing into the sheet, plus a
        # proportional trim on the force the tactile group actually reports
        if down and f_target > 0:
            f_cmd = f_target + self.force_gain * (f_target - f_meas)
            # the arm is position-controlled, so the press comes from the
            # vertical creep above, not from a J^T f feed-forward on the arm
            pass
        return self._to_action(pos, dq)

    def _act_chopsticks(self, obs):
        pos = unpack(obs["position"], "position")
        ext = unpack(obs["external"], "external")
        if self.q_target is None:
            self.q_target = pos["qpos"] + self.preload
        if self.q_hand0 is None:
            self.q_hand0 = self.q_target.copy()
        dq = np.zeros(NU)
        tip_h, obj, tgt = ext["tip_pos_in_hand"], ext["object_pos_in_hand"], ext["target_pos_in_hand"]
        pinched = self.f_meas > 0
        self.stage_t += self.dt
        if self.stage == 0:
            goal = obj - tip_h + np.array([0, 0, 0.05])
            if np.linalg.norm(goal) < 0.008:
                self.stage, self.stage_t = 1, 0.0
        elif self.stage == 1:
            goal = obj - tip_h
            if np.linalg.norm(goal) < 0.004 or self.stage_t > 4.0:
                self.stage, self.stage_t = 2, 0.0
        elif self.stage == 2:
            goal = np.zeros(3)
            self.squeeze = min(self.squeeze + 0.5 * self.dt, SQUEEZE_MAX)
            if pinched and self.stage_t > 0.4:
                self.stage, self.stage_t, self.z_lift0 = 3, 0.0, float(ext["tip_above_sheet"][0])
            elif self.stage_t > 2.5:
                self.squeeze, self.stage, self.stage_t = 0.0, 0, 0.0
        elif self.stage == 3:
            goal = np.array([0.0, 0.0, 0.05])
            if float(ext["tip_above_sheet"][0]) - self.z_lift0 > 0.045 or self.stage_t > 2.0:
                self.stage, self.stage_t = 4, 0.0
        elif self.stage == 4:
            goal = tgt - tip_h
            if np.linalg.norm(goal) < 0.008 or self.stage_t > 6.0:
                self.stage, self.stage_t = 5, 0.0
        elif self.stage == 5:
            goal = np.zeros(3)
            self.squeeze = max(self.squeeze - 1.5 * self.dt, 0.0)
            if self.stage_t > 0.8:
                self.stage, self.stage_t = 6, 0.0
        else:
            goal = np.array([0.0, 0.0, 0.04]) if self.stage_t < 1.0 else np.zeros(3)

        tip, R, jacp, J = self._arm_jacobian(pos["qpos"], tip_h)
        dq[:self.n_arm] = self._arm_step(self.gain * (R @ goal), R, J, pos)
        step = self.max_vel * self.dt
        h = slice(self.n_arm + len(S.WRIST_JOINTS), NU)
        want = self.q_hand0[h] + self.squeeze * self.close_dir[h]
        dq[h] = np.clip((want - self.q_target[h]) / step[h], -1, 1)
        return self._to_action(pos, dq)


def make_policy(**kw):
    return ScriptedTracker(**kw)
