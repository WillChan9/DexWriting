"""Hybrid controller: IK arm + learned grip, optionally improved by a residual policy.

    base   = HybridBase(grip_ckpt)              # arm from the scripted IK, wrist + fingers from a
                                                #   learned hold policy; no privileged signal anywhere
    policy = HybridPolicy(base, residual_ckpt, bounds)   # a = clip(base(obs) + B * residual(obs))

    DEXCRAFT_GRIP_CKPT=checkpoints/grasp_v10_format.pt [DEXCRAFT_RESIDUAL_CKPT=...] \\
        python -m dexcraft.evaluate --policy dexcraft.policies.hybrid:make_policy ...

The scripted expert's vertical creep is a force servo on the true contact force (`info`).
Here it is driven by the observed tip height instead: creep down until the tip is on the
sheet, then hold. How hard to press is left to the residual, learned from touch.
"""
import os

import numpy as np

from ..env import GROUP_DIM, LAYOUT, N_ARM, N_WRIST, NU, unpack

# The ten cursor entries of the external group. The grip policy is trained on L0 hold tasks, which
# have no strokes and so a zero cursor: its weights on these columns never receive a gradient and
# sit at their initialisation. Feeding them a live cursor perturbs the finger torques enough to
# drop the pen. The grip therefore sees the observation it was trained on; the cursor goes to the
# executor.
_EXT0 = GROUP_DIM["position"] + GROUP_DIM["tactile"] + GROUP_DIM["task"]
_CURSOR_SLOTS = np.concatenate([np.arange(_EXT0 + a, _EXT0 + a + n) for a, n in
                                [LAYOUT["external"][0][k] for k in
                                 ("write_target_in_hand", "write_ahead_near", "write_ahead_far", "write_pen_down")]])


def blind_grip(obs):
    """A copy of the observation with the cursor entries zeroed, for the frozen grip policy."""
    ext = np.asarray(obs["external"]["vec"], np.float32).copy()
    base = _EXT0 - (GROUP_DIM["position"] + GROUP_DIM["tactile"] + GROUP_DIM["task"])
    for a, n in [LAYOUT["external"][0][k] for k in
                 ("write_target_in_hand", "write_ahead_near", "write_ahead_far", "write_pen_down")]:
        ext[a:a + n] = 0.0
    out = dict(obs)
    out["external"] = dict(obs["external"], vec=ext)
    if "traj" in obs.get("task", {}):
        # the grip is trained on L0, where there is no trajectory. Same reason as the cursor.
        out["task"] = dict(obs["task"], traj=np.zeros_like(np.asarray(obs["task"]["traj"])))
    return out
from .base import Policy
from .ppo import PPOPolicy
from .scripted import ScriptedTracker

CONTACT_H = 0.0008          # tip this close to the sheet counts as touching (m)
DEPTH_MAX = 0.0006          # deeper than this into the sheet: back off (the expert's "press ran away" branch)


class HybridBase(Policy):
    architecture = "hybrid (IK arm + learned grip)"
    demos = "none"

    def __init__(self, grip_ckpt):
        self.tracker = ScriptedTracker()
        self.grip = PPOPolicy(grip_ckpt)
        self.control_mode, self.control_hz, self.obs_mode = self.grip.control_mode, self.grip.control_hz, self.grip.obs_mode

    def reset(self, task_info):
        self.tracker.reset(task_info)
        self.grip.reset(task_info)
        self.hold_task = task_info.get("level", "") == "L0"
        self.n_act = int(task_info.get("act_dim", NU))

    def observe(self, info):
        # nothing physical is read; the target's schedule (task input) may travel here
        if info.get("cursor") is not None:
            self.tracker.observe({"cursor": info["cursor"]})

    def act(self, obs):
        if self.hold_task:
            # hold tasks: the environment drives arm and wrist along its own trajectories;
            # a wrist torque from the tracker's PD fights that servo and shakes the pen loose.
            # Fingers only, as in the hold stage.
            a = np.full(self.n_act, -1.0, np.float32); a[:NU] = 0.0     # a done entry, if any, stays off
            a[N_ARM + N_WRIST:NU] = np.asarray(self.grip.act(blind_grip(obs)), np.float32)[N_ARM + N_WRIST:NU]
            return np.clip(a, -1.0, 1.0)
        # The tracker's vertical law reads a measured force: creep down while it is ~0, hold
        # while it is moderate, back off when it exceeds 2.5x the target. Drive those three
        # branches from the observed tip height instead of the privileged force.
        h = float(unpack(obs["external"], "external")["tip_above_sheet"][0])
        self.tracker.f_meas = 0.0 if h > CONTACT_H else (0.1 if h > -DEPTH_MAX else 10.0)
        a = np.asarray(self.tracker.act(obs), np.float32).copy()
        # arm AND wrist from the tracker: the hold policy's wrist rows were never trained
        # (the environment drove arm and wrist through the whole hold stage); its fingers were.
        a[N_ARM + N_WRIST:NU] = np.asarray(self.grip.act(blind_grip(obs)), np.float32)[N_ARM + N_WRIST:NU]
        return np.clip(a, -1.0, 1.0)


class GripBase(Policy):
    """The frozen grip alone. Fingers from the hold policy, arm and wrist ZERO - the
    executor policy trained on top supplies the whole arm and wrist command (bounds 1.0) and
    its finger rows are scaled to nothing, so the grip is protected by construction. The end
    of the episode is a rule on the cursor, not a learned output: finished, and the tip within
    4 mm of the last point."""
    architecture = "grip base (learned fingers, no arm)"
    demos = "none"

    def __init__(self, grip_ckpt, fingers=True):
        # fingers=False: the base supplies ONLY the end-of-episode rule and the policy owns all 34
        # joints, learning the fingers together with the arm; a grip frozen from the L0 hold stage
        # has never seen the arm press at 2.4 N.
        self.grip = PPOPolicy(grip_ckpt)
        self.fingers = bool(fingers)
        self.control_mode, self.control_hz, self.obs_mode = self.grip.control_mode, self.grip.control_hz, self.grip.obs_mode
        self._cursor = None

    def reset(self, task_info):
        self.grip.reset(task_info)
        self.hold_task = task_info.get("level", "") == "L0"
        self.n_act = int(task_info.get("act_dim", NU))
        self._cursor = None

    def observe(self, info):
        if info.get("cursor") is not None:
            self._cursor = info["cursor"]

    def act(self, obs):
        a = np.zeros(self.n_act, np.float32)
        if self.fingers:
            a[N_ARM + N_WRIST:NU] = np.asarray(self.grip.act(blind_grip(obs)), np.float32)[N_ARM + N_WRIST:NU]
        if self.n_act > NU:
            c = self._cursor
            tip_err = float(np.linalg.norm(unpack(obs["external"], "external")["write_target_in_hand"])) \
                if c is None else float(np.linalg.norm(np.asarray(c["write_target_in_hand"], float)))
            # DEXCRAFT_DONE_RULE=finished declares the end as soon as the trajectory's pointer has
            # passed its last point, without waiting for the tip to reach the lift-off point 15 mm up.
            # Otherwise a policy whose tip never climbs to the lift-off point idles on the sheet
            # while a pen that has already slipped in the hand creeps past the 20 mm drop threshold.
            rule = os.environ.get("DEXCRAFT_DONE_RULE", "reach")
            done = (not self.hold_task) and bool(c and c.get("finished")) and (rule == "finished" or tip_err < 0.004)
            a[NU:] = 1.0 if done else -1.0
        return np.clip(a, -1.0, 1.0)


class HybridPolicy(Policy):
    """base + bounded residual; the residual checkpoint is a train_ppo checkpoint."""
    architecture = "hybrid (IK arm + learned grip + residual RL)"
    demos = "none"

    def __init__(self, base, residual_ckpt=None, bounds=(0.1, 0.3, 0.3)):
        self.base = base
        self.residual = PPOPolicy(residual_ckpt) if residual_ckpt else None
        self.control_mode, self.control_hz, self.obs_mode = base.control_mode, base.control_hz, base.obs_mode
        from ..env import N_WRIST
        self.B = np.concatenate([np.full(N_ARM, bounds[0]), np.full(N_WRIST, bounds[1]),
                                 np.full(NU - N_ARM - N_WRIST, bounds[2])]).astype(np.float32)

    def reset(self, task_info):
        self.base.reset(task_info)
        if self.residual is not None:
            self.residual.reset(task_info)

    def observe(self, info):
        # the base needs the schedule (with the cursor out of the observation it arrives here);
        # without it the tracker aims at the hand origin for the whole episode
        self.base.observe(info)
        if self.residual is not None:
            self.residual.observe(info)

    def act(self, obs):
        a = self.base.act(obs)
        if self.residual is not None:
            # the residual checkpoint stores action_scale = the bounds it was trained with
            r = np.asarray(self.residual.act(obs), np.float32)
            n = min(len(r), NU)                  # the base declares the end; the residual moves joints
            a[:n] = a[:n] + r[:n]
        return np.clip(a, -1.0, 1.0)


def make_policy():
    # DEXCRAFT_BASE=grip: the executor stack (grip base + the arm/wrist policy as the "residual")
    kind = os.environ.get("DEXCRAFT_BASE", "hybrid")
    if kind == "done":                       # the end-of-episode rule only; the policy owns every joint
        base = GripBase(os.environ["DEXCRAFT_GRIP_CKPT"], fingers=False)
    else:
        base = (GripBase if kind == "grip" else HybridBase)(os.environ["DEXCRAFT_GRIP_CKPT"])
    res = os.environ.get("DEXCRAFT_RESIDUAL_CKPT")
    return HybridPolicy(base, res) if res else base
