"""Policy interface.

    class Policy:
        def reset(self, task_info: dict) -> None      # task_id, level, instruction, dt, control_hz, tool
        def act(self, obs: dict) -> np.ndarray        # (34,) - or an action chunk (K, 34) applied in order
        def observe(self, info: dict) -> None         # optional, privileged - see below

Every policy sees the same four observation groups and returns the same
34-float action; how it encodes them is entirely up to the model. The dimensions
below are fixed (`dexcraft.selftest` asserts them):

    obs["position"]  ( 75,)   qpos(34)  qvel x0.1(34)  arm_servo_target(7)
                              joint order: 7 arm (6 UR5e + forearm roll), 2 wrist, 25 hand
    obs["tactile"]   ( 34,)   per-joint torque as a fraction of that joint's limit
                              (motor + applied + contact reaction). The tool-sheet contact
                              force is NOT here: privileged, `info` only.
    obs["task"]      {"map": (300,400) uint8, "vec": (10,)}
                              the question. `map` IS the sheet: 400 x 300 mm at 1 mm/px,
                              the target drawn in sheet coordinates, darkness = target
                              press force (255 = FORCE_MAX 4 N), thickness = the ink width
                              that force implies. `vec`: time_frac, elapsed_s,
                              tool_onehot(3), force_max_N, map_half_m(2), map_center_m(2).
    obs["external"]  {"map": (300,400) uint8, "vec": (48,)}
                              `map`: the ink on the sheet so far, same geometry. `vec`:
                              hand pose, tool / tip / sheet / object poses in the hand
                              frame, and the stroke schedule (write_target_in_hand,
                              write_ahead_near/far, write_pen_down) and the brush state (7).
                              Standard group, on by default: the tool's pose in the hand
                              and the stroke direction / order live here. A policy
                              with obs_mode "deployable" runs without it (position +
                              tactile + task), which is the harder setting.
    obs["instruction"]        the task in words
    obs["images"]             {cam: (224,224,3) uint8} when obs_mode includes "pixels";
                              cameras default to ("top", "wrist"), side/front available

    action (34,) in [-1, 1], control_mode "hybrid" (what the benchmark scores):
      a[0:7]    arm: a step on the joint position target, target += a * max_vel * dt
                (what `servoj` takes on a real UR5e)
      a[7:34]   wrist (2) + hand (25): motor torque as a duty cycle - the fraction of the
                torque that joint can deliver AT ITS CURRENT SPEED (falls linearly to
                zero at 1.5x rated speed). No position setpoint: a policy that wants
                position control closes its own loop.
    The other control modes (pos, delta_pos, vel, acc, torque) drive every joint the
    same way; in those a (2*34,) action adds a feed-forward joint-torque channel.

`dexcraft.env.unpack(obs[group], group)` gives a named view of any group, and
`dexcraft.planning.map_to_strokes` is a reference decoder from the task map to an
ordered set of strokes - use it, replace it, or plan some other way entirely.

A policy is a python module exposing `make_policy() -> Policy`, or a
running policy server (see remote.py / serve.py).
"""
import importlib
import importlib.util
import os

import numpy as np


class Policy:
    control_mode = "delta_pos"
    obs_mode = "state"          # add "pixels" to also receive obs["images"]
    control_hz = 100            # policy rate; 20-50 Hz recommended for real-robot deployment
    # disclosure (written to the evaluate.py results json, never a restriction)
    architecture = "generalist"  # "generalist": one network for all tasks | "experts": per-task policies
    demos = "none"               # number / description of demonstrations used, or "none"
    compute = ""                 # e.g. "1 GPU, 6 h" / "64 CPU, 2 h"
    env_steps = ""               # environment steps used for RL, if any

    def reset(self, task_info):
        pass

    def observe(self, info):
        """Optional hook, called with the env `info` dict after every step.

        `info` carries privileged signals that are deliberately NOT in the
        observation (the true tool-paper force, per-finger contact force).
        Scripted teachers may use it; a learned policy under evaluation should not,
        and anything read from here should be disclosed."""

    def act(self, obs):
        from ..env import NU
        return np.zeros(NU, dtype=np.float32)


def load_policy(spec):
    """'module.path:ClassOrFactory' | '/path/to/policy.py' (exposes make_policy)."""
    if os.path.exists(spec):
        s = importlib.util.spec_from_file_location("submitted_policy", spec)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod.make_policy()
    modname, _, attr = spec.partition(":")
    mod = importlib.import_module(modname)
    obj = getattr(mod, attr or "make_policy")
    return obj() if callable(obj) else obj
