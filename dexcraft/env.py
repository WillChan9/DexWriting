"""DexCraftEnv: gymnasium environment for the DexCraft tool-use benchmark.

    env = DexCraftEnv(task_id="L2_write_word", control_mode="hybrid", control_hz=25)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

OBSERVATION - four groups, identical for every submission. How to encode them is
the model's business; what they contain is the benchmark's contract.

    position  ( 74,)  what the joint encoders report: joint angles and joint
                      velocities x0.1, plus the 6 commanded ARM joint targets (the
                      arm action integrates, so its setpoint must be observable;
                      the hand is torque-controlled and has none).
    tactile   ( 34,)  per-joint (motor torque + applied torque + contact
                      reaction) / that joint's limit. `qfrc_constraint` is the
                      only part a policy cannot derive from the position group.
                      (The per-finger and tool-sheet forces were removed - see
                      _tactile.)
    task      map (128x128 uint8) + vec (9,)
                      THE QUESTION. The map is the target drawn in a square window
                      aligned to it (centre `map_center_m`, half-size `map_half_m`
                      in the vec; bbox + 2 cm, at least 12.8 cm across, so at worst
                      1 mm/px); its DARKNESS is the target press force
                      (255 = FORCE_MAX N) and its thickness is the ink width that
                      force implies. vec carries time, tool id and the scale
                      constants needed to read the map.
    external  map (128x128 uint8) + vec (48,)
                      the world outside the hand: the ink actually on the sheet so
                      far (same frame as the task map; darkness = how hard the pen
                      pressed, on the task map's force scale), plus hand / tool / sheet /
                      object poses. On a real rig this is the camera and tracking.
                      With the brush, the last 7 entries are the tuft's footprint
                      state (a, b, cos/sin phi, ink left, F_n, |F_t|) from the
                      reduced-order tuft model (brush.py); zero for every other
                      tool. The brush tier is simulation-only and says so.

ACTION - arm position + hand torque, NU = 34 floats in [-1, 1]:

    a[:N_ARM] the arm joints (pan, lift, elbow, forearm roll): a step on the joint POSITION target,
            target += a * max_vel * dt. This is what `servoj` takes on a real
            UR5e, which does not accept joint torque.
    a[6:]   the 3 wrist + 25 hand joints: motor TORQUE, as a fraction of what
            that joint can deliver AT ITS CURRENT SPEED - a duty cycle in the
            same sense as PWM. Available torque falls linearly to zero at
            NO_LOAD_SPEED x max_vel, so stall torque at full speed is not on
            offer. There is no position setpoint for the hand: a policy that
            wants one closes its own loop (see policies/scripted.py).

Content is re-randomised every episode (position, scale, slant, rotation and the
force profile), and the server draws fresh seeds per submission, so the continuous
part of a task cannot be memorised; tasks.SPLITS holds out discrete content too.
"""
import json
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import scene as S
from . import tasks as T
from .fonts import resample

HERE = os.path.dirname(os.path.abspath(__file__))
N_ARM, N_WRIST, N_HAND = len(S.ARM_JOINTS), len(S.WRIST_JOINTS), len(S.HAND_JOINTS)
NU = N_ARM + N_WRIST + N_HAND                # 7 + 2 + 25 = 34 (forearm roll is in the ARM group)


GROUP_FIELDS = {
    "position": [("qpos", NU), ("qvel_x0.1", NU), ("arm_servo_target", N_ARM)],
    "tactile":  [("joint_torque_frac", NU)],
    "task":     [("time_frac", 1), ("elapsed_s", 1), ("tool_onehot", 3), ("force_max_N", 1),
                 ("map_half_m", 2), ("map_center_m", 2)],
    "external": [("hand_pos", 3), ("hand_rot6", 6), ("tool_pos_in_hand", 3), ("tool_axis_in_hand", 3),
                 ("tip_pos_in_hand", 3), ("tip_above_sheet", 1), ("sheet_origin_in_hand", 3),
                 ("sheet_normal_in_hand", 3), ("object_pos_in_hand", 3), ("target_pos_in_hand", 3),
                 ("write_target_in_hand", 3), ("write_ahead_near", 3),
                 ("write_ahead_far", 3), ("write_pen_down", 1), ("brush_state", 7)],
}


def _spans(fields):
    out, a = {}, 0
    for k, n in fields:
        out[k] = (a, n)
        a += n
    return out, a


LAYOUT = {g: _spans(f) for g, f in GROUP_FIELDS.items()}
GROUP_DIM = {g: LAYOUT[g][1] for g in LAYOUT}


def unpack(x, group):
    """Named view of one observation group.

        unpack(obs["position"], "position")     # flat groups
        unpack(obs["task"], "task")             # nested groups: the "vec" part
    """
    vec = x["vec"] if isinstance(x, dict) else x
    fields = LAYOUT[group][0]
    return {k: np.asarray(vec)[a:a + n] for k, (a, n) in fields.items()}


CONTROL_MODES = ("hybrid", "pos", "delta_pos", "vel", "acc", "torque")
# A motor cannot deliver stall torque at speed: PWM sets the voltage, and back-EMF
# eats the available current as the joint turns. Torque falls linearly to zero at
# the no-load speed, taken here as a multiple of each joint's rated max velocity.
NO_LOAD_SPEED = 1.5
def _load_teleop():
    """Recorded teleoperation wrist trajectories (flex, dev, roll), one row per control step, smoothed."""
    import csv, glob
    out = []
    root = os.path.join(HERE, "data", "teleop_wrist")
    for f in sorted(glob.glob(os.path.join(root, "*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 120:
            continue
        W = np.array([[float(r[c]) for c in ("wrist_flex", "wrist_dev", "wrist_roll")] for r in rows])
        k = np.ones(5) / 5
        W = np.stack([np.convolve(W[:, j], k, mode="same") for j in range(3)], axis=1)
        out.append(W)
    return out


_TELEOP = _load_teleop()

with open(os.path.join(HERE, "human_grip.json")) as _fh:
    _HUMAN_POSE = json.load(_fh)["hand_pose"]   # 2915 teleop frames, grip_w > 0.9
# How far the LINEAR tracking pull reaches. A short reach (2 cm) leaves a gradient
# of only 0.010 reward/mm beyond 5 cm, because `track` and `fine` are both saturated
# out there and only `far` is live; a policy then parks well off target.
# `approach` does not rescue this: it is potential-based, so it telescopes out of
# the return and reshapes credit assignment rather than the objective.
FINE_RANGE = 0.15
# The two observation maps ARE the sheet: same rectangle, same aspect, one pixel per
# millimetre. A coarser square window (e.g. 128x128 over 400x400 mm) wastes a quarter
# of the map on table beyond the 300 mm sheet, and at 3.1 mm/px a pixel is coarser
# than the 3 mm scoring tolerance and three times the 1 mm pen line, so the policy
# could not see the stroke it is asked to draw.
INK_MAP_MM_PER_PX = 1.0
INK_MAP_W = int(round(2 * S.PAPER_HALF[0] * 1000 / INK_MAP_MM_PER_PX))     # 400
INK_MAP_H = int(round(2 * S.PAPER_HALF[1] * 1000 / INK_MAP_MM_PER_PX))     # 300
INK_TIP_REACH = 0.002              # ink only if the tip itself is this close to the sheet
COVER_TOL_MM = 6.0                 # completion tolerance on the coverage stat [mm]
LIFT_CLEAR = 0.003                 # the tip must rise this far above the sheet for a
#                                    re-landing to count as a deliberate lift [m]
COVER_DILATE_MM = 2.0              # ink is dilated by this before its area is scored
WAYPOINT_SPACING = 0.0015          # m along strokes
PRESS_DEPTH = 0.0015               # pen-down waypoints sit this far into the paper
LIFT_HEIGHT = 0.015                # hop height between strokes
TL_NEAR = 0.05                     # tracking error: quadratic inside this radius, m
TL_FAR = 0.50                      # ... bounded linear ramp beyond it, out to here
PAINT_K = 6                        # L5 paint: palette slots in the observation
DIP_S = 0.2                        # L5 paint: the tip this long in a well = a dip (reservoir refilled)
PICK_LIFT = 0.020                  # L0_pick_up: this far off the table counts as lifted
PICK_HOLD_S = 1.0                  # ... and held this long counts as picked up
INK_ON_TARGET = 0.010              # ink bonus fades to zero this far from the live waypoint
GAP_LOST = 0.010                   # summed fingertip-to-shaft gap that counts as a lost grip (see _check_drop)
HOLD_FIT = 0.90                    # env-driven wrist sweep: how far toward a joint limit it may reach
HOLD_JITTER_HZ = (4.0, 9.0)        # ... and the band of the hand shake laid on top
SCRIBBLE_MARGIN = 0.03             # the driven scribble path keeps this far inside the sheet edge [m]
IK_TRACK_ITERS = 12                # per-step DLS iterations when tracking a driven path
DRIVE_Z_GAIN = 2.0                 # driven scribble: target pushed deeper by this x the arm's measured lift
DRIVE_Z_CLAMP = 0.005              # ... by at most this much [m]
INK_ON_TIMELINE = 0.030            # ... and this far (laterally) from the timeline point: the
                                   # target moves on its own clock, so "park and collect" is
                                   # impossible, and a 1 cm fade paid nothing to a policy whose
                                   # tracking error was 11-16 mm - so it hovered 10 mm up instead

PERTURB_PRESETS = {
    "none": {},
    "mild": dict(pen_mass=0.2, pen_friction=0.2, paper_friction=0.2, push_force=0.3, push_every=1.5,
                 obs_noise=0.002, action_delay=0, light_jitter=0.3, cam_jitter=0.005),
    "hard": dict(pen_mass=0.35, pen_friction=0.35, paper_friction=0.35, push_force=0.8, push_every=0.8,
                 obs_noise=0.005, action_delay=2, light_jitter=0.5, cam_jitter=0.01),
}
# Reward shaping for RL (metrics / scoring are separate - see metrics.py).
# Tracking is quadratic inside 5 cm and a bounded slope outside (always a
# gradient toward the target, never death-bait), a drop pays the rest of the
# episode at the worst rate (blocks "throw the tool to cut losses"), and the
# action-rate / joint-velocity / grip-pose terms keep the motion smooth.
# `approach` is potential-based shaping on the distance to the live waypoint. It
# matters more than it looks: `track` saturates beyond 5 cm, so with the target
# 20 cm away every state scores about the same and there is nothing to climb.
# Rewarding the *reduction* in distance is
# well scaled at any range and leaves the optimum unchanged.
REWARD_PRESETS = {
    # Holding tasks (L0): slipping is charged CONTINUOUSLY - position, rotation and
    # relative velocity - and the fingertips are required to be on the shaft, not
    # merely around it. A binary "did it fall out" check is far too permissive: a
    # zero-torque hand can creep 12.9 mm and open its fingertips 13.8 mm off the
    # shaft and still score a clean episode under it.
    # Writing. Three task terms: the target walks its own timeline, so `progress`,
    # `approach`, `far` and `fine` (which only matter for a cursor that moves when
    # the policy arrives) are off. Holding reward + tracking + an ink bonus.
    "write": dict(alive=1.0, timeline=800.0, timeline_far=10.0, timeline_fine=100.0,
                  ink=3.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                  slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                  track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                  arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v2 (arm frozen, wrist + fingers write): ink paid by contact DUTY over the control
    # step, and a charge on the deviation of the in-contact force from the map's target
    "write_v2": dict(alive=1.0, timeline=800.0, timeline_far=10.0, timeline_fine=100.0,
                     ink=3.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.015,   # half the arm-driven fade:
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,                     # 3 cm let "park at the centre" of a 3 cm target collect
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v3: write_v2 with the ink incentive rebalanced so that WRITING outbids "hold
    # and stay safe" at zero assist (otherwise drop risk dominates), and a tighter lateral
    # fade against backwards skimming.
    "write_v3": dict(alive=1.0, timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                     ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.012,
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v4: write_v3 with the ink corridor tightened to 5 mm and ink paid ONLY while
    # the tip moves ALONG the timeline direction - the corridor-scribble optimum (dense,
    # illegible loops inside the text corridor) collects nothing here.
    "write_v4": dict(alive=1.0, timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                     ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v5: PATIENT timeline - the target waits until
    # the current point is actually earned (pen-down points require real ink within
    # 3 mm; transit points require the tip within 6 mm), so progress = completion by
    # construction. The journey pays: advance per point, transit following, a re-entry
    # bonus per stroke, an incremental glyph payment; ink and pen-down advances are
    # SHARE-GATED (x wrist+finger share) so the creeping arm may reposition but earns
    # nothing by writing. alive drops to 0.2 - no wage for hiding.
    "write_v5": dict(alive=0.2, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                     share_gate=True,
                     timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                     ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v7: write_v5 with an explicit CHARGE on the tip motion the arm and the wrist
    # produce while the pen is engaged. write_v5's share_gate only scales the
    # progress terms by (1 - arm share) and leaves the wrist free, so with the arm
    # locked the wrist simply takes over the arm's job. Pricing both is the only term
    # in this file that asks for the FINGERS specifically.
    "write_v7": dict(alive=0.2, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                     share_gate=True,
                     timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                     ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=15.0, wrist=8.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # finger track: the same, plus a charge on the tip motion the arm / wrist produce
    # while the pen is engaged (see `fingers`), so strokes are biased onto the fingers
    "write_fingers": dict(alive=1.0, timeline=800.0, timeline_far=10.0, timeline_fine=100.0,
                          ink=3.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                          slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                          track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                          arm=8.0, wrist=3.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v8: write_v5 for the tasks that have a target, plus scribble's area and
    # lift terms so the L0 rungs kept in a curriculum mix still have a signal. One env
    # carries one reward, and a task with no strokes gets nothing at all from write_v5
    # alone, so a curriculum would silently un-learn the rung it just passed.
    "write_v8": dict(alive=0.2, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                     share_gate=True, area=0.5, lift=20.0,
                     timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                     ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True),
    # write_v9: write_v8 lets a policy sit 2-6 cm above the sheet for the whole episode,
    # even though the timeline terms pay +2.4/step within 5 mm and charge -11..-21/step
    # out there. Hovering is a local optimum because (a) `alive` pays 0.2/step for
    # surviving, and a hovering pen never drops, (b) nothing charges HEIGHT: the timeline
    # distance is mostly lateral out there, and the first touch off the line costs
    # slip/force with no ink to show for it. So: no alive pay; `hover` charges the tip's height above the
    # sheet on every pen-down step; `contact` pays for touching on pen-down steps at all,
    # ink still pays for touching ON the line; and `hover_end` ends the episode - charged
    # like a drop - once the tip has stayed `hover_h` above the sheet for `hover_t` s of a
    # pen-down segment. Training-only: the accept runs the default preset.
    "write_v9": dict(alive=0.0, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                     share_gate=True, area=0.5, lift=20.0,
                     timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                     ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                     force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                     track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                     arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                     hover=100.0, contact=0.3, hover_end=True, hover_h=0.010, hover_t=4.0),
    # write_v10: a time-driven recipe. Two things it does that the patient presets do not:
    #   * the target walks on ITS OWN CLOCK (`patient` off): "you are behind, go this way"
    #     at every step. The patient cursor parks on the pen-up lift point until the tip
    #     captures it, so for a policy that never gets there every pen-down-gated term -
    #     ink, and write_v9's hover charge and contact pay - stays silent forever;
    #   * ink pays for ANY sheet contact on a pen-down step (`contact`), not only for
    #     contact within 5 mm of the line moving along the stroke (`ink`, kept as the
    #     precision bonus on top).
    # Plus write_v9's rules: no alive pay, height charged on pen-down steps, and a tip
    # that stays > hover_h above the sheet for > hover_t s ends the episode like a drop.
    "write_v10": dict(alive=0.0, patient=False,
                      timeline=800.0, timeline_far=10.0, timeline_fine=150.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0),
    # write_v11: write_v10 closes the hover, but leaves the mirror image open - a policy
    # can carry the tip AWAY, far from the target and BELOW the sheet plane (off the
    # table edge). The height rule only looks upward and every distance term is capped at
    # FINE_RANGE / TL_FAR, so beyond 15 cm the per-step cost is flat and nothing ends the
    # episode. `far_end`: a tip that stays farther than `far_d` from the timeline target
    # for `far_t` s ends the episode like a drop (the teacher's transits are short: it
    # never trips at 15 cm / 3 s on L1 or L2). `timeline_far` 100: distance keeps hurting
    # out to TL_FAR, so the plateau slopes back toward the sheet.
    "write_v11": dict(alive=0.0, patient=False,
                      timeline=800.0, timeline_far=100.0, timeline_fine=150.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0),
    # write_v12: write_v11 plus the training-only start curriculum `place_start`
    # (the tip starts at the first stroke's first point, where the time-driven timeline
    # begins, instead of at the glyph centre). The accept preset never sets this.
    "write_v12": dict(alive=0.0, patient=False, place_start=True,
                      timeline=800.0, timeline_far=100.0, timeline_fine=150.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0),
    # write_v13: write_v12 with the timeline at half
    # speed (`tl_slow` 2: the training episode is twice as long, the glyph speed halves).
    "write_v13": dict(alive=0.0, patient=False, place_start=True, tl_slow=2.0,
                      timeline=800.0, timeline_far=100.0, timeline_fine=150.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.005, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0),
    # write_v14: under write_v12 a policy tracks the timeline but 1-5 cm behind it, and
    # at half speed (write_v13) the lag is the same, so it is precision, not pace. The
    # near-field pull there is weak: 0.15/mm (timeline_fine 150) against a flat +1/step
    # for touching anywhere. This preset sharpens it -
    # timeline 2000 (quadratic inside 5 cm), timeline_fine 600 (0.6/mm) - and lets the
    # on-line ink bonus reach from 10 mm instead of 5. Full speed.
    "write_v14": dict(alive=0.0, patient=False, place_start=True,
                      timeline=2000.0, timeline_far=100.0, timeline_fine=600.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.010, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0),
    # write_v15: write_v14 without `place_start`. The hand starts where the accept
    # starts it (the "default" preset's placement), so a demo-anchored policy trains on
    # the state distribution it is scored on and finds the first stroke on its own.
    "write_v15": dict(alive=0.0, patient=False,
                      timeline=2000.0, timeline_far=100.0, timeline_fine=600.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.010, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0),
    # write_v16 (policy-declared end): the policy ends the episode itself and is paid the strict
    # completion when it does; a per-step `time` charge makes it finish and stop; the cursor
    # is progress-driven (`patient`, advance / glyph / reentry pay per target point covered in
    # order) and is NOT in the observation (env kwarg cursor_obs=False) - only the expert
    # that demonstrates reads it. Tracking terms are weak on purpose: what pays is covering.
    "write_v16": dict(alive=0.0, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                      timeline=800.0, timeline_far=100.0, timeline_fine=150.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.010, ink_dir=True,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0, time=0.3, final=200.0),
    # write_v17: the reward pays what the final metrics measure.
    # Terminal: 300 x completion (1 mm, in order) + 100 x precision (1 mm) when the policy
    # ends the episode. Dense: ink pays only within 3 mm of the cursor's path and fades to
    # zero there (`ink_fade` 0.003), ink laid further than 1 mm off the line is
    # charged (`ink_off`), the patient cursor pays per target point covered in order, the
    # clock costs 0.3 a step. Grip, force, hover and drop terms as write_v14.
    "write_v17": dict(alive=0.0, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                      timeline=800.0, timeline_far=100.0, timeline_fine=150.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.003, ink_dir=True, ink_off=3.0,
                      force_reg=1.0, force=4.0, press_limit=3.0, idle_pose=20.0, idle_vel=0.5,
                      slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=5.0, drop=True,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0, time=0.3, final=300.0, final_precision=100.0),
    # write_v18: guards against a policy that learns to quit. The per-step timeline charges
    # (-12 .. -20 a step, -7k .. -12k an episode at a ~15 mm tracking error) are off, and the
    # episode's end is scored symmetrically - `final` x completion for what was written,
    # `quit` x (1 - completion) for what was not, at any end. Tracing beats quitting at every
    # skill level; the dense terms (advance / glyph / ink / ink_off / force / time) are shaping only.
    # The per-step grip / force / posture charges are write_v17's / 10 (they stay as gradients but
    # no longer add up to more than the terminal over an episode), and `drop` is off: a drop already
    # forfeits the unwritten part through `quit`, and "rest of the episode at the worst rate" makes
    # a policy that drops often prefer quitting to trying.
    "write_v18": dict(alive=0.0, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                      timeline=0.0, timeline_far=0.0, timeline_fine=0.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.003, ink_dir=True, ink_off=3.0,
                      force_reg=0.1, force=0.4, press_limit=0.3, idle_pose=2.0, idle_vel=0.05,
                      slip_pos=2.0, slip_rot=2.0, slip_vel=0.1, gap=2.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.03, qvel=0.002, pose=0.5, drop=False,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0, time=0.3, final=2000.0, final_precision=300.0, quit=2000.0),
    # v13_residual (residual RL on the IK expert): write_v18 with two changes for a
    # base that already tracks to 1-2 mm. `adv_tol` 1.5 mm pulls the dense per-point payment onto
    # the scored metric (completion = ink within 1 mm; at 3 mm a policy banks progress the metric
    # scores zero). The force / grip / posture charges are write_v17's / 3 instead of / 10: the /10
    # is sized for a policy with a 15 mm tracking error, and force_mae (0.5-0.9 N on the base) is
    # one of the numbers residual RL should improve. Terminal pay as write_v18: `final` x completion,
    # `quit` x (1 - completion) at any end, so the completion slope is 4000 per unit.
    "v13_residual": dict(alive=0.0, patient=True, advance=2.0, transit=0.5, reentry=8.0, glyph=40.0,
                      adv_tol=0.0015,
                      timeline=0.0, timeline_far=0.0, timeline_fine=0.0,
                      ink=6.0, ink_duty=True, ink_ramp=(0.1, 0.4), ink_fade=0.003, ink_dir=True, ink_off=3.0,
                      force_reg=0.3, force=1.2, press_limit=0.9, idle_pose=6.0, idle_vel=0.15,
                      slip_pos=6.0, slip_rot=6.0, slip_vel=0.3, gap=6.0,
                      track=0.0, far=0.0, progress=0.0, approach=0.0, fine=0.0,
                      arm=0.0, wrist=0.0, rate=0.1, qvel=0.006, pose=1.5, drop=False,
                      hover=100.0, contact=1.0, hover_end=True, hover_h=0.010, hover_t=4.0,
                      far_end=True, far_d=0.15, far_t=3.0, time=0.3, final=2000.0, final_precision=300.0, quit=2000.0),
    # v15_goal: reward only the final goal, without constraining how it is achieved. Four terms, each the evaluation metric paid as it is earned, all normalised per task so a word and a
    # line are on the same scale (see DexCraftEnv._reward_goal):
    #   completion  +w_completion x target points newly covered IN ORDER within 1 mm / all target points
    #   precision   -w_precision  x the change in (1 - precision), so it sums to -w_precision x (1 - precision)
    #   force       -w_force      x the change in force MAE (on target), so it sums to -w_force x MAE
    #   time        -w_time       x dt / the task's nominal duration
    # No terminal charge and no drop charge: a drop or a hover/far cutoff ends the episode, and what it
    # costs is the completion not yet earned. Nothing else is shaped.
    # v16_goal: v15_goal with the early-exit refund closed. Under v15_goal a policy can fly the tip
    # off the sheet, let the hover / far rule end the episode early, and pay only a fraction of what
    # the clock would have cost - quitting becomes the best strategy. So an episode that ends before
    # the schedule is finished pays the time it did not use. Not a new term: the same clock, no refund.
    "v16_goal": dict(goal=True, patient=True, adv_tol=0.0015, alive=0.0, drop=False, time_to_end=True,
                     prec_per_point=True,
                     w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                     hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense*: v16_goal plus ONE dense term - the
    # tip's distance to the current timeline point, paid as a potential difference (how much closer
    # it got this step, same target point both sides), so an episode's total is the net approach
    # and the optimum is unchanged. `goal_tol` is the curriculum knob: what counts as covering /
    # on-target ink, 5 mm -> 2 mm -> 1 mm; the pointer's `adv_tol` follows it.
    "v16_dense5": dict(goal=True, patient=True, adv_tol=0.005, goal_tol=0.005, w_track=5000.0, alive=0.0, drop=False,
                       time_to_end=True, prec_per_point=True,
                       w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                       hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_dense2": dict(goal=True, patient=True, adv_tol=0.002, goal_tol=0.002, w_track=5000.0, alive=0.0, drop=False,
                       time_to_end=True, prec_per_point=True,
                       w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                       hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_dense1": dict(goal=True, patient=True, adv_tol=0.0015, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                       time_to_end=True, prec_per_point=True,
                       w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                       hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense1d: v16_dense1 + a drop charge equal to the completion budget. Without it, a policy can
    # let go of the pen at the end, because after the schedule a drop pays only the rest of the clock
    # (~100) against the 1000 already earned. The metric fails a dropped episode outright.
    "v16_dense1d": dict(goal=True, patient=True, adv_tol=0.0015, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                        time_to_end=True, prec_per_point=True, drop_charge=1000.0,
                        w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                        hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense1c: a drop zeroes the completion - the reward takes back what the
    # episode has been paid for coverage, exactly the metric's rule (a dropped episode is a failed
    # one). Late drops cost the most.
    "v16_dense1c": dict(goal=True, patient=True, adv_tol=0.0015, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                        time_to_end=True, prec_per_point=True, drop_clawback=True,
                        w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                        hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense1s: + a slip term. Late "drops" are typically the pen creeping through the fingers
    # (the scripted expert's slip p95 is 3.7 mm) until it passes the metric's 20 mm threshold while
    # the hand idles. Paid as a
    # potential on the slip measured in units of the drop thresholds (20 mm translation, 30 deg
    # rotation), so an episode's total is -w_slip x the final slip: at the threshold on either axis,
    # -500. Intended for use with DEXCRAFT_DONE_RULE=finished.
    "v16_dense1s": dict(goal=True, patient=True, adv_tol=0.0015, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                        time_to_end=True, prec_per_point=True, drop_clawback=True, w_slip=500.0,
                        w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                        hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense1c5: v16_dense1c with the pointer's skip wait 1.5 s -> 0.5 s. A point the tip does not
    # cover within 1.5 mm holds the trajectory for the skip wait before moving on; with a long wait the
    # policy sits on the sheet waiting for "finished" and slips.
    "v16_dense1c5": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.5, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                         time_to_end=True, prec_per_point=True, drop_clawback=True,
                         w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                         hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense1c25: skip wait 0.25 s, the best value in a sweep over 0 - 1.5 s (completion / drops,
    # 50 seeds): 0 s 0.892 / 3, 0.1 s 0.903 / 2, 0.25 s 0.916 / 2, 0.5 s 0.899 / 2, 1.0 s 0.883 / 6,
    # 1.5 s 0.885 / 3.
    "v16_dense1c25": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                          time_to_end=True, prec_per_point=True, drop_clawback=True,
                          w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                          hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_dense1c25l: skip 0.25 s + a 10-point look-ahead in the pointer. Look-ahead (50 seeds)
    # 0 -> 0.916 / 2 drops, 5 -> 0.887 / 1, 10 / 20 / 40 -> 0.890 / 0. The look-ahead removes the drops
    # (no crawl through missed points with the pen resting) and costs the 2.5 % of the line the crawl
    # makes the policy go back for. Training under it asks PPO for exactly that last 2.5 %.
    "v16_dense1c25l": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, adv_lookahead=10, goal_tol=0.001,
                           w_track=5000.0, alive=0.0, drop=False, time_to_end=True, prec_per_point=True, drop_clawback=True,
                           w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                           hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # stage 2+ rungs: the tolerance curriculum with the 0.25 s skip and the claw-back (the settings the
    # line stage ends with), 5 mm and 2 mm
    # v16_c1: completion itself made continuous, with no tracking term. The two halves of
    # the metric share one ramp: a target point is worth 1 to completion inside the scored 1 mm and
    # ramps to 0 at 3 mm; an ink point is free to precision inside 1 mm and pays in full at 3 mm. The
    # 1-3 mm band is where a near-converged policy lives, and the hard rule gives it no
    # gradient there at all. force is charged per step instead of as the change in a running mean.
    # No shaping term is left: every term is an evaluation metric, made differentiable.
    # NOTE `w_track` is 0 here. It is the only term that pays while the pen is OFF the sheet, so this
    # preset is for fine-tuning a policy that already writes. Training from scratch under it behaves
    # like v15_goal: the tip flies up, completion stays at 0 and nothing points down.
    "v16_c1": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   goal_soft=0.003, force_per_step=True, w_track=0.0, alive=0.0, drop=False,
                   time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_c2: v16_c1 with the force baseline restored (`force_ref`, about the IK teacher's MAE).
    # Without a baseline the force term charges every contact step unconditionally and completion
    # collapses; see the note in _reward_goal. Everything else is unchanged.
    "v16_c2": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   goal_soft=0.003, force_per_step=True, force_ref=0.8, w_track=0.0, alive=0.0, drop=False,
                   time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_c3: v16_dense1c25 with the soft ramp, and NOTHING else changed. c1/c2 also drop w_track,
    # and w_track is the only term that pays while the pen is OFF the sheet; the ramp pays only where
    # ink lands. Removing w_track removes the restoring force - with no term of any sign off the
    # sheet, the policy diffuses off the writing manifold and completion collapses.
    # v16_f1 - the force term alone, charged per step against the teacher's level. The difference
    # form pays -w_force x (err - mae) / n, so the same error is worth 50x less at step 500 than at
    # step 10 and most of a word carries no force signal at all. `force_ref` is what the difference
    # form has for free: a baseline. Without one every contact step is a pure cost and the term
    # prices the QUANTITY of contact rather than its quality.
    "v16_f1": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   force_per_step=True, force_ref=0.585, w_track=5000.0, alive=0.0, drop=False,
                   time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_g1 - the same reward with the two terms that encode the TEACHER'S PLAN re-referenced to
    # the glyph: the restoring potential points at the nearest uncovered target point instead of
    # the timeline's k-th point, and "finished" can be the drawing being done instead of the
    # pointer reaching the end. Nothing else changes. The point: BC is by construction the
    # optimum of "follow this trajectory", so a reward built on that term cannot reward anything
    # better than BC.
    "v16_g1": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   w_track=0.0, w_track_goal=5000.0, done_frac=0.95, alive=0.0, drop=False,
                   time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v17_r0: v16_g1 - the glyph-referenced restoring potential, so the reward does not make the
    # teacher's trajectory its optimum - with the force term charged PER STEP against a baseline at
    # the fixed IK teacher's 0.53 N MAE. A BC clone of the teacher matches it on the drawing and is
    # behind it mainly on force. The difference form it replaces pays -w_force x (err - mae) / n,
    # which is 50x weaker at step 500 than at step 10. The baseline keeps the term pricing the
    # QUALITY of contact, not its quantity (without one every contact step is a pure cost); the
    # running total is clamped at >= 0, so beating the teacher refunds charges but never pays for
    # lingering.
    # v17_r1: tuned for RL to outperform the IK teacher (BC's ceiling). Under v17_r0 a policy can lift
    # off the paper, because contact costs (force baseline below the policy's own force) and hovering
    # does not (guidance aims at the sheet top, and at a target that moves as the tip wanders). So:
    #   guidance   `w_track_next`: the first uncovered glyph point ahead of the completion pointer, at the
    #              task's 1.5 mm press depth - pressing pays, wandering does not; still not the timeline
    #   force      back to the difference form, centred on the policy's own mean (stable)
    #   beyond the teacher: time 100 -> 300 (the teacher writes at the timeline's fixed pace; a faster
    #              equally good drawing should be worth something) and precision 300 -> 500 (the teacher
    #              covers 100 % of every glyph within 2 mm but 91.7 % within 1 mm: its losses are placement)
    "v17_r1": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   w_track=0.0, w_track_next=5000.0, goal_depth=0.0015, done_frac=0.95,
                   alive=0.0, drop=False, time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=500.0, w_force=200.0, w_time=300.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v17_r0": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   w_track=0.0, w_track_goal=5000.0, done_frac=0.95, force_per_step=True, force_ref=0.53,
                   alive=0.0, drop=False, time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # v16_c4: c3's ramp, but the credit outside the scored 1 mm is worth a quarter of on-target
    # instead of three quarters of it, so the boundary keeps its pull. See `goal_soft_gain`.
    "v16_c4": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   goal_soft=0.003, goal_soft_gain=0.25, w_track=5000.0, alive=0.0, drop=False,
                   time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_c3": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001,
                   goal_soft=0.003, w_track=5000.0, alive=0.0, drop=False,
                   time_to_end=True, prec_per_point=True, drop_clawback=True,
                   w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                   hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_dense5_25": dict(goal=True, patient=True, adv_tol=0.005, adv_skip=0.25, goal_tol=0.005, w_track=5000.0, alive=0.0, drop=False,
                          time_to_end=True, prec_per_point=True, drop_clawback=True,
                          w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                          hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_dense2_25": dict(goal=True, patient=True, adv_tol=0.002, adv_skip=0.25, goal_tol=0.002, w_track=5000.0, alive=0.0, drop=False,
                          time_to_end=True, prec_per_point=True, drop_clawback=True,
                          w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                          hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # stage 2 (line + circle): the line stays at 1 mm, the circle climbs 5 mm -> 2 mm -> 1 mm
    "v16_s2a": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                    time_to_end=True, prec_per_point=True, drop_clawback=True,
                    tol_by_task={"L1_trace_line": 0.001, "default": 0.005},
                    w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                    hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_s2b": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                    time_to_end=True, prec_per_point=True, drop_clawback=True,
                    tol_by_task={"L1_trace_line": 0.001, "default": 0.002},
                    w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                    hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # stage 2, rungs b' / c': the circle's completion tightens (2 mm, then 1 mm) while its pointer stays
    # loose (5 mm, then 3 mm) so the trajectory keeps flowing; the line's pointer stays at 1.5 mm
    "v16_s2b2": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                     time_to_end=True, prec_per_point=True, drop_clawback=True,
                     tol_by_task={"L1_trace_line": 0.001, "default": 0.002}, adv_tol_by_task={"L1_trace_line": 0.0015, "default": 0.005},
                     w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                     hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v16_s2c2": dict(goal=True, patient=True, adv_tol=0.0015, adv_skip=0.25, goal_tol=0.001, w_track=5000.0, alive=0.0, drop=False,
                     time_to_end=True, prec_per_point=True, drop_clawback=True,
                     tol_by_task={"L1_trace_line": 0.001, "default": 0.001}, adv_tol_by_task={"L1_trace_line": 0.0015, "default": 0.003},
                     w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                     hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    "v15_goal": dict(goal=True, patient=True, adv_tol=0.0015, alive=0.0, drop=False,
                     w_completion=1000.0, w_precision=300.0, w_force=200.0, w_time=100.0,
                     hover_end=True, hover_h=0.010, hover_t=4.0, far_end=True, far_d=0.15, far_t=3.0),
    # L0_scribble: keep the tool (the `hold` terms) and get paid for area and for
    # lifting. No timeline, no glyph, no tracking - there is nothing to trace.
    # L0_scribble: paid per metre the tip travels ALONG the sheet, charged for
    # contact above `impact_soft`. Travelling fast by ramming the tool into the paper is
    # the degenerate solution - with the arm released a policy can hit 23-118 N and lose
    # the pen inside a second - so the charge has to outweigh the payment there.
    # Episode returns over 4 seeds for three concrete behaviours - hover and never touch,
    # zero actions (rests on the sheet), draw 337 mm - under this preset:
    #   hover +47   rest +7   draw +468;  and a run that drops the tool, -212.
    # Paying for contact alone, or charging for airborne time, leaves hovering or resting
    # on top. The pose regularisers are also switched off for this rung - see _reward.
    "scribble": dict(alive=0.2, contact=0.3, travel=1500.0, impact=5.0, impact_soft=8.0,
                     slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0, press_limit=1.5,
                     track=0.0, far=0.0, ink=0.0, progress=0.0, approach=0.0, fine=0.0,
                     force=0.0, arm=0.0, wrist=0.0, rate=0.3, qvel=0.02,
                     pose=5.0, idle_pose=20.0, idle_vel=0.5, drop=True),
    "hold": dict(alive=1.0, slip_pos=20.0, slip_rot=20.0, slip_vel=1.0, gap=20.0,
                 track=0.0, far=0.0, ink=0.0, progress=0.0, approach=0.0, fine=0.0,
                 force=0.0, press_limit=1.0, arm=0.0, wrist=0.0,
                 rate=0.3, qvel=0.02, pose=5.0, drop=True),
    "default": dict(alive=1.0, track=800.0, far=10.0, ink=1.0, progress=300.0, approach=500.0,
                    fine=100.0, force=4.0, press_limit=1.0, arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=0.5, drop=True),
    # refinement stage: tighter tracking (mm-scale precision is what ink IoU
    # measures) and a writing-pressure limit rather than a "do not crash" one
    "precise": dict(alive=1.0, track=2500.0, far=10.0, ink=1.0, progress=300.0, approach=500.0,
                    fine=100.0, force=4.0, press_limit=0.8, arm=0.0, wrist=0.0, rate=0.3, qvel=0.02, pose=0.5, drop=True),
    # arm/wrist weights calibrated against the scripted teacher, which is 100 %
    # arm-driven and therefore takes the worst case: at force=6 it scored BELOW a
    # do-nothing policy (-41 on L2_write_word), i.e. the reward told PPO to stall.
    # force=4 with arm=8 leaves the teacher +157 ahead of stalling while still
    # charging clearly for arm-driven strokes (default is +230).
    "fingers_precise": dict(alive=1.0, track=2500.0, far=10.0, ink=1.0, progress=300.0, approach=500.0,
                            fine=100.0, force=4.0, press_limit=0.8, arm=8.0, wrist=3.0, rate=0.3, qvel=0.02,
                            pose=0.5, drop=True),
    # bias strokes onto the fingers: charge for the tip motion the arm and the
    # wrist produce while the pen is engaged (exactly what finger_share measures),
    # so gross repositioning still happens during pen-up hops
    "fingers": dict(alive=1.0, track=800.0, far=10.0, ink=1.0, progress=300.0, approach=500.0,
                    fine=100.0, force=4.0, press_limit=1.0, arm=15.0, wrist=5.0, rate=0.3, qvel=0.02, pose=0.5, drop=True),
    "fingers_hard": dict(alive=1.0, track=800.0, far=10.0, ink=1.0, progress=300.0, approach=500.0,
                         fine=100.0, force=4.0, press_limit=1.0, arm=30.0, wrist=10.0, rate=0.3, qvel=0.02,
                         pose=0.5, drop=True),
}
CONSTRAINT_PRESETS = {
    "none": {},
    "arm_lock": dict(arm_lock=True),            # finger track: the arm is frozen whenever the pen is down
    "slow_arm": dict(arm_speed=0.3),            # arm limited to 30 % of its max speed
    "fingers_only": dict(arm_speed=0.0),        # arm frozen for the whole episode
    # ring + pinky SERVOED to the captured grip pose, the policy's torque on them ignored:
    # they take no part in the tripod and a reward term alone left them curled 19-28 deg
    # off the human pose
    "idle_lock": dict(idle_lock=True),
    "fingers_only_lock": dict(arm_speed=0.0, idle_lock=True),
    # the arm may CREEP (8 % of max speed) to reposition while wrist + fingers
    # write - the human solution for glyphs larger than the wrist's touch envelope
    # (guaranteed touch only within ~2 cm of the start point)
    "micro_arm_lock": dict(arm_speed=0.08, idle_lock=True),
    # 8 % is too tight for L1: the scripted teacher itself only reaches completion
    # 0.66 / 0.05 / 0.39 on line / circle / shape under it, so a 0.5 gate is not
    # reachable by anything. 15 % is the smallest cap that clears it (0.78 / 0.82 /
    # 0.79); 25 % and 40 % buy nothing more (6 seeds each).
    "mini_arm_lock": dict(arm_speed=0.15, idle_lock=True),
    # in the fingers-forward pose the teacher's strokes are ARM-driven
    # (0.17-0.39 m of arm travel per L1 episode with the arm free) and the 15 % cap makes
    # L1 infeasible outright - the teacher drops the pen 6/6 and scores ~0 on every L1
    # task under it, against 0.69-0.81 with the arm free. Wider caps, same idle lock.
    "arm30_lock": dict(arm_speed=0.30, idle_lock=True),
    "arm50_lock": dict(arm_speed=0.50, idle_lock=True),
    "arm70_lock": dict(arm_speed=0.70, idle_lock=True),
    # if the env drove arm AND wrist through all of L0, a policy's wrist torque outputs were
    # never trained - released at L1 they flail and the pen is gone within seconds even
    # with the arm standing still. `wrist_hold` keeps the wrist on the env's position servo
    # at its placed angle (exactly what the hold rungs do) for a first RL stage in which the
    # arm learns to trace; a second stage releases it. Training-only: the accept runs the
    # track's constraint.
    "arm50_wristhold_lock": dict(arm_speed=0.50, idle_lock=True, wrist_hold=True),
    "arm25_wristhold_lock": dict(arm_speed=0.25, idle_lock=True, wrist_hold=True),   # = arm50 x scale-arm 0.5, feasibility check
    "no_wrist": dict(wrist_speed=0.0),
}


class DexCraftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, task_id="L2_write_word", control_mode="hybrid", obs_mode="state",
                 cameras=("top", "wrist"), image_size=(224, 224), control_hz=100,
                 perturb="none", constraint="none", reward="default",
                 force_gain=1.0, force_jitter=0.0, assist_steps=0, grip_assist=0.0, capture_radius=None,
                 wrist_teacher=0.0, glyph_scale=1.0, goal_on_paper=True, roll_offset=0.0, yaw_offset=None,
                 reach=None, start_height=None, start_wrist=False, soft_start=0.0, arm_seed=None, ik_iters=80, ik_retries=3,
                 pen_axis_roll=None,
                 tool_gravity=1.0, tool_gravity_min=None,
                 brush_cfg="wolf", paper_cfg="shuxuan",
                 render_mode=None, seed=None,
                 disabled_joints=(), end_mode="fixed", cursor_obs=True, cursor="auto",
                 traj_obs=0, traj_stride=2, adv_mode="ink"):
        assert control_mode in CONTROL_MODES, control_mode
        self.task_id, self.control_mode, self.obs_mode = task_id, control_mode, obs_mode
        self.cameras, self.image_size = list(cameras), image_size
        self.control_hz = control_hz
        frame_skip = max(1, int(round(500.0 / control_hz)))
        self.frame_skip = frame_skip
        self.perturb = dict(PERTURB_PRESETS[perturb]) if isinstance(perturb, str) else dict(perturb)
        self.constraint = dict(CONSTRAINT_PRESETS[constraint]) if isinstance(constraint, str) else dict(constraint)
        if isinstance(reward, dict) and "__preset__" in reward:      # a preset plus overrides, e.g. from --env-kwargs
            reward = dict(REWARD_PRESETS[reward["__preset__"]], **{k: v for k, v in reward.items() if k != "__preset__"})
        self.rw = dict(REWARD_PRESETS[reward]) if isinstance(reward, str) else dict(reward)
        if "tol_by_task" in self.rw:
            # stage 2+: the tolerance curriculum applies to the task being added; the tasks already
            # learned keep the 1 mm they were learned at. Loosening an already-learned task along with
            # the new one makes the policy unlearn it.
            t = self.rw["tol_by_task"].get(task_id, self.rw["tol_by_task"].get("default"))
            if t is not None:
                self.rw["goal_tol"], self.rw["adv_tol"] = float(t), max(float(t), 0.0015)
        if "adv_tol_by_task" in self.rw:
            # the POINTER's tolerance, decoupled from the reward's: on the circle at 2 mm, the tip
            # tracks 3-5 mm off around 100-135 deg, the pointer stalls (4 points/s against the timeline's
            # 37), the pen presses at 2-3 N while waiting and slips out. A looser pointer keeps the
            # trajectory flowing while the completion term (goal_tol) still only pays inside 2 mm.
            t = self.rw["adv_tol_by_task"].get(task_id, self.rw["adv_tol_by_task"].get("default"))
            if t is not None:
                self.rw["adv_tol"] = float(t)
        # end_mode="policy": the policy declares the end of a stroke task (an extra action
        # entry), the clock keeps running up to 2x the nominal time, and the cursor (where the
        # target's schedule stands) can be kept OUT of the observation - it then travels in
        # `info` for the scripted expert, whose job is to demonstrate. With a policy-declared
        # end the cursor is progress-driven ("patient"): it advances as the tip covers it.
        self.end_mode, self.cursor_obs = end_mode, bool(cursor_obs)
        # The task as a trajectory instead of a picture. `traj_obs` future points of the
        # timeline, every `traj_stride`-th, each (dx, dy, dz from the tip, target force, due time
        # from now). `adv_mode="tip"`: the pointer releases a pen-down point on the TIP being there
        # and touching, not on ink being there - a robot can compute the first and not the second.
        self.traj_obs, self.traj_stride = int(traj_obs), max(1, int(traj_stride))
        self.adv_mode = adv_mode
        self.progress_cursor = cursor == "progress" or (cursor == "auto" and end_mode == "policy")
        if self.progress_cursor:
            self.rw["patient"] = True
        self._cursor_priv, self._done_req = None, False
        self.force_gain = force_gain
        # curriculum: hold the tool in the hand for the first N control steps so the
        # policy can learn to squeeze before it has to catch. Under a torque action a
        # random policy drops the tool within ~2 s at ANY action scale (measured), so
        # without this there is no regime to bootstrap from. Anneal to 0.
        self.assist_steps = int(assist_steps)
        # Curriculum: the fraction of gravity the HELD TOOL feels. Pinning the tool
        # (assist_steps) makes the grip causally irrelevant - squeezing and not
        # squeezing look identical while it is pinned, so there is nothing to learn
        # and it simply drops the step the pin is released. Scaling its weight
        # instead keeps a weak grip failing, just more slowly.
        # Curriculum: a nominal grip torque the ENV supplies, which the policy's own
        # torque rides on top of. 1.0 = the default grip is held for you, 0.0 = you
        # hold it yourself (the benchmark contract). Anneal to 0.
        #
        # Neither of the other two knobs works here:
        #  - pinning the tool (assist_steps) makes the grip causally irrelevant, so
        #    the tool drops the step the pin is released;
        #  - gravity is not the mechanism at all. At ZERO gravity a zero-torque hand
        #    still loses the tool in 16 steps: the tool's reaction pushes the fingers
        #    open (0.1 -> 4.0 deg, contacts 5 -> 0 in 10 steps) and nothing resists.
        self.grip_assist = float(grip_assist)
        # Curriculum: widen the waypoint capture radius. The first positive feedback
        # on an ink task needs three things at once - tip inside the radius, contact
        # held, force near the target - and a policy that misses any one of them sees
        # a flat reward. Widening it is the cheapest way to get a non-zero completion
        # first and then tighten. None = the task's own value (2.5 mm).
        self.capture_radius = capture_radius
        # "arm frozen" training aids (None/False = the benchmark contract):
        #   reach        cap the target's extent [m]: strokes larger than this are scaled down
        #   start_height (lo, hi): at reset the ARM places the pen tip over the target, this
        #                high above the sheet (negative = pressed into it), and the grip
        #                transfers rigidly; with constraint "fingers_only" the wrist + fingers
        #                then have to write on the wrist sphere's flat cap (see _place_hand)
        #   start_wrist  also randomise the wrist start near the settled posture
        self.reach, self.start_height, self.start_wrist = reach, start_height, start_wrist
        self.soft_start = float(soft_start)      # [s] ramp on the arm/wrist action at the start of an episode
        # 6 arm joint angles [rad] to seed _place_hand's IK from. The damped-LS loop
        # converges to whichever elbow branch it starts on, and the settled grip pose
        # is elbow-up; seeding elbow-down is how the writing pose gets fixed.
        seed = S.ARM_SEED_DEFAULT if arm_seed is None else arm_seed
        self.arm_seed = None if seed is None else np.asarray(seed, float)
        self.ik_iters = int(ik_iters)
        self.ik_retries = int(ik_retries)      # alternative wrist_2 seeds tried if the first IK fails
        # Rotation [rad] of the whole hand+pen assembly about the PEN'S OWN AXIS at placement.
        # The grip is rigid, so the pen keeps its angle to the sheet while the hand swings
        # from fingers-down (the solved grip's dagger hold) toward palm-down, which is what
        # lets the wrist come down to the sheet.
        self.pen_axis_roll = float(S.PEN_AXIS_ROLL_DEFAULT if pen_axis_roll is None else pen_axis_roll)
        # sim2real: the pen-tip force the policy senses (and the ink model uses) gets a
        # per-episode gain error, so it learns to regulate RELATIVE pressure - darker vs
        # lighter - instead of trusting an absolute newton calibration
        self.force_jitter = float(force_jitter)
        self._f_gain = 1.0
        # blend a wrist-IK teacher into the WRIST duty only - the policy's own
        # fingers keep the grip. Teaches timeline-following (and stroke direction, by
        # construction) the way grip_assist taught holding. 0 = off.
        self.wrist_teacher = float(wrist_teacher)
        self.glyph_scale = float(glyph_scale)
        # visual only: the target glyph painted light-grey on the paper texture, merged
        # with the ink at UPLOAD time - never into self.ink, which feeds the observation
        self.goal_on_paper = bool(goal_on_paper)
        self._goal_tex = None
        self.roll_offset = float(roll_offset)
        # yaw the whole writing pose about the world z axis at reset. The classic
        # pose (0) points the pen's BACK end away from the arm base, so the tip trails
        # the wrist by 22 mm - the hand reaches past the point it is writing. pi turns
        # that around: tip 22 mm AHEAD of the wrist, barrel leaning back over the hand,
        # which is how a human holds it. roll_offset does NOT do this (it only
        # rotates the azimuth 25 deg within the same tip-trailing family).
        self.yaw_offset = float(S.YAW_OFFSET_DEFAULT if yaw_offset is None else yaw_offset)
        self._ik_scratch = None
        self._ik_q = None
        self._ik_zoff = 0.0
        self.placed_ok = True
        self.tool_gravity = float(tool_gravity)
        self.tool_gravity_min = tool_gravity if tool_gravity_min is None else float(tool_gravity_min)
        self.render_mode = render_mode
        self.grip = S.load_grip()          # remapped by joint/actuator name onto the current arm
        spec0 = T.sample(task_id, 0)
        self.tool = spec0.tool                                # tool is a property of the task family
        self.paint = spec0.palette is not None                # L5: paint wells + a brush rack in the scene
        self.model = S.build_model(grip=None, tool=self.tool, paint=self.paint)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep * frame_skip
        self.jn = S.names(m, mujoco.mjtObj.mjOBJ_JOINT)
        self.an = S.names(m, mujoco.mjtObj.mjOBJ_ACTUATOR)
        assert self.an == [S.ACTUATOR_OF[j] for j in S.ALL_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[self.jn.index(j)] for j in S.ALL_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[self.jn.index(j)] for j in S.ALL_JOINTS])
        self.lo = np.array([S.JOINT_RANGE[j][0] for j in S.ALL_JOINTS])
        self.hi = np.array([S.JOINT_RANGE[j][1] for j in S.ALL_JOINTS])
        self.max_vel = np.array([S.MAX_VEL[j] for j in S.ALL_JOINTS])
        self.tau = m.actuator_forcerange[:, 1].copy()
        # A cut tendon: the joint's actuator no longer acts. Its servo
        # gain and its torque channel are zeroed, so the joint is passive - its own damping
        # and limits only - and whatever the policy or the idle lock commands it is ignored.
        self.disabled_joints = tuple(disabled_joints)
        self.disabled_idx = np.array([S.ALL_JOINTS.index(j) for j in self.disabled_joints], dtype=int)
        self._act_mask = np.ones(NU)
        for i in self.disabled_idx:
            m.actuator_gainprm[i, :] = 0.0; m.actuator_biasprm[i, :] = 0.0; self._act_mask[i] = 0.0
        self.arm_idx = np.arange(0, N_ARM)
        self.wrist_idx = np.arange(N_ARM, N_ARM + N_WRIST)
        self.hand_idx = np.arange(N_ARM + N_WRIST, NU)
        # ring + pinky: no part in the tripod, but they must not flail while the others write
        self._idle_mask = np.array([j.startswith(("ring", "pinky")) for j in S.HAND_JOINTS])
        self.idle_idx = self.hand_idx[self._idle_mask]
        self.pq, self.pd = S.pen_joint(m)
        self.pen_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
        kp = m.actuator_gainprm[:, 0].copy(); kv = -m.actuator_biasprm[:, 2].copy()
        self._ga_kp, self._ga_kv = 0.5 * kp[N_ARM:], 0.10 * kv[N_ARM:]   # as the teacher's PD
        self.tool_bids = [b for b in range(m.nbody)
                          if b == self.pen_bid or m.body_rootid[b] == self.pen_bid]
        self.pen_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pen_geom")
        tool_bodies = {self.pen_bid} | {b for b in range(m.nbody) if m.body_parentid[b] == self.pen_bid} | \
                      {b for b in range(m.nbody) if m.body_parentid[b] in
                       {b2 for b2 in range(m.nbody) if m.body_parentid[b2] == self.pen_bid}}
        self.pen_gids = {g for g in range(m.ngeom) if m.geom_bodyid[g] in tool_bodies}
        self.paper_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "paper")
        self.tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")
        # L3: the tuft is a reduced-order model outside MuJoCo (brush.py). `brush_cfg` /
        # `paper_cfg` name presets in assets/brush_configs.json and are contract terms.
        self.brush = None
        if self.tool == "brush":
            from .brush import BrushTipModel, InkRenderer, brush_config, paper_config
            from skimage.transform import resize
            self.brush_cfg, self.paper_cfg = brush_config(brush_cfg), paper_config(paper_cfg)
            if self.paint:
                # L5: three brushes on the rack; the medium one (= the L3 brush) starts in the hand
                from .brush import BRUSHES, MEDIUM
                self.brush_cfgs = [brush_config(n) for n in BRUSHES]
                self.brush_id, self.brush_cfg = MEDIUM, self.brush_cfgs[MEDIUM]
                self.hair_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "hair")
                self._hair_rgba0 = m.mat_rgba[self.hair_mat].copy()
            self.brush = BrushTipModel(self.brush_cfg)
            self.brush_render = InkRenderer(S.PAPER_CENTER, S.PAPER_HALF, self.brush_cfg)
            self.ferrule_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pen_ferrule_geom")
            self._vel6 = np.zeros(6)
            # the dry-brush mask, resampled for the display texture and the observation map
            self._noise_tex = resize(self.brush_render.noise, (S.INK_RES, S.INK_RES), order=1).astype(np.float32)
            self._noise_map = resize(self.brush_render.noise, (INK_MAP_H, INK_MAP_W), order=1).astype(np.float32)
            self._brush_st, self._brush_centre, self._brush_fn, self._brush_ft = None, None, 0.0, 0.0
            self._brush_viol_steps, self._brush_viol_sub = 0, False
        self.hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
        self.grasp_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_dip") for f in S.GRASP_FINGERS]
        self.tip_nominal = np.asarray(self.grip["settled" if self.tool in ("pen", "marker") else f"settled_{self.tool}"]["tip_in_hand"])
        self.chop = self.tool == "chopsticks"
        self.chop_hold = self.chop and T.sample(task_id, 0).hold_motion is not None
        self.pickup = T.sample(task_id, 0).start == "table"     # L0_pick_up: start with an empty hand
        if self.chop:
            self.obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
            oj = [j for j in range(m.njnt) if m.jnt_bodyid[j] == self.obj_bid][0]
            self.oq, self.od = m.jnt_qposadr[oj], m.jnt_dofadr[oj]
            self.obj_gids = {sh: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"object_{sh}") for sh in S.OBJECT_SHAPES}
            self.bowl_mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")]
            self.lower_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lower_tip")
            self.lower_gids = {g for g in range(m.ngeom)
                               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("lower_")}
        self.tex_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "ink_tex")
        self.tex_adr = m.tex_adr[self.tex_id]
        self.finger_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] > self.hb
                            and m.geom_bodyid[g] != self.pen_bid]
        self.finger_gids = [g for g in self.finger_gids if not self.chop or g not in self.lower_gids]
        self._nominal = dict(pen_mass=m.body_mass[self.pen_bid].copy(),
                             pen_inertia=m.body_inertia[self.pen_bid].copy(),
                             pen_friction=m.geom_friction[self.pen_gid].copy(),
                             paper_friction=m.geom_friction[self.paper_gid].copy(),
                             light_diffuse=m.light_diffuse.copy(), cam_pos=m.cam_pos.copy())
        self.ink = np.full((S.INK_RES, S.INK_RES, 3), 255, dtype=np.uint8)
        self._inkmap = np.zeros((INK_MAP_H, INK_MAP_W), np.uint8)
        # L5 paint keys (present for every task, zero unless the task is a painting):
        # the target's palette index per pixel, the palette, the colour index of the paint laid
        self._task_color = np.zeros((INK_MAP_H, INK_MAP_W), np.uint8)
        self._colmap = np.zeros((INK_MAP_H, INK_MAP_W), np.uint8)
        self._palette = np.zeros((PAINT_K, 3), np.float32)
        self._paint_idx = 0
        self._renderer = None
        self.arm_dofs = S.arm_dofs(m)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (NU + (1 if end_mode == "policy" else 0),), np.float32)   # joint torque, as a fraction
                                                                       # of what the motor can give NOW
        self.tip_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_dip_geom")
                         for f in S.GRASP_FINGERS]
        self.tip_radii = [S.fingertip_radius(m, f) for f in S.GRASP_FINGERS]
        self.finger_geoms = {f: {g for g in range(m.ngeom)
                                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f + "_")}
                             for f in S.FINGERS}
        self.reset(seed=seed)
        mapspace = spaces.Box(0, 255, (INK_MAP_H, INK_MAP_W), np.uint8)
        self.observation_space = spaces.Dict({
            "position": spaces.Box(-np.inf, np.inf, (GROUP_DIM["position"],), np.float32),
            "tactile": spaces.Box(-np.inf, np.inf, (GROUP_DIM["tactile"],), np.float32),
            # L5 paint adds keys, never changes the existing ones: `color` maps are palette
            # indices (0 = none), `palette` the well colours, `paint` = colour on the brush
            # (RGB) + which brush is in the hand (one-hot 3). All zero outside a painting.
            "task": spaces.Dict(dict({"map": mapspace,
                                 "vec": spaces.Box(-np.inf, np.inf, (GROUP_DIM["task"],), np.float32),
                                 "color": mapspace, "palette": spaces.Box(0.0, 1.0, (PAINT_K, 3), np.float32)},
                                **({"traj": spaces.Box(-np.inf, np.inf, (self.traj_obs, 5), np.float32)} if self.traj_obs else {}))),
            "external": spaces.Dict({"map": mapspace,
                                     "vec": spaces.Box(-np.inf, np.inf, (GROUP_DIM["external"],), np.float32),
                                     "color": mapspace, "paint": spaces.Box(-np.inf, np.inf, (6,), np.float32)}),
        })

    # ------------------------------------------------------------ target path
    def _build_waypoints(self, task):
        """Strokes -> (N,3) world waypoints, pen_down flags, target widths, target forces.

        These are the env's own bookkeeping for progress and for the force metric.
        They are NOT part of the observation - a submission sees the task only as
        the map."""
        z_dn, z_up = S.PAPER_TOP - PRESS_DEPTH, S.PAPER_TOP + LIFT_HEIGHT
        pts, down, width, force = [], [], [], []
        tip = self._tip().copy()
        for k, s in enumerate(task.strokes):
            P = np.asarray(s) + S.PAPER_CENTER[:2]
            w = task.widths[k] if task.widths is not None else None
            f = task.forces[k] if task.forces is not None else None
            P3 = resample(np.column_stack([P, np.full(len(P), z_dn)]), WAYPOINT_SPACING)
            L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            Lr = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P3[:, :2], axis=0), axis=1))])
            if w is not None:
                w = np.interp(Lr, L, w)
            if f is not None:
                f = np.interp(Lr, L, f)
            if self.brush is not None and f is not None:
                # the brush presses to a DEPTH: its target force implies one (brush.py), and
                # the timeline built from these points then asks for that depth
                from .brush import depth_for_force
                cfg = self.brush_cfgs[task.brushes[k]] if self.paint else self.brush_cfg
                P3[:, 2] = S.PAPER_TOP - np.array([depth_for_force(cfg, v) for v in f])
            start_up = P3[0] + [0, 0, z_up - z_dn]
            for q in resample(np.stack([tip, start_up]), 0.004)[1:]:
                pts.append(q); down.append(False); width.append(0.0); force.append(0.0)
            for q in resample(np.stack([start_up, P3[0]]), 0.002)[1:]:
                pts.append(q); down.append(False); width.append(0.0); force.append(0.0)
            for i, q in enumerate(P3):
                pts.append(q); down.append(True); width.append(float(w[i]) if w is not None else 0.0)
                force.append(float(f[i]) if f is not None else 0.8)
            end_up = P3[-1] + [0, 0, z_up - z_dn]
            for q in resample(np.stack([P3[-1], end_up]), 0.002)[1:]:
                pts.append(q); down.append(False); width.append(0.0); force.append(0.0)
            tip = end_up
        if not pts:
            pts, down, width, force = [tip], [False], [0.0], [0.0]
        return (np.array(pts), np.array(down, dtype=bool), np.array(width), np.array(force))

    # ------------------------------------------------------------ reset / step
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        m, d = self.model, self.data
        self.task, (self.win_c, self.win_h), self._task_map, self._task_pts = \
            self._task_package(int(self.np_random.integers(1 << 31)))
        mujoco.mj_resetData(m, d)
        self._apply_perturb_params()             # mj_setConst resets qpos -> do it first
        if self.chop and not self.chop_hold:
            self._configure_object()             # (also mj_setConst)
        if self.pickup:
            self._reset_empty_hand()
        else:
            settled = self.grip["settled" if self.tool in ("pen", "marker") else f"settled_{self.tool}"]
            d.qpos[:] = settled["qpos"]
            d.ctrl[:] = settled["ctrl"]
            # The grip file's pen pose is a WORLD pose recorded on the 6-axis arm; on this arm
            # the same joint values put the hand elsewhere. Carry the pen at its hand-frame
            # pose instead (settled_hand), so the grip is exact whatever the arm is doing.
            self._pen_into_hand()
        if self.chop and not self.chop_hold:
            self._place_object()
        elif self.chop_hold:
            for g in self.obj_gids.values():     # rung 1 has no object at all
                m.geom_contype[g], m.geom_conaffinity[g], m.geom_rgba[g, 3] = 0, 0, 0.0
            d.qpos[self.oq:self.oq + 3] = S.OBJECT_HOME
        mujoco.mj_forward(m, d)
        if (self.start_height is not None and self.task.hold_motion in (None, "scribble") and not self.pickup
                and (self.task.strokes or self.task.place_at is not None)):
            self._place_hand()
        elif self.task.hold_motion is not None and not self.pickup:
            # hold rungs: same placement, tip hovering 10 cm over the sheet centre - the hand
            # orientation and pen-in-hand pose come from the solved grip, not from the arm
            self._place_hand(place_xy=np.zeros(2), tip_z=S.PAPER_TOP + 0.10)
        elif self.task.place_at is not None and not self.pickup:
            # a task with a spot but no strokes (L0_scribble_fingers): the arm is frozen for the
            # whole episode, so the tip must START over its spot - left at the home pose the
            # fingers would scribble in the air a metre from the sheet
            self._place_hand(place_xy=np.asarray(self.task.place_at, float), tip_z=S.PAPER_TOP + 0.003)
        self.target_q = d.ctrl.copy()            # servo targets the policy moves
        self.vel_cmd = np.zeros(m.nu)
        self.ink[:] = 255
        self._goal_tex = None
        if self.goal_on_paper and self.task.strokes:
            self._goal_tex = np.full_like(self.ink, 255)
            for st, wd in zip(self.task.strokes,
                              self.task.widths if self.task.widths is not None else
                              [np.full(len(np.atleast_2d(s_)), 1.0) for s_ in self.task.strokes]):
                P = np.asarray(st) + S.PAPER_CENTER[:2]
                for i, p in enumerate(P):
                    self._paint_into(self._goal_tex, p, float(np.atleast_1d(wd)[min(i, len(np.atleast_1d(wd)) - 1)]),
                                     np.array([185.0, 185.0, 185.0], np.float32))
        self._inkmap[:] = 0
        self.ink_points = []                      # (x, y, width_mm, t)
        self.tip_log, self.arm_log, self.action_log = [], [], []
        self.touch_log = []              # was the tool ON the sheet this step
        self.t = 0
        self.T = int(round(self.task.duration / self.dt))
        if self.rw.get("tl_slow"):
            # training-only curriculum (reward preset): a longer episode makes the time-driven
            # timeline proportionally slower (its speed is total length / budget), so the
            # target runs away from a learner more slowly. The accept preset never sets it.
            self.T = int(round(self.T * float(self.rw["tl_slow"])))
        if self.rw.get("patient") and self.task.strokes:
            # waiting needs budget - on stroke tasks only: the hold tasks' driven trajectories are
            # paced for the nominal length, and a 1.5x episode throws the pen
            self.T = int(round(self.T * 1.5))
        # policy-declared end: the schedule is paced for T, the episode may run to T_end
        self.T_end = int(round(self.T * 2)) if (self.end_mode == "policy" and self.task.strokes) else self.T
        self._done_req = False
        self.wp, self.wp_down, self.wp_width, self.wp_force = self._build_waypoints(self.task)
        self.wp_i = 0
        self.done_time = None
        self.dropped = False
        self._drop_timer = 0.0
        self.hovered, self._hover_timer, self._far_timer = False, 0.0, 0.0   # training rules, see _check_hover
        self._action_queue = []
        self._next_push = self.perturb.get("push_every", np.inf)
        R = d.xmat[self.hb].reshape(3, 3)
        self._tip_rel = R.T @ (self._tip() - d.xpos[self.hb])   # tool tip in the hand frame
        self._timeline, self._tl_down = self._build_timeline()
        self._tl_h, self._tl_f = self._timeline_zf()
        self._drive = None                      # set by _ik_track; online only for the driven scribble
        self._hold_traj = self._make_hold_motion() if self.task.hold_motion else None
        if self.task.hold_motion != "scribble":
            self._drive = None
        self._pen_rel0 = self._pen_in_hand()
        self._last_action = np.zeros(m.nu)
        self._prev_action = np.zeros(m.nu)
        self._tau_ff = np.zeros(m.nu)
        self._force_err, self._force_n = 0.0, 0
        self._wp_i_prev = 0
        self._e_prev = None
        self._done_prev = False
        self._vsplit = (np.zeros(3), np.zeros(3), np.zeros(3))
        self._force = 0.0                                    # cached tool-paper force for this step
        self._contact_hits, self._force_acc, self._contact_duty, self._force_mean = 0, 0.0, 0.0, 0.0
        self._f_gain = 1.0 + self.np_random.uniform(-self.force_jitter, self.force_jitter) \
            if self.force_jitter > 0 else 1.0
        self._ik_q, self._ik_zoff = None, 0.0
        self._tip_prev = None
        # patient-timeline state
        self._tl_k, self._tl_wait, self._ink_new = 0, 0.0, []
        self._goal_line, self._goal_tree, self._goal_cov, self._goal_j = None, None, None, 0     # v15_goal
        self._goal_ink_on = self._goal_ink_n = self._goal_ferr = self._goal_fn = 0
        self._goal_off_prev = self._goal_mae_prev = 0.0
        self._goal_prec_paid = self._goal_comp_paid = self._goal_force_paid = 0.0
        self._goal_slip_prev = None
        self._tl_hits = 0
        self._reentry_paid = set()
        self._tl_stroke_id = None
        self._tl_n_down = 0
        self._tl_adv_down = self._tl_adv_up = self._tl_reentry = 0
        self._peak_force = 0.0                               # for L0_scribble's impact limit
        self._travel_prev = None
        if self.brush is not None:
            self.brush.reset(float(self.task.meta.get("ink0", 1.0)))     # a full brush unless the task says otherwise
            self.brush_render.reset()
            self._brush_st, self._brush_centre, self._brush_fn, self._brush_ft = None, None, 0.0, 0.0
            self._brush_viol_steps, self._brush_viol_sub = 0, False
            if self.paint:
                from .brush import MEDIUM
                self._set_brush(MEDIUM)                                    # ... but a painter starts DRY
                self._dip_t, self._dip_done, self._rack_latch = 0.0, False, None
                self._dips = self._swaps = 0
                self._paint_idx = 0
                self._colmap[:] = 0
                self._task_color = self._rasterize_color_map(self.task, (self.win_c, self.win_h))
                self._palette[:] = 0.0
                self._palette[:len(self.task.palette)] = np.asarray(self.task.palette, np.float32)
        self._assist_rel = None                              # tool pose in the hand frame, for assist
        self._g_scale = float(self.np_random.uniform(min(self.tool_gravity_min, self.tool_gravity),
                                                     max(self.tool_gravity_min, self.tool_gravity)))
        self._duty = np.zeros(NU)                            # last motor command (torque mode)
        self._tau_applied = np.zeros(NU)                     # torque actually delivered last substep
        # The pose regulariser pulls toward the CAPTURED HUMAN GRIP, not toward
        # whatever pose the physics settled into. Without this the policy is free to
        # drift into a grip that holds well and looks nothing like a hand, e.g. the
        # pen held against the palm with the thumb at 0.00 N.
        self._q_grip_hand = np.array([_HUMAN_POSE.get(j, 0.0) for j in S.HAND_JOINTS])
        # the SERVO TARGET, not qpos: their difference is the squeeze preload that
        # actually holds the tool (up to 2.3 N.m on the grasp fingers). Targeting
        # qpos gives zero error at the settled pose, hence zero grip.
        self._q_grip_all = self.target_q[N_ARM:].copy()               # wrist + hand, for grip_assist
        self._share_fing = self._share_wrist = self._share_arm = self._share_den = 0.0
        self._n_lifts, self._was_down, self._lift_armed = 0, False, False
        self._cov_prev, self._lifts_paid, self._cov_base = 0, 0, None
        self._share_prev_tip = None
        self._idle_dev2, self._idle_vel2, self._idle_n = 0.0, 0.0, 0     # ring + pinky quietness
        self.phase, self.grasp_time, self.lift_time, self.obj_drops, self._obj_held = 0, None, None, 0, False
        self.pick_phase, self.pick_time, self._hold_steps, self._ever_held = 0, None, 0, not self.pickup
        self.obj_lost = False
        return self._obs(), self._info()

    # ------------------------------------------------------------ task packages
    @staticmethod
    def _task_window(task):
        """The maps cover the WHOLE board and nothing else: centred on the sheet, one
        half-extent per axis, so the image is the board 1:1 (absolute alignment, at the
        sheet's 4:3 aspect)."""
        return np.zeros(2), S.PAPER_HALF[:2].astype(float).copy()

    def _task_package(self, seed):
        """Everything about an episode's task that does not depend on the physics
        state: the sample (with `reach` scaling), its window, its map and the map's
        centreline."""
        task = T.sample(self.task_id, seed)
        if self.glyph_scale != 1.0 and task.strokes:
            P = np.concatenate(task.strokes)
            c = P.mean(axis=0)
            task.strokes = [(np.asarray(st) - c) * self.glyph_scale + c for st in task.strokes]
        if self.reach and task.strokes:
            P = np.concatenate([np.asarray(s) for s in task.strokes])
            ext = float((P.max(0) - P.min(0)).max())
            if ext > self.reach:
                c = P.mean(0)
                task.strokes = [(np.asarray(s) - c) * (self.reach / ext) + c for s in task.strokes]
        win = self._task_window(task)
        tmap = self._rasterize_task_map(task, win)
        return task, win, tmap, self._task_centreline(task, tmap, win)


    def _reset_empty_hand(self):
        """Arm at its home pose, fingers straight, tool lying on the table beside
        the sheet at the spot this episode drew."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[self.qadr[self.hand_idx]] = 0.0              # zero = a straight, open hand
        d.ctrl[:] = d.qpos[self.qadr]
        mujoco.mj_forward(m, d)
        meta = self.task.meta
        x = meta.get("rest_x", np.mean(S.TOOL_REST_X))
        y = meta.get("rest_y", np.mean(S.TOOL_REST_Y))
        yaw = meta.get("rest_yaw", 0.0)
        axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])    # lying flat, random heading
        d.qpos[self.pq:self.pq + 3] = [x, y, S.TABLE_TOP + S.PEN_RADIUS]
        d.qpos[self.pq + 3:self.pq + 7] = S.pen_quat(axis)
        d.qvel[self.pd:self.pd + 6] = 0.0
        self.target_q = d.ctrl.copy()

    def _update_pickup(self):
        """not touching -> fingers on it -> lifted off the table -> held steadily."""
        d = self.data
        touching = any(c.geom1 in self.finger_gids or c.geom2 in self.finger_gids
                       for c in (d.contact[i] for i in range(d.ncon))
                       if c.geom1 in self.pen_gids or c.geom2 in self.pen_gids)
        lifted = d.qpos[self.pq + 2] > S.TABLE_TOP + PICK_LIFT
        if touching and self.pick_phase < 1:
            self.pick_phase = 1
        if touching and lifted:
            self._ever_held = True
            self._hold_steps += 1
            self.pick_phase = max(self.pick_phase, 2)
            if self._hold_steps * self.dt >= PICK_HOLD_S:
                self.pick_phase = 3
                if self.done_time is None:
                    self.done_time = self.t * self.dt
                if self.pick_time is None:
                    self.pick_time = self.t * self.dt
        else:
            self._hold_steps = 0

    # ------------------------------------------------------------ L5 objects
    def _configure_object(self):
        """Enable the task's object shape and set its size / mass / friction
        (model edits, then mj_setConst - which resets qpos)."""
        m, d, meta = self.model, self.data, self.task.meta
        sz, shape = meta["size"], meta["shape"]
        P = self.perturb
        rng = self.np_random
        s = lambda key: 1.0 + rng.uniform(-P[key], P[key]) if key in P else 1.0
        for sh, g in self.obj_gids.items():
            on = sh == shape
            m.geom_contype[g], m.geom_conaffinity[g] = (8, 15) if on else (0, 0)
            m.geom_rgba[g, 3] = 1.0 if on else 0.0
            m.geom_group[g] = 1 if on else 4
            m.geom_friction[g] = [meta["friction"] * s("pen_friction"), 0.02, 0.001]
        g = self.obj_gids[shape]
        if shape == "sphere":
            m.geom_size[g] = [sz, 0, 0]; half_h = sz
        elif shape == "cube":                                   # aspect > 1 makes it an oblong block
            m.geom_size[g] = [sz * meta.get("aspect", 1.0), sz, sz]; half_h = sz
        elif shape == "cylinder":
            m.geom_size[g] = [sz, 0.8 * sz, 0]; half_h = 0.8 * sz
        else:                                                   # capsule
            m.geom_size[g] = [0.7 * sz, 0.6 * sz, 0]; half_h = 0.7 * sz
        m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g][:3])) + 1e-3
        m.geom_solref[g][:2] = meta.get("solref", [0.005, 1.0])   # soft = a rubbery, forgiving pinch
        mass = meta["mass"] * s("pen_mass")
        m.body_mass[self.obj_bid] = mass
        m.body_inertia[self.obj_bid] = np.full(3, 0.4 * mass * sz ** 2)
        self.obj_half_h = half_h
        self.obj_rest_z = S.PAPER_TOP + half_h
        mujoco.mj_setConst(m, d)

    def _place_object(self):
        """Object on the sheet, bowl (mocap) at its task position."""
        d, meta = self.data, self.task.meta
        d.qpos[self.oq:self.oq + 3] = np.r_[np.asarray(meta["object_xy"]) + S.PAPER_CENTER[:2], self.obj_rest_z + 0.0005]
        d.qpos[self.oq + 3:self.oq + 7] = [1, 0, 0, 0]
        d.qvel[self.od:self.od + 6] = 0.0
        d.mocap_pos[self.bowl_mid] = np.r_[np.asarray(meta["bowl_xy"]) + S.PAPER_CENTER[:2], 0.0]

    def _object_contacts(self):
        d = self.data
        upper = lower = False
        force = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.obj_bid not in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]):
                continue
            other = c.geom2 if self.model.geom_bodyid[c.geom1] == self.obj_bid else c.geom1
            if other in self.pen_gids:
                upper = True
            elif other in self.lower_gids:
                lower = True
            else:
                continue
            mujoco.mj_contactForce(self.model, d, i, f6)
            force += abs(f6[0])
        return upper, lower, force

    def _update_phase(self):
        d = self.data
        obj = d.qpos[self.oq:self.oq + 3]
        upper, lower, _ = self._object_contacts()
        pinched = upper and lower
        lifted = pinched and obj[2] > self.obj_rest_z + 0.015
        if pinched and self.grasp_time is None:
            self.grasp_time = self.t * self.dt
        if lifted and self.lift_time is None:
            self.lift_time = self.t * self.dt
        if self._obj_held and not pinched and obj[2] > self.obj_rest_z + 0.01 and not self._in_bowl(obj):
            self.obj_drops += 1                                   # let go in mid-air outside the bowl
        self._obj_held = pinched
        if obj[2] < S.PAPER_TOP - 0.03 or np.any(np.abs(obj[:2] - S.PAPER_CENTER[:2]) > S.PAPER_HALF[:2] + 0.15):
            self.obj_lost = True                                  # knocked off the table: over
        placed = (self.lift_time is not None and not upper and not lower and self._in_bowl(obj)
                  and np.linalg.norm(d.qvel[self.od:self.od + 3]) < 0.03)
        self.phase = 3 if (placed or self.done_time is not None) else 2 if self.lift_time is not None else 1 if pinched else 0
        if placed and self.done_time is None:
            self.done_time = self.t * self.dt

    def _in_bowl(self, obj):
        bowl = self.data.mocap_pos[self.bowl_mid]
        return (np.linalg.norm(obj[:2] - bowl[:2]) < S.BOWL_R - 0.5 * self.obj_half_h
                and obj[2] < S.PAPER_TOP + 0.003 + 2 * self.obj_half_h + 0.01)

    def _l5_targets(self):
        """(object, drop target above the bowl, lower-stick tip) in world coordinates."""
        d = self.data
        obj = d.qpos[self.oq:self.oq + 3].copy()
        bowl = d.mocap_pos[self.bowl_mid] + [0, 0, S.PAPER_TOP + 0.003 + self.obj_half_h + 0.025]
        return obj, bowl, d.site_xpos[self.lower_sid].copy()

    def _apply_perturb_params(self):
        m, rng, P = self.model, self.np_random, self.perturb
        s = lambda key: 1.0 + rng.uniform(-P[key], P[key]) if key in P else 1.0
        m.body_mass[self.pen_bid] = self._nominal["pen_mass"] * s("pen_mass")
        m.body_inertia[self.pen_bid] = self._nominal["pen_inertia"] * s("pen_mass")
        m.geom_friction[self.pen_gid] = self._nominal["pen_friction"] * s("pen_friction")
        m.geom_friction[self.paper_gid] = self._nominal["paper_friction"] * s("paper_friction")
        # vision domain randomisation: light intensity and small camera pose jitter
        m.light_diffuse[:] = self._nominal["light_diffuse"] * s("light_jitter")
        if "cam_jitter" in P:
            m.cam_pos[:] = self._nominal["cam_pos"] + rng.uniform(-P["cam_jitter"], P["cam_jitter"], m.cam_pos.shape)
        mujoco.mj_setConst(m, self.data)

    def _build_timeline(self):
        """One target point per control step: approach, descend, traverse each
        stroke at a constant speed, hop between strokes, lift off. Returns (T,3) targets and a
        pen-down flag, or None for tasks with no strokes.

        The target moves on its own clock. That is the whole difference from a
        waypoint cursor: the policy is told at every step where it should be and
        which way that point is going, instead of only being told when it happens
        to arrive somewhere."""
        if not self.task.strokes or self.task.hold_motion is not None:
            return None, None
        lift = np.array([0.0, 0.0, LIFT_HEIGHT])
        pts, down = [], []
        wp = [np.asarray(s_, dtype=np.float64) for s_ in self._stroke_points()]
        # L5 paint: the wells and the rack are part of the job, so the timeline visits them
        # (dip before the first stroke, again on a colour change or a low reservoir, the
        # rack on a brush change) instead of hopping straight to the next stroke
        if self.paint:
            self._swap_before, self._dip_before = self._paint_plan()
        det = [self._paint_detours(k) if self.paint else [] for k in range(len(wp))]
        V_DET, V_DIP = 0.12, 0.06                                 # m/s: across the table / down into a station

        def leg_speed(a, b):
            return V_DIP if np.linalg.norm((b - a)[:2]) < 0.01 else V_DET

        def travel(a, b, v):
            n = max(2, int(round(float(np.linalg.norm(b - a)) / v / self.dt)))
            for i in range(1, n + 1):
                pts.append(a + (i / n) * (b - a)); down.append(0.0)

        def visit(stations):
            for p, dwell in stations:
                travel(np.asarray(pts[-1]), p, leg_speed(np.asarray(pts[-1]), p))
                for _ in range(int(round(dwell / self.dt))):
                    pts.append(p.copy()); down.append(0.0)

        def visit_time(a, stations, b):
            """Seconds a visit costs: legs at their speeds plus the dwells, from a via the stations to b."""
            t, cur = 0.0, np.asarray(a)
            for p, dwell in stations:
                t += float(np.linalg.norm(p - cur)) / leg_speed(cur, p) + dwell
                cur = p
            return t + float(np.linalg.norm(np.asarray(b) - cur)) / V_DET

        # approach: the tip starts ~10 cm above the sheet, so the timeline walks it
        # to the lift point above stroke 1 first (15 cm/s) instead of opening 10 cm away
        tip0, start_up = self._tip().copy(), wp[0][0] + lift
        pts.append(tip0); down.append(0.0)
        if det[0]:
            visit(det[0])
            travel(np.asarray(pts[-1]), start_up, V_DET)
            t_appr = len(pts) * self.dt
        else:
            t_appr = max(float(np.linalg.norm(start_up - tip0)) / 0.15, 0.2)
            n = max(2, int(round(t_appr / self.dt)))
            for i in range(1, n):
                pts.append(tip0 + (i / (n - 1)) * (start_up - tip0)); down.append(0.0)
        n = max(2, int(round(0.5 / self.dt)))                     # descend onto stroke 1
        for i in range(1, n):
            pts.append(wp[0][0] + (1 - i / (n - 1)) * lift); down.append(0.0)
        # One speed for the WHOLE drawing, so the timeline finishes inside the episode.
        # Per-stroke speeds would make every stroke take 60 % of the episode on its own,
        # so with four strokes the timeline would be truncated to the opening of the first.
        total = sum(float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()) for x in wp)
        t_det = sum(visit_time(wp[k - 1][-1] + lift, det[k], wp[k][0] + lift) + 0.3
                    for k in range(1, len(wp)) if det[k])            # station visits between strokes
        budget = max(self.T * self.dt - t_appr - 0.5 - 0.3 * (len(wp) - 1) - 0.3 - t_det, 0.5)
        speed = max(total / budget, 0.005)
        for k, s_ in enumerate(wp):
            seg = np.linalg.norm(np.diff(s_, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            m_ = max(2, int(round(cum[-1] / (speed * self.dt))))
            si = np.linspace(0.0, cum[-1], m_)
            pts.extend(np.stack([np.interp(si, cum, s_[:, j]) for j in range(3)], axis=1))
            down.extend([1.0] * m_)
            if k + 1 < len(wp):                                   # hop to the next
                nx = wp[k + 1][0]
                if det[k + 1]:                                    # ... via the stations it needs
                    pts.append(s_[-1] + lift); down.append(0.0)
                    visit(det[k + 1])
                    travel(np.asarray(pts[-1]), nx + lift, V_DET)
                    n = max(2, int(round(0.3 / self.dt)))
                    for i in range(1, n):
                        pts.append(nx + (1 - i / (n - 1)) * lift); down.append(0.0)
                    continue
                n = max(2, int(round(0.3 / self.dt)))
                for i in range(1, n + 1):
                    u = i / n
                    pts.append(s_[-1] * (1 - u) + nx * u + np.sin(np.pi * u) * lift)
                    down.append(0.0)
        P = np.asarray(pts)
        if len(P) < self.T + 1:                                   # hold the last point
            P = np.vstack([P, np.repeat(P[-1:], self.T + 1 - len(P), axis=0)])
            down = down + [0.0] * (self.T + 1 - len(down))
        return P[:self.T + 1], np.asarray(down[:self.T + 1])

    def _timeline_zf(self):
        """Per timeline point: height above the sheet (0 while writing) and target press force (0
        while the pen is up). The timeline aims pen-down points PRESS_DEPTH into the paper - that is
        the IK teacher's aim point for pressing, not part of the job, so the trajectory reports 0."""
        if self._timeline is None:
            return None, None
        h = np.maximum(self._timeline[:, 2] - S.PAPER_TOP, 0.0)
        f = np.array([self._target_force_at(p) if d > 0.5 else 0.0
                      for p, d in zip(self._timeline, self._tl_down)])
        return h * (self._tl_down < 0.5), f

    def _traj_vec(self):
        """The job as a trajectory. K points from the pointer, every `traj_stride`-th:
        (x - tip x, y - tip y, height above sheet - tip height, target force, due time - now)."""
        K = self.traj_obs
        out = np.zeros((K, 5), np.float32)
        if self._timeline is None:
            return out
        last = len(self._timeline) - 1
        k0 = min(self._tl_k, last) if self.rw.get("patient") else min(self.t, last)
        tip = self._tip()
        tip_h = float(tip[2] - S.PAPER_TOP)
        for i in range(K):
            k = min(k0 + i * self.traj_stride, last)
            p = self._timeline[k]
            out[i] = (p[0] - tip[0], p[1] - tip[1], self._tl_h[k] - tip_h,
                      self._tl_f[k], (k - self.t) * self.dt)
        return out

    def _stroke_points(self):
        """The task's strokes as world-frame 3-D point lists at pen-down height: the
        runs of pen-down waypoints. `wp` cannot be sliced by the raw stroke lengths:
        the approach / descent / lift waypoints between strokes are interleaved, and
        the strokes are resampled."""
        out, cur = [], []
        for p, dn in zip(self.wp, self.wp_down):
            if dn:
                cur.append(p)
            elif cur:
                out.append(np.asarray(cur))
                cur = []
        if cur:
            out.append(np.asarray(cur))
        return [o for o in out if len(o) >= 2] or [np.asarray(self.wp)]

    def _make_hold_motion(self):
        """L0: arm + wrist targets the environment drives (policy arm actions ignored)."""
        Tn = self.T + 1
        n = N_ARM + N_WRIST
        if self.task.hold_motion == "static":
            return np.repeat(self.target_q[None, :n], Tn, axis=0)
        if self.task.hold_motion == "scribble":
            return self._scribble_motion(Tn)
        if self.task.hold_motion == "moving":
            t = self._teleop_motion(Tn, n)      # a real wrist, when we have one
            if t is not None:
                if getattr(self.task, "hold_box", None) is not None:
                    t = self._swing_arm(t)      # the arm carries the wrist through a box
                return t
        rng = self.np_random
        n_way = max(2, int(getattr(self.task, "hold_waypoints", 5)))
        sc = float(getattr(self.task, "hold_scale", 1.0))
        # forearm roll is in the ARM group, so the wrist is N_WRIST wide, not 3:
        # this array has to be n long or the broadcast below throws.
        lim = sc * np.concatenate([np.full(N_ARM, 0.25),
                                   np.array([0.5, 0.4, 0.6])[:N_WRIST]])        # rad
        assert len(lim) == n, (len(lim), n)
        way = rng.uniform(-lim, lim, (n_way, n)) + self.target_q[:n]
        way[0] = self.target_q[:n]
        ts = np.linspace(0, Tn - 1, n_way)
        out = np.zeros((Tn, n))
        for k in range(n):
            for i in range(n_way - 1):
                a, b = int(ts[i]), int(ts[i + 1])
                uu = np.linspace(0, 1, b - a + 1)[:-1]
                out[a:b, k] = way[i, k] + (way[i + 1, k] - way[i, k]) * (1 - np.cos(np.pi * uu)) / 2
            out[-1, k] = way[-1, k]
        return out

    def _teleop_motion(self, Tn, n):
        """Wrist trajectory taken from a random segment of a real teleop session.

        A random-waypoint motion is smooth in
        a way a human wrist is not, and the whole point of the hold tasks is to keep
        the tool through the motion a person actually makes. Random segment, random
        speed, optional mirror, shifted so it starts at the current pose."""
        S_ = _TELEOP
        if not S_:
            return None
        rng = self.np_random
        seg = S_[rng.integers(len(S_))]
        sc = float(getattr(self.task, "hold_scale", 1.0))
        speed = float(rng.uniform(0.6, 1.4)) * sc
        need = int(Tn * speed) + 2
        if len(seg) < need + 2:
            return None
        i0 = int(rng.integers(0, len(seg) - need))
        u = i0 + np.arange(Tn) * speed
        w = np.stack([np.interp(u, np.arange(len(seg)), seg[:, k]) for k in range(3)], axis=1)
        if rng.random() < 0.5:
            w[:, 1] *= -1.0                                   # mirror the deviation
        # the recorded columns are (flex, dev, roll) but forearm roll lives in
        # the ARM group (its last slot) - offset each recorded axis to where it went
        base = np.concatenate([self.target_q[N_ARM:N_ARM + N_WRIST], self.target_q[N_ARM - 1:N_ARM]]) \
            if N_WRIST == 2 else self.target_q[N_ARM:N_ARM + N_WRIST]
        w = (w[:, :len(base)] - w[0, :len(base)]) * sc + base   # no jump at t=0; sc widens the sweep
        # FIT the sweep inside the joint range instead of clipping it. Clipped, a wide
        # segment turns into a bang-bang between the two limits (target pinned on a limit
        # 53-98 % of the episode, 17-25 rad/s peaks). Each axis is scaled about its start value so that its
        # extreme lands HOLD_FIT of the way to the nearer limit; the shake below has the rest.
        slots = ([N_ARM, N_ARM + 1, N_ARM - 1] if N_WRIST == 2 else list(range(N_ARM, N_ARM + N_WRIST)))[:w.shape[1]]
        for k, j in enumerate(slots):
            dev = w[:, k] - base[k]
            up, dn = max(float(dev.max()), 1e-9), max(float(-dev.min()), 1e-9)
            fit = min(1.0, HOLD_FIT * (self.hi[j] - base[k]) / up, HOLD_FIT * (base[k] - self.lo[j]) / dn)
            w[:, k] = base[k] + dev * fit
        jd = np.radians(float(getattr(self.task, "hold_jitter_deg", 0.0)))
        if jd > 0.0:                          # a hand shake: three tones, 4-9 Hz, random phase
            tt = np.arange(Tn) * self.dt
            for k in range(min(2, w.shape[1])):
                f, ph = rng.uniform(*HOLD_JITTER_HZ, 3), rng.uniform(0.0, 2 * np.pi, 3)
                w[:, k] += (jd / 3.0) * sum(np.sin(2 * np.pi * f[i] * tt + ph[i]) for i in range(3))
        out = np.repeat(self.target_q[None, :n], Tn, axis=0).copy()
        if N_WRIST == 2:                       # (flex, dev) -> wrist, roll -> last arm slot
            out[:, N_ARM:N_ARM + 2] = np.clip(w[:, :2], self.lo[N_ARM:N_ARM + 2], self.hi[N_ARM:N_ARM + 2])
            out[:, N_ARM - 1] = np.clip(w[:, 2], self.lo[N_ARM - 1], self.hi[N_ARM - 1])
        else:
            out[:, N_ARM:N_ARM + N_WRIST] = np.clip(
                w, self.lo[N_ARM:N_ARM + N_WRIST], self.hi[N_ARM:N_ARM + N_WRIST])
        return out

    @staticmethod
    def _ease_path(way, Tn):
        """Cosine-eased piecewise path through the rows of `way`, sampled at Tn steps."""
        way = np.asarray(way, float)
        n_way, k = way.shape
        ts = np.linspace(0, Tn - 1, n_way)
        out = np.zeros((Tn, k))
        for i in range(n_way - 1):
            a, b = int(ts[i]), int(ts[i + 1])
            uu = np.linspace(0, 1, b - a + 1)[:-1]
            out[a:b] = way[i] + (way[i + 1] - way[i]) * ((1 - np.cos(np.pi * uu)) / 2)[:, None]
        out[-1] = way[-1]
        return out

    def _swing_arm(self, traj):
        """hold_moving: the ARM carries the wrist mount through `hold_box` (random
        waypoints, cosine-eased) while the wrist sweeps as `traj` says. The mount keeps its
        reset orientation, so the sweep is all wrist."""
        rng = self.np_random
        box = np.asarray(self.task.hold_box, float)
        Tn = len(traj)
        p0 = self._mount_pos()
        n_way = max(2, int(getattr(self.task, "hold_waypoints", 5)))
        way = p0 + rng.uniform(-box, box, (n_way, 3))
        way[:, 2] += 0.5 * box[2]             # more room up than down: the tip must clear the sheet
        way[0] = p0
        return self._ik_track(self._ease_path(way, Tn), traj, point="mount")

    def _scribble_motion(self, Tn):
        """L0_scribble: the env drags the TIP along a random fast path inside the sheet,
        pressed `scribble_depth_m` into it, with a wrist wobble on top; the arm keeps the
        tip on that path whatever the wrist does."""
        rng = self.np_random
        n = N_ARM + N_WRIST
        tip0 = self._tip().copy()
        lo = S.PAPER_CENTER[:2] - S.PAPER_HALF[:2] + SCRIBBLE_MARGIN
        hi = S.PAPER_CENTER[:2] + S.PAPER_HALF[:2] - SCRIBBLE_MARGIN
        n_way = max(2, int(getattr(self.task, "hold_waypoints", 5)))
        way = np.zeros((n_way, 3))
        way[:, :2] = rng.uniform(lo, hi, (n_way, 2))
        way[0, :2] = np.clip(tip0[:2], lo, hi)
        way[:, 2] = S.PAPER_TOP - float(getattr(self.task, "scribble_depth_m", 0.0))
        pts = self._ease_path(way, Tn)
        k = max(2, int(round(0.4 / self.dt)))      # settle onto the sheet from the start band
        pts[:k, 2] = np.linspace(tip0[2], pts[k, 2], k)
        traj = np.repeat(self.target_q[None, :n], Tn, axis=0).copy()
        wd = np.radians(float(getattr(self.task, "scribble_wrist_deg", 0.0)))
        if wd > 0.0:
            wway = rng.uniform(-wd, wd, (n_way, 2))
            wway[0] = 0.0
            W = slice(N_ARM, N_ARM + 2)
            traj[:, W] = np.clip(traj[:, W] + self._ease_path(wway, Tn), self.lo[W], self.hi[W])
        return self._ik_track(pts, traj, point="tip")

    def _mount_body(self):
        """The body the wrist flexion joint hangs from: the forearm side of the wrist."""
        m = self.model
        jw = int(np.where(m.jnt_qposadr == self.qadr[N_ARM])[0][0])
        return int(m.body_parentid[m.jnt_bodyid[jw]])

    def _mount_pos(self):
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self._mount_body()].copy()

    def _ik_point(self, p_goal, point, Rm0, mount, A, cols, jacp, jacr, q4, e_r):
        """Damped-least-squares from the CURRENT d.qpos on the joints `A` until `point`
        (the tool tip, or the wrist mount) sits at p_goal with the mount at Rm0. True on
        convergence within IK_TRACK_ITERS."""
        m, d = self.model, self.data
        nu = len(A)
        for _ in range(IK_TRACK_ITERS):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            if point == "tip":
                body = self.hb
                p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ self._tip_rel
            else:
                body = mount
                p = d.xpos[body]
            e_p = p_goal - p
            mujoco.mju_mat2Quat(q4, np.ascontiguousarray(Rm0 @ d.xmat[mount].reshape(3, 3).T).ravel())
            mujoco.mju_quat2Vel(e_r, q4, 1.0)
            if np.linalg.norm(e_p) < 5e-4 and np.linalg.norm(e_r) < 5e-3:
                return True
            mujoco.mj_jac(m, d, jacp, None, p, body)
            mujoco.mj_jac(m, d, None, jacr, d.xpos[mount], mount)
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_p, e_r]))
            d.qpos[A] = np.clip(d.qpos[A] + np.clip(dq, -0.2, 0.2), self.lo[:nu], self.hi[:nu])
        return False

    def _ik_track(self, pts, traj, point="tip"):
        """Fill the six UR joints of `traj` step by step so that `point` (the tool tip, or
        the wrist mount) follows `pts` while the wrist columns and forearm roll of `traj`
        are applied as given and the mount keeps its reset orientation. Warm-started from
        the previous step, so a few damped-least-squares iterations per step suffice.
        Restores the sim state; the count of steps that did not converge is kept in
        `_drive_ik_miss`. Also stashes the path and mount frame for `_drive_step`."""
        m, d = self.model, self.data
        q0, v0, c0 = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        nu = N_ARM - 1                                   # pan, lift, elbow, wrist_1/2/3
        A, cols = self.qadr[:nu], self.dadr[:nu]
        D = self.qadr[nu:N_ARM + N_WRIST]                 # roll + wrist: driven, not solved
        mount = self._mount_body()
        mujoco.mj_kinematics(m, d)
        Rm0 = d.xmat[mount].reshape(3, 3).copy()
        out = np.asarray(traj, float).copy()
        jacp, jacr, q4, e_r = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros(4), np.zeros(3)
        miss = 0
        for i in range(len(pts)):
            d.qpos[D] = out[i, nu:N_ARM + N_WRIST]
            if not self._ik_point(pts[i], point, Rm0, mount, A, cols, jacp, jacr, q4, e_r):
                miss += 1
            out[i, :nu] = d.qpos[A]
        self._drive_ik_miss = miss
        self._drive = dict(pts=np.asarray(pts, float), point=point, Rm0=Rm0, mount=mount)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q0, v0, c0
        mujoco.mj_forward(m, d)
        return out

    def _drive_step(self):
        """Online re-solve of the driven arm target from the CURRENT joint state.

        The precomputed joint trajectory keeps the tip on its path only when every joint
        is AT its target. The UR position servos lag a fast path by 6 deg on average and
        up to 19 deg on the driven scribble, and a lagging joint combination is off
        the plane: the rigid-FK tip rises 1.7-3.4 mm above the sheet at the reversals
        and contact is lost 21-34 % of the time. Solving from where the arm
        actually is, for the NEXT path point, keeps the commanded configuration on the
        plane whatever the lag does to the timing. Roll and wrist stay as driven."""
        dr = self._drive
        if dr is None:
            return
        m, d = self.model, self.data
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        nu = N_ARM - 1
        A, cols = self.qadr[:nu], self.dadr[:nu]
        D = self.qadr[nu:N_ARM + N_WRIST]
        d.qpos[D] = self.target_q[nu:N_ARM + N_WRIST]
        i = min(self.t + 1, len(dr["pts"]) - 1)
        goal = dr["pts"][i].copy()
        if dr["point"] == "tip" and DRIVE_Z_GAIN > 0.0:
            # z-servo in Cartesian space: the joint servos alone let the arm rise 1.7-3.4 mm
            # off the plane at the reversals (contact lost ~30 % of the time). The
            # ARM-side tip (rigid FK from the grip pose, so a pen tilting in the hand is not
            # "corrected" - that is the policy's job) is compared with the path and the
            # target pushed the other way, clamped.
            mujoco.mj_kinematics(m, d)
            z_fk = float(d.xpos[self.hb][2] + (d.xmat[self.hb].reshape(3, 3) @ self._tip_rel)[2])
            goal[2] -= float(np.clip(DRIVE_Z_GAIN * (z_fk - dr["pts"][i][2]), -DRIVE_Z_CLAMP, DRIVE_Z_CLAMP))
        jacp, jacr, q4, e_r = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros(4), np.zeros(3)
        self._ik_point(goal, dr["point"], dr["Rm0"], dr["mount"], A, cols, jacp, jacr, q4, e_r)
        self.target_q[:nu] = d.qpos[A]
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_forward(m, d)

    def _apply_action(self, action):
        m, d = self.model, self.data
        a_full = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        if self.end_mode == "policy" and a_full.size in (NU + 1, 2 * NU + 1):
            self._done_req = bool(a_full[-1] > 0.5)    # the policy says it is finished
            a_full = a_full[:-1]
        if a_full.size == NU:                       # torque mode, or position with no force channel
            a_full = np.concatenate([a_full, np.zeros(NU)])
        a, a_force = a_full[:NU], a_full[NU:]
        if self.perturb.get("action_delay", 0):
            self._action_queue.append(a)
            if len(self._action_queue) > self.perturb["action_delay"]:
                a = self._action_queue.pop(0)
            else:
                a = np.zeros_like(a)
        self._last_action = a.copy()
        # force channel: a feed-forward joint torque on top of the servo, so the
        # policy can choose a press force instead of driving position error into
        # the sheet. Constraints below zero it wherever they zero the position one.
        self._tau_ff = a_force * self.tau * self.force_gain
        speed = self.max_vel.copy()
        speed[self.arm_idx] *= self._arm_speed()
        speed[self.wrist_idx] *= self._wrist_speed()
        mode = self.control_mode
        prev_target = self.target_q.copy()
        if mode == "hybrid":
            # arm: position control (what a UR5e actually accepts - `servoj`), the
            # action moving its joint target. hand + wrist: motor torque.
            self._duty = a.copy()
            self._duty[:N_ARM] = 0.0
            if self.constraint.get("idle_lock"):
                self._duty[self.idle_idx] = 0.0
            self._tau_ff = np.zeros(NU)
            # Soft start: the first pen-down target sits 1.5 mm INSIDE the sheet and the tip
            # starts a few mm above it, so the opening move closes 4-6 mm in one or two control
            # steps and lands at 12-15 N, against 0.6 N for the same depth while writing: it is
            # the speed of arrival, not the depth. 14 N is full black and the full 3 mm width in
            # the ink model, so it prints a dot before a stroke is written. Zeroing the first
            # action only moves the impact one step later. The arm's joint-target rate AND the
            # wrist's torque ramp from 1/n to 1 over the first `soft_start` seconds; the fingers
            # are left alone (they hold the pen and play no part in the impact).
            if self.soft_start > 0.0:
                g = min(1.0, (self.t + 1) / max(1.0, round(self.soft_start / self.dt)))
                self._duty[self.wrist_idx] *= g
            else:
                g = 1.0
            self.target_q[:N_ARM] = np.clip(
                self.target_q[:N_ARM] + a[:N_ARM] * speed[:N_ARM] * self.dt * g,
                self.lo[:N_ARM], self.hi[:N_ARM])
            if self.constraint.get("arm_lock") and self._pen_engaged():
                self.target_q[self.arm_idx] = prev_target[self.arm_idx]
            if self._hold_traj is not None:
                n = N_ARM + N_WRIST
                self.target_q[:n] = self._hold_traj[min(self.t, len(self._hold_traj) - 1)]
                self._drive_step()
                self._duty[:n] = 0.0
            if self.constraint.get("wrist_hold"):
                self._duty[self.wrist_idx] = 0.0          # the env's servo has the wrist
            if self.wrist_teacher > 0.0 and self._timeline is not None and self._hold_traj is None:
                b = self.wrist_teacher
                self._duty[self.wrist_idx] = ((1.0 - b) * self._duty[self.wrist_idx]
                                              + b * self._teacher_wrist_duty())
            return
        if mode == "torque":
            # the action IS the motor command: a duty cycle in [-1, 1], scaled at
            # every substep by the torque actually available at that joint speed
            self._duty = a.copy()
            self._tau_ff = np.zeros(NU)
            if self.constraint.get("arm_lock") and self._pen_engaged():
                self._duty[self.arm_idx] = 0.0
            if self._hold_traj is not None:
                # L0 hold: the ENV carries the arm through a set motion while the
                # policy only has to keep hold of the tool. Those joints keep their
                # position servo even here, or the arm would simply never move.
                n = N_ARM + N_WRIST
                self.target_q[:n] = self._hold_traj[min(self.t, len(self._hold_traj) - 1)]
                self._drive_step()
                self._duty[:n] = 0.0
            return
        if mode == "pos":
            self.target_q = self.lo + (a + 1) / 2 * (self.hi - self.lo)
        elif mode == "delta_pos":
            self.target_q = self.target_q + a * speed * self.dt
        elif mode == "vel":
            self.vel_cmd = a * speed
            self.target_q = self.target_q + self.vel_cmd * self.dt
        elif mode == "acc":
            self.vel_cmd = np.clip(self.vel_cmd + a * speed * 4.0 * self.dt, -speed, speed)
            self.target_q = self.target_q + self.vel_cmd * self.dt
        self.target_q = np.clip(self.target_q, self.lo, self.hi)
        if self.constraint.get("arm_lock") and self._pen_engaged():
            # finger track: the arm may only reposition while the pen is up
            self.target_q[self.arm_idx] = prev_target[self.arm_idx]
            self.vel_cmd[self.arm_idx] = 0.0
            self._tau_ff[self.arm_idx] = 0.0
        self._tau_ff[self.arm_idx] *= self._arm_speed()
        self._tau_ff[self.wrist_idx] *= self._wrist_speed()
        if self._hold_traj is not None:
            n = N_ARM + N_WRIST
            self.target_q[:n] = self._hold_traj[min(self.t, len(self._hold_traj) - 1)]
            self._drive_step()
        d.ctrl[:] = self.target_q

    def _teacher_wrist_duty(self):
        """Wrist teacher: damped-LS IK on the two wrist joints toward the next
        timeline point (z-admittance backs off above 2.5 N), closed by the scripted
        teacher's PD, returned as a duty cycle for the wrist slots only."""
        m, d = self.model, self.data
        if self._ik_scratch is None:
            self._ik_scratch = mujoco.MjData(m)
        d2 = self._ik_scratch
        if self._force > 2.5:
            self._ik_zoff = min(self._ik_zoff + 0.0002, 0.003)
        elif self._force < 0.5:
            self._ik_zoff = max(self._ik_zoff - 0.0001, 0.0)
        k = min((self._tl_k if self.rw.get("patient") else self.t) + 1, len(self._timeline) - 1)
        goal = self._timeline[k] + np.array([0.0, 0.0, self._ik_zoff])
        Wq, Wd = self.qadr[self.wrist_idx], self.dadr[self.wrist_idx]
        lo, hi = self.lo[self.wrist_idx], self.hi[self.wrist_idx]
        q = self.data.qpos[Wq].copy() if self._ik_q is None else self._ik_q
        d2.qpos[:] = self.data.qpos
        jacp = np.zeros((3, m.nv))
        step = self.max_vel[self.wrist_idx] * self.dt * 0.5
        for _ in range(3):
            d2.qpos[Wq] = q
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_comPos(m, d2)
            tip = d2.xpos[self.hb] + d2.xmat[self.hb].reshape(3, 3) @ self._tip_rel
            e = goal - tip
            if np.linalg.norm(e) < 2e-4:
                break
            mujoco.mj_jac(m, d2, jacp, None, tip, self.hb)
            J = jacp[:, Wd]
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), e), -step, step)
            q = np.clip(q + dq, lo, hi)
        self._ik_q = q
        kp = m.actuator_gainprm[self.wrist_idx, 0] * 0.5           # scripted KP_SCALE
        kv = -m.actuator_biasprm[self.wrist_idx, 2] * 0.10         # scripted KV_SCALE
        qw, qdw = d.qpos[Wq], d.qvel[Wd]
        tau = kp * (q - qw) - kv * qdw
        avail = self.tau[self.wrist_idx] * np.clip(
            1.0 - np.abs(qdw) / (NO_LOAD_SPEED * self.max_vel[self.wrist_idx]), 0.0, 1.0)
        return np.clip(tau / np.maximum(avail, 1e-9), -1.0, 1.0)

    def _assist_capture(self):
        d = self.data
        R = d.xmat[self.hb].reshape(3, 3)
        self._assist_rel = (R.T @ (d.qpos[self.pq:self.pq + 3] - d.xpos[self.hb]),
                            R.T @ d.xmat[self.pen_bid].reshape(3, 3))

    def _pen_into_hand(self):
        """Put the tool at the solved grip's hand-frame pose relative to the CURRENT hand."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        _, p_rel, q_rel = S.settled_hand(self.tool)
        R = d.xmat[self.hb].reshape(3, 3)
        d.qpos[self.pq:self.pq + 3] = d.xpos[self.hb] + R @ p_rel
        Rp = np.zeros(9); mujoco.mju_quat2Mat(Rp, q_rel)
        q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, np.ascontiguousarray(R @ Rp.reshape(3, 3)).ravel())
        d.qpos[self.pq + 3:self.pq + 7] = q4
        d.qvel[self.pd:self.pd + 6] = 0.0
        mujoco.mj_forward(m, d)

    def _place_hand(self, place_xy=None, tip_z=None):
        """Put the pen tip over the target (or `place_xy` on the sheet, at `tip_z`) with the
        hand in the SOLVED grip's orientation (settled_hand, then yaw_offset and
        pen_axis_roll on top) and the pen at its hand-frame pose. Damped-least-squares IK on
        the six placement joints - pan, lift, elbow, forearm roll, wrist flex, wrist dev -
        for the full hand_base pose. Nothing here depends on the arm's joint values."""
        m, d, rng = self.model, self.data, self.np_random
        if place_xy is not None:
            c0 = np.asarray(place_xy, float) + S.PAPER_CENTER[:2]
        elif self.task.strokes:
            P = np.concatenate([np.asarray(s) for s in self.task.strokes]) + S.PAPER_CENTER[:2]
            c0 = P.mean(0)
            if self.rw.get("place_start"):
                # training-only curriculum (reward preset flag): start the tip at the FIRST
                # stroke's first point instead of the glyph centre, so the time-driven
                # timeline begins where the tip already is. The accept preset never sets it.
                c0 = np.asarray(self.task.strokes[0])[0] + S.PAPER_CENTER[:2]
        else:                                    # no target to trace (L0_scribble)
            c0 = np.asarray(self.task.place_at, float) + S.PAPER_CENTER[:2]
        j = float(getattr(self.task, "place_jitter", 0.01))
        c = c0 + rng.uniform(-j, j, 2)
        z = S.PAPER_TOP + float(rng.uniform(*self.start_height)) if tip_z is None else float(tip_z)
        tip_goal = np.array([c[0], c[1], z])
        R_settled, p_rel, q_rel = S.settled_hand(self.tool)
        # Orientation target: the SOLVED grip's hand orientation, with yaw_offset and
        # pen_axis_roll applied on top - what every caller (evaluate, viewer, training
        # scripts, all with yaw_offset=pi) assumes. HAND_TARGET is an absolute pose for the
        # 4-DoF variant; on the shipped arm it is only used when explicitly set.
        R0 = S.hand_target_rot(**S.HAND_TARGET) if S.HAND_TARGET else R_settled
        R_rel = np.zeros(9); mujoco.mju_quat2Mat(R_rel, q_rel); R_rel = R_rel.reshape(3, 3)
        # tip in the hand frame: from the pen's hand-frame pose and the pen's own tip offset
        self._pen_into_hand()
        Rh = d.xmat[self.hb].reshape(3, 3)
        tip_rel = Rh.T @ (self._tip() - d.xpos[self.hb])
        W, A, cols = self.qadr[N_ARM:N_ARM + N_WRIST], self.qadr[:N_ARM + N_WRIST], self.dadr[:N_ARM + N_WRIST]
        if self.start_wrist:
            # Three concurrent wrist axes move a rigidly held tip on a SPHERE (Jacobian
            # rank 2), so with the arm frozen the tip can only write where
            # that sphere meets the sheet. At the settled -57 deg flex the tip sits at
            # the bottom of the sphere, a cap that stays within 2-3 mm of the plane over
            # a 5-6 cm disc - the grip's compliance absorbs that if the arm PRE-PRESSES
            # the tip into the sheet (start_height < 0). Pitching the wrist further
            # tilts the cap and one side lifts off, so the start band stays near it;
            # more height range than this needs finger-driven pen extension.
            # seed the wrist near the grip's OWN settled flexion (the tripod grip settles at -15 deg),
            # plus a little scatter
            wf0 = float(self.grip["settled" if self.tool in ("pen", "marker") else f"settled_{self.tool}"].get("wrist_flex", np.radians(-52)))
            qw = np.array([wf0 + np.radians(rng.uniform(-7, 7)), np.radians(rng.uniform(-5, 5))])   # flex, dev
            d.qpos[W] = np.clip(qw, self.lo[N_ARM:N_ARM + N_WRIST], self.hi[N_ARM:N_ARM + N_WRIST])
            # forearm roll (last ARM slot): roll_offset=+0.3 gives the tip-outward human
            # pose (azimuth -31 -> -2 deg, elev 45 -> 52). Default 0.
            d.qpos[self.qadr[N_ARM - 1]] = np.clip(self.roll_offset + np.radians(rng.uniform(-5, 5)),
                                                   self.lo[N_ARM - 1], self.hi[N_ARM - 1])
            mujoco.mj_kinematics(m, d)                               # (wrist joints are IK columns now: this is their seed)
        if self.yaw_offset:
            cy, sy = np.cos(self.yaw_offset), np.sin(self.yaw_offset)
            R0 = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ R0
        if self.pen_axis_roll:
            u = R0 @ (R_rel @ np.array([0.0, 0.0, 1.0]))           # pen shaft axis in the world
            u /= np.linalg.norm(u)
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            c, sn = np.cos(self.pen_axis_roll), np.sin(self.pen_axis_roll)
            R0 = (np.eye(3) + sn * K + (1 - c) * K @ K) @ R0       # Rodrigues about the pen axis
        p_goal = tip_goal - R0 @ tip_rel
        jacp, jacr, q4, e_r = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros(4), np.zeros(3)
        # Seeds: the configured one, then (if it fails) the same pan/lift/elbow with wrist_2
        # moved to other points of the convergence basin. The DLS below is deterministic and
        # basin-sensitive - a seed on a basin edge fails silently and the policy then starts
        # with the tip in the air - so a bad first seed must not be the end of it.
        if self.arm_seed is None:
            seeds = [None]                      # start from the current (settled) arm pose
        else:
            seeds = [np.asarray(self.arm_seed, float)]
            for w2 in (-40.0, -65.0, -30.0)[:self.ik_retries]:
                alt = seeds[0].copy(); alt[4] = np.radians(w2); seeds.append(alt)
        ok = False
        for seed in seeds:
            if seed is not None:
                k = len(seed)
                d.qpos[A[:k]] = np.clip(seed, self.lo[:k], self.hi[:k])
            ok = self._ik_to(m, d, A, cols, p_goal, R0, jacp, jacr, q4, e_r)
            if ok:
                break
        e_r[:] = self._ik_last_er
        mujoco.mj_kinematics(m, d)
        self.ik_residual = (float(np.linalg.norm(p_goal - d.xpos[self.hb])), float(np.linalg.norm(e_r)))
        self._pen_into_hand()
        d.qvel[:] = 0.0
        d.ctrl[:N_ARM + N_WRIST] = d.qpos[self.qadr[:N_ARM + N_WRIST]]
        mujoco.mj_forward(m, d)
        self.placed_ok = ok

    def _ik_to(self, m, d, A, cols, p_goal, R0, jacp, jacr, q4, e_r):
        """Damped-least-squares on the six arm joints to put hand_base at p_goal with
        orientation R0, from the CURRENT d.qpos. True on convergence."""
        for _ in range(self.ik_iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.hb].reshape(3, 3)
            e_p = p_goal - d.xpos[self.hb]
            mujoco.mju_mat2Quat(q4, np.ascontiguousarray(R0 @ R.T).ravel())
            mujoco.mju_quat2Vel(e_r, q4, 1.0)
            self._ik_last_er = e_r.copy()
            if np.linalg.norm(e_p) < 5e-4 and np.linalg.norm(e_r) < 5e-3:
                return True
            mujoco.mj_jac(m, d, jacp, jacr, d.xpos[self.hb], self.hb)
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_p, e_r]))
            n = len(A)
            d.qpos[A] = np.clip(d.qpos[A] + np.clip(dq, -0.2, 0.2), self.lo[:n], self.hi[:n])
        return False

    def _assist_pin(self):
        """Hold the tool at its captured pose in the hand frame (the hand may move)."""
        d = self.data
        if self._assist_rel is None:
            self._assist_capture()
        p_rel, R_rel = self._assist_rel
        R = d.xmat[self.hb].reshape(3, 3)
        d.qpos[self.pq:self.pq + 3] = d.xpos[self.hb] + R @ p_rel
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(R @ R_rel).ravel())
        d.qpos[self.pq + 3:self.pq + 7] = q
        d.qvel[self.pd:self.pd + 6] = 0.0

    def _tau_now(self):
        """Torque each motor can deliver at its current speed (the speed-torque
        curve). Without it the action is a torque source with no motor behind it,
        and a policy will happily ask for stall torque at full speed."""
        w = np.abs(self.data.qvel[self.dadr])
        return self.tau * np.clip(1.0 - w / (NO_LOAD_SPEED * self.max_vel), 0.0, 1.0)

    def step(self, action):
        m, d = self.model, self.data
        self._contact_hits, self._force_acc = 0, 0.0
        self._ink_new = []
        self._apply_action(action)
        for _ in range(self.frame_skip):
            d.qfrc_applied[:] = 0.0
            S.compensate_arm_gravity(m, d, self.arm_dofs)
            if self.control_mode == "hybrid":
                self._tau_applied = self._duty * self._tau_now() * self._act_mask
                d.qfrc_applied[self.dadr] += self._tau_applied
                d.ctrl[:] = d.qpos[self.qadr]        # hand: parked, torque only
                if self.constraint.get("idle_lock"):
                    d.ctrl[self.idle_idx] = self._q_grip_hand[self._idle_mask]   # ring + pinky: held
                d.ctrl[:N_ARM] = self.target_q[:N_ARM]        # arm: position servo
                if self._hold_traj is not None:
                    d.ctrl[:N_ARM + N_WRIST] = self.target_q[:N_ARM + N_WRIST]
                elif self.constraint.get("wrist_hold"):
                    d.ctrl[N_ARM:N_ARM + N_WRIST] = self.target_q[N_ARM:N_ARM + N_WRIST]   # held where placed
            elif self.control_mode == "torque":
                self._tau_applied = self._duty * self._tau_now() * self._act_mask
                d.qfrc_applied[self.dadr] += self._tau_applied
                d.ctrl[:] = d.qpos[self.qadr]       # servo follows, no restoring force
                if self._hold_traj is not None:     # except the env-driven hold joints
                    n = N_ARM + N_WRIST
                    d.ctrl[:n] = self.target_q[:n]
            else:
                self._tau_applied = self._tau_ff
                d.qfrc_applied[self.dadr] += self._tau_ff
            self._maybe_push()
            if self.grip_assist > 0.0:
                # env-supplied nominal grip, on the FINGERS only. Holding the wrist as well
                # (dadr[N_ARM:]) fights a policy that needs the wrist to write.
                hw = self.dadr[N_ARM + N_WRIST:]
                qh = self.qadr[N_ARM + N_WRIST:]
                d.qfrc_applied[hw] += self.grip_assist * (
                    self._ga_kp[-N_HAND:] * (self._q_grip_all[-N_HAND:] - d.qpos[qh])
                    - self._ga_kv[-N_HAND:] * d.qvel[hw])
            if self._g_scale < 1.0:                 # partial weight on the tool
                lift = (1.0 - self._g_scale) * 9.81
                for b in self.tool_bids:
                    d.xfrc_applied[b, 2] += m.body_mass[b] * lift
            if self.t < self.assist_steps:
                self._assist_pin()
            if self.brush is not None:
                self._brush_apply()
            mujoco.mj_step(m, d)
            self._ink_step()
        self.t += 1
        if self.brush is not None:
            self._brush_viol_steps += int(self._brush_viol_sub)     # a step with any violating substep
            self._brush_viol_sub = False
        self._contact_duty = self._contact_hits / self.frame_skip      # fraction of substeps in contact
        self._force_mean = self._force_acc / self.frame_skip           # mean tool-sheet force over the step
        self._force = self._pen_paper_force()              # cached: _state / _reward reuse it
        self._peak_force = max(self._peak_force, float(self._force))
        qi = d.qpos[self.qadr[self.idle_idx]] - self._q_grip_hand[self._idle_mask]
        self._idle_dev2 += float(np.mean(qi ** 2))
        self._idle_vel2 += float(np.mean(d.qvel[self.dadr[self.idle_idx]] ** 2))
        self._idle_n += 1
        # A lift only counts once the tip has actually cleared the sheet. Counting every
        # rising edge of the contact force instead counts chatter: a do-nothing policy
        # resting on the paper scores ~3.7 "lifts" an episode that way.
        down = self._force > 0.0
        if down and self._cov_base is None:
            # Everything a pen inks just by resting where it landed is not scribbling.
            # Without this a do-nothing policy scores 0.52 on the finger-only task,
            # whose whole achievable range is 63 mm2 (resting) to 102 (open-loop sweep).
            self._cov_base = self._inked_area_px()
        if self._tip()[2] - S.PAPER_TOP > LIFT_CLEAR:
            self._lift_armed = True
        if down and not self._was_down and self._lift_armed:
            self._n_lifts += 1
            self._lift_armed = False
        self._was_down = down
        if self._force > 0.0:                              # motion shares: only while writing (L5: while touching the object)
            v_arm, v_wrist, v_fing = self._tip_velocity_split()
            self._vsplit = (v_arm, v_wrist, v_fing)
            # Project each group onto the direction the tip is ACTUALLY going, and
            # divide by the distance it actually covers. The shares then decompose
            # the real path exactly and sum to 1, and a group that fights the motion
            # scores negative - which is the honest reading.
            #
            # Sum-of-norms over sum-of-norms would count motion that cancels: a hand
            # whose fingers push while the arm pulls back moves the tip nowhere but
            # scores on both (per-group norms can sum to 2.6x the tip's real travel).
            # The direction is the tip's REAL motion, and the denominator is the
            # distance it really covered - not the Jacobian model's own sum. The
            # model treats the tool as rigid with the fingertips, which holds while
            # the arm drags the pen (residual 15 % of the motion) and
            # fails when the fingers drive it and the tool rolls between the pads
            # (residual 124 %). Normalising by the real path makes the three shares
            # fall short of 1 by exactly that much, so a broken split reports itself
            # instead of decomposing a motion that never happened.
            tip = self._tip()
            if self._share_prev_tip is not None:
                step = tip - self._share_prev_tip
                n = float(np.linalg.norm(step))
                if n > 1e-9:
                    u = step / n
                    self._share_fing += float(v_fing @ u) * self.dt
                    self._share_wrist += float(v_wrist @ u) * self.dt
                    self._share_arm += float(v_arm @ u) * self.dt
                    self._share_den += n
            self._share_prev_tip = tip.copy()
        else:
            self._vsplit = (np.zeros(3), np.zeros(3), np.zeros(3))
            self._share_prev_tip = None          # pen-up: do not bridge across the hop
        if self.rw.get("patient") and self._timeline is not None:
            self._tl_advance()
        if self.pickup:
            self._update_pickup()
        elif self.chop and not self.chop_hold:
            self._update_phase()
        else:
            if self._force > 0.0:
                # order-free force tracking: what does the map ask for HERE, at the
                # pixel the tip is on? Comparing against a waypoint index would
                # assume the policy draws in the generator's stroke order.
                tgt_f = self._target_force_at(self._tip())
                if tgt_f > 0:
                    self._force_err += abs(self._force - tgt_f)
                    self._force_n += 1
            self._advance_waypoints()
        self._check_drop()
        self._check_hover()
        tip = self._tip()
        self.tip_log.append(tip.copy())
        self.touch_log.append(self._force > 0)
        self.arm_log.append(d.xpos[self.hb].copy())          # hand position (arm travel metric)
        self.action_log.append(self._last_action.copy())
        reward = self._reward()
        terminated = self.dropped or self.hovered
        if self.end_mode == "policy" and self.task.strokes:
            # the policy ends the episode (or the 2x cap does); the schedule finishing does not
            finished = self._done_req
            if finished and self.done_time is None:
                self.done_time = self.t * self.dt
            truncated = self.t >= self.T_end or finished or (self.chop and self.obj_lost)
            if (truncated and not terminated and (self.rw.get("final") or self.rw.get("quit"))) \
                    or (terminated and self.rw.get("quit")):        # presets without `quit` pay nothing on termination
                # terminal pay = the strict, order-aware completion the task is scored on; `quit` charges
                # the part left unwritten at ANY end (declared done, cap, hover / far-away, drop) - without
                # it, declaring done at once is the cheapest policy
                from .metrics import episode_metrics
                fm = episode_metrics(self.episode_record())
                compl = float(fm.get("completion") or 0.0)
                reward += float(self.rw.get("final", 0.0)) * compl \
                    + float(self.rw.get("final_precision", 0.0)) * float(fm.get("precision") or 0.0) \
                    - float(self.rw.get("quit", 0.0)) * (1.0 - compl)
        else:
            finished = self.done_time is not None and self.t * self.dt >= self.done_time + 0.5
            truncated = self.t >= self.T or finished or (self.chop and self.obj_lost)
        if truncated and not self.task.strokes and not self.dropped and not self.pickup \
                and (not self.chop or self.chop_hold):
            # hold tasks are done by surviving; the pick-up task is not - it is done
            # only when the tool is actually in hand (see _update_pickup)
            self.done_time = self.t * self.dt
        return self._obs(), reward, terminated, truncated, self._info()

    def _inked_area_px(self):
        """Ink dilated by COVER_DILATE_MM, in pixels. One pixel is one square mm, and
        this is exactly what metrics scores L0_scribble on."""
        h = self._inkmap > 0
        for _ in range(int(COVER_DILATE_MM / INK_MAP_MM_PER_PX)):
            h = (h | np.roll(h, 1, 0) | np.roll(h, -1, 0) | np.roll(h, 1, 1) | np.roll(h, -1, 1))
        return int(h.sum())

    def _arm_speed(self):
        """How fast the arm may move. A task may freeze it outright (L0_scribble asks
        for hand-driven marks, so an arm that helps is not doing the task); otherwise
        it is the track's business."""
        if getattr(self.task, "arm_frozen", False):
            return 0.0
        return self.constraint.get("arm_speed", 1.0)

    def _wrist_speed(self):
        """As _arm_speed, for the wrist. L0_scribble_fingers freezes it because the
        wrist out-reaches the fingers in every direction measured, so leaving it free
        means the fingers are never needed."""
        if getattr(self.task, "wrist_frozen", False):
            return 0.0
        return self.constraint.get("wrist_speed", 1.0)

    def _arm_gate(self):
        """1 - arm share of this step's tip motion; 1.0 when the tip is still."""
        va, vw, vf = (np.linalg.norm(v) for v in self._vsplit)
        tot = va + vw + vf
        if tot < 1e-6:
            return 1.0
        g = 1.0 - va / tot
        return 0.0 if g < 0.1 else 1.0 if g > 0.9 else g

    def _tl_advance(self):
        """Patient timeline: earn the current point to move on. Pen-down points need
        real ink within 3 mm while current; transit points need the tip within 6 mm.
        A point that resists for `adv_skip` seconds is skipped (no pay, no hit).

        The default is 0.25 s. The pointer skips ONE point per wait, so with a long wait
        (e.g. 1.5 s) a run of points the tip cannot reach costs that much each: the teacher
        can spend half an episode frozen in the air between strokes, or never get going at
        all before the hover rule ends the episode."""
        if self._tl_stroke_id is None:
            down = self._tl_down > 0.5
            rising = np.concatenate([[down[0]], down[1:] & ~down[:-1]])
            self._tl_stroke_id = np.cumsum(rising) * down          # 0 on transit points
            self._tl_n_down = max(1, int(down.sum()))
        self._tl_adv_down = self._tl_adv_up = self._tl_reentry = 0
        last = len(self._timeline) - 1
        tip = self._tip()
        ink = np.asarray(self._ink_new) if self._ink_new else None
        for _ in range(6):                                          # fast writing covers several points
            if self._tl_k > last:
                break
            k = self._tl_k
            p = self._timeline[k]
            if self._tl_down[k] > 0.5:
                # the tolerance is a preset key so the dense payment can be pulled onto the
                # scored metric (completion is ink within 1 mm); 3 mm is the default
                tol = float(self.rw.get("adv_tol", 0.003))
                if self.adv_mode == "tip":
                    # what a robot can compute - the tip is there and touching the sheet
                    ok = (float(np.linalg.norm(tip[:2] - p[:2])) <= tol
                          and float(tip[2] - S.PAPER_TOP) <= 0.0008)
                else:
                    ok = ink is not None and bool((np.linalg.norm(ink - p[:2], axis=1) <= tol).any())
            else:
                # transit points sit 1 cm above the sheet but the wrist z envelope is
                # +-3 mm - judge them in XY with a loose z band, or they are unreachable
                ok = (float(np.linalg.norm(tip[:2] - p[:2])) <= 0.006
                      and abs(float(tip[2] - p[2])) <= 0.015)
            if not ok:
                break
            if self._tl_down[k] > 0.5:
                self._tl_adv_down += 1
                self._tl_hits += 1
                sid = int(self._tl_stroke_id[k])
                if sid and sid not in self._reentry_paid:
                    self._reentry_paid.add(sid)
                    self._tl_reentry += 1
            else:
                self._tl_adv_up += 1
            self._tl_k += 1
            self._tl_wait = 0.0
        else:
            pass
        if self._tl_adv_down == 0 and self._tl_adv_up == 0 and self._tl_k <= last:
            # `adv_lookahead`: the current point is not covered, but a later one is - the tip
            # skipped a bit of the line and is writing further on. Jump to it, as the completion
            # metric's in-order alignment would (it looks 40 points ahead): the trajectory keeps
            # pointing to where the pen actually is instead of crawling through the missed points
            # one skip wait at a time while the network waits for "finished".
            L = int(self.rw.get("adv_lookahead", 0))
            jumped = False
            if L > 0:
                tol = float(self.rw.get("adv_tol", 0.003))
                for j in range(self._tl_k + 1, min(self._tl_k + L, last) + 1):
                    p = self._timeline[j]
                    if self._tl_down[j] > 0.5:
                        ok = (self.adv_mode == "tip" and float(np.linalg.norm(tip[:2] - p[:2])) <= tol
                              and float(tip[2] - S.PAPER_TOP) <= 0.0008) or \
                             (self.adv_mode != "tip" and ink is not None and bool((np.linalg.norm(ink - p[:2], axis=1) <= tol).any()))
                    else:
                        ok = float(np.linalg.norm(tip[:2] - p[:2])) <= 0.006 and abs(float(tip[2] - p[2])) <= 0.015
                    if ok:
                        self._tl_k, self._tl_wait, jumped = j + 1, 0.0, True
                        break
            if not jumped:
                self._tl_wait += self.dt
                if self._tl_wait > float(self.rw.get("adv_skip", 0.25)):
                    self._tl_k += 1                                     # skipped: no pay, no hit (after `adv_skip` s)
                    self._tl_wait = 0.0
        if self._tl_k > last and self.done_time is None and self.task.strokes:
            self.done_time = self.t * self.dt                       # schedule finished

    # ------------------------------------------------------------ mechanics
    def _tip(self):
        return self.data.site_xpos[self.tip_sid]

    def _pen_engaged(self):
        """Is the tool working the sheet right now? The finger track freezes the arm
        exactly while this holds.

        Order-free on purpose: it does not consult `wp_down[wp_i]`, the generator's
        own stroke order, because the task is a picture a policy may draw in any
        order. Touching the sheet is the honest test. L6: the object is being shoved by a single stick (a proper
        two-stick pinch frees the arm)."""
        if self.chop:
            upper, lower, _ = self._object_contacts()
            return upper != lower
        return self._force > 0.0

    def _tip_velocity_split(self):
        """Tip velocity attributed to the arm, the wrist and the fingers.

        The reference body has to be a FINGERTIP, not hand_base: the tool is a
        free body held by the three fingertips, so its motion follows theirs, and
        a point attached to hand_base has structurally zero Jacobian columns for
        every finger joint (they are downstream of it). Anchoring at hand_base
        would make finger_share identically 0.00 for every policy.
        """
        m, d = self.model, self.data
        tip = self._tip()
        jac = np.zeros((3, m.nv))
        tmp = np.zeros((3, m.nv))
        for b in self.grasp_bids:
            mujoco.mj_jac(m, d, tmp, None, tip, b)
            jac += tmp
        jac /= max(len(self.grasp_bids), 1)
        v_arm = jac[:, self.dadr[self.arm_idx]] @ d.qvel[self.dadr[self.arm_idx]]
        v_wrist = jac[:, self.dadr[self.wrist_idx]] @ d.qvel[self.dadr[self.wrist_idx]]
        v_fing = jac[:, self.dadr[self.hand_idx]] @ d.qvel[self.dadr[self.hand_idx]]
        return v_arm, v_wrist, v_fing

    def _pen_in_hand(self):
        d = self.data
        R = d.xmat[self.hb].reshape(3, 3)
        p = R.T @ (d.xpos[self.pen_bid] - d.xpos[self.hb])
        ax = R.T @ d.xmat[self.pen_bid].reshape(3, 3)[:, 2]
        return np.concatenate([p, ax])

    def _slip(self):
        """How far the tool has moved IN THE HAND since the episode began.

        A binary "did it fall out" check is far too permissive: a zero-torque hand
        can let the pen creep 12.9 mm and slide from 5 contacts down to 1 and still
        score a clean episode. Rotation matters as much as translation - a pen that
        has turned 30 deg between the fingers is unusable even if it has not moved."""
        rel = self._pen_in_hand()
        dp = float(np.linalg.norm(rel[:3] - self._pen_rel0[:3]))
        dax = float(np.linalg.norm(rel[3:] - self._pen_rel0[3:]))       # chord, not angle
        R = self.data.xmat[self.hb].reshape(3, 3)
        v = float(np.linalg.norm(R.T @ self.data.qvel[self.pd:self.pd + 3]))
        return dp, dax, v

    def _finger_gaps(self):
        """Sum of the distances from the three grasp fingertips to the shaft. Zero
        when the pads are actually on the tool; a grip that merely traps the pen in
        the hollow between the fingers scores badly here, which is the point."""
        d, m = self.data, self.model
        a = d.xpos[self.pen_bid]
        b = a + d.xmat[self.pen_bid].reshape(3, 3)[:, 2] * S.PEN_HALF
        a = a - d.xmat[self.pen_bid].reshape(3, 3)[:, 2] * S.PEN_HALF
        tot = 0.0
        for gid, r in zip(self.tip_gids, self.tip_radii):
            p1, p2 = S.capsule_segment(m, d, gid)
            tot += max(0.0, S.seg_seg_dist(p1, p2, a, b) - r - S.PEN_RADIUS)
        return tot

    def _brush_apply(self):
        """The reduced-order tuft (brush.py), once per substep: the nominal tip's pose
        and velocity go in, the paper's force on the shaft comes out and is applied at
        the tip through mj_applyFT (qfrc_applied is zeroed at the top of each substep,
        so nothing accumulates). The ferrule's own paper contact is a scored violation."""
        m, d = self.model, self.data
        tip = d.site_xpos[self.tip_sid]
        axis = d.xmat[self.pen_bid].reshape(3, 3)[:, 2]                   # tip -> root
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.tip_sid, self._vel6, 0)
        on_sheet = bool(abs(tip[0] - S.PAPER_CENTER[0]) <= S.PAPER_HALF[0]
                        and abs(tip[1] - S.PAPER_CENTER[1]) <= S.PAPER_HALF[1])
        surface = S.PAPER_TOP if on_sheet else S.TABLE_TOP
        if self.paint and self._in_well(tip) is not None:
            surface = S.WELL_PAINT_Z                                 # pressing into paint, not the table
        st, f, tau, centre = self.brush.step(tip, self._vel6[3:], axis, m.opt.timestep, surface, on_sheet)
        if st.depth > 0.0:
            mujoco.mj_applyFT(m, d, f, tau, tip, self.pen_bid, d.qfrc_applied)
        self._brush_st, self._brush_centre = st, centre
        self._brush_fn, self._brush_ft = st.f_normal, float(np.linalg.norm(f[:2]))
        ferrule = any(self.paper_gid in (c.geom1, c.geom2) and self.ferrule_gid in (c.geom1, c.geom2)
                      for c in (d.contact[i] for i in range(d.ncon)))
        self._brush_viol_sub |= self.brush.violation(st, ferrule)
        if self.paint:
            self._paint_stations(tip, m.opt.timestep)

    # ------------------------------------------------------------ L5 paint stations
    @staticmethod
    def _in_well(tip):
        for k, (x, y) in enumerate(S.WELL_XY):
            if np.hypot(tip[0] - x, tip[1] - y) < S.WELL_R and tip[2] < S.WELL_TOP:
                return k
        return None

    def _paint_stations(self, tip, dt):
        """Wells and rack, judged on where the nominal tip is: DIP_S in a well loads the
        brush with that colour; entering a rack slot hands over that (dry) brush."""
        w = self._in_well(tip)
        if w is None:
            self._dip_t, self._dip_done = 0.0, False
        else:
            self._dip_t += dt
            if self._dip_t >= DIP_S and not self._dip_done:
                self.brush.dip(self.task.palette[w])
                self._paint_idx, self._dips, self._dip_done = w + 1, self._dips + 1, True
                self._hair_color(self.brush.color)
        r = next((k for k, (x, y) in enumerate(S.RACK_XY)
                  if np.hypot(tip[0] - x, tip[1] - y) < S.RACK_R and tip[2] < S.RACK_TOP + 0.015), None)
        if r is None:
            self._rack_latch = None
        elif r != self._rack_latch:
            self._rack_latch = r
            if r != self.brush_id:
                self._set_brush(r)
                self._swaps += 1

    def _set_brush(self, k):
        self.brush_id, self.brush_cfg = k, self.brush_cfgs[k]
        self.brush.set_config(self.brush_cfg)
        self._paint_idx = 0
        self._hair_color(None)

    def _hair_color(self, rgb):
        """The tuft is drawn in the paint it carries (visual only)."""
        self.model.mat_rgba[self.hair_mat] = np.r_[self._hair_rgba0[:3] if rgb is None else np.clip(rgb, 0, 1), 1.0]

    def _paint_plan(self):
        """Which strokes need the rack / a well before them: a brush change, a colour change,
        or a reservoir estimate (footprint area x length x c_ink) about to run low. The
        timeline visits the stations, so `write_target_in_hand` shows the whole job."""
        from .brush import MEDIUM, depth_for_force, footprint_target, grip_axis
        t, theta = self.task, float(np.arccos(np.clip(grip_axis()[2], -1, 1)))
        swap, dip, res, brush, col = set(), set(), 0.0, MEDIUM, None
        for k, s in enumerate(t.strokes):
            cfg = self.brush_cfgs[t.brushes[k]]
            if t.brushes[k] != brush:
                swap.add(k)
                brush, res, col = t.brushes[k], 0.0, None
            a, b = footprint_target(cfg, depth_for_force(cfg, float(np.mean(t.forces[k]))), theta)
            use = cfg.c_ink * np.pi * a * b * float(np.linalg.norm(np.diff(np.asarray(s), axis=0), axis=1).sum())
            if col != t.colors[k] or res - use < 0.2:
                dip.add(k)
                res, col = 1.0, t.colors[k]
            res -= use
        return swap, dip

    def _paint_detours(self, k):
        """Station visits before stroke k: [(xyz, dwell_s), ...] - above the slot / well,
        down into it, up again."""
        out = []
        if k in self._swap_before:
            x, y = S.RACK_XY[self.task.brushes[k]]
            # dwells are long on purpose: a follower that trails the timeline by half a second
            # must still spend DIP_S inside the station
            out += [((x, y, S.RACK_TOP + 0.06), 0.0), ((x, y, S.RACK_TOP - 0.002), 0.8), ((x, y, S.RACK_TOP + 0.06), 0.0)]
        if k in self._dip_before:
            x, y = S.WELL_XY[self.task.colors[k] - 1]
            out += [((x, y, S.WELL_TOP + 0.05), 0.0), ((x, y, S.WELL_PAINT_Z - 0.002), DIP_S + 0.8),
                    ((x, y, S.WELL_TOP + 0.05), 0.0)]
        return [(np.asarray(p, float), dw) for p, dw in out]

    def _paint_vec(self):
        """(6,): the colour on the brush (RGB, zero when dry) and which brush is in the hand."""
        v = np.zeros(6, np.float32)
        if self.paint:
            if self.brush.color is not None and self.brush.m > 0.0:
                v[:3] = self.brush.color
            v[3 + self.brush_id] = 1.0
        return v

    def _rasterize_color_map(self, task, win):
        """L5: the target's palette index per pixel (0 = none). Later strokes OVERWRITE
        earlier ones, as paint does, so a door painted over a wall keeps its colour."""
        win_c, win_h = win
        img = np.zeros((INK_MAP_H, INK_MAP_W), np.uint8)
        px_per_m = INK_MAP_W / (2 * win_h[0])
        for k, s_ in enumerate(task.strokes):
            P = np.asarray(s_) + S.PAPER_CENTER[:2]
            w = task.widths[k] if task.widths is not None else np.full(len(P), 1.0)
            for i in range(len(P)):
                seg = [P[i]]
                if i > 0:
                    n = int(np.linalg.norm(P[i] - P[i - 1]) * px_per_m) + 1
                    seg = [P[i - 1] + (P[i] - P[i - 1]) * j / n for j in range(1, n + 1)]
                rad = max(1, int(round(0.5 * w[i] / 1000 * px_per_m)))
                for q in seg:
                    px, py = self._to_px(q, win_c, win_h)
                    if 0 <= px < INK_MAP_W and 0 <= py < INK_MAP_H:
                        y0, y1, x0, x1 = max(0, py - rad), min(INK_MAP_H, py + rad + 1), max(0, px - rad), min(INK_MAP_W, px + rad + 1)
                        yy, xx = np.mgrid[y0:y1, x0:x1]
                        img[y0:y1, x0:x1][(yy - py) ** 2 + (xx - px) ** 2 <= rad * rad] = int(task.colors[k])
        return img

    def _brush_ink(self):
        """Stamp the footprint into the metric canvas, the display texture and the
        observation map, keeping the per-step contact statistics the pen path keeps.
        A painting stamps in the colour the brush carries; a dry brush lays nothing."""
        from .brush import ellipse_mask
        st, centre = self._brush_st, self._brush_centre
        f = st.f_normal * self._f_gain if st is not None else 0.0
        self._force_acc += f
        self._contact_hits += f > 0.0
        color = self.brush.color if self.paint else None
        if st is None or centre is None or st.depth <= 0.0 or st.ink <= 0.0 or (self.paint and color is None):
            self.brush_render.lift()
            return
        self.brush_render.stamp(st, centre, color)
        v = S.INK_V_REF                              # footprint speed, for the deposition factor
        if self.ink_points:
            dt_ink = float(self.data.time - self.ink_points[-1][3])
            if 0.0 < dt_ink < 0.05:
                v = float(np.linalg.norm(np.asarray(centre[:2], float)
                                         - np.asarray(self.ink_points[-1][:2], float))) / dt_ink
        self.ink_points.append((centre[0], centre[1], 2000.0 * st.b, self.data.time))
        self._ink_new.append((centre[0], centre[1]))
        keep = 1.0 if st.ink >= self.brush_cfg.m_dry else st.ink / self.brush_cfg.m_dry
        rgb8 = np.asarray(S.TOOLS["brush"]["ink_rgb"], np.uint8) if color is None else \
            (np.clip(color, 0, 1) * 255).astype(np.uint8)
        # display texture: a square image over the 4:3 sheet, so its pixels are not square
        u = (centre[0] - S.PAPER_CENTER[0] + S.PAPER_HALF[0]) / (2 * S.PAPER_HALF[0])
        v = (centre[1] - S.PAPER_CENTER[1] + S.PAPER_HALF[1]) / (2 * S.PAPER_HALF[1])
        ppm = S.INK_RES / (2 * S.PAPER_HALF[0])
        mk = ellipse_mask(self.ink.shape[:2], u * S.INK_RES, (1 - v) * S.INK_RES, st.a * ppm, st.b * ppm,
                          -st.phi, self._noise_tex, keep, aspect=S.PAPER_HALF[0] / S.PAPER_HALF[1])
        if mk is not None:
            y0, y1, x0, x1, mask = mk
            self.ink[y0:y1, x0:x1][mask] = rgb8
        # observation maps (1 mm/px, square): ink so far, and for a painting its palette index
        u = (centre[0] - S.PAPER_CENTER[0] - self.win_c[0] + self.win_h[0]) / (2 * self.win_h[0])
        v = (centre[1] - S.PAPER_CENTER[1] - self.win_c[1] + self.win_h[1]) / (2 * self.win_h[1])
        ppm = INK_MAP_W / (2 * self.win_h[0])
        mk = ellipse_mask(self._inkmap.shape, u * INK_MAP_W, (1 - v) * INK_MAP_H, st.a * ppm, st.b * ppm,
                          -st.phi, self._noise_map, keep)
        if mk is not None:
            y0, y1, x0, x1, mask = mk
            sub = self._inkmap[y0:y1, x0:x1]
            # the tuft's ink load and the stroke speed decide how dark the mark comes out;
            # a flat 255 would throw that away for the brush tier
            dark = float(np.clip(keep * S.ink_speed_factor(v), 0.05, 1.0))
            np.maximum(sub, np.where(mask, int(round(255 * dark)), 0).astype(np.uint8), out=sub)
            if self.paint:
                self._colmap[y0:y1, x0:x1][mask] = self._paint_idx

    def _pen_paper_force(self):
        if self.chop:
            return self._object_contacts()[2]
        if self.brush is not None:
            return self._brush_fn * self._f_gain                  # the tuft model's normal force
        f6 = np.zeros(6)
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.paper_gid in (c.geom1, c.geom2) and (c.geom1 in self.pen_gids or c.geom2 in self.pen_gids):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])
        return total * self._f_gain

    def _ink_step(self):
        if self.chop:
            return
        if self.brush is not None:
            self._brush_ink()
            return
        f = self._pen_paper_force()
        self._force_acc += f
        self._contact_hits += f > 0.0
        if f <= 0.0:
            return
        tip = self._tip()
        # the force is over every tool geom, so without this a policy inks by
        # pressing the pen's BODY (or the brush handle) onto the sheet while the
        # tip is centimetres above it
        if tip[2] - S.PAPER_TOP > INK_TIP_REACH:
            return
        prev = self.ink_points[-1] if self.ink_points else None
        # tip speed along the sheet, measured from the last mark: the same stroke drawn
        # faster is laid over more paper, so it comes out lighter and slightly thinner
        v = S.INK_V_REF
        if prev is not None:
            dt_ink = float(self.data.time - prev[3])
            if 0.0 < dt_ink < 0.05:
                v = float(np.linalg.norm(tip[:2] - np.asarray(prev[:2], float))) / dt_ink
        w, a = S.ink_deposit(self.tool, f, v)
        self.ink_points.append((tip[0], tip[1], w, self.data.time))
        self._ink_new.append((tip[0], tip[1]))
        # paint a dot, joined to the previous dot when it is recent and close
        pts = [tip[:2]]
        if prev is not None and self.data.time - prev[3] < 0.02 and \
                np.linalg.norm(tip[:2] - prev[:2]) < 0.005:
            n = int(np.linalg.norm(tip[:2] - prev[:2]) / 0.0004) + 1
            pts = [np.asarray(prev[:2]) + (tip[:2] - prev[:2]) * k / n for k in range(1, n + 1)]
        for p in pts:
            self._paint(p, w, a)
            self._stamp(self._inkmap, p, max(1, int(round(0.5 * w / 1000 * INK_MAP_W / (2 * self.win_h[0])))),
                        int(round(255 * a)), self.win_c, self.win_h)

    def _paint_into(self, tex, p, w, rgb, a=1.0):
        """Stamp a disc of colour `rgb` into an arbitrary texture array (goal layer)."""
        u = (p[0] - S.PAPER_CENTER[0] + S.PAPER_HALF[0]) / (2 * S.PAPER_HALF[0])
        v = (p[1] - S.PAPER_CENTER[1] + S.PAPER_HALF[1]) / (2 * S.PAPER_HALF[1])
        if not (0 <= u < 1 and 0 <= v < 1):
            return
        px, py = int(u * S.INK_RES), int((1 - v) * S.INK_RES)
        r = max(1, int(round(0.5 * w / 1000 / (2 * S.PAPER_HALF[0]) * S.INK_RES)))
        y0, y1 = max(0, py - r), min(S.INK_RES, py + r + 1)
        x0, x1 = max(0, px - r), min(S.INK_RES, px + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - py) ** 2 + (xx - px) ** 2 <= r * r
        sub = tex[y0:y1, x0:x1]
        sub[mask] = ((1.0 - a) * sub[mask].astype(np.float32) + a * rgb).astype(np.uint8)

    def _tex_merged(self):
        """What the paper LOOKS like: darker-of(goal grey, real ink)."""
        return self.ink if self._goal_tex is None else np.minimum(self.ink, self._goal_tex)

    def refresh_texture(self, handle=None):
        """Push the merged paper texture to a live viewer (mjpython launch_passive)."""
        tex = self._tex_merged()
        self.model.tex_data[self.tex_adr:self.tex_adr + tex.size] = tex.ravel()
        if handle is not None:
            try:
                handle.update_texture(self.tex_id)
            except Exception:
                pass

    def _paint(self, p, w, a=1.0):
        # paper texture spans the sheet; u along +x, v along +y
        u = (p[0] - S.PAPER_CENTER[0] + S.PAPER_HALF[0]) / (2 * S.PAPER_HALF[0])
        v = (p[1] - S.PAPER_CENTER[1] + S.PAPER_HALF[1]) / (2 * S.PAPER_HALF[1])
        if not (0 <= u < 1 and 0 <= v < 1):
            return
        px, py = int(u * S.INK_RES), int((1 - v) * S.INK_RES)   # texture row 0 = far (+y) edge
        r = max(1, int(round(0.5 * w / 1000 / (2 * S.PAPER_HALF[0]) * S.INK_RES)))
        y0, y1 = max(0, py - r), min(S.INK_RES, py + r + 1)
        x0, x1 = max(0, px - r), min(S.INK_RES, px + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - py) ** 2 + (xx - px) ** 2 <= r * r
        sub = self.ink[y0:y1, x0:x1]
        rgb = np.asarray(S.TOOLS[self.tool]["ink_rgb"], np.float32)
        sub[mask] = ((1.0 - a) * sub[mask].astype(np.float32) + a * rgb).astype(np.uint8)   # light touch = light line

    def _advance_waypoints(self):
        if self.wp_i >= len(self.wp):
            return
        tip = self._tip()
        touching = self._force > 0
        # Capture the furthest of the next K waypoints the tip can reach. Pen-up
        # hops may skip freely (K=10, ~15 mm) - they lay no ink and only reposition.
        # Pen-DOWN waypoints may not: with a large K a policy sprints along the
        # stroke just inside the capture radius, banks full progress and leaves
        # almost no ink (e.g. a line "traced" in 1.28 s vs the scripted 2.9 s,
        # ink IoU 0.34). Drawing a stroke means passing over it, so K=2 there.
        if self.wp_down[min(self.wp_i, len(self.wp) - 1)]:
            hi = min(self.wp_i + 2, len(self.wp))
        else:
            # Pen-up hops lay no ink and only reposition, so the cursor may skip the
            # WHOLE run of them and land on the stroke they lead to. A fixed K=10 is
            # not enough: an approach ramp is ~20 points, and a policy that simply
            # drops onto the sheet (as the scripted teacher does) would leave the
            # cursor stranded on a descent it is free to shortcut, so `progress`
            # would pay it exactly what it pays stalling.
            end = self.wp_i
            while end < len(self.wp) and not self.wp_down[end]:
                end += 1
            hi = min(end + 1, len(self.wp))
        hit = None
        for j in range(self.wp_i, hi):
            # pen-up hops need no precision: 3x radius; pen-down: radius (+ press depth)
            cr = self.task.capture_radius if self.capture_radius is None else self.capture_radius
            r = (cr + PRESS_DEPTH) if self.wp_down[j] else 3 * cr
            if np.linalg.norm(self.wp[j] - tip) < r and (touching or not self.wp_down[j]):
                hit = j
        if hit is not None:
            self.wp_i = hit + 1
        # A painting is done when its SCHEDULE is: coverage is colour-blind and counts a
        # neighbouring band's ink, so three wide arcs of five already "cover" a rainbow
        # (which would end the episode with two colours never dipped)
        schedule_done = not self.paint or self._timeline is None or self.t >= len(self._timeline) - 1
        if self.done_time is None and self.task.strokes and schedule_done and \
                self.coverage() >= self.task.coverage_for_success:
            self.done_time = self.t * self.dt

    def _check_drop(self):
        d = self.data
        touching = any(c.geom1 in self.finger_gids or c.geom2 in self.finger_gids
                       for c in (d.contact[i] for i in range(d.ncon))
                       if c.geom1 in self.pen_gids or c.geom2 in self.pen_gids)
        # Thresholds: 2 cm of travel or 30 deg of rotation in the hand. 6 cm with
        # no rotation check lets a visibly failed grip pass. Fingertips off the
        # shaft count too (GAP_LOST): a relaxed hand lying on a pen that the
        # PAPER holds up never trips the slip test (gaps 4-10 mm, the pen
        # dragged across the target and inking). Good policies stay
        # under 6 mm (scripted p95 3.7).
        dp, dax, _ = self._slip()
        far = dp > 0.02 or dax > 2 * np.sin(np.radians(30) / 2) or self._finger_gaps() > GAP_LOST
        self._drop_timer = 0.0 if touching and not far else self._drop_timer + self.dt
        # in the pick-up task an empty hand is the starting condition, not a failure
        if self._drop_timer > 0.3 and self._ever_held:
            self.dropped = True

    def _check_hover(self):
        """Training rule (reward preset `hover_end`): on a pen-down segment of the
        timeline, a tip that stays more than `hover_h` above the sheet for `hover_t`
        seconds ends the episode, charged like a drop. A hovering pen never drops, so
        without this the safest policy under any per-step charge is to hover and wait
        out the clock. Never active in the
        default preset, so scoring is untouched."""
        w = self.rw
        if not w.get("hover_end") or self._timeline is None or self.hovered:
            return
        # Cursor-agnostic on purpose. The patient timeline parks its cursor on the first
        # point - the LIFT above the stroke start, a pen-up point - until the tip captures
        # it, so a pen-down gate never engages for a policy that never gets there. The rule
        # is simply: the tip may not stay `hover_h` above the sheet for more than `hover_t`
        # seconds at a stretch. The teacher's approach (lift to the first stroke start at
        # 0.15 m/s, land) is 2.5 s in the air, so 2 s would trip it on every task; 4 s never
        # does, and still ends a hovering episode inside its first third.
        high = float(self._tip()[2] - S.PAPER_TOP) > float(w.get("hover_h", 0.01))
        self._hover_timer = self._hover_timer + self.dt if high else 0.0
        if self._hover_timer > float(w.get("hover_t", 1.0)):
            self.hovered = True
        if w.get("far_end"):
            # ... and it may not stay `far_d` away from the timeline target for `far_t`
            # s either (otherwise a policy can fly the tip far off, below the sheet plane)
            k = min(self._tl_k if w.get("patient") else self.t, len(self._timeline) - 1)
            far = float(np.linalg.norm(self._timeline[k] - self._tip())) > float(w.get("far_d", 0.15))
            self._far_timer = self._far_timer + self.dt if far else 0.0
            if self._far_timer > float(w.get("far_t", 3.0)):
                self.hovered = True

    def _maybe_push(self):
        P = self.perturb
        if "push_force" not in P or self.data.time < self._next_push:
            return
        f = self.np_random.normal(size=3)
        f *= P["push_force"] / np.linalg.norm(f)
        self.data.xfrc_applied[self.pen_bid, :3] = f
        self._next_push = self.data.time + P["push_every"] * self.np_random.uniform(0.7, 1.3)
        self._push_off = self.data.time + 0.05
        if self.data.time > getattr(self, "_push_off", 0):
            self.data.xfrc_applied[self.pen_bid] = 0.0

    # ------------------------------------------------------------ observation
    def _position(self):
        """What the joint encoders report, plus the ARM's commanded joint targets.

        The arm is position-controlled (`servoj` on a real UR5e) and its action
        integrates, so those 6 setpoints have to be observable or the problem is
        not Markov. The hand is torque-controlled and has no setpoint to report."""
        d = self.data
        return np.concatenate([d.qpos[self.qadr], d.qvel[self.dadr] * 0.1,
                               self.target_q[:N_ARM]]).astype(np.float32)

    def _tactile(self):
        """Joint torque as a fraction of each joint's limit.

        The per-finger and tool-paper contact forces are not included: on scripted
        rollouts, 70-94% of their variance is already recoverable from these joint
        torques, one channel (pinky) is identically zero, and the tool-paper force
        is the quantity the task is scored on. Both are still reported in `info` for teachers and diagnostics."""
        d = self.data
        # what a joint torque sensor transmits: the motor's own contribution plus
        # whatever the contacts push back with. Under a torque action the servo is
        # parked, so `qfrc_constraint` is the only part the policy does not already know.
        tau = (d.actuator_force + self._tau_applied + d.qfrc_constraint[self.dadr]) \
            / np.maximum(self.tau, 1e-9)
        return np.clip(tau, -4, 4).astype(np.float32)

    def _task_vec(self):
        tool = [float(self.tool == t) for t in ("pen", "brush", "chopsticks")]
        return np.concatenate([[self.t / max(self.T, 1), self.t * self.dt], tool,
                               [S.FORCE_MAX], self.win_h, self.win_c]).astype(np.float32)

    def _external(self):
        """Everything outside the hand: poses of hand, tool, sheet and object."""
        d = self.data
        R, hb = d.xmat[self.hb].reshape(3, 3), d.xpos[self.hb]
        tip = self._tip()
        tool_p = R.T @ (d.xpos[self.pen_bid] - hb)
        tool_ax = R.T @ d.xmat[self.pen_bid].reshape(3, 3)[:, 2]
        sheet_o = R.T @ (S.PAPER_CENTER + [0, 0, S.PAPER_HALF[2]] - hb)
        sheet_n = R.T @ np.array([0.0, 0.0, 1.0])
        if self.chop and not self.chop_hold:
            obj = R.T @ (d.qpos[self.oq:self.oq + 3] - hb)
            tgt = R.T @ (d.mocap_pos[self.bowl_mid] - hb)
        else:
            obj = tgt = np.zeros(3)
        # Where the stroke wants the tip to be RIGHT NOW, and which way it is
        # heading. The target walks a precomputed timeline, so there is a dense
        # "you are behind, go this way" signal at every step. A waypoint cursor
        # that only advances on capture gives none: standing still freezes the
        # cursor and with it the gradient.
        if self._timeline is None:
            wt, w1, w2 = np.zeros(3), np.zeros(3), np.zeros(3); pd = 0.0
        else:
            last = len(self._timeline) - 1
            k = min(self._tl_k, last) if self.rw.get("patient") else min(self.t, last)
            # TWO lookaheads, not a unit tangent: +0.05 s and +0.2 s deltas carry the target's speed and curvature as well as its
            # direction. A normalised tangent throws both away.
            k1 = min(k + max(1, int(round(0.05 / self.dt))), last)
            k2 = min(k + max(1, int(round(0.20 / self.dt))), last)
            wt = R.T @ (self._timeline[k] - hb)
            w1 = R.T @ (self._timeline[k1] - self._timeline[k])
            w2 = R.T @ (self._timeline[k2] - self._timeline[k])
            pd = float(self._tl_down[k])
        # the cursor as the expert sees it (info); zeroed in the observation when cursor_obs is off
        self._cursor_priv = dict(write_target_in_hand=np.asarray(wt, float).copy(), ahead_near=np.asarray(w1, float).copy(),
                                 ahead_far=np.asarray(w2, float).copy(), pen_down=float(pd),
                                 finished=bool(self._timeline is not None and self._tl_k > len(self._timeline) - 1))
        if not self.cursor_obs:
            wt, w1, w2, pd = np.zeros(3), np.zeros(3), np.zeros(3), 0.0
        if self.brush is not None:
            # the tuft's footprint state: no real sensor reports this, the tier is simulation-only
            b = self.brush
            br = [b.a, b.b, np.cos(b.phi), np.sin(b.phi), b.m, self._brush_fn, self._brush_ft]
        else:
            br = np.zeros(7)
        return np.concatenate([hb, R[:, 0], R[:, 1], tool_p, tool_ax, R.T @ (tip - hb),
                               [tip[2] - S.PAPER_TOP], sheet_o, sheet_n, obj, tgt,
                               wt, w1, w2, [pd], br]).astype(np.float32)

    def _noisy(self, v):
        n = self.perturb.get("obs_noise", 0)
        return v + self.np_random.normal(0, n, v.shape).astype(np.float32) if n else v

    def _rasterize_task_map(self, task, win):
        """The question, as an image: the target strokes drawn in the task-aligned
        window with pixel value = 255 * target_force / FORCE_MAX and thickness = the
        ink width that force implies. 0 is background, so any target pixel is >= 1."""
        win_c, win_h = win
        m = np.zeros((INK_MAP_H, INK_MAP_W), np.uint8)
        px_per_m = INK_MAP_W / (2 * win_h[0])
        for k, s_ in enumerate(task.strokes):
            P = np.asarray(s_) + S.PAPER_CENTER[:2]
            f = task.forces[k] if task.forces is not None else np.full(len(P), 0.8)
            w = task.widths[k] if task.widths is not None else np.full(len(P), 1.0)
            for i in range(len(P)):
                seg = [P[i]] if i == 0 else None
                if i > 0:
                    n = int(np.linalg.norm(P[i] - P[i - 1]) * px_per_m) + 1
                    seg = [P[i - 1] + (P[i] - P[i - 1]) * j / n for j in range(1, n + 1)]
                val = int(np.clip(255.0 * f[i] / S.FORCE_MAX, 1, 255))
                rad = max(1, int(round(0.5 * w[i] / 1000 * px_per_m)))
                for q in seg:
                    self._stamp(m, q, rad, val, win_c, win_h)
        return m

    @staticmethod
    def _to_px(p, win_c, win_h):
        """World xy -> pixel in the sheet-aligned map (row 0 = the sheet's +y edge).

        win_h is (half_x, half_y) - the map is the sheet's rectangle, not a square
        around it, so the two axes carry their own extent even though the scale is
        the same on both (one pixel per millimetre)."""
        u = (p[0] - S.PAPER_CENTER[0] - win_c[0] + win_h[0]) / (2 * win_h[0])
        v = (p[1] - S.PAPER_CENTER[1] - win_c[1] + win_h[1]) / (2 * win_h[1])
        return int(u * INK_MAP_W), int((1 - v) * INK_MAP_H)

    def _sheet_to_px(self, p):
        return self._to_px(p, self.win_c, self.win_h)

    @classmethod
    def _stamp(cls, img, p, rad, val, win_c, win_h):
        px, py = cls._to_px(p, win_c, win_h)
        if not (0 <= px < INK_MAP_W and 0 <= py < INK_MAP_H):
            return
        y0, y1 = max(0, py - rad), min(INK_MAP_H, py + rad + 1)
        x0, x1 = max(0, px - rad), min(INK_MAP_W, px + rad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - py) ** 2 + (xx - px) ** 2 <= rad * rad
        sub = img[y0:y1, x0:x1]
        np.maximum(sub, np.where(mask, val, 0).astype(np.uint8), out=sub)

    def _target_force_at(self, tip):
        """Target press force the task map asks for at the tool tip, 0 off-target."""
        px, py = self._sheet_to_px(tip[:2])
        if not (0 <= px < INK_MAP_W and 0 <= py < INK_MAP_H):
            return 0.0
        y0, y1 = max(0, py - 1), min(INK_MAP_H, py + 2)
        x0, x1 = max(0, px - 1), min(INK_MAP_W, px + 2)
        patch = self._task_map[y0:y1, x0:x1]
        v = int(patch.max())
        return v / 255.0 * S.FORCE_MAX if v else 0.0

    def _task_centreline(self, task, tmap, win):
        """Pixels along the target's centre lines, decoded once at reset."""
        if not task.strokes:
            return np.zeros((0, 2), int)
        from .planning import map_to_strokes
        win_c, win_h = win
        pts = []
        for st in map_to_strokes(tmap, tool=self.tool, half_m=win_h, center=win_c):
            for xy in st.xy:
                px, py = self._to_px(xy + S.PAPER_CENTER[:2], win_c, win_h)
                if 0 <= px < INK_MAP_W and 0 <= py < INK_MAP_H:
                    pts.append((py, px))
        return np.array(pts, int) if pts else np.zeros((0, 2), int)

    def coverage(self):
        """Fraction of the target's CENTRE LINES the tool has actually passed along.

        Order-free on purpose: the task is a picture, so a policy may draw it in any
        order. Centre lines rather than filled area on purpose too - asking for the
        ink to cover every pixel of a stroke also demands that it be exactly as wide
        as the target, so a legible word drawn a little thin would read as
        unfinished. Width and press fidelity are `ink_iou` and `force_mae_N`."""
        if not len(self._task_pts):
            return 1.0
        hit = self._inkmap > 0
        for _ in range(int(round(COVER_TOL_MM / INK_MAP_MM_PER_PX))):   # dilate by the tolerance
            hit = (hit | np.roll(hit, 1, 0) | np.roll(hit, -1, 0)
                   | np.roll(hit, 1, 1) | np.roll(hit, -1, 1))
        return float(hit[self._task_pts[:, 0], self._task_pts[:, 1]].mean())

    def _obs(self):
        obs = {
            "position": self._noisy(self._position()),
            "tactile": self._noisy(self._tactile()),
            "task": {"map": self._task_map, "vec": self._task_vec(),
                     "color": self._task_color, "palette": self._palette},     # L5 paint keys, zero elsewhere
            "instruction": self.task.instruction,
        }
        if self.traj_obs:
            obs["task"]["traj"] = self._traj_vec()
        if "deployable" not in self.obs_mode:      # the default; "deployable" runs without the group and is flagged
            obs["external"] = {"map": self._inkmap, "vec": self._noisy(self._external()),
                               "color": self._colmap, "paint": self._paint_vec()}   # L5 paint keys, zero elsewhere
        if "pixels" in self.obs_mode:
            obs["images"] = {c: self.render(camera=c) for c in self.cameras}
        return obs

    def camera_calibration(self):
        """Intrinsics / extrinsics of every camera: in the world frame and in the
        UR base frame (what you would reproduce on the real rig)."""
        m, d = self.model, self.data
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        Rb, pb = d.xmat[base].reshape(3, 3), d.xpos[base]
        out = {}
        for c in range(m.ncam):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
            Rw, pw = d.cam_xmat[c].reshape(3, 3), d.cam_xpos[c]
            out[name] = {"fovy_deg": float(m.cam_fovy[c]), "resolution": list(self.image_size),
                         "world": {"pos": pw.tolist(), "xmat": Rw.tolist()},
                         "ur5e_base": {"pos": (Rb.T @ (pw - pb)).tolist(), "xmat": (Rb.T @ Rw).tolist()},
                         "mounted_on": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.cam_bodyid[c])}
        return out


    def _info(self):
        return {"task_id": self.task.task_id, "level": self.task.level, "t": self.t * self.dt,
                "cursor": self._cursor_priv,           # the target's schedule; task input, not physics
                "finger_share": (self._share_fing / self._share_den if self._share_den > 0 else None),
                "wrist_share": (self._share_wrist / self._share_den if self._share_den > 0 else None),
                "arm_share": (self._share_arm / self._share_den if self._share_den > 0 else None),
                "progress": (self.pick_phase / 3.0 if self.pickup else
                             self.phase / 3.0 if (self.chop and not self.chop_hold) else
                             self.coverage()), "dropped": self.dropped, "hovered": self.hovered,
                "done_time": self.done_time, "n_ink": len(self.ink_points),
                "idle_rms_deg": float(np.degrees(np.sqrt(self._idle_dev2 / max(self._idle_n, 1)))),
                "brush_state": (self._brush_st.__dict__ if self.brush is not None and self._brush_st is not None else None),
                "brush_violation_steps": (self._brush_viol_steps if self.brush is not None else None),
                "paint_state": (dict(brush=self.brush_id, color=(None if self.brush.color is None else self.brush.color.tolist()),
                                     ink=self.brush.m, dips=self._dips, swaps=self._swaps) if self.paint else None),
                # privileged: NOT in the observation. Teachers and diagnostics only.
                "tool_contact_force": float(self._force),
                "finger_force": [float(S.contact_normal_force(self.model, self.data, self.finger_geoms[f]))
                                 for f in S.FINGERS]}

    def _reward(self):
        """Shaped reward for RL; see REWARD_PRESETS. Scoring never uses it."""
        if self.rw.get("goal"):
            return self._reward_goal()
        w, d = self.rw, self.data
        a = self._last_action
        r = w["alive"] - w.get("time", 0.0)          # `time`: the clock costs, finish and stop
        r -= w["rate"] * float(np.mean((a - self._prev_action) ** 2))
        r -= w["qvel"] * float(np.mean(d.qvel[self.dadr[self.hand_idx]] ** 2))
        # The two pose regularisers hold the hand at the captured grip. That is right for
        # the hold rungs, which share this preset in the L0 mix, and backwards on a rung
        # scored by how fast the tool moves: with them on, a policy that draws 337 mm
        # returns -47 against +3.8 for one that hovers and never touches. Off, the same
        # pair is +468 against +47 (4 seeds).
        fast = getattr(self.task, "path_ref_mm", 0.0) > 0
        if not fast:
            r -= w["pose"] * float(np.mean((d.qpos[self.qadr[self.hand_idx]] - self._q_grip_hand) ** 2))
        if (w.get("idle_pose") or w.get("idle_vel")) and not fast:
            # ring + pinky take no part in the tripod but must not flail while the others
            # write: charged for leaving the captured grip pose and for moving at all
            qi = d.qpos[self.qadr[self.idle_idx]] - self._q_grip_hand[self._idle_mask]
            r -= w.get("idle_pose", 0.0) * float(np.mean(qi ** 2))
            r -= w.get("idle_vel", 0.0) * float(np.mean(d.qvel[self.dadr[self.idle_idx]] ** 2))
        # L0_scribble: there is no target, so the signal is AREA newly inked this step
        # (dilated, so spreading out pays and re-inking the same spot does not) and a
        # bounty per deliberate lift-and-land. Both are dense; the task's own score is
        # the same area, so the reward is not a proxy for the metric, it is the metric.
        # ...and ONLY where there is no target: on L1/L2 these two terms fight the
        # timeline: with them on, `trace_line` lays 92 % of its ink further than 3 mm
        # from the line (precision 0.081, ink_iou 0.042) and every L1 task comes out
        # as a knot of scribble.
        # ...and ONLY on the rung scored that way. The same preset drives the two hold
        # tasks in the L0 mix, and paying them per metre of tip travel near the sheet
        # has the same problem.
        # The driven scribble gets `contact` (keep the tip on the sheet) and `impact`
        # (do not let it jam and slam) but no `travel` - the env moves it, not the policy.
        driven = getattr(self.task, "hold_motion", None) == "scribble"
        if (w.get("travel") or w.get("impact")) and (getattr(self.task, "path_ref_mm", 0.0) > 0 or driven):
            tip = self._tip()
            if self._force > 0:
                # `contact` is the rung down to the sheet - `travel` pays only while
                # touching, so on its own it is unreachable from a policy that starts
                # airborne. Kept small: at 1.0 it pays resting on the paper 170 an
                # episode, which is most of what drawing earns.
                r += w.get("contact", 0.0)
                if self._travel_prev is not None and not driven:
                    r += w.get("travel", 0.0) * float(np.linalg.norm(tip - self._travel_prev))
            self._travel_prev = tip.copy()
            r -= w.get("impact", 0.0) * max(0.0, self._force - w.get("impact_soft", 8.0))
        if (w.get("area") or w.get("lift")) and not self.task.strokes:
            cov = self._inked_area_px() - (self._cov_base or 0)
            r += w.get("area", 0.0) * max(0, cov - self._cov_prev)
            self._cov_prev = cov
            r += w.get("lift", 0.0) * (self._n_lifts - self._lifts_paid)
            self._lifts_paid = self._n_lifts
        # Which shaping family this preset uses. Not gated on the `timeline` weight alone, so a
        # preset that switches the distance charges off but keeps advance / glyph / ink_off / force_reg
        # does not fall through to the waypoint treadmill, where those terms pay nothing.
        tl_active = self._timeline is not None and any(
            w.get(k) for k in ("timeline", "advance", "glyph", "reentry", "transit", "ink_off", "force_reg"))
        if tl_active:
            k = min(self._tl_k, len(self._timeline) - 1) if w.get("patient") else min(self.t, len(self._timeline) - 1)
            e_t = float(np.linalg.norm(self._timeline[k] - self._tip()))
            if w.get("patient"):
                g = self._arm_gate() if w.get("share_gate") else 1.0
                r += w.get("advance", 0.0) * (self._tl_adv_down * g + self._tl_adv_up)
                r += w.get("glyph", 0.0) / self._tl_n_down * self._tl_adv_down * g
                r += w.get("reentry", 0.0) * self._tl_reentry
                if self._tl_down[k] <= 0.5:
                    r += w.get("transit", 0.0) * max(0.0, 1.0 - e_t / 0.006)
            # Quadratic inside TL_NEAR for precision, bounded LINEAR outside so there
            # is always a gradient pointing back. A quadratic capped at 10 cm is flat
            # beyond the cap, so a tip that starts farther off has no signal at all.
            # The linear term reaches FINE_RANGE (15 cm), not TL_NEAR: the quadratic
            # saturates at 5 cm and `timeline_far` alone is 0.01/mm, so a policy
            # that starts 10 cm off would have nothing to climb and drift away
            # without ever touching the sheet.
            r -= (w["timeline"] * min(e_t * e_t, TL_NEAR ** 2)
                  + w.get("timeline_far", 0.0) * min(max(e_t - TL_NEAR, 0.0), TL_FAR)
                  + w.get("timeline_fine", 0.0) * min(e_t, FINE_RANGE))
            if self._tl_down[k] > 0.5 and (w.get("hover") or w.get("contact")):
                # the sheet is where the pen should be on a pen-down step - charge
                # height, pay any contact (ink below pays for contact ON the line)
                h = float(self._tip()[2] - S.PAPER_TOP)
                r -= w.get("hover", 0.0) * float(np.clip(h, 0.0, 0.05))
                if self._force > 0.0:
                    r += w.get("contact", 0.0)
            # Ink pays for touching ON the timeline point and fades out by 1 cm, and
            # it has to outweigh what contact costs (impact force spikes, slip). A
            # flat +1 does not: a policy then hovers 1-2 cm above the sheet and a few
            # cm off the timeline, rarely touching - hovering and writing score about
            # the same per step.
            if self._tl_down[k] > 0.5:
                # `ink_duty`: pay by the fraction of the control step the tip was in
                # contact, not by the end-of-step force - even the scripted teacher's
                # contact flickers in ~6 ms gaps, which makes the per-step bonus a coin toss,
                # and a policy can learn to skim at 0.15 N with contact on 20 % of steps.
                touch = self._contact_duty if w.get("ink_duty") else float(self._force > 0.0)
                if touch > 0.0 and w.get("ink_ramp"):
                    # no pay for a pen merely resting on the sheet (0.03-0.06 N) or skimming
                    # (0.15 N): full pay from `ink_ramp[1]` N of mean in-contact force
                    lo_f, hi_f = w["ink_ramp"]
                    touch *= float(np.clip((self._force_mean / self._contact_duty - lo_f) / (hi_f - lo_f), 0.0, 1.0))
                if touch > 0.0 and w.get("ink_dir") and self._tip_prev is not None:
                    k2 = min(k + 1, len(self._timeline) - 1)
                    tl_dir = (self._timeline[k2] - self._timeline[max(k - 1, 0)])[:2]
                    if float(tl_dir @ (self._tip() - self._tip_prev)[:2]) <= 0.0:
                        touch = 0.0                      # ink pays only moving ALONG the stroke
                if touch > 0.0:
                    e_lat = float(np.linalg.norm((self._timeline[k] - self._tip())[:2]))
                    gi = self._arm_gate() if (w.get("patient") and w.get("share_gate")) else 1.0
                    r += w["ink"] * touch * gi * max(0.0, 1.0 - e_lat / w.get("ink_fade", INK_ON_TIMELINE))
                if w.get("ink_off") and self._force > 0.0:
                    # ink laid off the line is what `precision` (1 mm) charges; charge it here too
                    e_lat = float(np.linalg.norm((self._timeline[k] - self._tip())[:2]))
                    r -= w["ink_off"] * min(max(e_lat - 0.001, 0.0) / 0.002, 1.0)
                if w.get("force_reg") and self._contact_duty > 0.0:
                    # regular ink = regular pressure: charge the deviation of the mean
                    # in-contact force from what the map asks for at the tip
                    tgt = self._target_force_at(self._tip())
                    f_c = self._force_mean / self._contact_duty
                    r -= w["force_reg"] * abs(f_c - (tgt if tgt > 0.0 else 0.8))
            r -= w["force"] * max(self._force - w["press_limit"], 0.0)
            if (w["arm"] or w["wrist"]) and self._pen_engaged():
                v_arm, v_wrist, _ = self._vsplit          # tip motion the arm / wrist produced
                r -= w["arm"] * float(np.linalg.norm(v_arm))
                r -= w["wrist"] * float(np.linalg.norm(v_wrist))
        if w.get("slip_pos") or w.get("gap"):
            # Slipping is charged continuously, not only when the tool finally leaves
            # the hand. Rotation and relative velocity are charged too: a pen that is
            # turning between the fingers is already failing, and penalising the
            # velocity catches it while it is still happening.
            dp, dax, v = self._slip()
            r -= w.get("slip_pos", 0.0) * dp * dp
            r -= w.get("slip_rot", 0.0) * dax * dax
            r -= w.get("slip_vel", 0.0) * v * v
            r -= w.get("gap", 0.0) * self._finger_gaps()
        if self.chop and not self.chop_hold:             # L5: approach -> pinch -> lift -> place
            obj, bowl, lower = self._l5_targets()
            mid = 0.5 * (self._tip() + lower)
            r -= 20.0 * min(float(np.linalg.norm(mid - obj)), 0.05) if self.phase < 2 else \
                20.0 * min(float(np.linalg.norm(obj - bowl)), 0.05)
            r += 0.5 * self.phase + 10.0 * float(self.done_time is not None)
        elif self.task.strokes and self.task.hold_motion is None and not tl_active:   # ink tasks: waypoint treadmill
            # (the timeline reward above replaces this whole block - running both
            # would charge `force` twice and pay `ink` twice)
            i = min(self.wp_i, len(self.wp) - 1)
            e = float(np.linalg.norm(self.wp[i] - self._tip()))
            r -= w["track"] * min(e * e, 0.0025)             # quadratic inside 5 cm
            r -= w["fine"] * min(e, FINE_RANGE)              # linear: a quadratic vanishes
            #                                                 exactly at the mm scale that
            #                                                 decides whether a waypoint is
            #                                                 captured at all
            r -= w["far"] * min(max(e - 0.05, 0.0), 0.5)     # bounded slope outside
            r += w["progress"] * (self.wp_i - self._wp_i_prev) / len(self.wp)
            if self._e_prev is not None and self.wp_i == self._wp_i_prev:
                r += w["approach"] * (self._e_prev - e)      # skipped on capture: e jumps to the next waypoint
            self._e_prev = e
            if self.wp_down[i] and self._force > 0.0:
                # Paid for inking ON the target, not merely for touching the sheet:
                # a flat contact bonus is a "hover and collect" trap: a policy can park
                # just outside the capture radius, touch at 0.2 N and stall for most
                # of the episode while banking the bonus every step.
                r += w["ink"] * max(0.0, 1.0 - e / INK_ON_TARGET)
            # Writing is a sub-newton job (the scripted baseline averages 0.33 N).
            # The charge has to outweigh what a waypoint capture pays (~3), or the
            # policy simply buys progress with force: at weight 1.0 it presses at
            # 3.16 N mean / 10 N peak, touches the sheet on only 33 % of pen-down
            # steps and covers 15 % of the target.
            # Without this the policy dives at the sheet at 20+ N; with the limit
            # set loosely it just presses right up against it (1.80 N at limit
            # 2.0), and the ink comes out 3.7 mm wide and blotchy.
            r -= w["force"] * max(self._force - w["press_limit"], 0.0)
            if (w["arm"] or w["wrist"]) and self._pen_engaged():
                v_arm, v_wrist, _ = self._vsplit          # tip motion the arm / wrist produced
                r -= w["arm"] * float(np.linalg.norm(v_arm))
                r -= w["wrist"] * float(np.linalg.norm(v_wrist))
        # Finishing ends the episode, so without this the alive bonus makes
        # stalling beat solving the task outright (on L2_write_word:
        # 1217 for doing nothing vs 978 for the scripted teacher, which writes
        # the whole word). Collecting the unused alive bonus on completion is the
        # mirror image of the drop penalty and restores the ordering.
        if self.done_time is not None and not self._done_prev:
            r += w["alive"] * max(self.T - self.t, 0)
        self._done_prev = self.done_time is not None
        if (self.dropped or self.hovered) and w["drop"]:
            # a drop pays the rest of the episode at the worst per-step rate, so
            # letting go is strictly worse than holding on with any tracking error
            # every capped per-step penalty has to appear here, or letting go becomes
            # cheaper than holding on at a large tracking error (death-bait).
            worst = (w["alive"] + w["track"] * 0.0025 + w["fine"] * FINE_RANGE
                     + w["far"] * 0.5 + w["ink"]
                     + w.get("advance", 0.0) * 6 + w.get("transit", 0.0)
                     + w.get("timeline", 0.0) * TL_NEAR ** 2
                     + w.get("timeline_far", 0.0) * TL_FAR
                     + w.get("timeline_fine", 0.0) * FINE_RANGE)
            r -= worst * max(self.T - self.t, 0)
        self._prev_action = a.copy()
        self._wp_i_prev = self.wp_i
        self._tip_prev = self._tip().copy()
        return float(r)

    GOAL_TOL = 0.001            # 1 mm: completion and precision are scored at this tolerance
    GOAL_LOOKAHEAD = 40         # target points (0.5 mm apart) an ink point may jump ahead: 20 mm

    def _reward_goal(self):
        """v15_goal: the evaluation metrics, paid as they are earned, and nothing else.

        completion is in-order coverage of the target centre line, resampled at 0.5 mm exactly as
        metrics._target_strokes does. Each new ink point covers the EARLIEST not-yet-passed target
        points within 1 mm, looking at most GOAL_LOOKAHEAD points ahead; coverage never moves
        backwards, so strokes drawn out of order or in the wrong direction earn nothing - the rule the
        monotone alignment in metrics._stroke_scores applies in hindsight. precision and force use the
        same 1 mm centre line and the same on-target force lookup as the metrics."""
        w = self.rw
        tip = self._tip()
        tip_prev = self._tip_prev                    # where the tip was last step (dense terms)
        self._prev_action = self._last_action.copy()
        self._wp_i_prev = self.wp_i
        self._tip_prev = tip.copy()
        self._done_prev = self.done_time is not None
        if not self.task.strokes or self._timeline is None:
            return 0.0
        if self._goal_line is None:
            from scipy.spatial import cKDTree
            from .metrics import _resample_arc, _target_strokes
            strokes = _target_strokes({"timeline": self._timeline, "tl_down": self._tl_down})
            self._goal_line = (np.concatenate([_resample_arc(t) for t in strokes]) if strokes
                               else np.zeros((1, 2)))
            self._goal_tree = cKDTree(self._goal_line)
            self._goal_cov = np.zeros(len(self._goal_line), float)     # credit, not a flag
            self._goal_j = 0
        n_line = len(self._goal_line)
        tol = float(w.get("goal_tol", self.GOAL_TOL))
        soft = float(w.get("goal_soft", 0.0))        # > tol: the ramp's outer radius (see below)
        gain = float(w.get("goal_soft_gain", 1.0))   # credit outside `tol`, as a fraction of on-target
        r = -w["w_time"] * self.dt / max(float(self.task.duration), 1e-6)
        if w.get("w_track_goal") and tip_prev is not None:
            # A reward that hard-codes tracking the trajectory has BC as its optimum, so the policy
            # must be able to explore outside it. Same restoring force as w_track (the only term that
            # pays while the pen is OFF the sheet, without which the policy diffuses away from the
            # writing and completion collapses), but referenced to the GLYPH instead of the teacher's plan: the
            # nearest target point that is still uncovered, at the sheet. It says "go do the work that
            # is left", not "be at the teacher's k-th point now", so the route between strokes, the
            # speed along them and the order within what the metric allows are the policy's to choose.
            # Restricted to points at or after the completion pointer so it can never pull backwards
            # into coverage that is already forfeited.
            un = np.nonzero(self._goal_cov[self._goal_j:] < 1.0)[0]
            if len(un):
                q = self._goal_line[self._goal_j + un]
                p = np.array([*q[int(np.argmin(np.linalg.norm(q - tip[:2], axis=1)))], S.PAPER_TOP])
                r += w["w_track_goal"] * (float(np.linalg.norm(tip_prev - p)) - float(np.linalg.norm(tip - p)))
        if w.get("w_track_next") and tip_prev is not None:
            # w_track_goal with its two faults removed:
            #  - it aims AT the sheet top, while every pen-down point of the task's trajectory sits
            #    1.5 mm INTO it; hovering half a millimetre up then pays about as well as pressing, and
            #    once contact costs anything the policy lifts off. This target is at `goal_depth`.
            #  - it aims at the NEAREST uncovered point, which changes as the tip wanders, and every
            #    change resets the potential in the wanderer's favour. This target is the FIRST
            #    uncovered point at or after the completion pointer: it moves only as coverage does.
            # Still the glyph, not the teacher's timeline - the order is the task's stroke order, which
            # the metric scores; the timing, route and transits are left to the policy.
            un = np.nonzero(self._goal_cov[self._goal_j:] < 1.0)[0]
            if len(un):
                q = self._goal_line[self._goal_j + int(un[0])]
                p = np.array([q[0], q[1], S.PAPER_TOP - float(w.get("goal_depth", 0.0015))])
                r += w["w_track_next"] * (float(np.linalg.norm(tip_prev - p)) - float(np.linalg.norm(tip - p)))
        if w.get("w_track") and tip_prev is not None:
            # the dense term: how much closer the tip got to the CURRENT timeline point (3-D, so it
            # also pulls the pen down to the pressing depth). Same point on both sides, so the
            # pointer moving on is never charged, and the episode total is the net approach.
            k = min(self._tl_k, len(self._timeline) - 1)
            p_k = self._timeline[k]
            r += w["w_track"] * (float(np.linalg.norm(tip_prev - p_k)) - float(np.linalg.norm(tip - p_k)))
        sched_done = self._tl_k > len(self._timeline) - 1          # the trajectory's pointer is past its end
        if w.get("done_frac") and self._goal_cov is not None and len(self._goal_cov):
            # ... or the GLYPH is written. "Finished" should be a fact about the drawing, not about
            # how far the teacher's pointer got; tying it to the pointer makes chasing the teacher's
            # schedule the only way to stop without paying the rest of the clock. Strictly a second
            # way to be finished - it never makes the pointer rule stricter.
            sched_done = sched_done or bool(float((self._goal_cov >= 1.0).mean()) >= float(w["done_frac"]))
        if w.get("time_to_end") and (self.dropped or ((self.hovered or self._done_req) and not sched_done)):
            # the episode is ending without the schedule finished: pay the rest of the clock, so
            # being terminated early is worth nothing. A DROP pays it even after the schedule is
            # finished - otherwise a policy can cover the line and then let go of the pen, at no cost
            # to the reward and every cost to the metric. A done the POLICY declares before the
            # trajectory is finished pays it too, so quitting early is never a refund.
            r -= w["w_time"] * max(0, self.T_end - self.t) * self.dt / max(float(self.task.duration), 1e-6)
        if self.dropped and w.get("drop_charge"):
            r -= float(w["drop_charge"])                 # the metric fails a dropped episode; so does the reward
        if self._ink_new:
            ink = np.asarray(self._ink_new, float)[:, :2]
            dist, _ = self._goal_tree.query(ink)
            # precision is a FRACTION of the ink laid; pay the change in (1 - precision) so the episode
            # sums to exactly -w_precision x (1 - precision). A per-step charge instead grows with time:
            # a pen resting off the line for the whole episode would cost -900 against a -300 ceiling.
            self._goal_ink_on += int((dist <= tol).sum())
            self._goal_ink_n += len(dist)
            if w.get("prec_per_point"):
                # charge for HOW MUCH ink missed the target, not for the ratio. As a ratio, the
                # first exploratory touch that lands 2 mm off pays the whole -w_precision, so never
                # touching the sheet is cheaper than learning to write and the policy hovers.
                # Normalised by the ink a faithful drawing lays (the ink model emits ~20 points per
                # contact step, over the timeline's pen-down steps), a drawing that is off-target
                # from start to finish costs -w_precision, one stray dot a few hundredths of that,
                # and the episode's total is capped at 2 x w_precision so a pen parked on the sheet
                # cannot run it to -20 000 and make lifting off the only sane move.
                n_down = max(int((self._tl_down > 0.5).sum()), 1)
                # the same ramp as completion, from the other side: an ink point inside the scored
                # tolerance is free, one at `goal_soft` or beyond pays in full, and the band between
                # is charged in proportion. One function for both halves of the metric - the recall
                # side pays +credit, the precision side charges -(1 - credit).
                miss = (1.0 - np.where(dist <= tol, 1.0,
                                       gain * np.clip((soft - dist) / (soft - tol), 0.0, 1.0))
                        if soft > tol else (dist > tol).astype(float))
                charge = w["w_precision"] * float(miss.sum()) / (20.0 * n_down)
                room = 2.0 * w["w_precision"] - self._goal_prec_paid
                charge = min(charge, max(room, 0.0))
                self._goal_prec_paid += charge
                r -= charge
            else:
                off = 1.0 - self._goal_ink_on / self._goal_ink_n
                r -= w["w_precision"] * (off - self._goal_off_prev)
                self._goal_off_prev = off
            gained = 0.0
            for p in ink:
                j = self._goal_j
                if j >= n_line:
                    break
                seg = self._goal_line[j:j + self.GOAL_LOOKAHEAD]
                d = np.linalg.norm(seg - p, axis=1)
                if soft > tol:
                    # Continuous completion, without moving what the metric pays for. A target point
                    # is worth 1 inside the scored tolerance, and its worth ramps to 0 at `goal_soft`
                    # outside it, so ink at 1.2 mm earns something where the hard rule pays it the same
                    # as ink at 20 mm. Credit is a high-water mark per target point and the window only
                    # moves forward, so the reward still cannot be farmed and still refuses strokes
                    # drawn out of order. With `goal_soft` unset this is the 0/1 rule.
                    # `goal_soft_gain` scales the credit OUTSIDE the scored tolerance. At gain 1 the
                    # ramp is continuous but the boundary is diluted: ink at 1.5 mm is then worth 0.75
                    # of ink on target, so closing the last half millimetre is worth only 0.25 and the
                    # pull that the 0/1 rule has at 1 mm is three-quarters gone; tasks that are already
                    # precise lose precision. A gain of 0.25 keeps a 0.81 step at the boundary and
                    # still leaves a gradient out to `goal_soft`.
                    c = np.where(d <= tol, 1.0,
                                 gain * np.clip((soft - d) / (soft - tol), 0.0, 1.0))
                else:
                    c = (d <= tol).astype(float)
                cov = self._goal_cov[j:j + len(seg)]      # a view: the writes below land in _goal_cov
                up = c > cov
                if up.any():
                    gained += float((c[up] - cov[up]).sum())
                    cov[up] = c[up]
                near = np.nonzero(d <= tol)[0]
                if len(near) == 0:
                    continue
                i = j + int(near[0])                     # the earliest target point ahead that it reaches
                k = i
                while k < n_line and np.linalg.norm(self._goal_line[k] - p) <= tol:
                    k += 1
                self._goal_j = k
            r += w["w_completion"] * gained / n_line
            self._goal_comp_paid += w["w_completion"] * gained / n_line
        if self.dropped and w.get("drop_clawback"):
            r -= self._goal_comp_paid                    # a drop zeroes the completion
            self._goal_comp_paid = 0.0
        if w.get("w_slip"):
            # the pen's slip in the hand, in units of the drop thresholds; paid as a potential
            dp, dax, _ = self._slip()
            slip = dp / 0.02 + float(np.degrees(2.0 * np.arcsin(min(dax / 2.0, 1.0)))) / 30.0
            if self._goal_slip_prev is not None:
                r -= w["w_slip"] * (slip - self._goal_slip_prev)
            self._goal_slip_prev = slip
        if self._force > 0.0:
            tgt = self._target_force_at(tip)
            if tgt > 0.0:
                # force_mae is a mean over on-target contact steps: pay its change, the sum is -w_force x MAE
                err = abs(self._force - tgt)
                self._goal_ferr += err
                self._goal_fn += 1
                if w.get("force_per_step"):
                    # the difference form below pays -w_force x (err - mae) / n, so
                    # the SAME physical force error is worth 50x less at step 500 than at step 10 and
                    # the second half of a word carries almost no force signal at all. Charged per step
                    # the rate is constant. Force is not a side quantity here: the task map's darkness
                    # IS the press force it asks for, so this term is how dark the writing comes out.
                    #
                    # `force_ref` is what the difference form has for free: a BASELINE. Without it every
                    # contact step is a pure cost (err >= 0), so the term stops pricing the quality of
                    # contact and starts pricing its quantity - and for a policy whose ink misses the line
                    # 28 % of the time, a quarter of its contact steps earn no completion and are then
                    # strictly negative, so the policy learns to lift off and completion collapses.
                    # A step at `force_ref` costs nothing however long the pen stays down; only
                    # being worse than it costs. The running total is clamped at >= 0 so beating the
                    # reference refunds what was charged but never pays a bonus for lingering.
                    n_down = max(int((self._tl_down > 0.5).sum()), 1)
                    charge = w["w_force"] * (err - float(w.get("force_ref", 0.0))) / n_down
                    charge = max(charge, -self._goal_force_paid)
                    self._goal_force_paid += charge
                    r -= charge
                else:
                    mae = self._goal_ferr / self._goal_fn
                    r -= w["w_force"] * (mae - self._goal_mae_prev)
                    self._goal_mae_prev = mae
        return float(r)

    # ------------------------------------------------------------ rendering
    def render(self, camera="side", width=None, height=None):
        w, h = (width or self.image_size[0]), (height or self.image_size[1])
        if self._renderer is None or self._renderer.width != w or self._renderer.height != h:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
            self._vis = mujoco.MjvOption()
            self._vis.sitegroup[:] = 0
        # upload the ink texture
        tex = self._tex_merged()
        self.model.tex_data[self.tex_adr:self.tex_adr + tex.size] = tex.ravel()
        mujoco.mjr_uploadTexture(self.model, self._renderer._mjr_context, self.tex_id)
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._vis)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------ for metrics
    def episode_record(self):
        return dict(task=self.task, ink_points=np.array(self.ink_points).reshape(-1, 4),
                    tip=np.array(self.tip_log).reshape(-1, 3), arm=np.array(self.arm_log).reshape(-1, 3),
                    touch=np.array(self.touch_log, bool),
                    actions=np.array(self.action_log), dropped=self.dropped, done_time=self.done_time, tool=self.tool,
                    n_lifts=self._n_lifts, cov_base=self._cov_base or 0,
                    peak_force=float(self._peak_force),
                    finger_share=(self._share_fing / self._share_den if self._share_den > 0 else None),
                    wrist_share=(self._share_wrist / self._share_den if self._share_den > 0 else None),
                    arm_share=(self._share_arm / self._share_den if self._share_den > 0 else None),
                    duration=self.t * self.dt,
                    progress=(self.pick_phase / 3.0 if self.pickup else
                              self.phase / 3.0 if (self.chop and not self.chop_hold) else
                              self.coverage()),
                    pickup=self.pickup, pick_phase=self.pick_phase, pick_time=self.pick_time,
                    wp=self.wp, wp_down=self.wp_down, wp_width=self.wp_width, wp_force=self.wp_force,
                    timeline=(self._timeline.copy() if self._timeline is not None else None),
                    tl_down=(self._tl_down.copy() if self._timeline is not None else None),
                    force_mae=(self._force_err / self._force_n) if self._force_n else None,
                    idle_rms_deg=float(np.degrees(np.sqrt(self._idle_dev2 / max(self._idle_n, 1)))),
                    idle_vel_rps=float(np.sqrt(self._idle_vel2 / max(self._idle_n, 1))),
                    task_map=self._task_map, ink_map=self._inkmap.copy(), dt=self.dt,
                    map_win_c=self.win_c.copy(), map_win_h=self.win_h,
                    **(self._l5_record() if (self.chop and not self.chop_hold) else {}),
                    **(dict(brush_canvas=self.brush_render.alpha.copy(), brush_violation_steps=self._brush_viol_steps,
                            brush_steps=self.t, brush_ran_dry=self.brush.ran_dry, brush_ink_left=self.brush.m,
                            brush_cfg=self.brush_cfg.name, paper_cfg=self.paper_cfg.name)
                       if self.brush is not None else {}),
                    **(dict(paint=True, brush_rgb=(None if self.brush_render.rgb is None else self.brush_render.rgb.copy()),
                            palette=self.task.palette, dips=self._dips, brush_swaps=self._swaps)
                       if self.paint else {}))

    def _l5_record(self):
        obj, bowl, _ = self._l5_targets()
        bowl_c = self.data.mocap_pos[self.bowl_mid]
        return dict(phase=self.phase, placed=self.done_time is not None, grasp_time=self.grasp_time,
                    lift_time=self.lift_time, object_drops=self.obj_drops, object_lost=self.obj_lost,
                    object_to_bowl_mm=1000 * float(np.linalg.norm(obj[:2] - bowl_c[:2])), bowl_r_mm=1000 * S.BOWL_R)
