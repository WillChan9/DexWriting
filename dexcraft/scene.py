"""Scene assembly: dex_hand hand + arm, a free pen, a sheet of paper
with a paintable ink texture. Everything task-independent lives here."""
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_XML_6AXIS = os.path.join(os.path.dirname(HERE), "dex_hand", "dex_hand_ur5e.xml")   # the shipped UR5e
# The benchmark arm: the UR5e as shipped (all six joints). The pose over the sheet is set
# by where the base stands (ARM_BASE_POS), not by removing joints.
HAND_XML = HAND_XML_6AXIS
UR_WRISTS_REMOVED = ()                          # no arm joints are removed
HAND_JSON = os.path.join(os.path.dirname(HERE), "dex_hand", "dex_hand.json")
GRIP_JSON = os.path.join(HERE, "grip_pocket.json")

from . import gen_pen as P       # pen dimensions + visual meshes (assets/pen_*.stl)

PEN_RADIUS = P.GRIP_R            # where the fingers hold it (rubber grip)
PEN_LEN = P.PEN_LEN
PEN_HALF = P.PEN_HALF
GRIP_FROM_TIP = 0.028            # where the fingertips sit, measured from the pen tip
ORIGIN_ABOVE_GRIP = PEN_HALF - GRIP_FROM_TIP   # tool body origin, along +z from the grip point (both tools)
PEN_PITCH_DEG = 50.0             # writing posture: pen axis vs paper plane at reset
PEN_DENSITY = 1100.0             # ~8.5 g pen
# ink width [mm] as a function of the tool-paper normal force [N]
TOOLS = {
    # ONE writing tool for every ink task, so every ink task is scored at the same 1 mm
    # tolerance on the same line width. 1 mm at a light touch, 3 mm leaned on.
    "pen":   dict(ink_w0=1.00, ink_k=0.50, ink_max=3.00, ink_rgb=(20, 20, 90)),
    # alias of "pen", kept so records and tiers that name it still load
    "marker": dict(ink_w0=1.00, ink_k=0.50, ink_max=3.00, ink_rgb=(20, 20, 90)),
    # brush: width comes from the footprint model (brush.py); only the colour is used here
    "brush": dict(ink_w0=0.0, ink_k=0.0, ink_max=0.0, ink_rgb=(15, 15, 15)),
    "chopsticks": dict(ink_w0=0.0, ink_k=0.0, ink_max=0.0, ink_rgb=(0, 0, 0)),   # no ink: pick and place
}
# L5 chopsticks: the upper stick is the free "pen" body (same tripod pocket); the
# lower stick is fixed to the hand (wedged in the thumb web, resting on the ring
# finger) - it does not move relative to the palm in a proper grip. grip.py lays
# it out from the *settled* upper stick (grip["chop_lower"]): parallel in the
# horizontal direction at the reset pose so both tips are level, converging
# toward the tips (back ends CHOP_GAP_BACK apart, tips CHOP_GAP_TIP) so a pinched
# object is wedged toward the closed tips instead of squeezed out.
CHOP_GAP_DIR = np.array([0.0, -0.6, -1.0])      # fallback direction (hand frame) before grip.py has laid it out
CHOP_GAP_BACK, CHOP_GAP_TIP = 0.014, 0.020
OBJECT_SHAPES = ("sphere", "cube", "cylinder", "capsule")
OBJECT_HOME = np.array([0.15, 0.20, 0.03])      # parked in the keyframe; tasks place it
BOWL_R, BOWL_WALL_H = 0.035, 0.015              # inner radius / wall height [m]
BOWL_HOME = np.array([-0.12, 0.20, 0.0])


def ink_width_mm(tool, force):
    if tool == "brush":
        # the reduced-order tuft: width = 2 b(d) with d the depth that force implies
        from .brush import brush_config, width_mm
        return float(max(0.8, width_mm(brush_config(), force)))
    t = TOOLS[tool]
    return float(np.clip(t["ink_w0"] + t["ink_k"] * force, 0.8, t["ink_max"]))


def ink_speed_factor(speed):
    """How much ink a unit of paper gets at this tip speed, relative to INK_V_REF.

    The tool feeds ink at a rate set by the contact, not by the distance travelled, so
    the same stroke drawn twice as fast is laid over twice the paper: deposition per unit
    length goes as (v_ref / v) ** INK_V_P. Clipped so a stall does not blot and a sprint
    does not vanish."""
    v = max(float(speed), 1e-4)
    return float(np.clip((INK_V_REF / v) ** INK_V_P, *INK_V_CLIP))


def ink_deposit(tool, force, speed):
    """(width [mm], darkness [0..1]) of the mark this tool leaves.

    Both depend on how hard and how fast the tip is moving: force widens the footprint
    (ink_k) and darkens it on the task map's force scale, speed thins and lightens it through
    ink_speed_factor. Width is the weaker of the two speed effects (INK_V_W), which is
    what a ballpoint does - it goes pale long before it goes thin."""
    s = ink_speed_factor(speed)
    w = ink_width_mm(tool, force) * (1.0 - INK_V_W + INK_V_W * s)
    # Darkness is on the task map's own scale (255 * force / FORCE_MAX), so what the map ASKS
    # for and what the paper RECORDS are the same number: a policy can see that it laid 90
    # where 138 was wanted. The speed factor is the physical part - the same pressure drawn
    # faster leaves a paler line.
    dark = np.clip(force / FORCE_MAX, 0.02, 1.0) * s
    return float(w), float(np.clip(dark, 0.01, 1.0))
FORCE_MAX = 4.0                         # N; full-scale of the task map's darkness and of the force channel
                                        # (a Wacom Intuos pen's full scale; human handwriting ~1.6 N median, 0.9-2.5 N p10-p90)
INK_V_REF = 0.05                        # m/s; the tip speed the ink model is calibrated at (the
                                        # scripted teacher writes at 30-80 mm/s)
INK_V_P = 0.5                           # deposition per unit length ~ (INK_V_REF / v) ** INK_V_P
INK_V_CLIP = (0.45, 1.25)               # bounds on that factor: no blot when stalled, no invisible sprint
INK_V_W = 0.15                          # how much of the speed factor reaches the WIDTH (darkness takes all)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PAPER_FRICTION = [0.4, 0.005, 0.0001]   # pen-on-paper sliding
PAPER_CENTER = np.array([0.0, 0.20, 0.0])
PAPER_HALF = np.array([0.20, 0.15, 0.005])      # 40 x 30 cm sheet
PAPER_TOP = PAPER_CENTER[2] + PAPER_HALF[2]
# The sheet rests on a rectangular table, and the tool starts lying on that table
# beside it for the pick-up task. The table collides with the fingers as well as
# the tool (contype 4 / conaffinity 3) - you cannot reach through it for the pen.
TABLE_TOP = PAPER_CENTER[2] - PAPER_HALF[2]        # sheet bottom == table top
TABLE_HALF = np.array([0.42, 0.30, 0.18])
TABLE_CENTER = np.array([PAPER_CENTER[0], PAPER_CENTER[1], TABLE_TOP - TABLE_HALF[2]])
TOOL_REST_X = (0.26, 0.34)                          # beside the sheet, on the table
TOOL_REST_Y = (0.10, 0.30)
# L5 paint: six paint wells beside the sheet (+x side, fixed palette order) and a
# three-slot brush rack on the other side (fine / medium / broad, see brush.BRUSHES).
# Neither collides with anything: a dip or a brush change is a matter of where the
# nominal tip is, not of contact.
PAINT_PALETTE = [(0.85, 0.16, 0.16),   # 1 red
                 (0.95, 0.55, 0.10),   # 2 orange
                 (0.96, 0.84, 0.16),   # 3 yellow
                 (0.20, 0.62, 0.26),   # 4 green
                 (0.20, 0.42, 0.86),   # 5 blue
                 (0.45, 0.28, 0.12)]   # 6 brown
WELL_XY = [(0.285, 0.06 + 0.055 * k) for k in range(len(PAINT_PALETTE))]
WELL_R, WELL_H = 0.017, 0.012
WELL_TOP = TABLE_TOP + WELL_H                       # dish rim; the paint surface sits 4 mm below it
WELL_PAINT_Z = WELL_TOP - 0.004
RACK_XY = [(-0.285, 0.12 + 0.08 * k) for k in range(3)]
RACK_R, RACK_TOP = 0.022, TABLE_TOP + 0.010
RACK_STUB_R = (0.0025, 0.005, 0.008)                # the tuft radius each slot shows
INK_RES = 512                                   # paper texture (square, covers the sheet)

# friction = [slide, torsion, roll]. Torsion and roll are LENGTHS in MuJoCo: the
# torque a contact can carry is friction[i] * normal_force, so they encode the size
# of the contact patch the point contact stands for. Physically, torsional friction
# over a Hertzian patch of radius a is mu*(3*pi/16)*a, and the rolling-resistance arm
# is a small fraction of a. From the measured penetrations (pad 1.45 mm on a 2.63 mm
# sphere -> a = 2.8 mm; capsule 1.09 mm on 4.51 mm -> a = 3.1 mm) that gives ~0.002.
# Values an order of magnitude larger would hold the grip up partly by a torsional
# stiction that a real 3 mm contact patch cannot provide.
CONTACT = dict(condim=6, friction=[1.2, 0.0022, 0.00063], solref=[0.005, 1])
PAD_CONTACT = dict(condim=6, friction=[1.2, 0.0020, 0.00055], solref=[0.009, 1])
GRASP_FINGERS = ["thumb", "index", "middle"]

HAND_SPEC = json.load(open(HAND_JSON))
ACTUATOR_OF_FULL = {j["name"]: j["actuator"] for j in HAND_SPEC["joints"]}   # before any removal
HAND_SPEC["joints"] = [j for j in HAND_SPEC["joints"] if j["name"] not in UR_WRISTS_REMOVED]
HAND_JOINTS = [j["name"] for j in HAND_SPEC["joints"] if j["group"] == "hand"]
WRIST_JOINTS = [j["name"] for j in HAND_SPEC["joints"] if j["group"] == "wrist"]
ARM_JOINTS = [j["name"] for j in HAND_SPEC["joints"] if j["group"] == "arm"]
ALL_JOINTS = ARM_JOINTS + WRIST_JOINTS + HAND_JOINTS          # = actuator order
ACTUATOR_OF = {j["name"]: j["actuator"] for j in HAND_SPEC["joints"]}
JOINT_RANGE = {j["name"]: (np.deg2rad(j["range_deg"]) if "range_deg" in j else np.array(j["range"]))
               for j in HAND_SPEC["joints"]}
MAX_VEL = {j["name"]: j["max_vel"] for j in HAND_SPEC["joints"]}
ARM_HOME = np.array([j["home"] for j in HAND_SPEC["joints"] if j["group"] == "arm"])
# Where the UR5e is bolted down, overriding the XML's <body name="base" pos>. The XML
# mounts it at table height (shoulder 113 mm above the sheet), which forces every IK
# solution over the sheet onto the elbow-UP branch: the arm arches over the paper and
# the hand comes down backwards.
ARM_BASE_XML = np.array([0.0, -0.42, -0.05])   # where the XML mounts it - grip_pocket.json's settled
#                                                 qpos (incl. the pen's WORLD pose) was recorded here
# The base stays on the ground, set back from the XML spot so the arm reaches the sheet
# extended instead of arched. With the tripod hand (fingers forward), y = -0.82 is the
# flattest setback that still reaches the whole sheet on L1/L2 (upper arm +31 deg, elbow
# ~340 mm, wrist stack >= 42 mm from the table); -0.87 loses the far edge.
ARM_BASE_POS = ARM_BASE_XML + np.array([0.0, -0.40, 0.0])
# IK seed for placing the hand over the sheet. None = do not touch the arm before IK: the
# settled grip pose is the seed. With the base on the ground an elbow-down seed puts the
# elbow through the table, and overwriting the joints with ARM_HOME - nominally the same
# values - moves placements onto a different IK solution.
ARM_SEED_DEFAULT = None
# Roll of the rigid hand+pen about the pen's own axis at placement; the tripod grip needs none.
PEN_AXIS_ROLL_DEFAULT = 0.0
HAND_TARGET = None   # optional absolute hand orientation (see hand_target_rot); None = target the settled grip orientation
# Yaw (about world z) applied to the settled hand orientation at placement; callers may pass
# yaw_offset explicitly. With the tripod grip (pen 43 deg off the finger axis, thumb loaded -
# grip_pocket.json, solved from a human capture) yaw 0 puts the fingers straight at the far
# edge of the sheet (azimuth 0), palm down, pen tip forward-left at ~53 deg: the human
# writing hand.
YAW_OFFSET_DEFAULT = 0.0


def hand_target_rot(fingers_azim_deg=0.0, fingers_elev_deg=-15.0, twist_deg=-30.0):
    """World rotation of hand_base: +y (fingers) at (azimuth, elevation) - azimuth 0 = toward
    the far edge of the sheet (+y), positive = to the left; +z (back of hand) as far up as
    the twist about +y allows (twist 0 = straight up)."""
    a, e, t = np.radians(fingers_azim_deg), np.radians(fingers_elev_deg), np.radians(twist_deg)
    y = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    up = np.array([0.0, 0.0, 1.0]); z0 = up - (up @ y) * y; z0 /= np.linalg.norm(z0)
    x0 = np.cross(y, z0); z = np.cos(t) * z0 + np.sin(t) * x0; x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def pen_quat(shaft_dir):
    """Quaternion taking the pen's local z axis to shaft_dir (tip -> back)."""
    u = np.asarray(shaft_dir, float)
    u = u / np.linalg.norm(u)
    a = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, u)
    s = np.linalg.norm(v)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / max(s, 1e-9), np.arctan2(s, a @ u))
    return q


def build_model(grip=None, pen_density=PEN_DENSITY, pad_solref=None, tool="pen", chop_lower=True,
                chop_grip=None, paint=False):
    """Compile the scene. `grip` (dict from grip_pocket.json) places the pen in
    the hand at the solved pocket; without it the pen lies on the paper.
    tool = "pen" | "brush" | "chopsticks" (adds the fixed lower stick, the object
    and the bowl; chop_lower=False omits the lower stick - used by grip.py to lay it out).
    chop_grip: grip dict to read the lower-stick layout from, when grip is None.
    paint: add the L5 paint wells and brush rack (visual only)."""
    spec = mujoco.MjSpec.from_file(HAND_XML)
    spec.body("base").pos = np.asarray(ARM_BASE_POS, float).tolist()
    return _build_from_spec(spec, grip, pen_density, pad_solref, tool, chop_lower, chop_grip, paint)


def _add_paint_stations(spec):
    """L5: paint wells (a grey dish with a disc of paint on top) and the brush rack
    (a plate per slot with a stub showing that brush's tuft size). No collision."""
    spec.add_material(name="dish_mat", rgba=[0.82, 0.82, 0.84, 1], specular=0.4, shininess=0.5)
    spec.add_material(name="rack_mat", rgba=[0.35, 0.33, 0.30, 1], specular=0.2, shininess=0.2)
    spec.add_material(name="rack_hair", rgba=[0.25, 0.22, 0.20, 1], specular=0.1, shininess=0.1)
    vis = dict(contype=0, conaffinity=0, group=1)
    for k, (x, y) in enumerate(WELL_XY):
        spec.add_material(name=f"paint_{k}", rgba=[*PAINT_PALETTE[k], 1], specular=0.3, shininess=0.4)
        h = WELL_PAINT_Z - TABLE_TOP                    # the dish, filled to the paint surface
        spec.worldbody.add_geom(name=f"well_{k}_dish", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                pos=[x, y, TABLE_TOP + h / 2], size=[WELL_R, h / 2, 0], material="dish_mat", **vis)
        spec.worldbody.add_geom(name=f"well_{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                pos=[x, y, WELL_PAINT_Z + 0.0006], size=[WELL_R - 0.002, 0.0006, 0],
                                material=f"paint_{k}", **vis)
    for k, (x, y) in enumerate(RACK_XY):
        spec.worldbody.add_geom(name=f"rack_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[x, y, TABLE_TOP + 0.003], size=[0.026, 0.026, 0.003], material="rack_mat", **vis)
        spec.worldbody.add_geom(name=f"rack_{k}_stub", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                fromto=[x, y, TABLE_TOP + 0.006, x, y, TABLE_TOP + 0.006 + 0.03],
                                size=[RACK_STUB_R[k], 0, 0], material="rack_hair", **vis)


def _build_from_spec(spec, grip=None, pen_density=PEN_DENSITY, pad_solref=None, tool="pen",
                     chop_lower=True, chop_grip=None, paint=False):
    """The scene assembly behind build_model, on an already-loaded arm+hand spec."""
    spec.option.noslip_iterations = 5        # kills static-friction creep of the held pen
    # elliptic cone is the true Coulomb cone; the pyramidal default over-estimates
    # friction in the corners. impratio is left at its default: raising it to 5 leaves
    # completion unchanged (0.744 vs 0.734) but triples the force error on
    # L2_write_word (2.46 N vs 0.77 N), and it is solver tuning, not physics.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # soft fingertip pads on the grasp-finger capsules, regular contact elsewhere
    for g in spec.geoms:
        if (g.name.endswith("_geom") or "_pad" in g.name) and any(g.name.startswith(f) for f in GRASP_FINGERS):
            g.condim = PAD_CONTACT["condim"]
            g.friction[:] = PAD_CONTACT["friction"]
            g.solref[:2] = pad_solref or PAD_CONTACT["solref"]
        elif g.name.endswith("_geom"):
            g.condim = CONTACT["condim"]
            g.friction[:] = CONTACT["friction"]
            g.solref[:2] = CONTACT["solref"]
    # paper: a white box with a paintable texture on top, pen-only collision
    tex = spec.add_texture(name="ink_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=INK_RES, height=INK_RES, nchannel=3)
    tex.data = np.full(INK_RES * INK_RES * 3, 255, dtype=np.uint8).tobytes()
    mat = spec.add_material(name="paper_mat", specular=0.0, shininess=0.0, reflectance=0.0)
    spec.add_material(name="pen_blue", rgba=[0.10, 0.14, 0.45, 1], specular=0.6, shininess=0.7)
    spec.add_material(name="pen_rubber", rgba=[0.08, 0.08, 0.08, 1], specular=0.1, shininess=0.1)
    spec.add_material(name="pen_metal", rgba=[0.80, 0.80, 0.82, 1], specular=0.9, shininess=0.9)
    spec.add_material(name="bamboo", rgba=[0.72, 0.55, 0.32, 1], specular=0.3, shininess=0.3)
    spec.add_material(name="hair", rgba=[0.08, 0.07, 0.06, 1], specular=0.1, shininess=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ink_tex"
    paper = spec.worldbody.add_geom(
        name="paper", type=mujoco.mjtGeom.mjGEOM_BOX, pos=PAPER_CENTER.tolist(),
        size=PAPER_HALF.tolist(), material="paper_mat", contype=4, conaffinity=2,
        priority=1, condim=3, friction=PAPER_FRICTION, solref=[0.005, 1])   # priority: paper sets mu
    spec.add_material(name="table_mat", rgba=[0.55, 0.47, 0.40, 1], specular=0.15, shininess=0.2)
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=TABLE_CENTER.tolist(),
        size=TABLE_HALF.tolist(), material="table_mat", contype=4, conaffinity=3,
        condim=3, friction=[0.6, 0.005, 0.0001], solref=[0.005, 1])
    spec.worldbody.add_geom(
        name="paper_frame", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(PAPER_CENTER - [0, 0, 0.0015]).tolist(), size=(PAPER_HALF + [0.01, 0.01, -0.001]).tolist(),
        rgba=[0.55, 0.55, 0.6, 1], contype=0, conaffinity=0)
    if paint:
        _add_paint_stations(spec)
    # pen: free capsule, collides with fingers (1) and paper (2)
    if grip is not None:
        u = np.asarray(grip["pen_shaft_dir"])
        center = np.asarray(grip["pen_grip_point"]) + (PEN_HALF - GRIP_FROM_TIP) * u
        quat = pen_quat(u)
    else:                                   # no grip: the tool lies on the table
        center = np.array([np.mean(TOOL_REST_X), np.mean(TOOL_REST_Y), TABLE_TOP + PEN_RADIUS])
        quat = pen_quat([0, 1, 0])
    pen = spec.worldbody.add_body(name="pen", pos=center.tolist(), quat=quat.tolist())
    pen.add_freejoint()
    if tool in ("pen", "marker"):                # marker: alias of the pen
        _add_pen(spec, pen, pen_density)
    elif tool == "brush":
        _add_brush(spec, pen)
    else:
        # `chop_grip` supplies the lower-stick layout when the caller passes grip=None
        # (i.e. "do not pre-place the tool"). Without it the layout falls back to the
        # grip file on disk.
        _add_chopsticks(spec, pen, chop_grip or grip or load_grip(), chop_lower)
    # cameras for the vision track (fixed rig cameras + a wrist camera on the forearm cuff)
    spec.worldbody.add_camera(name="top", pos=[0.0, 0.20, 0.75], xyaxes=[1, 0, 0, 0, 1, 0], fovy=45)
    spec.worldbody.add_camera(name="side", pos=[0.55, 0.20, 0.30], xyaxes=[0, 1, 0, -0.45, 0, 0.9], fovy=45)
    spec.worldbody.add_camera(name="front", pos=[0.0, 0.75, 0.35], xyaxes=[-1, 0, 0, 0, -0.5, 0.85], fovy=45)
    # wrist camera: on the back of the hand (hand_base frame: +y distal, +z dorsal),
    # offset to the thumb side so the pen tip is not hidden by the hand; aimed at the tip region
    cam_pos = np.array([-0.075, -0.03, 0.075])
    look = np.array([-0.01, 0.17, -0.07]) - cam_pos
    z_cam = -look / np.linalg.norm(look)
    x_cam = np.cross(np.array([0, 0, 1.0]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    spec.body("hand_base").add_camera(name="wrist", pos=cam_pos.tolist(),
                                      xyaxes=np.concatenate([x_cam, y_cam]).tolist(), fovy=75)
    # same-finger mcp-dip never truly collide
    for f in GRASP_FINGERS:
        spec.add_exclude(bodyname1=f"{f}_mcp", bodyname2=f"{f}_dip")
    # extend the arm's home keyframe with the pen's free joint (pos, quat)
    extra = np.concatenate([OBJECT_HOME, [1, 0, 0, 0]]) if tool == "chopsticks" else np.zeros(0)
    for k in spec.keys:
        k.qpos = np.concatenate([np.asarray(k.qpos), center, quat, extra]).tolist()
    return spec.compile()


def _add_pen(spec, pen, pen_density):
    # collision: body + rubber grip capsules (fingers hold these), cone, ball tip
    pen.add_geom(name="pen_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, P.GRIP_Z, 0, 0, P.CAP_Z], size=[P.BODY_R, 0, 0],
                 density=pen_density, contype=2, conaffinity=3,
                 rgba=[0.12, 0.12, 0.55, 0.3], group=3, condim=6, friction=[1.5, 0.05, 0.002],
                 solref=[0.005, 1])
    pen.add_geom(name="pen_grip_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, P.CONE_Z + 0.003, 0, 0, P.GRIP_Z - 0.003], size=[P.GRIP_R, 0, 0],
                 density=pen_density, contype=2, conaffinity=3,
                 rgba=[0.1, 0.1, 0.1, 0.3], group=3, condim=6, friction=[1.8, 0.08, 0.003],
                 solref=[0.005, 1])
    pen.add_geom(name="pen_cone_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, P.TIP_Z + 0.005, 0, 0, P.CONE_Z - 0.002], size=[0.0022, 0, 0],
                 density=pen_density, contype=2, conaffinity=2,
                 rgba=[0.7, 0.7, 0.7, 0.3], group=3, condim=3, friction=[0.4, 0.005, 0.0001],
                 solref=[0.005, 1])
    pen.add_geom(name="pen_nib", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, P.TIP_Z + P.TIP_R, 0, 0, P.TIP_Z + 0.004], size=[P.TIP_R, 0, 0],
                 density=pen_density, contype=2, conaffinity=2,
                 rgba=[0.05, 0.05, 0.05, 1], group=3, condim=3, friction=[0.4, 0.005, 0.0001],
                 solref=[0.005, 1])
    pen.add_site(name="pen_tip", pos=[0, 0, P.TIP_Z], size=[0.0015, 0, 0])   # ball tip
    # visual: revolved meshes + clip
    for name, mat in (("pen_body", "pen_blue"), ("pen_grip", "pen_rubber"), ("pen_cone", "pen_metal")):
        spec.add_mesh(name=name, file=os.path.join(P.ASSETS, name + ".stl"))
        pen.add_geom(name=name + "_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                     material=mat, contype=0, conaffinity=0, density=0, group=1)
    pen.add_geom(name="pen_clip_vis", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[P.BODY_R + 0.0012, 0, P.CAP_Z - 0.022], size=[0.0008, 0.0025, 0.022],
                 material="pen_metal", contype=0, conaffinity=0, density=0, group=1)


def _add_brush(spec, pen):
    """Calligraphy brush, two layers (see brush.py): the bamboo handle and metal
    ferrule are rigid bodies the fingers hold; the tuft is NOT a MuJoCo body. It is
    drawn (visual meshes only, no collision) and its mechanics - footprint, normal
    force, friction, ink - come from the reduced-order model in brush.py, which the
    env applies to this body every substep. Two sites: `tip_root` where the tuft
    leaves the ferrule and `pen_tip` at the tuft's natural length along the axis.
    The ferrule keeps its paper collision: ferrule-on-paper is a scored violation."""
    hard = dict(contype=2, conaffinity=3, group=3, condim=6, friction=[1.8, 0.08, 0.003], solref=[0.005, 1])
    # ~18 g in all, centre of mass toward the head: handle 700, ferrule 2000 kg/m^3
    pen.add_geom(name="pen_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, -0.012, 0, 0, P.BRUSH_TOP_Z - 0.005], size=[P.BRUSH_HANDLE_R, 0, 0],
                 density=700.0, rgba=[0.6, 0.45, 0.25, 0.3], **hard)
    pen.add_geom(name="pen_grip_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, P.BRUSH_FERRULE_Z + 0.004, 0, 0, -0.012], size=[P.BRUSH_GRIP_R, 0, 0],
                 density=700.0, rgba=[0.6, 0.45, 0.25, 0.3], **hard)
    pen.add_geom(name="pen_ferrule_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, 0, P.BRUSH_TUFT_Z + 0.004, 0, 0, P.BRUSH_FERRULE_Z - 0.002], size=[0.0044, 0, 0],
                 density=7500.0, rgba=[0.7, 0.7, 0.7, 0.3], contype=2, conaffinity=2, group=3,
                 condim=3, friction=[0.4, 0.005, 0.0001], solref=[0.005, 1])
    for name, mat in (("brush_handle", "bamboo"), ("brush_ferrule", "pen_metal"),
                      ("brush_tuft1", "hair"), ("brush_tuft2", "hair")):
        spec.add_mesh(name=name, file=os.path.join(P.ASSETS, name + ".stl"))
        pos = [0, 0, P.BRUSH_TUFT_Z] if name == "brush_tuft1" else \
              [0, 0, P.BRUSH_TUFT_Z - P.BRUSH_TUFT1_LEN] if name == "brush_tuft2" else [0, 0, 0]
        pen.add_geom(name=name + "_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=pos,
                     material=mat, contype=0, conaffinity=0, density=0, group=1)
    pen.add_site(name="tip_root", pos=[0, 0, P.BRUSH_TUFT_Z], size=[0.001, 0, 0])
    pen.add_site(name="pen_tip", pos=[0, 0, P.BRUSH_TIP_Z], size=[0.0015, 0, 0])


def chopstick_lower_segment(grip):
    """Back end and tip of the fixed lower stick in the hand_base frame."""
    if "chop_lower" in grip:
        return np.asarray(grip["chop_lower"]["back"]), np.asarray(grip["chop_lower"]["tip"])
    u = np.asarray(grip["pen_shaft_dir"]); u = u / np.linalg.norm(u)          # tip -> back
    G = np.asarray(grip["pen_grip_point"])
    e = CHOP_GAP_DIR - (CHOP_GAP_DIR @ u) * u
    e /= np.linalg.norm(e)
    back_u = G + (P.CHOP_TOP_Z + ORIGIN_ABOVE_GRIP) * u
    tip_u = G + (P.CHOP_TIP_Z + ORIGIN_ABOVE_GRIP) * u
    return back_u + CHOP_GAP_BACK * e, tip_u + CHOP_GAP_TIP * e


def _add_chopsticks(spec, pen, grip, chop_lower=True):
    """Upper stick on the free body (held in the pen pocket), lower stick welded
    to hand_base, a free object on the sheet and a mocap bowl."""
    hard = dict(contype=2, conaffinity=3, group=3, condim=6, friction=[1.8, 0.08, 0.003], solref=[0.005, 1])
    spec.add_material(name="chop_wood", rgba=[0.78, 0.62, 0.40, 1], specular=0.3, shininess=0.3)
    spec.add_material(name="bowl_mat", rgba=[0.92, 0.93, 0.95, 1], specular=0.4, shininess=0.5)
    spec.add_material(name="object_mat", rgba=[0.85, 0.25, 0.20, 1], specular=0.4, shininess=0.4)
    spec.add_mesh(name="chopstick", file=os.path.join(P.ASSETS, "chopstick.stl"))
    # collision pieces along the stick (top -> tip), radii follow the taper
    pieces = [("pen_geom", -0.022, P.CHOP_TOP_Z - 0.004, P.BODY_R, 700.0),     # same section as the pen in the pocket
              ("pen_grip_geom", -0.062, -0.022, P.GRIP_R, 700.0),
              ("pen_mid_geom", -0.132, -0.062, 0.0030, 700.0),
              ("pen_low_geom", P.CHOP_TIP_Z + 0.014, -0.132, 0.0019, 700.0),
              ("pen_nib", P.CHOP_TIP_Z + P.CHOP_TIP_R, P.CHOP_TIP_Z + 0.014, P.CHOP_TIP_R, 700.0)]
    for name, z0, z1, r, rho in pieces:
        kw = dict(hard)
        if name in ("pen_mid_geom", "pen_low_geom", "pen_nib"):
            kw.update(conaffinity=2 | 8, friction=[1.0, 0.02, 0.001])     # tip region: object + table, not fingers
        pen.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, z0, 0, 0, z1],
                     size=[r, 0, 0], density=rho, rgba=[0.6, 0.45, 0.25, 0.3], **kw)
    pen.add_geom(name="chop_upper_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="chopstick",
                 material="chop_wood", contype=0, conaffinity=0, density=0, group=1)
    pen.add_site(name="pen_tip", pos=[0, 0, P.CHOP_TIP_Z], size=[0.0015, 0, 0])
    # lower stick: fixed to the hand, collides with the object, the table and the upper stick only
    back, tip = chopstick_lower_segment(grip)
    if not chop_lower:
        tip, back = tip + [0, 0, 1.0], back + [0, 0, 1.0]              # parked far away
    axis = (back - tip) / np.linalg.norm(back - tip)               # tip -> back = local +z
    origin = tip - P.CHOP_TIP_Z * axis                              # body-frame origin along the stick
    quat = pen_quat(axis)
    hb = spec.body("hand_base")
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    low = dict(contype=8 | 2, conaffinity=8 | 4 | 2, group=3, condim=6, friction=[1.0, 0.02, 0.001],
               solref=[0.005, 1], rgba=[0.6, 0.45, 0.25, 0.3])
    for name, z0, z1, r, _ in pieces:
        a, b = origin + R @ [0, 0, z0], origin + R @ [0, 0, z1]
        hb.add_geom(name="lower_" + name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=np.concatenate([a, b]).tolist(), size=[r, 0, 0], density=0, **low)
    hb.add_geom(name="chop_lower_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="chopstick", pos=origin.tolist(),
                quat=quat.tolist(), material="chop_wood", contype=0, conaffinity=0, density=0, group=1)
    hb.add_site(name="lower_tip", pos=tip.tolist(), size=[0.0015, 0, 0])
    # the object: one geom per shape, the task enables one of them at reset
    obj = spec.worldbody.add_body(name="object", pos=OBJECT_HOME.tolist())
    obj.add_freejoint()
    okw = dict(contype=8, conaffinity=15, condim=6, friction=[1.0, 0.02, 0.001], solref=[0.005, 1],
               material="object_mat", density=600.0, group=1)
    obj.add_geom(name="object_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.010, 0, 0], **okw)
    obj.add_geom(name="object_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.008, 0.008], **okw)
    obj.add_geom(name="object_cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.008, 0.008, 0], **okw)
    obj.add_geom(name="object_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0.008, 0], **okw)
    obj.add_site(name="object_site", pos=[0, 0, 0], size=[0.002, 0, 0])
    # the bowl: mocap body, flat floor + 10 wall segments; pen-style collision (object and sticks)
    bowl = spec.worldbody.add_body(name="bowl", pos=BOWL_HOME.tolist(), mocap=True)
    bkw = dict(contype=4, conaffinity=2 | 8, condim=3, friction=[0.6, 0.005, 0.0001], solref=[0.005, 1],
               material="bowl_mat", group=1)
    bowl.add_geom(name="bowl_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, PAPER_TOP + 0.0015],
                  size=[BOWL_R + 0.006, 0.0015, 0], **bkw)
    n_wall = 12
    for i in range(n_wall):
        th = 2 * np.pi * i / n_wall
        seg = np.pi * (BOWL_R + 0.003) / n_wall
        bowl.add_geom(name=f"bowl_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[(BOWL_R + 0.003) * np.cos(th), (BOWL_R + 0.003) * np.sin(th), PAPER_TOP + 0.003 + BOWL_WALL_H / 2],
                      quat=[np.cos(th / 2), 0, 0, np.sin(th / 2)], size=[0.003, seg, BOWL_WALL_H / 2], **bkw)
    bowl.add_site(name="bowl_site", pos=[0, 0, PAPER_TOP + 0.003], size=[0.002, 0, 0])


def contact_normal_force(model, data, geom_set, other_set=None, out=None):
    """Sum of contact normal forces on `geom_set` (optionally only against `other_set`).

    This is what a real fingertip / joint force sensor reports, and it is computed
    from the solver's contact forces rather than from a MuJoCo touch site so the
    grouping is exactly per finger."""
    f6 = out if out is not None else np.zeros(6)
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        a_in, b_in = c.geom1 in geom_set, c.geom2 in geom_set
        if not (a_in or b_in):
            continue
        if other_set is not None:
            other = c.geom2 if a_in else c.geom1
            if other not in other_set:
                continue
        mujoco.mj_contactForce(model, data, i, f6)
        total += abs(f6[0])
    return total


def names(model, objtype):
    n = {mujoco.mjtObj.mjOBJ_JOINT: model.njnt, mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
         mujoco.mjtObj.mjOBJ_GEOM: model.ngeom, mujoco.mjtObj.mjOBJ_BODY: model.nbody}[objtype]
    return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


def fingertip_geoms(model, finger):
    """Every collision geom on `finger`'s distal phalanx.

    With soft pads the tip is a thin core capsule plus a grid of pad spheres, and
    the pads are what actually touch the tool, so anything that counts fingertip
    contact has to look at the whole body, not just `<finger>_dip_geom`."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_dip")
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def fingertip_radius(model, finger):
    """Outer contact radius of the fingertip: how far the pad envelope reaches
    from the phalanx axis. Equals the bare capsule radius on a pad-less model."""
    return max(float(np.hypot(model.geom_pos[g][0], model.geom_pos[g][2]) + model.geom_size[g][0])
               for g in fingertip_geoms(model, finger))


def capsule_segment(model, data, gid):
    c = data.geom_xpos[gid]
    z = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    h = model.geom_size[gid][1]
    return c - h * z, c + h * z


def seg_seg_dist(p1, p2, q1, q2):
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c = u @ u, u @ v, v @ v
    d_, e_ = u @ w, v @ w
    D = a * c - b * b
    if D < 1e-12:
        s = 0.0
        t = np.clip(e_ / c if c > 1e-12 else 0.0, 0.0, 1.0)
    else:
        s = np.clip((b * e_ - c * d_) / D, 0.0, 1.0)
        t = np.clip((a * s + e_) / c if c > 1e-12 else 0.0, 0.0, 1.0)
        s = np.clip((b * t - d_) / a if a > 1e-12 else 0.0, 0.0, 1.0)
    return np.linalg.norm((p1 + s * u) - (q1 + t * v))


def load_grip():
    g = json.load(open(GRIP_JSON)) if os.path.exists(GRIP_JSON) else None
    if g is None:
        return None
    for key in [k for k in g if k.startswith("settled")]:
        st = g[key]
        if "joint_names" not in st:
            continue
        # the grip file was recorded on the 6-axis arm: keep entries by NAME, drop the removed
        # joints/actuators, keep the trailing free-joint entries (pen pose; reset re-places it)
        jn, an = st["joint_names"], st["actuator_names"]
        keep_j = [i for i, n in enumerate(jn) if n not in UR_WRISTS_REMOVED]
        n_hinge = len(jn)
        q = np.asarray(st["qpos"], float)
        st["qpos"] = np.concatenate([q[keep_j], q[n_hinge:]]).tolist()
        act_drop = {ACTUATOR_OF_FULL[j] for j in UR_WRISTS_REMOVED}     # empty when nothing is removed
        keep_a = [i for i, n in enumerate(an) if n not in act_drop]
        st["ctrl"] = np.asarray(st["ctrl"], float)[keep_a].tolist()
        st["joint_names"] = [jn[i] for i in keep_j]; st["actuator_names"] = [an[i] for i in keep_a]
    return g


_SETTLED_HAND = {}
def settled_hand(tool="pen"):
    """(R_hand_world, p_pen_in_hand, quat_pen_in_hand) of the SOLVED grip, from the 6-axis model
    the grip was settled on. Placement targets this hand orientation and carries the pen at
    this hand-frame pose, so it does not depend on which arm the hand is mounted on."""
    key = "settled" if tool in ("pen", "marker") else f"settled_{tool}"
    if key in _SETTLED_HAND:
        return _SETTLED_HAND[key]
    g = json.load(open(GRIP_JSON)); st = g[key]
    spec = mujoco.MjSpec.from_file(HAND_XML_6AXIS)
    spec.body("base").pos = np.asarray(ARM_BASE_XML, float).tolist()
    m6 = _build_from_spec(spec, tool=tool)
    d = mujoco.MjData(m6); d.qpos[:] = st["qpos"]; d.ctrl[:] = st["ctrl"]; mujoco.mj_forward(m6, d)
    hb, pen = m6.body("hand_base").id, m6.body("pen").id
    R = d.xmat[hb].reshape(3, 3).copy(); p_rel = R.T @ (d.xpos[pen] - d.xpos[hb])
    q_rel = np.zeros(4); mujoco.mju_mat2Quat(q_rel, np.ascontiguousarray(R.T @ d.xmat[pen].reshape(3, 3)).ravel())
    _SETTLED_HAND[key] = (R, p_rel, q_rel)
    return _SETTLED_HAND[key]


def arm_dofs(model):
    jn = names(model, mujoco.mjtObj.mjOBJ_JOINT)
    return np.array([model.jnt_dofadr[jn.index(j)] for j in ARM_JOINTS])


def compensate_arm_gravity(model, data, dofs):
    """Real arms run gravity compensation in their joint controllers; emulate
    it so the position servos see no gravity load (call before each mj_step)."""
    data.qfrc_applied[dofs] = data.qfrc_bias[dofs]


def reset_home(model, data):
    """Keyframe 'home': arm above the sheet, hand open; everything else zero."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def pen_joint(model):
    jid = int(np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0])
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def place_pen(model, data, grip, settle_iters=80):
    """Put the pen into the solved pocket relative to the CURRENT hand_base
    pose, then nudge it out of any residual interpenetration. Returns the
    pen centre (world)."""
    hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base")
    R = data.xmat[hb].reshape(3, 3)
    p = data.xpos[hb]
    u = R @ np.asarray(grip["pen_shaft_dir"])
    center = p + R @ np.asarray(grip["pen_grip_point"]) + ORIGIN_ABOVE_GRIP * u
    quat = pen_quat(u)
    pq, pd = pen_joint(model)
    pen_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pen_set = {g for g in range(model.ngeom) if model.geom_bodyid[g] == pen_bid}
    pos = center.copy()
    for _ in range(settle_iters):
        data.qpos[pq:pq + 3] = pos
        data.qpos[pq + 3:pq + 7] = quat
        data.qvel[pd:pd + 6] = 0.0
        mujoco.mj_forward(model, data)
        worst, move = 0.0, np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 not in pen_set and c.geom2 not in pen_set) or c.dist >= 0:
                continue
            n = c.frame[:3].copy()
            step = n * (-c.dist) if c.geom2 in pen_set else -n * (-c.dist)
            if -c.dist > worst:
                worst, move = -c.dist, step
        if worst < 0.0008:
            break
        pos = pos + 0.7 * move
    return pos.copy()
