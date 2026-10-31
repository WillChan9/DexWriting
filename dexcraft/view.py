"""Open the full DexCraft scene in the MuJoCo viewer: UR5e + hand + tool + sheet.

macOS needs `mjpython` (the viewer must own the main thread); Linux can use python.
The managed `-m mujoco.viewer` can crash on macOS, so this uses launch_passive.

    mjpython -m dexcraft.view                     # pen, seated in the solved grip
    mjpython -m dexcraft.view --tool brush        # calligraphy brush
    mjpython -m dexcraft.view --tool chopsticks   # L6: two sticks, object, bowl
    mjpython -m dexcraft.view --task L2_write_word --play   # run the scripted policy live

    # watch the brush tuft work the paper, contacts drawn, camera locked on the tip:
    mjpython -m dexcraft.view --task L3_brush_stroke --play --seed 2 --contacts --follow

Sliders in the right-hand "Control" panel drive one position servo per DOF
(`<joint>_act`). Press 3 to show the collision capsules (geom group 3), and
Tab / the left panel for the rest of the viewer options.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from . import scene as S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="pen", choices=list(S.TOOLS))
    ap.add_argument("--task", default=None, help="load this task's sheet and targets (implies --play)")
    ap.add_argument("--play", action="store_true", help="run a policy instead of just holding the grip")
    ap.add_argument("--policy", default="dexcraft.policies.scripted:make_policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", action="store_true", help="freeze physics and just show the pose")
    ap.add_argument("--contacts", action="store_true",
                    help="draw contact points and force arrows, and show the collision geoms")
    ap.add_argument("--follow", action="store_true", help="keep the camera locked on the tool tip")
    ap.add_argument("--zoom", type=float, default=None, help="camera distance [m] (default 0.75)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    args = ap.parse_args(argv)

    if args.task or args.play:
        from .env import DexCraftEnv
        from .policies.base import load_policy
        task = args.task or {"pen": "L2_write_word", "brush": "L3_brush_chinese",
                             "chopsticks": "L6_chop_cube"}[args.tool]
        env = DexCraftEnv(task, obs_mode="state", control_hz=25)
        pol = load_policy(args.policy)
        obs, info = env.reset(seed=args.seed)
        pol.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"],
                   "dt": env.dt, "control_hz": env.control_hz, "tool": env.tool})
        model, data = env.model, env.data
        print(f"task: {obs['instruction']}")
    else:
        env = pol = None
        grip = S.load_grip()
        model = S.build_model(grip=None, tool=args.tool)
        data = mujoco.MjData(model)
        key = "settled" if args.tool == "pen" else f"settled_{args.tool}"
        n = len(grip[key]["qpos"])
        data.qpos[:n] = grip[key]["qpos"]
        data.ctrl[:] = grip[key]["ctrl"]
        mujoco.mj_forward(model, data)
        print(f"tool: {args.tool} | drag the *_act sliders to move joints; press 3 for collision capsules")

    if args.contacts:
        # the defaults are sized for the whole arm: one contact disc then covers the
        # whole sheet. These are metres, so keep them at tool scale.
        model.vis.scale.contactwidth = 0.004
        model.vis.scale.contactheight = 0.0015
        model.vis.scale.forcewidth = 0.0015
        model.vis.map.force = 30.0

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = S.PAPER_CENTER + [0.0, -0.02, 0.06]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = args.zoom or 0.75, 130, -22
        if args.contacts:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            v.opt.geomgroup[3] = 1          # the collision capsules live in group 3
            print("contacts on: red discs are contact points, arrows are the normal force;"
                  " group 3 shows the collision geoms (press 3 to toggle)")
        tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")
        obs_ = obs if env is not None else None
        while v.is_running():
            t0 = time.time()
            if env is not None:
                a = pol.act(obs_)
                obs_, r, term, trunc, info = env.step(a)
                pol.observe(info)
                if term or trunc:
                    obs_, info = env.reset()
                    pol.reset({"task_id": info["task_id"], "level": info["level"],
                               "instruction": obs_["instruction"], "dt": env.dt,
                               "control_hz": env.control_hz, "tool": env.tool})
            elif not args.hold:
                S.compensate_arm_gravity(model, data, S.arm_dofs(model))
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            if env is not None:
                # env.render() uploads the ink to its offscreen context, which the
                # viewer does not share - push it into the viewer's context too,
                # otherwise the paper stays blank while the tool writes on it
                model.tex_data[env.tex_adr:env.tex_adr + env.ink.size] = env.ink.ravel()
                v.update_texture(env.tex_id)
            if args.follow and tip_sid >= 0:
                v.cam.lookat[:] = data.site_xpos[tip_sid]
            v.sync()
            dt = (env.dt if env is not None else model.opt.timestep) / max(args.speed, 1e-3) \
                - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
