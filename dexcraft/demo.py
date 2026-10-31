"""Watch a policy write.

    mjpython -m dexcraft.demo --task L2_write_word --view            # interactive viewer (macOS: mjpython)
    python   -m dexcraft.demo --task L2_write_word --gif out.gif     # top | side GIF at 20 fps
    python   -m dexcraft.demo --task L1_trace_circle --policy path/to/policy.py --seed 3

In the viewer the ink is drawn live as line segments on the sheet and the
target strokes are shown as a faint grey guide (the policy never sees that
guide; it only gets the waypoints in its observation).
"""
import argparse
import time

import mujoco
import numpy as np

from . import scene as S
from .env import DexCraftEnv
from .policies.base import load_policy

INK_Z = S.PAPER_TOP + 0.0004
GUIDE_Z = S.PAPER_TOP + 0.0002


def _add_segment(scn, p0, p1, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def draw_overlay(viewer, env, guide=True):
    """Rebuild the viewer's user geoms: target guide + ink so far."""
    scn = viewer.user_scn
    scn.ngeom = 0
    if guide:
        for s in env.task.strokes:
            P = np.asarray(s) + S.PAPER_CENTER[:2]
            for a, b in zip(P[:-1], P[1:]):
                _add_segment(scn, [*a, GUIDE_Z], [*b, GUIDE_Z], 0.0006, (0.55, 0.55, 0.6, 0.9))
    rgb = np.array(S.TOOLS[env.tool]["ink_rgb"]) / 255.0
    pts = env.ink_points
    prev = None
    for x, y, w, t in pts:
        p = np.array([x, y, INK_Z])
        if prev is not None and t - prev[3] < 0.02 and np.linalg.norm(p[:2] - prev[:2]) < 0.005:
            _add_segment(scn, prev[:3], p, 0.0005 * w, (*rgb, 1.0))
        else:
            _add_segment(scn, p, p + [0, 0, 1e-5], 0.0005 * w, (*rgb, 1.0))
        prev = (x, y, INK_Z, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="L2_write_word")
    ap.add_argument("--policy", default="dexcraft.policies.scripted:make_policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-guide", action="store_true", help="hide the grey target guide in the viewer")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--realtime", type=float, default=0.5, help="playback speed factor in the viewer")
    args = ap.parse_args()
    policy = load_policy(args.policy)
    env = DexCraftEnv(args.task, control_mode=policy.control_mode, obs_mode=getattr(policy, "obs_mode", "state"),
                      control_hz=getattr(policy, "control_hz", 100))
    obs, info = env.reset(seed=args.seed)
    policy.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"],
                  "dt": env.dt, "control_hz": env.control_hz, "tool": env.tool})
    print("task:", obs["instruction"])
    frames, every = [], max(1, int(round(1.0 / (args.fps * env.dt))))
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [0.0, 0.2, 0.05]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.75, 160, -30
        viewer.opt.sitegroup[:] = 0
    t_wall = time.time()
    done = False
    while not done:
        chunk = np.asarray(policy.act(obs), dtype=np.float32)
        for a in (chunk if chunk.ndim == 2 else chunk[None]):
            obs, r, term, trunc, info = env.step(a)
            policy.observe(info)
            if args.gif and env.t % every == 0:
                top = env.render(camera="top", width=360, height=270)
                side = env.render(camera="side", width=360, height=270)
                frames.append(np.concatenate([top, side], axis=1))
            if viewer is not None:
                if env.t % 5 == 0:
                    draw_overlay(viewer, env, guide=not args.no_guide)
                viewer.sync()
                lag = env.t * env.dt / args.realtime - (time.time() - t_wall)
                if lag > 0:
                    time.sleep(lag)
                if not viewer.is_running():
                    done = True
                    break
            if term or trunc:
                print(f"done: progress {info['progress']:.2f}, dropped {info['dropped']}, "
                      f"completion time {info['done_time']}")
                if viewer is not None:
                    draw_overlay(viewer, env, guide=not args.no_guide)
                    viewer.sync()
                    time.sleep(3.0)
                done = True
                break
    if args.gif and frames:
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=int(1000 / args.fps), loop=0)
        print("wrote", args.gif, f"({len(ims)} frames)")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
