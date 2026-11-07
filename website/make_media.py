"""Media for the project page, rendered from the shipped checkpoints.

    MUJOCO_GL=glfw python website/make_media.py [--only tasks,method,results] [--redo]

Writes website/site/media/:
    tasks/<task>.mp4|jpg    one episode per task family, oblique camera at natural speed. Paper tasks run the
                            released policies (hold tasks: the learned grasp; writing: the final RL policy);
                            the other families run the scripted controller.
    grasp/                  the grasp policy on the three hold tasks, with per-finger contact forces
    dart/clean              the IK teacher writing with the learned grasp
    progress/               one glyph written by the BC initialization and by RL checkpoints
    compare/                IK, BC and RL on the same episodes
    zeroshot/               held-out content written by the final policy
    data/*.json             per-step force logs and the results tables behind the page's interactive charts
The method figure is rendered from website/assets/method_overview.pdf; the paper is copied from website/assets/CoRL2026_PhysWM.pdf.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zlib

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from dexcraft import scene as S  # noqa: E402
from dexcraft import tasks as T  # noqa: E402
from dexcraft.env import DexCraftEnv  # noqa: E402
from dexcraft.metrics import episode_metrics  # noqa: E402

OUT = os.path.join(HERE, "site", "media")
FPS_STEP = 2                     # record every 2nd control step: 12.5 frames/s plays back in real time
FPS = 25 / FPS_STEP
EK_WRITE = dict(control_hz=25, control_mode="hybrid", obs_mode="state", grip_assist=0.0, start_height=(0.002, 0.006),
                start_wrist=True, constraint="idle_lock", reward={"__preset__": "v15_goal", "adv_skip": 0.25},
                end_mode="policy", cursor_obs=False, traj_obs=16, adv_mode="tip")
EK_GRASP = dict(control_hz=25, control_mode="hybrid", obs_mode="state", grip_assist=0.0, start_height=(-0.0025, 0.0015),
                start_wrist=True, constraint="mini_arm_lock", reward="scribble", traj_obs=16)
PAPER_TASKS = {"L0_hold_static", "L0_hold_moving", "L0_scribble", "L1_trace_line", "L1_trace_circle", "L1_trace_shape",
               "L2_write_digits", "L2_write_letters", "L2_write_word", "L2_write_chinese"}
SCRIPTED_SEED = {"L5_paint_scene": 3}


def eval_seed(task, k):
    return 1000 * k + zlib.crc32(task.encode()) % 1000


# ------------------------------------------------------------------ policies
def ppo(path):
    from dexcraft.policies.ppo import PPOPolicy
    return PPOPolicy(path)


def ik_teacher():
    os.environ["DEXCRAFT_BASE"] = "hybrid"
    os.environ["DEXCRAFT_GRIP_CKPT"] = "checkpoints/grasp_v10_format.pt"
    from dexcraft.policies.hybrid import make_policy
    return make_policy()


def scripted():
    from dexcraft.policies.scripted import make_policy
    return make_policy()


def setup(task):
    """(policy, env kwargs) the page shows for a task family."""
    if task.startswith("L0_") and task in PAPER_TASKS:
        return ppo("checkpoints/grasp.pt"), EK_GRASP
    if task in PAPER_TASKS or task.startswith("E2_"):
        return ppo("checkpoints/rl.pt"), EK_WRITE
    pol = scripted()
    return pol, dict(control_mode=pol.control_mode, obs_mode="state", control_hz=getattr(pol, "control_hz", 25))


# ------------------------------------------------------------------ cameras
def cam_oblique(env):
    """Close oblique view on the work: the glyph, the chopsticks' object and bowl, or the hand."""
    if env.chop and not env.chop_hold:
        look = 0.5 * (env.data.qpos[env.oq:env.oq + 3] + env.data.mocap_pos[env.bowl_mid]) + [0, 0, 0.02]
        dist = 0.50
    elif env.task.strokes:
        c = np.concatenate([np.asarray(s) for s in env.task.strokes]).mean(0)
        look, dist = np.array([c[0] + S.PAPER_CENTER[0], c[1] + S.PAPER_CENTER[1] - 0.03, S.PAPER_TOP + 0.03]), 0.48
    else:
        hb = env.data.xpos[env.hb]
        look, dist = np.array([hb[0], hb[1], 0.5 * (hb[2] + S.PAPER_TOP)]), 0.70
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, 155.0, -33.0
    return cam


def cam_follow(env):
    """Oblique view that follows the hand (the arm is driven in the hold tasks)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    hb = env.data.xpos[env.hb]
    cam.lookat[:] = [hb[0], hb[1] + 0.06, hb[2] - 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.42, 200.0, -18.0
    return cam


def cam_top(env, margin=(0.17, 0.19)):
    """Near top-down view framed on the glyph, text upright, the arm entering from below."""
    P = np.concatenate([np.asarray(s) for s in env.task.strokes]) + S.PAPER_CENTER[:2]
    c, ext = (P.max(0) + P.min(0)) / 2, P.max(0) - P.min(0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [c[0] + 0.015, c[1] - 0.045, S.PAPER_TOP]
    cam.distance = float(max((ext[0] + margin[0]) / 1.243, (ext[1] + margin[1]) / 0.828))
    cam.azimuth, cam.elevation = 90.0, -86.0
    return cam


# ------------------------------------------------------------------ rendering
def run(task, seed, policy, ek, cam_fn, size, cam_every_frame=False, max_steps=None):
    """One episode -> (frames, per-step log, metrics, instruction)."""
    env = DexCraftEnv(task, **ek)
    obs, info = env.reset(seed=seed)
    policy.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"], "dt": env.dt,
                  "control_hz": env.control_hz, "tool": env.tool, "act_dim": int(env.action_space.shape[0])})
    w, h = size
    env.render(camera="top", width=w, height=h)          # builds the renderer and uploads the ink texture
    cam = cam_fn(env)
    frames, log = [], []

    def snap():
        nonlocal cam
        env.render(camera="top", width=w, height=h)      # refreshes the ink texture
        if cam_every_frame:
            cam = cam_fn(env)
        env._renderer.update_scene(env.data, camera=cam, scene_option=env._vis)
        frames.append(env._renderer.render().copy())

    snap()
    while True:
        a = np.asarray(policy.act(obs), np.float32)
        a = a if a.ndim == 1 else a[0]
        obs, r, te, tr, info = env.step(np.clip(a, -1, 1))
        policy.observe(info)
        k = min(getattr(env, "_tl_k", 0), len(env._timeline) - 1) if getattr(env, "_timeline", None) is not None else 0
        down = env._timeline is not None and float(env._tl_h[k]) <= 1e-4
        log.append(dict(t=round(env.t * env.dt, 3), f_req=round(float(env._tl_f[k]) if down else 0.0, 3),
                        f_pen=round(float(info.get("tool_contact_force") or 0.0), 3),
                        fingers=[round(float(x), 2) for x in info.get("finger_force") or []],
                        progress=round(float(info.get("progress") or 0.0), 3)))
        if env.t % FPS_STEP == 0:
            snap()
        if te or tr or (max_steps and env.t >= max_steps):
            break
    snap()
    try:
        m = episode_metrics(env.episode_record())
        metrics = {k: (None if m.get(k) is None else round(float(m[k]), 4)) for k in
                   ("completion", "precision", "force_mae_N", "time_s")}
        metrics["dropped"] = bool(m.get("dropped"))
    except Exception:
        metrics = {}
    text = obs["instruction"]
    env.close()
    return frames, log, metrics, text


def save_clip(frames, path, hold_s=1.2, max_s=None):
    """Write an mp4 and a poster. `max_s` speeds a long episode up to that many seconds (frames subsampled)."""
    if max_s and len(frames) > max_s * FPS:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, int(max_s * FPS)).round().astype(int)]
    frames = frames + [frames[-1]] * int(hold_s * FPS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path + ".mp4", fps=FPS, codec="libx264", quality=6, pixelformat="yuv420p",
                            macro_block_size=None, ffmpeg_params=["-movflags", "+faststart"]) as wr:
        for f in frames:
            wr.append_data(f)
    imageio.imwrite(path + ".jpg", frames[-1], quality=85)


def dump(obj, name):
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", name + ".json"), "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
    # the pages read the data from one script, so they also work opened straight from disk (no fetch)
    data = {}
    for f in sorted(os.listdir(os.path.join(OUT, "data"))):
        if f.endswith(".json"):
            data[f[:-5]] = json.load(open(os.path.join(OUT, "data", f)))
    with open(os.path.join(OUT, "data", "data.js"), "w") as fh:
        fh.write("window.SITE_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")


# ------------------------------------------------------------------ brush couplet preview
# The brush-layout preview writes a two-column couplet instead of a random column: 好好学习 (left) and
# 天天向上 (right). 习 is not in the package's stroke data, so its Make Me a Hanzi medians are added here,
# at render time only (adding it to dexcraft/data would change the task pools drawn from that file).
COUPLET = ["好好学习", "天天向上"]
PREVIEW_MAX_S = 10.0             # coming-next previews longer than this are sped up to it


def couplet_layout(rng):
    from dexcraft import fonts as F
    F.HANZI.update(json.load(open(os.path.join(HERE, "hanzi_extra.json"))))
    strokes = []
    for col, text in enumerate(COUPLET):                                  # left column first
        for k, ch in enumerate(text):
            for st in F.chinese_strokes(ch):
                strokes.append(np.asarray(st) + [1.6 * col, -1.3 * k])
    st = T._place(strokes, rng, 0.25, rot=0.0)                          # as tall as the sheet allows, columns upright
    fo = T._brush_forces(st, rng, ["press_release"] * len(st), depth=T.BRUSH_DEPTH_LAYOUT)
    n, text = len(st), "，".join(COUPLET)
    return T.TaskSpec("L3_brush_layout", "L3",
                      f"Write the couplet '{text}' ({n} strokes) in two columns with one brushful of ink - no re-dipping.",
                      st, forces=fo, widths=T.widths_from_forces(fo, "brush"), duration=8.0 + 1.2 * n, tool="brush",
                      meta={"text": text, "n_strokes": n, "ink0": 3.0, "layout": "columns"})   # ink for 40 strokes


T.TASKS["L3_brush_layout"] = couplet_layout


def cam_upright(env):
    """Seen from the writer's side, so the columns read top to bottom."""
    P = np.concatenate([np.asarray(s) for s in env.task.strokes]) + S.PAPER_CENTER[:2]
    c = (P.max(0) + P.min(0)) / 2
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [c[0], c[1] - 0.02, S.PAPER_TOP]
    cam.distance, cam.azimuth, cam.elevation = 0.55, 90.0, -62.0
    return cam


# ------------------------------------------------------------------ sections
TILE = (480, 360)
TOP = (640, 432)


def make_tasks(only=None):
    path = os.path.join(OUT, "data", "tasks.json")
    old = {r["id"]: r for r in json.load(open(path))} if only and os.path.exists(path) else {}
    rows = []
    for task in list(T.TASKS) + ["E2_write_word_unseen", "E2_write_idiom", "E2_write_chinese_unseen"]:
        if only and task not in only:
            if task in old:
                rows.append(old[task])
            continue
        pol, ek = setup(task)
        seed = SCRIPTED_SEED.get(task, 0) if task not in PAPER_TASKS and not task.startswith("E2_") else eval_seed(task, 0)
        frames, log, m, text = run(task, seed, pol, ek, cam_upright if task == "L3_brush_layout" else cam_oblique, TILE)
        preview = task not in PAPER_TASKS and not task.startswith("E2_")
        save_clip(frames, os.path.join(OUT, "tasks", task), max_s=PREVIEW_MAX_S if preview else None)
        sp = T.sample(task, seed)
        rows.append(dict(id=task, level=task.split("_")[0], instruction=text, tool=sp.tool, duration=sp.duration,
                         strokes=len(sp.strokes), paper=task in PAPER_TASKS or task.startswith("E2_"),
                         zero_shot=task.startswith("E2_"),
                         policy="grasp" if ek is EK_GRASP else ("rl" if ek is EK_WRITE else "scripted")))
        print(f"tasks/{task}: {len(frames)} frames, {text!r}", flush=True)
    dump(rows, "tasks")


def make_method():
    # stage 1: the grasp on the three hold tasks, with finger contact forces
    logs = {}
    for task in ("L0_hold_static", "L0_hold_moving", "L0_scribble"):
        frames, log, m, text = run(task, eval_seed(task, 0), ppo("checkpoints/grasp.pt"), EK_GRASP, cam_follow, TILE,
                                   cam_every_frame=True)
        save_clip(frames, os.path.join(OUT, "grasp", task))
        logs[task] = [dict(t=r["t"], fingers=r["fingers"]) for r in log]
        print(f"grasp/{task}: {len(frames)} frames", flush=True)
    dump(logs, "grasp_forces")
    # stage 2: a demonstration from the IK teacher
    task, seed = "L2_write_letters", eval_seed("L2_write_letters", 6)
    frames, log, m, text = run(task, seed, ik_teacher(), EK_WRITE, cam_top, TOP)
    save_clip(frames, os.path.join(OUT, "dart", "clean"))
    print(f"dart/clean: {len(frames)} frames, {text!r}", flush=True)
    # stage 3: one glyph through RL training
    steps = 512 * 1024 / 1e6
    out = []
    for name, ck, it in (("bc", "checkpoints/bc.pt", 0), ("it24", "checkpoints/rl_snapshots/rl_it24.pt", 24),
                         ("it36", "checkpoints/rl_snapshots/rl_it36.pt", 36), ("it48", "checkpoints/rl_snapshots/rl_it48.pt", 48),
                         ("it63", "checkpoints/rl_snapshots/rl_it63.pt", 63), ("final", "checkpoints/rl.pt", 76)):
        task, seed = "L2_write_digits", eval_seed("L2_write_digits", 0)
        frames, log, m, text = run(task, seed, ppo(ck), EK_WRITE, cam_top, TOP)
        save_clip(frames, os.path.join(OUT, "progress", name))
        out.append(dict(name=name, msteps=round(it * steps, 1), **m))
        print(f"progress/{name}: {m}", flush=True)
    dump(out, "progress")


def make_results():
    # IK / BC / RL on the same episodes, with force logs for the synced plot
    logs = {}
    for ep, (task, k) in {"word": ("L2_write_word", 8), "chinese": ("L2_write_chinese", 6)}.items():
        for name, mk in (("ik", ik_teacher), ("bc", lambda: ppo("checkpoints/bc.pt")), ("rl", lambda: ppo("checkpoints/rl.pt"))):
            frames, log, m, text = run(task, eval_seed(task, k), mk(), EK_WRITE, cam_top, TOP)
            save_clip(frames, os.path.join(OUT, "compare", f"{ep}_{name}"))
            logs[f"{ep}_{name}"] = dict(metrics=m, text=text, t=[r["t"] for r in log], f_req=[r["f_req"] for r in log],
                                        f_pen=[r["f_pen"] for r in log])
            print(f"compare/{ep}_{name}: {m}", flush=True)
    dump(logs, "compare")
    # zero-shot content
    for name, task, k in (("word", "E2_write_word_unseen", 4), ("idiom", "E2_write_idiom", 5),
                          ("chinese", "E2_write_chinese_unseen", 1)):
        frames, log, m, text = run(task, eval_seed(task, k), ppo("checkpoints/rl.pt"), EK_WRITE, cam_top, TOP)
        save_clip(frames, os.path.join(OUT, "zeroshot", name))
        print(f"zeroshot/{name}: {text!r} {m}", flush=True)
    # result tables: the paper's 50-seed evaluations
    runs = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "results", "r1_accept50.csv"))):
        runs.setdefault(r["run"], {})[r["task"]] = {k: float(r[k]) for k in
                                                    ("completion", "precision", "force_mae_N", "time_s")}
    names = {"V17b_teacher50": "ik", "f8_self50": "bc", "AX_r1_final": "rl"}
    curve = [("f8_self50", 0), ("AX_r1_it15", 15), ("AX_r1_it24", 24), ("AX_r1_it36", 36), ("AX_r1_it48", 48),
             ("AX_r1_it63", 63), ("AX_r1_it72", 72), ("AX_r1_final", 76)]
    mean = lambda run, key: float(np.mean([v[key] for v in runs[run].values()]))
    dump(dict(per_task={names[k]: runs[k] for k in names},
              curve=[dict(msteps=round(it * 512 * 1024 / 1e6, 1), **{key: round(mean(r, key), 4) for key in
                                                                      ("completion", "precision", "force_mae_N", "time_s")})
                     for r, it in curve],
              ik_mean={key: round(mean("V17b_teacher50", key), 4) for key in ("completion", "precision", "force_mae_N", "time_s")}),
         "results")


def make_figures():
    os.makedirs(OUT, exist_ok=True)
    # pdftocairo cannot crop SVG output, so trim the blank page area below the figure via the viewBox
    svg = os.path.join(OUT, "pipeline.svg")
    subprocess.run(["pdftocairo", "-svg", os.path.join(HERE, "assets", "method_overview.pdf"), svg], check=True)
    with open(svg) as f:
        text = f.read()
    text = re.sub(r'width="[^"]*" height="[^"]*" viewBox="[^"]*"', 'width="2110" height="712" viewBox="14 4 2110 712"', text, count=1)
    with open(svg, "w") as f:
        f.write(text)
    shutil.copy(os.path.join(HERE, "assets", "CoRL2026_PhysWM.pdf"), os.path.join(OUT, "paper.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="figures,tasks,method,results")
    ap.add_argument("--tasks", default="", help="with --only tasks: re-render just these task families")
    args = ap.parse_args()
    for part in args.only.split(","):
        if part == "tasks":
            make_tasks([t for t in args.tasks.split(",") if t] or None)
        else:
            {"figures": make_figures, "method": make_method, "results": make_results}[part]()
