"""Collect (obs, action) pairs from a scripted teacher (scripts/collect_demos.sh has the paper's settings).

    python scripts/bc_collect.py out.npz --episodes 120 --workers 6
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS = ["L0_hold_static", "L0_hold_moving", "L0_scribble",
         "L1_trace_line", "L1_trace_circle", "L1_trace_shape",
         "L2_write_letters", "L2_write_word", "L2_write_digits", "L2_write_chinese",
         "L2_write_cursive"]


def run_episode(job):
    task, seed, noise, actor_ck, yaw, polkw, envkw, teacher, label_cut = job
    from dexcraft.env import DexCraftEnv
    from dexcraft.policies.scripted import make_policy
    env = DexCraftEnv(task, obs_mode="state", control_hz=25,
                      start_height=tuple(envkw.pop("start_height", (-0.0025, 0.0015))), start_wrist=True,
                      yaw_offset=yaw, **envkw)
    if teacher.startswith("hybrid:"):
        # The demonstrator is the hybrid base (IK arm + the learned grip), so the clone's fingers
        # are its own init's and only arm/wrist are new
        from dexcraft.policies.hybrid import HybridBase
        pol = HybridBase(teacher.split(":", 1)[1])
        if "max_step" in polkw:                  # DAgger: a gentler teacher step (m per tick) so labels off the track do not saturate
            pol.tracker.max_step = polkw["max_step"]
    else:
        pol = make_policy(**polkw)
    actor = None
    if actor_ck:
        from dexcraft.policies.ppo import PPOPolicy
        actor = PPOPolicy(actor_ck)
    obs, info = env.reset(seed=seed)
    pol.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"],
               "dt": env.dt, "control_hz": 25, "tool": env.tool, "act_dim": int(env.action_space.shape[0])})
    if actor is not None:
        actor.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"]})
    pol.observe(info)       # the reset info already carries the cursor: without this the first label aims at the hand origin
    # Storing flatten_obs per step would be 938 KB a frame with the maps covering the
    # full sheet at 1 mm/px - over a terabyte for a full collection. But 99.9 % of that is
    # map, the task map never changes within an episode, and the ink map gains only a
    # few dozen pixels a step. So: the vector every step, the task map once, and the
    # ink map as sparse deltas. bc_train replays them.
    V, A, ink_d, TJ = [], [], [], []
    task_map = np.asarray(obs["task"]["map"]).copy()
    prev_ink = np.zeros_like(obs["external"]["map"])
    rng = np.random.default_rng(seed)
    done, t, drop_at, touched = False, 0, None, False
    from dexcraft.env import unpack
    while not done and t < env.T:
        a = np.asarray(pol.act(obs), np.float32)
        touched = touched or float(unpack(obs["external"], "external")["tip_above_sheet"][0]) < 0.001
        V.append(np.concatenate([obs["position"], obs["tactile"],
                                 obs["task"]["vec"], obs["external"]["vec"]]).astype(np.float32))
        if "traj" in obs["task"]:
            TJ.append(np.asarray(obs["task"]["traj"], np.float32))     # the job, as a trajectory
        cur = np.asarray(obs["external"]["map"])
        r, c = np.nonzero(cur != prev_ink)
        ink_d.append(np.stack([r, c, cur[r, c]], 1).astype(np.uint16) if len(r)
                     else np.zeros((0, 3), np.uint16))
        prev_ink = cur.copy()
        A.append(a)
        if label_cut and touched and drop_at is None and float(np.abs(a[:7]).max()) > label_cut:
            # DAgger: where the actor has strayed far, the teacher's arm step is several times its
            # clean size and a clone trained on it overshoots; keep the gentle corrections only, cut the rest.
            # Only once the pen has touched: the approach from the start pose is large in clean demos too.
            drop_at = len(V) - 1
            break
        if actor is not None:
            a_exec = np.asarray(actor.act(obs), np.float32)
        else:
            a_exec = a if noise <= 0 else np.clip(a + rng.normal(0.0, noise, a.shape).astype(np.float32), -1, 1)
        obs, r, te, tr, infos = env.step(a_exec)
        pol.observe(infos)
        t += 1; done = te or tr
        if infos.get("dropped") and drop_at is None:
            drop_at = len(V) - 1        # only reached when the pen is dropped
    m_hits = getattr(env, "_tl_hits", 0)
    env.close()
    # TRUNCATE at the drop, do not discard the episode. Everything after the pen
    # leaves the hand is a hand waving at nothing and cloning it teaches waving,
    # but the steps before it are the most valuable labels DART collects - the
    # teacher recovering from a state its own clean rollouts never visit.
    # Discarding whole episodes instead would throw away most of the noisy sets
    # along with exactly those labels.
    n = len(V) if drop_at is None else max(drop_at, 0)
    return (np.array(V[:n]), np.array(A[:n]), task_map, ink_d[:n],
            m_hits, int(drop_at is not None), np.array(TJ[:n]) if TJ else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--yaw", type=float, default=None, help="env yaw_offset [rad]; None = scene default (YAW_OFFSET_DEFAULT)")
    ap.add_argument("--wrist-cost", type=float, default=None, help="teacher: price of wrist motion")
    ap.add_argument("--leak", type=float, default=None, help="teacher: pull back to the grip pose")
    ap.add_argument("--span", type=float, default=None, help="teacher: driver box, 0 = arm-only")
    ap.add_argument("--max-step", type=float, default=None, help="teacher: metres of tip error chased per step")
    ap.add_argument("--reach", type=float, default=0.06, help="cap the glyph extent [m]")
    ap.add_argument("--constraint", default="none", help="env constraint preset (e.g. arm_lock)")
    # With a nonzero assist the clone learns finger torques that only hold WITH the assist
    # and drops the pen at the contract's grip_assist 0, while the teacher at 0 does not.
    # Default is the contract.
    ap.add_argument("--ga", type=float, default=0.0, help="grip_assist the teacher runs with (contract: 0)")
    # The teacher plans from the task map and ignores the cursor, but the CLONE sees the
    # cursor: under the patient presets it parks until ink lands, under "default" (what
    # evaluate runs) it walks on its own clock. Collect under the preset the clone is scored in.
    ap.add_argument("--reward", default="write_v5", help="env reward preset the demos are observed under")
    ap.add_argument("--end-mode", default="fixed", choices=["fixed", "policy"], help="policy: the expert declares the end (35-D actions)")
    ap.add_argument("--cursor-obs", type=int, default=1, help="0: the schedule is not in the observation (the expert reads it from info)")
    ap.add_argument("--traj-obs", type=int, default=0, help="store this many future trajectory points per step")
    ap.add_argument("--start-height", type=float, nargs=2, default=(-0.0025, 0.0015), metavar=("LO", "HI"),
                    help="tip height above the sheet at reset [m]")
    ap.add_argument("--adv-mode", default="ink", choices=["ink", "tip"], help="how the timeline pointer advances")
    ap.add_argument("--teacher", default="scripted", help="scripted | hybrid:<grip ckpt> (IK arm + learned grip)")
    ap.add_argument("--label-cut", type=float, default=0.0, help="DAgger: cut the episode at the first frame whose teacher arm label exceeds this (0 = off)")
    args = ap.parse_args()
    tasks = args.tasks.split(",") if args.tasks else TASKS
    polkw = {}
    if args.wrist_cost is not None: polkw["wrist_cost"] = args.wrist_cost
    if args.leak is not None: polkw["leak"] = args.leak
    if args.span is not None: polkw["finger_span"] = args.span
    if args.max_step is not None: polkw["max_step"] = args.max_step
    envkw = {"reach": args.reach, "constraint": args.constraint, "grip_assist": args.ga, "reward": args.reward,
             "end_mode": args.end_mode, "cursor_obs": bool(args.cursor_obs),
             "traj_obs": args.traj_obs, "adv_mode": args.adv_mode, "start_height": tuple(args.start_height)}
    jobs = [(tasks[i % len(tasks)], args.seed_base + i, args.noise, args.actor, args.yaw, polkw, envkw, args.teacher, args.label_cut)
            for i in range(args.episodes)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_episode, jobs)
    results = [r for r in results if len(r[0])]
    V = np.concatenate([r[0] for r in results])
    A = np.concatenate([r[1] for r in results])
    ep_len = np.array([len(r[0]) for r in results], np.int32)
    task_maps = np.stack([r[2] for r in results])
    # ink deltas: one ragged (k,3) block per step, flattened with an index
    flat = np.concatenate([d for r in results for d in r[3]]) if any(
        len(d) for r in results for d in r[3]) else np.zeros((0, 3), np.uint16)
    ink_n = np.array([len(d) for r in results for d in r[3]], np.int32)
    hits = sum(r[4] for r in results)
    dropped = sum(r[5] for r in results)
    # Stamp the env this was collected under, so bc_train writes the true settings
    # into the checkpoint instead of copying --init's.
    meta = dict(obs_mode="state", control_mode="hybrid", control_hz=25,
                reward=args.reward, grip_assist=args.ga,
                start_height=list(args.start_height), start_wrist=True,
                yaw_offset=args.yaw, reach=args.reach, constraint=args.constraint,
                end_mode=args.end_mode, cursor_obs=bool(args.cursor_obs), teacher=args.teacher,
                traj_obs=args.traj_obs, adv_mode=args.adv_mode)
    extra = {"traj": np.concatenate([r[6] for r in results])} if args.traj_obs else {}
    np.savez_compressed(args.out, vec=V, act=A, ep_len=ep_len, task_map=task_maps,
                        ink_delta=flat, ink_n=ink_n, meta=np.array(json.dumps(meta)), **extra)
    print(f"collected {len(V)} pairs from {len(jobs)} episodes "
          f"({dropped} truncated at a drop) (teacher tl_hits total {hits}) -> {args.out}")


if __name__ == "__main__":
    main()
