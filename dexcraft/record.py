"""Record demonstrations into a dataset you can train on.

The benchmark ships no fixed demonstration set on purpose - every episode
re-randomises the task, so a frozen dataset would age badly. You generate as
much as you want, from whatever teacher you like, and disclose how much you used.

    python -m dexcraft.record --tasks write --episodes 200 --out data/scripted
    python -m dexcraft.record --tasks L6 --policy my_teacher.py --episodes 50 --out data/l6

Layout: one .npz per shard plus a meta.json.

    position     (N,  86) float32     per step, the four observation groups
    tactile      (N,  43) float32
    task_vec     (N,   8) float32
    external_vec (N,  31) float32
    action       (N,  43) float32     what was actually applied
    reward       (N,)     float32
    ep_start     (E,) int32           index of each episode's first step
    ep_len       (E,) int32
    task_map     (E, 128, 128) uint8  constant within an episode, so stored once
    ep_task      (E,) int32           index into meta["task_ids"]
    ep_completion, ep_success, ep_force_mae   (E,) per-episode metrics
    ink_map      (N, 64, 64) uint8    only with --with-ink (it is replayable, so off by default)

Only clean episodes are kept by default (`--min-completion`), so the set is a
demonstration set rather than a log.
"""
import argparse
import json
import os
import time

import numpy as np

from .env import DexCraftEnv
from .metrics import episode_metrics
from .policies.base import load_policy
from .tasks import LEVELS, TASKS
from .train_ppo import resolve_tasks


def record(args):
    task_ids = resolve_tasks(args.tasks)
    os.makedirs(args.out, exist_ok=True)
    policy = load_policy(args.policy)
    control_hz = args.control_hz or getattr(policy, "control_hz", 25)
    kept = rejected = shard = attempt = 0
    buf, ep, t0 = _new_buffer(), [], time.time()

    while kept < args.episodes:
        tid = task_ids[kept % len(task_ids)]
        # the seed follows the ATTEMPT, not the kept count: retrying a rejected
        # episode with the same seed just replays the same failure for ever
        seed = args.seed + attempt
        attempt += 1
        env = DexCraftEnv(tid, control_mode=policy.control_mode, control_hz=control_hz,
                          perturb=args.perturb)
        obs, info = env.reset(seed=seed)
        policy.reset({"task_id": info["task_id"], "level": info["level"],
                      "instruction": obs["instruction"], "dt": env.dt,
                      "control_hz": env.control_hz, "tool": env.tool})
        steps = _new_buffer()
        done = False
        while not done:
            a = np.asarray(policy.act(obs), dtype=np.float32).ravel()
            steps["position"].append(obs["position"])
            steps["tactile"].append(obs["tactile"])
            steps["task_vec"].append(obs["task"]["vec"])
            steps["external_vec"].append(obs["external"]["vec"])
            steps["action"].append(a)
            if args.with_ink:
                steps["ink_map"].append(obs["external"]["map"].copy())
            obs, r, term, trunc, info = env.step(a)
            policy.observe(info)
            steps["reward"].append(np.float32(r))
            done = term or trunc
        rec = env.episode_record()
        m = episode_metrics(rec)
        env.close()
        if (m["completion"] or 0.0) < args.min_completion or m["dropped"]:
            rejected += 1
            if rejected > 20 * max(args.episodes, 1):
                raise SystemExit(f"teacher too weak on {args.tasks}: {rejected} rejected, {kept} kept "
                                 f"(lower --min-completion or use a better teacher)")
            continue
        start = len(buf["position"])
        for k in steps:
            buf[k].extend(steps[k])
        ep.append(dict(start=start, length=len(steps["position"]), task=task_ids.index(tid),
                       completion=float(m["completion"] or 0.0), success=bool(m["success"]),
                       force_mae=float(m["force_mae_N"]) if m.get("force_mae_N") else float("nan"),
                       task_map=rec["task_map"]))
        kept += 1
        if args.verbose and kept % 10 == 0:
            n = len(buf["position"])
            print(f"  {kept}/{args.episodes} episodes ({rejected} rejected), {n:,} steps, "
                  f"{n / (time.time() - t0):.0f} steps/s", flush=True)
        if args.shard_episodes and kept % args.shard_episodes == 0:
            _flush(args, buf, ep, shard, task_ids)
            buf, ep, shard = _new_buffer(), [], shard + 1
    if ep:
        _flush(args, buf, ep, shard, task_ids)
    meta = dict(task_ids=task_ids, episodes=kept, rejected=rejected, control_hz=control_hz,
                control_mode=policy.control_mode, perturb=args.perturb,
                policy=args.policy, with_ink=bool(args.with_ink))
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"wrote {kept} episodes ({rejected} rejected) to {args.out}", flush=True)


def _new_buffer():
    return {k: [] for k in ("position", "tactile", "task_vec", "external_vec",
                            "action", "reward", "ink_map")}


def _flush(args, buf, ep, shard, task_ids):
    out = {k: np.asarray(v, dtype=np.float32) for k, v in buf.items() if v and k != "ink_map"}
    if buf["ink_map"]:
        out["ink_map"] = np.asarray(buf["ink_map"], dtype=np.uint8)
    out["ep_start"] = np.array([e["start"] for e in ep], np.int32)
    out["ep_len"] = np.array([e["length"] for e in ep], np.int32)
    out["ep_task"] = np.array([e["task"] for e in ep], np.int32)
    out["ep_completion"] = np.array([e["completion"] for e in ep], np.float32)
    out["ep_success"] = np.array([e["success"] for e in ep], bool)
    out["ep_force_mae"] = np.array([e["force_mae"] for e in ep], np.float32)
    out["task_map"] = np.asarray([e["task_map"] for e in ep], np.uint8)
    path = os.path.join(args.out, f"shard_{shard:04d}.npz")
    np.savez_compressed(path, **out)
    print(f"  -> {path}  {len(ep)} episodes, {len(out['action']):,} steps, "
          f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="write", help="task id, comma list, a level (L2), 'write' or 'all'")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--policy", default="dexcraft.policies.scripted:make_policy",
                    help="the teacher: module:factory or path/to/policy.py")
    ap.add_argument("--out", default="data/scripted")
    ap.add_argument("--perturb", default="none")
    ap.add_argument("--control-hz", type=int, default=None)
    ap.add_argument("--min-completion", type=float, default=0.6,
                    help="reject episodes the teacher botched, so this is a demo set not a log")
    ap.add_argument("--with-ink", action="store_true", help="also store the 64x64 ink map per step")
    ap.add_argument("--shard-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1)
    record(ap.parse_args(argv))


if __name__ == "__main__":
    main()
