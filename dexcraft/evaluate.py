"""Run a policy over the benchmark suite and write a results json.

    python -m dexcraft.evaluate --policy dexcraft.policies.scripted:make_policy --seeds 3 \
        --out results/scripted.json
    python -m dexcraft.evaluate --policy-url http://localhost:8100 --tasks L1_trace_circle,L2_write_word

Tracks: nominal (no perturbation), robust (perturb preset "mild"), constrained =
the finger track (preset "arm_lock": the arm may only move while the pen is
up, so strokes must be made with the wrist and fingers). All tracks reuse the
same task seeds.
"""
import argparse
import json
import os
import secrets
import sys
import time
import zlib

import numpy as np

from . import benchmark as B

from . import tasks as T
from .env import DexCraftEnv
from .metrics import aggregate, episode_metrics
from .policies.base import load_policy

TRACKS = {"nominal": dict(perturb="none", constraint="none"),
          "robust": dict(perturb="mild", constraint="none"),
          "constrained": dict(perturb="none", constraint="arm_lock"),   # finger track: arm frozen while the pen is down
          "frozen": dict(perturb="none", constraint="fingers_only"),    # diagnostic: the arm never moves
          "frozen_lock": dict(perturb="none", constraint="fingers_only_lock"),   # arm never moves, ring/pinky servoed to the grip pose
          "micro_lock": dict(perturb="none", constraint="micro_arm_lock")}   # arm creeps at 8%, ring/pinky locked


def run_episode(env, policy, seed, record_video=None):
    obs, info = env.reset(seed=seed)
    policy.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"],
                  "act_dim": int(env.action_space.shape[0]),
                  "dt": env.dt, "control_hz": env.control_hz, "tool": env.tool})
    # The reset info carries the first cursor. Without this a policy that reads info - the
    # scripted expert, the IK expert - computes its first action with no target and the
    # tracker aims at the hand origin, which at a 6 mm step cap is a shove the wrong way.
    policy.observe(info)
    done = False
    while not done:
        chunk = np.asarray(policy.act(obs), dtype=np.float32)
        for a in (chunk if chunk.ndim == 2 else chunk[None]):     # (K, NU) action chunks are applied in order
            obs, r, term, trunc, info = env.step(a)
            policy.observe(info)
            if term or trunc:
                done = True
                break
    rec = env.episode_record()
    m = episode_metrics(rec)
    if record_video is not None:                 # final frame: top view | side view
        from PIL import Image
        top = env.render(camera="top", width=400, height=300)
        side = env.render(camera="side", width=400, height=300)
        Image.fromarray(np.concatenate([top, side], axis=1)).save(record_video)
    return m, rec


def evaluate(policy, task_ids, seeds, tracks, obs_mode=None, verbose=True, snapshot_dir=None, control_hz=None,
             seed_base=0, env_kwargs=None):
    """seed_base: offset added to every episode seed; a freshly drawn base
    (`--fresh-seeds`) scores episodes that have never been seen before. Position,
    scale, slant, rotation and the force profile are re-randomised from the seed,
    so the continuous part of a task cannot be memorised even if the content pool
    were known."""
    episodes = []
    obs_mode = obs_mode or getattr(policy, "obs_mode", "state")
    control_hz = control_hz or getattr(policy, "control_hz", 100)
    for track in tracks:
        for tid in task_ids:
            env = DexCraftEnv(tid, control_mode=policy.control_mode, obs_mode=obs_mode, control_hz=control_hz,
                              **{**TRACKS[track], **(env_kwargs or {})})
            for k in range(seeds):
                seed = seed_base + 1000 * k + zlib.crc32(tid.encode()) % 1000     # deterministic
                snap = None
                if snapshot_dir and k == 0 and track == "nominal":
                    snap = os.path.join(snapshot_dir, f"{tid}.png")
                t0 = time.time()
                m, _ = run_episode(env, policy, seed, record_video=snap)
                episodes.append({"track": track, "task_id": tid, "seed": seed, "metrics": m})
                if verbose:
                    print(f"[{track:11s}] {tid:20s} seed {k}: success={m['success']} "
                          f"prog={m['progress']:.2f} iou={m['ink_iou']} finger_share={m['finger_share']} "
                          f"t_done={m['completion_time']} ({time.time() - t0:.1f}s)", flush=True)
            env.close()
    return episodes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="module:factory or path/to/policy.py")
    ap.add_argument("--policy-url", help="remote policy server base URL")
    ap.add_argument("--tasks", default="all", help="comma list of task ids, a level (L2) or 'all'")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tracks", default="nominal,robust,constrained")
    ap.add_argument("--obs-mode", default=None)
    ap.add_argument("--control-hz", type=int, default=None, help="policy rate (default: policy.control_hz or 100)")
    ap.add_argument("--seed-base", type=int, default=0, help="evaluation seed offset")
    ap.add_argument("--fresh-seeds", action="store_true",
                    help="draw a new random seed base for this run, so the episodes have never been "
                         "scored before")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                    help="disclosure fields: architecture=generalist|experts demos=<n|none> "
                         "compute=<text> env_steps=<n> (recorded in the results json)")
    ap.add_argument("--name", default=None, help="name recorded in the results json")
    ap.add_argument("--env-kwargs", default=None, help='JSON of extra DexCraftEnv kwargs, e.g. {"reach": 0.04, "start_height": [0, 0.02]}')
    ap.add_argument("--out", default=None)
    ap.add_argument("--snapshots", default=None, help="dir for final-frame PNGs (nominal, seed 0)")
    args = ap.parse_args(argv)
    if args.policy_url:
        from .policies.remote import RemotePolicy
        policy = RemotePolicy(args.policy_url)
    else:
        policy = load_policy(args.policy)
    if args.tasks == "all":
        task_ids = list(T.TASKS)
    elif args.tasks in T.LEVELS:
        task_ids = T.LEVELS[args.tasks]
    else:
        task_ids = args.tasks.split(",")
    tracks = args.tracks.split(",")
    if args.snapshots:
        os.makedirs(args.snapshots, exist_ok=True)
    seed_base = secrets.randbelow(1 << 30) if args.fresh_seeds else args.seed_base
    ekw = json.loads(args.env_kwargs) if args.env_kwargs else None
    # A run that relaxes the contract may still be useful, but it must say so.
    # See dexcraft/benchmark.py.
    deviation = B.deviations(ekw)
    if deviation:
        print("!! OFF CONTRACT - not a benchmark number:")
        for k, v in sorted(deviation.items()):
            print(f"     {k}: contract {v['contract']!r}, running with {v['ran_with']!r}")
    episodes = evaluate(policy, task_ids, args.seeds, tracks, args.obs_mode, snapshot_dir=args.snapshots,
                        control_hz=args.control_hz, seed_base=seed_base,
                        env_kwargs=ekw)
    obs_mode = args.obs_mode or getattr(policy, "obs_mode", "state")
    disclosure = {"architecture": getattr(policy, "architecture", "generalist"),
                  "demos": getattr(policy, "demos", "none"), "compute": getattr(policy, "compute", ""),
                  "env_steps": getattr(policy, "env_steps", ""),
                  "privileged_obs": "state" in obs_mode and "deployable" not in obs_mode}
    for kv in args.meta:
        k, _, v = kv.partition("=")
        disclosure[k] = v
    result = {"name": args.name or (args.policy or args.policy_url), "policy": args.policy or args.policy_url,
              "control_mode": policy.control_mode, "obs_mode": obs_mode,
              "control_hz": args.control_hz or getattr(policy, "control_hz", 100),
              "fresh_seeds": bool(args.fresh_seeds), "disclosure": disclosure,
              "env_kwargs": ekw, "contract_deviations": deviation, "on_contract": not deviation,
              "tasks": task_ids, "seeds": args.seeds, "tracks": tracks,
              "date": time.strftime("%Y-%m-%d"), "episodes": episodes, **aggregate(episodes)}
    print(json.dumps(result["score"], indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1, default=_json_default)
        print("wrote", args.out)
    return result


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
