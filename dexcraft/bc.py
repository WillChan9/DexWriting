"""Behaviour cloning from recorded demonstrations, into a train_ppo checkpoint.

Used to warm-start the HOLDING skill before any RL: on the hybrid interface a
zero-torque hand loses the tool in 0.6 s, and no action scale creates a survivable
random regime (measured), so PPO from scratch spends its whole budget learning not
to drop and never reaches the task.

    python -m dexcraft.bc --data data/hold --out runs/bc/agent.pt --epochs 40

The output loads directly with `train_ppo --init`, so the RL stages continue from it.

NOTE on action scale: the teacher exceeds the usual ScaleAction caps on 22-28 % of
steps (wrist 0.5, hand 0.25), so a scaled agent cannot reproduce it. This writes
`action_scale = 1` and expects the RL stages to control exploration with
`--init-logstd-*` instead, which is the knob that actually matters once the policy
starts from a competent init rather than from noise.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .train_ppo import ACT_DIM, MAP_DIM, OBS_DIM, VEC_DIM, Agent


def load(data_dir):
    X, Y = [], []
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {data_dir}")
    for f in shards:
        d = np.load(f)
        n = len(d["action"])
        vec = np.concatenate([d["position"], d["tactile"], d["task_vec"], d["external_vec"]], axis=1)
        if vec.shape[1] < VEC_DIM:
            # shards recorded without the brush block of the external group:
            # those entries are zero on every non-brush task, so padding is exact
            vec = np.concatenate([vec, np.zeros((n, VEC_DIM - vec.shape[1]), vec.dtype)], axis=1)
        assert vec.shape[1] == VEC_DIM, f"{vec.shape[1]} != {VEC_DIM}"
        # the task map is constant within an episode; the ink map is not stored
        # (hold tasks lay no ink, so it is zero by construction)
        maps = np.zeros((n, MAP_DIM), np.float32)
        for k, (s, L) in enumerate(zip(d["ep_start"], d["ep_len"])):
            tm = np.asarray(d["task_map"][k]).ravel().astype(np.float32) / 255.0
            maps[s:s + L, :tm.size] = tm
        X.append(np.concatenate([vec, maps], axis=1).astype(np.float32))
        Y.append(d["action"].astype(np.float32))
        print(f"  {os.path.basename(f)}: {n:,} steps", flush=True)
    return np.concatenate(X), np.concatenate(Y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hold")
    ap.add_argument("--out", default="runs/bc/agent.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--cuda", type=int, default=1)
    args = ap.parse_args(argv)

    X, Y = load(args.data)
    assert X.shape[1] == OBS_DIM and Y.shape[1] == ACT_DIM, (X.shape, Y.shape)
    print(f"dataset {X.shape[0]:,} steps  obs {X.shape[1]}  act {Y.shape[1]}", flush=True)

    mean, var = X.mean(0), X.var(0)
    Xn = (X - mean) / np.sqrt(var + 1e-8)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xn))
    nval = max(1, int(args.val_frac * len(idx)))
    va, tr = idx[:nval], idx[nval:]

    dev = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    agent = Agent(OBS_DIM, ACT_DIM, args.hidden).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    xt = torch.as_tensor(Xn, device=dev); yt = torch.as_tensor(Y, device=dev)

    for ep in range(args.epochs):
        agent.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        for k in range(0, len(tr), args.batch):
            b = torch.as_tensor(tr, device=dev)[perm[k:k + args.batch]]
            pred = agent.actor_mean(agent.enc_pi(xt[b]))
            loss = nn.functional.mse_loss(pred, yt[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        agent.eval()
        with torch.no_grad():
            vb = torch.as_tensor(va, device=dev)
            vl = float(nn.functional.mse_loss(agent.actor_mean(agent.enc_pi(xt[vb])), yt[vb]))
        print(f"epoch {ep:3d}  train {tot/len(tr):.5f}  val {vl:.5f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": agent.state_dict(), "obs_mean": mean.astype(np.float64),
                "obs_var": var.astype(np.float64), "obs_mode": "state",
                "control_mode": "hybrid", "control_hz": 25, "hidden": args.hidden,
                "obs_dim": OBS_DIM, "min_logstd": -3.0, "iteration": 0,
                "tasks": json.load(open(os.path.join(args.data, "meta.json")))["task_ids"],
                "action_scale": np.ones(ACT_DIM, np.float32)}, args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
