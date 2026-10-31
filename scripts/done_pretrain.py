"""Teach the network's done output from its own rollouts, then let PPO refine it.

    PYTHONPATH=. python scripts/done_pretrain.py runs/bc_s0_merged.pt runs/bc_s0_final.pt [episodes]

Left for PPO to discover from scratch (std 0.5 around -1), the done output fires randomly early and
the idle phase slips and drops the pen; each zeroes the completion, which makes writing look
unprofitable and collapses progress. So the done row is fitted first: the network is rolled out (its done
never fires at bias -1, so episodes run past the line), every frame is labelled "the trajectory's
pointer is past its end", and the done row - a linear readout of the arm trunk's features - is fitted
by ridge regression to +1 / -1. Nothing else changes. The network then declares the end itself, from
its own observation, and PPO can move the readout wherever the reward says.
"""
import os
import sys
import numpy as np
import torch
from dexcraft.env import DexCraftEnv, NU
from dexcraft.policies.ppo import PPOPolicy
from dexcraft.train_ppo import pack_obs

SRC, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 120
EK = dict(control_hz=25, control_mode="hybrid", obs_mode="state", grip_assist=0.0, start_height=(-0.0025, 0.0015),
          start_wrist=True, constraint="idle_lock", reward=os.environ.get("DP_REWARD", "v16_dense1c"), end_mode="policy", cursor_obs=False,
          traj_obs=16, adv_mode="tip")
pol = PPOPolicy(SRC)
X, Y, ep_id = [], [], []
TASKS = os.environ.get("DP_TASKS", "L1_trace_line").split(",")
for i in range(N):
    seed = 300000 + i
    env = DexCraftEnv(TASKS[i % len(TASKS)], **EK); obs, info = env.reset(seed=seed)
    pol.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"], "dt": env.dt,
               "control_hz": 25, "tool": env.tool, "act_dim": int(env.action_space.shape[0])})
    while True:
        vec, _ = pack_obs(obs, True)
        x = np.clip((vec - pol.mean) / pol.std, -10, 10).astype(np.float32)
        X.append(x); Y.append(1.0 if env._tl_k > len(env._timeline) - 1 else -1.0); ep_id.append(i)
        a = np.asarray(pol.act(obs), np.float32); a[NU:] = -1.0          # never declare: see the idle too
        obs, r, te, tr, info = env.step(np.clip(a, -1, 1)); pol.observe(info)
        if te or tr: break
X, Y, ep_id = np.array(X), np.array(Y), np.array(ep_id)
print(f"{len(X):,} frames from {N} episodes, {100 * (Y > 0).mean():.1f} % after the trajectory's end")
agent = pol.agent
with torch.no_grad():
    F = agent.arm_trunk(torch.as_tensor(X)).numpy()                     # (n, 256) features the done row reads
train = ep_id < int(0.8 * N)
A = np.concatenate([F, np.ones((len(F), 1))], 1)
lam = 1.0
w = np.linalg.solve(A[train].T @ A[train] + lam * np.eye(A.shape[1]), A[train].T @ Y[train])
pred = A @ w
for name, m in (("train", train), ("held-out", ~train)):
    acc = ((pred[m] > 0.5) == (Y[m] > 0)).mean()
    early = ((pred[m] > 0.5) & (Y[m] < 0)).sum() / max(1, (Y[m] < 0).sum())
    late = ((pred[m] <= 0.5) & (Y[m] > 0)).sum() / max(1, (Y[m] > 0).sum())
    print(f"{name}: accuracy {100 * acc:.2f} %, fires before the end on {100 * early:.2f} % of unfinished frames, "
          f"misses {100 * late:.2f} % of finished frames")
ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck["model"]
sd["arm_head.weight"][NU] = torch.as_tensor(w[:-1], dtype=torch.float32)
sd["arm_head.bias"][NU] = float(w[-1])
sd["actor_logstd"][0, NU] = float(np.log(0.2))
ck["iteration"] = f"done_pretrain({SRC})"
torch.save(ck, OUT)
print("saved", OUT)
