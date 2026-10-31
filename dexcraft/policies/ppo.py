"""Wrap a PPO checkpoint (train_ppo.py) as a benchmark Policy.

    DEXCRAFT_PPO_CKPT=runs/word/agent.pt python -m dexcraft.evaluate \
        --policy dexcraft.policies.ppo:make_policy --tasks L2_write_word

The checkpoint carries the observation normalisation and the control settings it
was trained with, so evaluation reproduces training exactly.
"""
import os

import numpy as np
import torch

from ..train_ppo import ACT_DIM, TRAJ_ARCHS, VEC_DIM, Agent, adapt_checkpoint, pack_obs, tip_px_from_vec, vec_dim_for
from .base import Policy


class PPOPolicy(Policy):
    architecture = "generalist"
    demos = "none"

    def __init__(self, ckpt, deterministic=True):
        ck = adapt_checkpoint(torch.load(ckpt, map_location="cpu", weights_only=False))
        # No defaults for the control contract. Guessing it is silent and fatal: a
        # hybrid / 25 Hz policy run as delta_pos / 100 Hz drops the tool on every
        # episode and looks like a training failure.
        missing = [k for k in ("control_mode", "control_hz") if k not in ck]
        if missing:
            raise ValueError(f"{ckpt}: checkpoint does not say how it was controlled "
                             f"(missing {missing}). Re-train with a dataset that stamps "
                             f"them, or set them on the checkpoint before evaluating.")
        self.obs_mode = ck.get("obs_mode", "state")
        self.control_mode = ck["control_mode"]
        self.control_hz = ck["control_hz"]
        obs_dim = ck.get("obs_dim") or ck["obs_mean"].shape[0]
        out_dim = int(ck["model"]["actor_mean.2.bias"].shape[0])       # act_dim, x the chunk length
        self.chunk = int(ck.get("chunk", 1))                           # K actions per forward
        act_dim = out_dim // self.chunk                                # 34, or 35 with a done entry
        self.arch, self.hist = ck.get("arch", "conv"), int(ck.get("hist", 0))
        self.agent = Agent(obs_dim, act_dim, hidden=ck.get("hidden", 256), min_logstd=ck.get("min_logstd", -2.5),
                           arch=self.arch, hist=self.hist, crop=int(ck.get("crop", 96)), chunk=self.chunk)
        self.agent.load_state_dict(ck["model"])
        self.agent.eval()
        # normalisation applies to the VECTOR group; older checkpoints stored statistics
        # for the whole flattened observation, so take the leading VEC_DIM entries.
        pv = vec_dim_for(self.arch)              # `traj`: 137 state entries + the 80-entry window
        self.mean = np.asarray(ck["obs_mean"], dtype=np.float32)[:pv]
        self.std = np.sqrt(np.asarray(ck["obs_var"], dtype=np.float32)[:pv] + 1e-8)
        self.deterministic = deterministic
        self.tasks = ck.get("tasks", [])
        sc = ck.get("action_scale")
        self.action_scale = np.ones(act_dim, dtype=np.float32) if sc is None else np.asarray(sc, dtype=np.float32)
        if len(self.action_scale) < act_dim:                             # a done entry is never scaled
            self.action_scale = np.concatenate([self.action_scale, np.ones(act_dim - len(self.action_scale), np.float32)])
        self.env_steps = ck.get("iteration", "")
        self._h = np.zeros((max(self.hist, 1), pv), np.float32)

    def reset(self, task_info):
        self._h[:] = 0.0                                   # a fresh episode has no history

    def act(self, obs):
        vec, maps = pack_obs(obs, self.arch in TRAJ_ARCHS)
        x = np.clip((vec - self.mean) / self.std, -10, 10).astype(np.float32)
        with torch.no_grad():
            tv = torch.as_tensor(x).unsqueeze(0)
            tm = torch.as_tensor(maps).unsqueeze(0)
            tt = torch.as_tensor(np.zeros(2, np.float32) if self.arch in TRAJ_ARCHS
                                 else tip_px_from_vec(vec)).unsqueeze(0)   # the tip pixel is the xf branch's input
            th = torch.as_tensor(self._h).unsqueeze(0)
            a = (self.agent.mean(tv, tm, tt, th) if self.deterministic
                 else self.agent.action_and_value(tv, tm, tt, th)[0])
        if self.hist:
            # one push per act() call, so the history stride equals the chunk - bc_train
            # builds its history windows the same way
            self._h = np.concatenate([self._h[1:], x[None]], 0)
        a = np.clip(a.squeeze(0).numpy(), -1, 1)
        if self.chunk > 1:
            # (K, act_dim); evaluate applies the rows in order, one per control step
            return a.reshape(self.chunk, -1) * self.action_scale
        return a * self.action_scale


def make_policy(ckpt=None):
    return PPOPolicy(ckpt or os.environ["DEXCRAFT_PPO_CKPT"])
