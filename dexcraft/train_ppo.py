"""PPO trainer for DexCraft (cleanrl-style, CPU-parallel MuJoCo envs).

Observation normalisation lives in the trainer (one exact RunningMeanStd, saved
in the checkpoint), the policy log-std has a floor so entropy cannot collapse,
updates stop on a KL threshold, and a dropped tool pays the rest of the episode
(env.REWARD_PRESETS) so "throw the pen to cut losses" is not an exploit.

    # single task, local smoke test
    python -m dexcraft.train_ppo --task L1_trace_circle --steps 50000 --n-envs 4 --vec sync

    # full run (96 cores)
    python -m dexcraft.train_ppo --task L2_write_word --n-envs 96 --steps 30_000_000 \
        --reward fingers --out runs/word_fingers

    # curriculum: warm-start from an easier stage
    python -m dexcraft.train_ppo --task L2_write_word --init runs/trace/agent.pt ...

Multi-task: --task takes a comma list, a level ("L2"), or "write" (L1+L2); each
env is pinned to one task id and they are dealt round-robin.

Evaluate a checkpoint with
    DEXCRAFT_PPO_CKPT=runs/.../agent.pt python -m dexcraft.evaluate \
        --policy dexcraft.policies.ppo:make_policy --tasks L2_write_word
"""
import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import scene as S
from . import tasks as T
from .env import GROUP_DIM, INK_MAP_H, INK_MAP_W, LAYOUT, N_ARM, N_HAND, N_WRIST, NU, DexCraftEnv

VEC_DIM = sum(GROUP_DIM[g] for g in ("position", "tactile", "task", "external"))
MAP_DIM = 2 * INK_MAP_H * INK_MAP_W     # informational; the maps are never flattened
OBS_DIM = VEC_DIM + MAP_DIM

# `--arch traj`: the task arrives as a trajectory, not as a picture, so the packed vector is
# the state entries that vary in a writing task plus the trajectory window. Dropped: the cursor
# (replaced), the brush tuft state and the chopstick object / bowl (other tiers), the brush and
# chopstick tool slots, the force scale (always FORCE_MAX) and the map window's size and centre
# (only there to read the map image). All 30 are constant or zero in every writing task, so a
# transplanted first layer folds their contribution into its bias and loses nothing.
def _state_keep():
    g0, a = {}, 0
    for g in ("position", "tactile", "task", "external"):
        g0[g] = a; a += GROUP_DIM[g]
    drop = []
    for g, fields in (("task", ("force_max_N", "map_half_m", "map_center_m")),
                      ("external", ("object_pos_in_hand", "target_pos_in_hand", "write_target_in_hand",
                                    "write_ahead_near", "write_ahead_far", "write_pen_down", "brush_state"))):
        for f in fields:
            o, n = LAYOUT[g][0][f]
            drop += list(range(g0[g] + o, g0[g] + o + n))
    o, _ = LAYOUT["task"][0]["tool_onehot"]
    drop += [g0["task"] + o + 1, g0["task"] + o + 2]          # brush / chopsticks slots; the pen slot stays
    return np.array([i for i in range(VEC_DIM) if i not in set(drop)]), np.array(sorted(drop))


STATE_KEEP, STATE_DROP = _state_keep()
TRAJ_K, TRAJ_COLS = 16, 5                                     # points, (dx, dy, dz, force, due time)
TRAJ_DIM = TRAJ_K * TRAJ_COLS
PVEC_TRAJ = len(STATE_KEEP) + TRAJ_DIM                        # 137 + 80 = 217


TRAJ_ARCHS = ("traj", "traj2")           # traj2 = the same input, two trunks


def vec_dim_for(arch):
    return PVEC_TRAJ if arch in TRAJ_ARCHS else VEC_DIM

ACT_DIM = NU                       # joint torque, one duty cycle per joint

# ---------------------------------------------------------------- where the tip is on the map
_TASK0 = GROUP_DIM["position"] + GROUP_DIM["tactile"]
_EXT0 = _TASK0 + GROUP_DIM["task"]


def _span(group, name, base):
    a, n = LAYOUT[group][0][name]
    return slice(base + a, base + a + n)


SL_HAND_POS, SL_ROT6, SL_TIP_H = _span("external", "hand_pos", _EXT0), _span("external", "hand_rot6", _EXT0), _span("external", "tip_pos_in_hand", _EXT0)
SL_WIN_H, SL_WIN_C = _span("task", "map_half_m", _TASK0), _span("task", "map_center_m", _TASK0)


def tip_px_from_vec(vec):
    """Raw (unnormalised) observation vector(s) (..., VEC_DIM) -> the tip's map pixel (x, y), continuous.

    The env's _to_px without the int(): tip_world = hand_pos + R @ tip_in_hand, R = [c0 c1 c0 x c1] from
    hand_rot6 (the rotation's first two columns), then the sheet window (map_center_m / map_half_m, in
    the task group). Every term is in the standard observation - nothing privileged."""
    v = np.asarray(vec, np.float32)
    flat = v.reshape(-1, v.shape[-1])
    hp, r6, th = flat[:, SL_HAND_POS], flat[:, SL_ROT6], flat[:, SL_TIP_H]
    c0, c1 = r6[:, :3], r6[:, 3:]
    c2 = np.cross(c0, c1)
    tip = hp + c0 * th[:, :1] + c1 * th[:, 1:2] + c2 * th[:, 2:3]
    wh, wc = np.maximum(flat[:, SL_WIN_H], 1e-6), flat[:, SL_WIN_C]
    u = (tip[:, 0] - S.PAPER_CENTER[0] - wc[:, 0] + wh[:, 0]) / (2 * wh[:, 0])
    w = (tip[:, 1] - S.PAPER_CENTER[1] - wc[:, 1] + wh[:, 1]) / (2 * wh[:, 1])
    out = np.stack([u * INK_MAP_W, (1 - w) * INK_MAP_H], 1).astype(np.float32)
    return out.reshape(v.shape[:-1] + (2,))


# ---------------------------------------------------------------- env plumbing
def pack_obs(obs, traj=False):
    """Env observation -> what THIS trainer hands its network: (vec float32, maps uint8).

    Model-side by design. The env fixes what an observation CONTAINS; how a model reads
    it is the model's business, and another submission may pack it completely
    differently. Nothing here belongs in env.py.

    The maps stay uint8 in their natural (2, H, W) shape: they are ink on paper, 0..255,
    and there is nothing to gain by widening them to float32 four steps before the
    network - it only costs 4x the rollout memory (23.6 GB against 5.9). They also stay
    at full resolution: max-pooling to 64x64 would put a 6.25 mm pixel in front of a
    policy asked to lay a 1 mm line.
    """
    vec = np.concatenate([obs["position"], obs["tactile"],
                          obs["task"]["vec"], obs["external"]["vec"]]).astype(np.float32)
    if traj:
        # the kept state entries + the trajectory window. The maps are not read, so a 1x1
        # placeholder goes through the rollout buffers instead of 240 000 pixels per step.
        vec = np.concatenate([vec[STATE_KEEP], np.asarray(obs["task"]["traj"], np.float32).ravel()])
        return vec, np.zeros((2, 1, 1), np.uint8)
    maps = np.stack([obs["task"]["map"], obs["external"]["map"]]).astype(np.uint8)
    return vec, maps


class PackObs(gym.ObservationWrapper):
    """DexCraftEnv's four-group observation -> this trainer's (vec, maps) pair."""

    def __init__(self, env, key=None, traj=False):
        super().__init__(env)
        self.traj = bool(traj)
        shape = (2, 1, 1) if traj else (2, INK_MAP_H, INK_MAP_W)
        self.observation_space = gym.spaces.Dict({
            "vec": gym.spaces.Box(-np.inf, np.inf, (PVEC_TRAJ if traj else VEC_DIM,), np.float32),
            "maps": gym.spaces.Box(0, 255, shape, np.uint8)})

    def observation(self, obs):
        vec, maps = pack_obs(obs, self.traj)
        return {"vec": vec, "maps": maps}


class ResidualBase(gym.Wrapper):
    """The policy's action is a residual on a base controller (policies/hybrid.py).

    Sits directly on DexCraftEnv (the base needs the dict observation). The incoming
    action is already scaled by the per-group bounds (ScaleAction above it); it is added
    to the base action and clipped. The residual's size is charged (`cost` x mean r^2, in
    bound-scaled units) so it stays near zero unless it pays.
    """

    def __init__(self, env, base, cost=0.0):
        super().__init__(env)
        self.base, self.cost = base, cost
        self._obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.base.reset({"task_id": info["task_id"], "level": info["level"], "instruction": obs["instruction"],
                         "dt": self.env.unwrapped.dt, "control_hz": self.env.unwrapped.control_hz,
                         "tool": self.env.unwrapped.tool,
                         "act_dim": int(self.env.action_space.shape[0])})
        # with cursor_obs=False the schedule reaches the base only through info, and the
        # reset info carries the first one - without this the base's first action aims at the
        # hand origin
        self.base.observe(info)
        self._obs = obs
        return obs, info

    def step(self, r):
        b = np.asarray(self.base.act(self._obs), np.float32)
        a = np.clip(b + np.asarray(r, np.float32), -1.0, 1.0)
        if a.size > NU:
            # the base declares the end of the episode; a residual that could shift the done
            # entry can learn to quit
            a[NU:] = b[NU:]
        obs, rew, term, trunc, info = self.env.step(a)
        self.base.observe(info)
        self._obs = obs
        if self.cost:
            rew -= self.cost * float(np.mean(np.square(r)))
        return obs, rew, term, trunc, info


class ScaleAction(gym.ActionWrapper):
    """Shrink the policy's action per joint group before it reaches the env.

    The action is joint torque, and full-scale torque on 34 finger dims tears the
    tool out of the hand: measured, a random policy keeps the pen for a whole
    episode only below ~0.05 of full scale on the fingers. Running PPO at that
    std instead makes the log-prob hypersensitive to the mean (34 finger dims,
    KL in the tens per update). Scaling the action decouples the two: the policy
    explores at a healthy std ~0.2 while the fingers still move only 0.05 of
    full scale. The scale is saved with the checkpoint, so a submitted policy is
    simply one that outputs smaller numbers - the env contract is unchanged.
    """

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = np.asarray(scale, dtype=np.float32)

    def action(self, a):
        a = np.asarray(a)
        if a.shape[-1] > len(self.scale):                # a done entry (policy-declared end) is unscaled
            return np.concatenate([np.clip(a[..., :len(self.scale)], -1.0, 1.0) * self.scale, a[..., len(self.scale):]], -1)
        return np.clip(a, -1.0, 1.0) * self.scale


def group_scale(arm, wrist, hand, force=1.0):
    """Per-joint-group action scale. `force` is accepted and ignored: the action is
    a single torque channel now, so there is no separate force channel to scale."""
    return np.concatenate([np.full(N_ARM, arm), np.full(N_WRIST, wrist),
                           np.full(N_HAND, hand)]).astype(np.float32)


def resolve_tasks(spec):
    if spec == "all":
        return list(T.TASKS)
    if spec == "write":                       # the ink tasks that use the pen
        return [t for t in T.TASKS if t.startswith(("L1", "L2"))]
    if spec in T.LEVELS:
        return T.LEVELS[spec]
    return [s for s in spec.split(",") if s]


def make_env(task_id, args, idx, constraint=None):
    def thunk():
        env = DexCraftEnv(task_id, control_mode=args.control_mode, obs_mode=args.obs_mode,
                          control_hz=args.control_hz, perturb=args.perturb,
                          constraint=constraint if constraint is not None else args.constraint,
                          reward=args.reward, assist_steps=args.assist_steps,
                          grip_assist=args.grip_assist, capture_radius=args.capture_radius,
                          reach=args.reach, start_height=tuple(args.start_height) if args.start_height else None,
                          start_wrist=bool(args.start_wrist), force_jitter=args.force_jitter,
                          wrist_teacher=args.wrist_teacher, glyph_scale=args.glyph_scale,
                          yaw_offset=args.yaw_offset,
                          end_mode=args.end_mode, cursor_obs=bool(args.cursor_obs),
                          traj_obs=(TRAJ_K if args.arch in TRAJ_ARCHS else 0), adv_mode=args.adv_mode,
                          seed=args.seed * 1000 + idx)
        if args.base in ("hybrid", "grip", "done"):
            from .policies.hybrid import GripBase, HybridBase
            base = (HybridBase(args.grip_ckpt) if args.base == "hybrid"
                    else GripBase(args.grip_ckpt, fingers=(args.base == "grip")))
            env = ResidualBase(env, base, cost=args.residual_cost)
        env = PackObs(env, args.obs_mode, traj=(args.arch in TRAJ_ARCHS))
        env = ScaleAction(env, group_scale(args.scale_arm, args.scale_wrist, args.scale_hand, args.scale_force))
        env = gym.wrappers.RecordEpisodeStatistics(env)      # logs the RAW return
        env = gym.wrappers.ClipAction(env)
        # Reward normalisation is not optional here: a drop pays the rest of the
        # episode, so raw returns swing by thousands and the advantages become
        # noise (measured: KL 100+ per update, entropy inflating, drop rate 1.00).
        # Dividing by the running std of the discounted return and clipping turns
        # that spike into a single bounded step.
        if args.reward_scale:
            # a FIXED scale. NormalizeReward's running std is one more moving statistic under a
            # warm-started critic (its target's units drift); with a fitted critic the units must hold.
            k = float(args.reward_scale)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r * k, -10, 10)))
        elif args.norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        return env
    return thunk


# ---------------------------------------------------------------- agent
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    """Small conv stack on the two maps, an MLP on the vector groups, concatenated.

    This is only the trainer's choice of architecture - the benchmark fixes what
    the observation contains, not how a model reads it."""

    def __init__(self, vec_dim=VEC_DIM, hidden=256, map_feat=256, use_maps=True):
        super().__init__()
        # `--arch mlp`: the vector only. The executor gets its target through the cursor entries of
        # the vector; the maps carry nothing it needs.
        self.use_maps = bool(use_maps)
        if not self.use_maps:
            self.vec = nn.Sequential(layer_init(nn.Linear(vec_dim, hidden)), nn.Tanh())
            self.out_dim = hidden
            return
        # Five stride-2 stages take the 300x400 sheet down to ~10x13, then an adaptive
        # MAX pool to 4x5 keeps a coarse layout. Max, not average: the target is a thin
        # bright glyph on a mostly empty sheet and averaging washes it out.
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(2, 16, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 5)), nn.Flatten(),
            layer_init(nn.Linear(64 * 20, map_feat)), nn.Tanh())
        self.vec = nn.Sequential(layer_init(nn.Linear(vec_dim, hidden)), nn.Tanh())
        self.out_dim = hidden + map_feat

    def forward(self, vec, maps):
        if not self.use_maps:
            return self.vec(vec)
        # uint8 in, float here: converting is the model's job, done once at the edge
        # of the network rather than four steps earlier in the data pipeline.
        m = maps.to(self.conv[0].weight.dtype) / 255.0
        return torch.cat([self.vec(vec), self.conv(m)], dim=-1)


class XfBranch(nn.Module):
    """Transformer branch with trajectory history. Tokens: a C x C mm crop of
    the two maps centred on the tip as P x P patches, one coarse global-map token (the conv stack), the
    state vector (+ the tip's map position), and the last K state vectors. The crop is what makes "where is
    the stroke relative to my pen" a local pattern: the conv stack alone summarises the sheet into 4 x 5
    cells of ~8 cm."""

    def __init__(self, vec_dim=VEC_DIM, hist=8, crop=96, patch=8, d=256, layers=4, heads=4):
        super().__init__()
        self.hist, self.crop, self.patch = int(hist), int(crop), int(patch)
        n = (self.crop // self.patch) ** 2
        self.patch_proj = nn.Linear(2 * self.patch * self.patch, d)
        self.pos_crop = nn.Parameter(torch.zeros(1, n, d))
        self.glob = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 5)), nn.Flatten(), nn.Linear(64 * 20, d))
        self.vec = nn.Sequential(nn.Linear(vec_dim + 4, d), nn.Tanh(), nn.Linear(d, d))
        self.hvec = nn.Sequential(nn.Linear(vec_dim, d), nn.Tanh(), nn.Linear(d, d))
        self.pos_hist = nn.Parameter(torch.zeros(1, max(self.hist, 1), d))
        self.type_emb = nn.Parameter(torch.zeros(5, d))          # cls, vec, global, history, crop
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        for prm in (self.pos_crop, self.pos_hist, self.type_emb, self.cls):
            nn.init.normal_(prm, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.xf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def crop_tokens(self, m, tip):
        # m float (B, 2, H, W); tip (B, 2) continuous px (x, y) in the env's _to_px units (pixel i spans [i, i+1)).
        # Sample crop x crop pixel centres around the tip; align_corners=False maps px x to 2x/W - 1.
        B, C = m.shape[0], self.crop
        r = torch.arange(C, device=m.device, dtype=m.dtype) - C / 2 + 0.5
        gx = 2 * (tip[:, 0:1] + r[None]) / m.shape[3] - 1
        gy = 2 * (tip[:, 1:2] + r[None]) / m.shape[2] - 1
        grid = torch.stack([gx[:, None, :].expand(B, C, C), gy[:, :, None].expand(B, C, C)], -1)
        win = F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False)   # (B, 2, C, C)
        P = self.patch
        tok = win.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 2 * P * P)
        return self.patch_proj(tok) + self.pos_crop

    def forward(self, vec, maps, tip, hist):
        m = maps.to(self.patch_proj.weight.dtype) / 255.0
        B = vec.shape[0]
        tipn = torch.stack([tip[:, 0] / INK_MAP_W, tip[:, 1] / INK_MAP_H,
                            tip[:, 0] - torch.floor(tip[:, 0]), tip[:, 1] - torch.floor(tip[:, 1])], 1)
        toks = [self.cls.expand(B, -1, -1) + self.type_emb[0],
                (self.vec(torch.cat([vec, tipn], 1)) + self.type_emb[1])[:, None],
                (self.glob(m) + self.type_emb[2])[:, None]]
        if self.hist and hist is not None:
            toks.append(self.hvec(hist) + self.pos_hist[:, :hist.shape[1]] + self.type_emb[3])
        toks.append(self.crop_tokens(m, tip) + self.type_emb[4])
        x = self.xf(torch.cat(toks, 1))
        return self.norm(x[:, 0])


def adapt_checkpoint(ck):
    """A checkpoint trained before the external group grew (the brush block, +7 entries)
    has a narrower first vec layer. Pad it with zero columns and the
    statistics with mean 0 / var 1: the new entries are identically zero on every
    non-brush task, so the loaded policy computes exactly what it did."""
    sd = ck["model"]
    for key in ("enc_pi.vec.0.weight", "enc_v.vec.0.weight"):
        w = sd.get(key)
        if w is not None and w.shape[1] < VEC_DIM:
            n_old = w.shape[1]
            sd[key] = torch.cat([w, torch.zeros(w.shape[0], VEC_DIM - n_old, dtype=w.dtype, device=w.device)], 1)
            for k, fill in (("obs_mean", 0.0), ("obs_var", 1.0)):
                v = ck.get(k)
                if v is not None and len(v) == n_old:
                    ck[k] = np.concatenate([np.asarray(v, np.float64), np.full(VEC_DIM - n_old, fill)])
    return ck


class Agent(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256, min_logstd=-3.5, init_logstd=None, arch="conv", hist=0, crop=96,
                 chunk=1):
        super().__init__()
        # `chunk` > 1 makes the head emit K consecutive actions per forward (action chunking).
        # The policy then decides once per K control steps instead of every 40 ms, which is what
        # compounding error feeds on, and it has to commit to one continuation of a stroke instead
        # of averaging the ones the ink map leaves ambiguous. chunk=1 is the plain one-action head.
        self.arch, self.hist = arch, int(hist)
        self.chunk, self.act_dim = int(chunk), int(act_dim)
        out_dim = int(act_dim) * self.chunk
        vd = vec_dim_for(arch)
        self.enc_pi = Encoder(vec_dim=vd, hidden=hidden, use_maps=(arch not in ("mlp",) + TRAJ_ARCHS))
        self.enc_v = Encoder(vec_dim=vd, hidden=hidden, use_maps=(arch not in ("mlp",) + TRAJ_ARCHS))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.enc_v.out_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.enc_pi.out_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, out_dim), std=0.01))
        if init_logstd is None:
            init_logstd = np.full(act_dim, -1.0)
        ls = np.tile(np.asarray(init_logstd, np.float32), self.chunk) if self.chunk > 1 else np.asarray(init_logstd, np.float32)
        self.actor_logstd = nn.Parameter(torch.as_tensor(ls, dtype=torch.float32).reshape(1, out_dim))
        self.min_logstd = min_logstd
        self.split = arch == "traj2"
        if self.split:
            # ONE network, TWO trunks. The fingers keep the grip's trunk (transplanted whole from
            # the L0 stage); the arm, the wrist and the done entry get their own. On L1_trace_line
            # one shared trunk clones to 0.741 with 8 drops in 24, two trunks to 0.876 with 2 -
            # the arm has to move the trunk to learn and the grip has to hold it still.
            self.arm_trunk = nn.Sequential(
                layer_init(nn.Linear(vd, hidden)), nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)), nn.Tanh())
            self.arm_head = layer_init(nn.Linear(hidden, out_dim), std=0.01)
            m = torch.zeros(act_dim)
            m[:N_ARM + N_WRIST] = 1.0
            m[NU:] = 1.0
            self.register_buffer("arm_mask", m.repeat(self.chunk) if self.chunk > 1 else m)
        if arch == "xf":
            # the transformer branch adds to the conv actor's arm / wrist / done rows only; the fingers
            # stay with the conv actor, which is the grip backbone
            self.xf = XfBranch(hist=self.hist, crop=crop, d=hidden)
            self.xf_head = layer_init(nn.Linear(hidden, out_dim), std=0.01)
            mask = torch.ones(act_dim)
            mask[N_ARM + N_WRIST:NU] = 0.0
            self.register_buffer("xf_mask", mask.repeat(self.chunk) if self.chunk > 1 else mask)

    def mean(self, vec, maps, tip=None, hist=None):
        mu = self.actor_mean(self.enc_pi(vec, maps))
        if self.split:
            mu = (1.0 - self.arm_mask) * mu + self.arm_mask * self.arm_head(self.arm_trunk(vec))
        if self.arch == "xf":
            mu = mu + self.xf_mask * self.xf_head(self.xf(vec, maps, tip, hist))
        return mu

    def value(self, vec, maps, tip=None, hist=None):
        return self.critic(self.enc_v(vec, maps))

    def action_and_value(self, vec, maps, tip=None, hist=None, action=None):
        mean = self.mean(vec, maps, tip, hist)
        logstd = torch.clamp(self.actor_logstd, min=self.min_logstd).expand_as(mean)
        dist = torch.distributions.Normal(mean, logstd.exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(self.enc_v(vec, maps)).squeeze(-1)


# alias used by policies/ppo.py
ActorCritic = Agent


# ---------------------------------------------------------------- training
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        try:                                  # a GPU too new for this torch build
            torch.zeros(8, device="cuda").sum().item()   # (e.g. sm_120) only fails here
            device = torch.device("cuda")
        except RuntimeError as e:
            print(f"CUDA unusable ({str(e).splitlines()[0]}); training on CPU", flush=True)
    print(f"device: {device} | envs: {args.n_envs} | tasks: {args.task}", flush=True)

    task_ids = resolve_tasks(args.task)
    env_tasks = [task_ids[i % len(task_ids)] for i in range(args.n_envs)]
    uniq_tasks = list(dict.fromkeys(task_ids))              # a task listed twice is one task
    env_task_idx = np.array([uniq_tasks.index(t) for t in env_tasks])
    # One policy is scored on all three tracks, so train it on a mix: --constraint
    # "none,none,arm_lock" makes a third of the envs the finger track (arm frozen
    # while the pen is down), which is where the scripted baseline scores 0.13.
    cons = [c for c in args.constraint.split(",") if c]
    env_cons = [cons[i % len(cons)] for i in range(args.n_envs)]
    VecEnv = gym.vector.SyncVectorEnv if args.vec == "sync" else gym.vector.AsyncVectorEnv
    if args.vec == "async" and args.base != "none":
        # the base controller runs torch inside every worker; forked children of a process
        # that has already touched CUDA die silently (broken pipe at _check_spaces) - spawn them
        VecEnv = lambda fns: gym.vector.AsyncVectorEnv(fns, context="spawn")   # noqa: E731
    # Double-buffering: the envs are split into `groups` vector envs stepped out of
    # phase - while one group's workers step, the main process normalises, runs the
    # policy and sends the other group. Same batch (n_envs x rollout), same per-env
    # trajectories, same update; it only removes the idle time of the lock-step loop
    # (measured: 9.4 of 27 ms per 64-env vector step was serial main-process work
    # during which every worker waited).
    G = max(1, min(args.groups, args.n_envs))
    idx_groups = np.array_split(np.arange(args.n_envs), G)
    envs = [VecEnv([make_env(env_tasks[i], args, int(i), env_cons[i]) for i in idx]) for idx in idx_groups]
    obs_dim, act_dim = OBS_DIM, int(envs[0].single_action_space.shape[0])   # 34, or 35 with a done entry
    PV = vec_dim_for(args.arch)                  # packed vector: 217 for `traj`, else the 167 entries

    # Exploration noise is per joint group. delta_pos integrates the action, so
    # hand noise is what makes the tool fall: a random policy keeps the pen for a
    # whole episode only below std ~0.06, while the arm needs real noise to learn
    # to track. The policy can widen these later - they are only the starting point.
    init_logstd = np.concatenate([np.full(N_ARM, args.init_logstd_arm),
                                  np.full(N_WRIST, args.init_logstd_wrist),
                                  np.full(N_HAND, args.init_logstd_hand),
                                  np.full(act_dim - NU, args.init_logstd_hand if args.init_logstd_done is None else args.init_logstd_done)])
    agent = Agent(PV if args.arch in TRAJ_ARCHS else obs_dim, act_dim, args.hidden, args.min_logstd, init_logstd, arch=args.arch, hist=args.hist, crop=args.crop).to(device)
    if args.freeze_grip:
        # PPO trains the arm trunk and the critic only; the grip trunk and its finger rows stay
        # as transplanted. Unfrozen, even tiny updates (sigma 0.005 / 0.05, lr 1e-5) wear the
        # grip down: return 734 -> 27 and drops 0.05 -> 0.23 over 100 iterations.
        params = [p for n, p in agent.named_parameters()
                  if not (n.startswith("enc_pi.") or n.startswith("actor_mean."))]
        print(f"grip frozen: {sum(p.numel() for p in params):,} of {sum(p.numel() for p in agent.parameters()):,} parameters train", flush=True)
    else:
        params = list(agent.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    _pids = {id(p) for p in params}
    actor_params = [p for n, p in agent.named_parameters()         # --full-step rescales these
                    if id(p) in _pids and not (n.startswith("enc_v.") or n.startswith("critic."))]

    from gymnasium.wrappers.normalize import RunningMeanStd
    obs_rms = RunningMeanStd(shape=(PV,))        # the VECTOR only: the maps are
                                                # images, and per-pixel whitening
                                                # destroys what the conv is for

    start_iter = 0
    if args.init:
        ck = adapt_checkpoint(torch.load(args.init, map_location=device, weights_only=False))
        sd = ck["model"]
        n_old = sd["actor_mean.2.bias"].shape[0]
        if n_old < act_dim:                      # a 34-output checkpoint into a trainer with a done entry
            pad = act_dim - n_old
            sd["actor_mean.2.weight"] = torch.cat([sd["actor_mean.2.weight"], torch.zeros(pad, sd["actor_mean.2.weight"].shape[1], device=sd["actor_mean.2.weight"].device)])
            sd["actor_mean.2.bias"] = torch.cat([sd["actor_mean.2.bias"], torch.full((pad,), -1.0, device=sd["actor_mean.2.bias"].device)])
            sd["actor_logstd"] = torch.cat([sd["actor_logstd"], torch.full((1, pad), args.init_logstd_hand, device=sd["actor_logstd"].device)], 1)
        fresh = args.arch == "xf" and not any(k.startswith("xf") for k in sd)
        agent.load_state_dict(sd, strict=not fresh)
        if fresh:
            print("transformer branch starts fresh (the init has none)", flush=True)
        obs_rms.mean = np.asarray(ck["obs_mean"], dtype=np.float64)
        obs_rms.var = np.asarray(ck["obs_var"], dtype=np.float64)
        obs_rms.count = 1e12 if args.freeze_obs_stats else 1e6     # trust the restored statistics
        if args.freeze_obs_stats:
            # a warm start's statistics come from the demonstrations and must not move. At
            # count 1e6 the running mean still travels 12 % of the way to the live batch mean in
            # 8 iterations of 64 envs, and a precise clone degraded to half its completion during
            # the critic warm-up, before the actor had changed at all.
            print("observation statistics frozen", flush=True)
        if args.resume and "iteration" in ck:
            start_iter = int(ck["iteration"])
        print(f"warm-started from {args.init} (iteration {start_iter})", flush=True)
        if args.reset_logstd:
            # RL from a BC clone - the clone carries its init's exploration std (0.3 everywhere);
            # restart every row at the CLI values so exploration is small where the clone is precise
            with torch.no_grad():
                agent.actor_logstd.copy_(torch.as_tensor(init_logstd, dtype=torch.float32, device=device).reshape(1, act_dim))
            print("exploration std reset to the CLI values", flush=True)
    if args.zero_head:
        # start a warm-started policy with chosen output rows silent - the hold backbone's arm and
        # wrist rows were never trained (the environment drove them), and the done entry must start off
        groups = {"arm": range(0, N_ARM), "wrist": range(N_ARM, N_ARM + N_WRIST), "hand": range(N_ARM + N_WRIST, NU),
                  "done": range(NU, act_dim)}
        rows = [i for g in args.zero_head.split(",") for i in groups[g]]
        std0 = {"arm": args.init_logstd_arm, "wrist": args.init_logstd_wrist, "hand": args.init_logstd_hand, "done": args.init_logstd_hand}
        with torch.no_grad():
            agent.actor_mean[-1].weight[rows] = 0.0
            agent.actor_mean[-1].bias[rows] = 0.0
            if "done" in args.zero_head and act_dim > NU:
                agent.actor_mean[-1].bias[NU:] = -1.0
            for g in args.zero_head.split(","):          # silenced rows also restart at the CLI's exploration std
                agent.actor_logstd[0, list(groups[g])] = float(std0[g])
        print(f"zeroed head rows: {args.zero_head} ({len(rows)} rows)", flush=True)
    if args.residual_init == "zero":
        # start exactly at the base controller: the residual mean is zero everywhere
        with torch.no_grad():
            agent.actor_mean[-1].weight.zero_(); agent.actor_mean[-1].bias.zero_()
            if args.arch == "xf":        # the transformer branch adds to arm / wrist / done as well
                agent.xf_head.weight.zero_(); agent.xf_head.bias.zero_()
        print("residual head zeroed", flush=True)

    def norm(o, update=True):
        if update:
            obs_rms.update(o)
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
               for k, v in vars(args).items()} | {"tasks": task_ids, "constraints": cons},
              open(os.path.join(args.out, "args.json"), "w"), indent=1)

    def save(name, iteration):
        torch.save({"model": agent.state_dict(), "obs_mean": obs_rms.mean, "obs_var": obs_rms.var,
                    "obs_mode": args.obs_mode, "control_mode": args.control_mode,
                    "control_hz": args.control_hz, "hidden": args.hidden, "obs_dim": PV if args.arch in TRAJ_ARCHS else obs_dim,
                    "min_logstd": args.min_logstd, "iteration": iteration, "tasks": task_ids,
                    "arch": args.arch, "hist": args.hist, "crop": args.crop,
                    "action_scale": group_scale(args.scale_arm, args.scale_wrist, args.scale_hand, args.scale_force),
                    "env_kwargs": {"reach": args.reach, "start_height": args.start_height, "start_wrist": bool(args.start_wrist),
                                   "force_jitter": args.force_jitter, "glyph_scale": args.glyph_scale,
                                   "yaw_offset": args.yaw_offset, "grip_assist": args.grip_assist,
                                   "constraint": args.constraint, "reward": args.reward},
                    # these ARE this run's own args, so they are trustworthy - without the
                    # stamp a viewer falls back to its defaults and plays the policy in an
                    # env it never trained in
                    "env_kwargs_source": "collection"},
                   os.path.join(args.out, name))

    # Demo anchor: a behaviour-cloning term on teacher demos added to every PPO minibatch
    # (DAPG's recipe). Without it, plain RL polish of a BC writer can forget the writing or
    # drift to loose inking the strict metric rejects, because nothing holds the policy near
    # the demonstrations while the return climbs. The demos
    # are the bc_collect shards (vector every step, task map once, ink as deltas), replayed
    # by bc_train's loader with the trainer's own observation statistics.
    demo, demo_it = None, None
    if args.bc_data:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from bc_train import episodes as _bc_episodes, batches as _bc_batches
        ds = [np.load(p) for p in args.bc_data]
        Vd = np.concatenate([d["vec"] for d in ds]).astype(np.float32)
        if Vd.shape[1] < VEC_DIM:                     # shards from before the vec grew: pad like adapt_checkpoint
            Vd = np.concatenate([Vd, np.zeros((len(Vd), VEC_DIM - Vd.shape[1]), np.float32)], 1)
        _scale = group_scale(args.scale_arm, args.scale_wrist, args.scale_hand, args.scale_force)
        if len(_scale) < act_dim:                     # a done entry is never scaled
            _scale = np.concatenate([_scale, np.ones(act_dim - len(_scale), np.float32)])
        Ad = np.clip(np.clip(np.concatenate([d["act"] for d in ds]), -1, 1) / np.maximum(_scale, 1e-6), -1, 1).astype(np.float32)
        demo = (_bc_episodes(ds), Vd, Ad)
        demo_rng = np.random.default_rng(args.seed)
        print(f"demo anchor: {len(Vd):,} steps from {len(args.bc_data)} shards, coef {args.bc_coef}, batch {args.bc_batch}", flush=True)

        def _demo_batches():
            while True:
                yield from _bc_batches(demo[0], demo[1], demo[2], obs_rms.mean.astype(np.float32),
                                       np.sqrt(obs_rms.var + 1e-8).astype(np.float32), args.bc_batch, demo_rng, device,
                                       hist=args.hist)
        demo_it = _demo_batches()

    batch = args.n_envs * args.rollout
    # Size the minibatch, do NOT fix their count: with a small batch a fixed
    # count (64) gives 16-sample minibatches, and 10 epochs then take 640 gradient
    # steps on 1024 samples - the policy moves so far per update that the KL hits
    # the hundreds and the run never recovers.
    minibatch = min(batch, args.minibatch)
    n_iters = max(1, args.steps // batch)

    OV = torch.zeros((args.rollout, args.n_envs, PV), device=device)
    MH, MW = (1, 1) if args.arch in TRAJ_ARCHS else (INK_MAP_H, INK_MAP_W)   # traj: the maps are a placeholder
    OM = torch.zeros((args.rollout, args.n_envs, 2, MH, MW),
                     dtype=torch.uint8, device=device)
    KH = max(args.hist, 1)
    OT = torch.zeros((args.rollout, args.n_envs, 2), device=device)                 # tip map px
    OH = torch.zeros((args.rollout, args.n_envs, KH, PV), device=device)       # the K vectors before each step
    A = torch.zeros((args.rollout, args.n_envs, act_dim), device=device)
    LP = torch.zeros((args.rollout, args.n_envs), device=device)
    R = torch.zeros((args.rollout, args.n_envs), device=device)
    D = torch.zeros((args.rollout, args.n_envs), device=device)
    V = torch.zeros((args.rollout, args.n_envs), device=device)

    next_vec = torch.zeros((args.n_envs, PV), device=device)
    next_maps = torch.zeros((args.n_envs, 2, MH, MW), dtype=torch.uint8, device=device)
    next_done = torch.zeros(args.n_envs, device=device)
    idx_t = [torch.as_tensor(idx, device=device) for idx in idx_groups]

    next_tip = torch.zeros((args.n_envs, 2), device=device)
    cur_hist = torch.zeros((args.n_envs, KH, PV), device=device)

    def store(ig, raw, done=None):
        if args.hist:
            # the incoming observation's history = the K vectors before it; a fresh episode has none
            cur_hist[ig] = torch.roll(cur_hist[ig], -1, dims=1)
            cur_hist[ig, -1] = next_vec[ig]
            if done is not None and np.any(done):
                cur_hist[ig[torch.as_tensor(done, device=device)]] = 0.0
        next_vec[ig] = torch.tensor(norm(raw["vec"]), device=device)
        next_tip[ig] = torch.as_tensor(tip_px_from_vec(raw["vec"]), device=device)
        next_maps[ig] = torch.as_tensor(np.asarray(raw["maps"]), device=device)

    for g, idx in enumerate(idx_groups):
        raw, _ = envs[g].reset(seed=args.seed + g)
        store(idx_t[g], raw)
    hist, t0, gstep = [], time.time(), start_iter * batch

    def act_and_send(step, g):
        """Policy on group g's current observation -> buffers[step]; send it stepping."""
        ig = idx_t[g]
        OV[step, ig], OM[step, ig], D[step, ig] = next_vec[ig], next_maps[ig], next_done[ig]
        OT[step, ig], OH[step, ig] = next_tip[ig], cur_hist[ig]
        with torch.no_grad():
            a, lp, _, v = agent.action_and_value(next_vec[ig], next_maps[ig], next_tip[ig], cur_hist[ig])
        A[step, ig], LP[step, ig], V[step, ig] = a, lp, v
        envs[g].step_async(a.cpu().numpy())

    for it in range(start_iter, n_iters):
        if args.anneal_lr:
            for pg in opt.param_groups:
                pg["lr"] = args.lr * (1.0 - it / n_iters)
        for g in range(G):
            act_and_send(0, g)
        for step in range(args.rollout):
            for g in range(G):
                idx, ig = idx_groups[g], idx_t[g]
                raw, rew, term, trunc, infos = envs[g].step_wait()      # the other group is stepping meanwhile
                R[step, ig] = torch.tensor(rew, dtype=torch.float32, device=device)
                dn = np.logical_or(term, trunc)
                store(ig, raw, dn)
                next_done[ig] = torch.tensor(dn, dtype=torch.float32, device=device)
                gstep += len(idx)
                for i, fi in enumerate(infos.get("final_info") if infos.get("final_info") is not None else []):
                    if fi and "episode" in fi:
                        hist.append((float(np.ravel(fi["episode"]["r"])[0]), float(np.ravel(fi["episode"]["l"])[0]),
                                     float(fi.get("progress", 0.0)), float(bool(fi.get("dropped", False))),
                                     float(fi.get("finger_share") or 0.0),
                                     float(fi["done_time"] is not None) if "done_time" in fi else 0.0,
                                     float(env_task_idx[idx[i]])))
                if step + 1 < args.rollout:
                    act_and_send(step + 1, g)

        with torch.no_grad():
            next_v = agent.value(next_vec, next_maps).squeeze(-1)
            adv = torch.zeros_like(R)
            gae = 0.0
            for t in reversed(range(args.rollout)):
                nonterm = 1.0 - (next_done if t == args.rollout - 1 else D[t + 1])
                nv = next_v if t == args.rollout - 1 else V[t + 1]
                delta = R[t] + args.gamma * nv * nonterm - V[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + V

        b_ov = OV.reshape(-1, PV)
        b_om = OM.reshape(-1, 2, MH, MW)
        b_a = A.reshape(-1, act_dim)
        b_ot, b_oh = OT.reshape(-1, 2), OH.reshape(-1, KH, PV)
        b_lp, b_adv, b_ret, b_v = LP.reshape(-1), adv.reshape(-1), ret.reshape(-1), V.reshape(-1)
        vy = b_ret.var()
        ev = float(1.0 - (b_ret - b_v).var() / vy) if vy > 0 else 0.0   # how much of the return the critic explains
        idx = np.arange(batch)
        kl, stop, n_upd, n_vupd = 0.0, False, 0, 0
        bc_sum, bc_n = 0.0, 0
        if args.full_step:
            # The actor takes ONE step per pass, from a gradient estimated on the WHOLE rollout,
            # and the trust region is spent on that single direction. At sigma 0.005 one Adam step
            # on a 1024-sample minibatch already exceeds KL 0.01, so with per-minibatch steps the
            # brake fires after a few of 192 minibatches and the actor's direction is estimated
            # from ~1/48 of the data. Averaging over the whole batch cuts the gradient
            # noise by ~sqrt(48); the rescale below then places the step exactly on the KL budget
            # (KL is quadratic in the step, so one correction lands it) instead of overshooting into
            # the brake. This is the TRPO arrangement, with Adam's scaling kept as the metric.
            b_advn = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            # The KL the rescale below aims at has to be measured on a sample of the WHOLE batch.
            # `idx[:8192]` is not one: the buffer is flattened (t, env), so it is the first 8192 / n_envs
            # timesteps of the rollout - 16 of 2048 at 512 envs, 0.78 % of the data and all of it from the
            # opening of an episode - and steps sized on it land far outside the trust region.
            kidx = np.random.choice(batch, min(16384, batch), replace=False)

            def _kl_now():
                with torch.no_grad():
                    _, lp, _, _ = agent.action_and_value(b_ov[kidx], b_om[kidx], b_ot[kidx], b_oh[kidx], b_a[kidx])
                    lr_ = lp - b_lp[kidx]
                    return float(((lr_.exp() - 1) - lr_).mean().item())

            for _ in range(0 if it - start_iter < args.critic_warmup else max(int(args.full_step), 1)):
                opt.zero_grad()
                for s in range(0, batch, minibatch):
                    mb = idx[s:s + minibatch]
                    _, lp, ent, _ = agent.action_and_value(b_ov[mb], b_om[mb], b_ot[mb], b_oh[mb], b_a[mb])
                    ratio = (lp - b_lp[mb]).exp()
                    madv = b_advn[mb]
                    pg = torch.max(-madv * ratio, -madv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    ((pg - args.ent * ent.mean()) * len(mb) / batch).backward()   # accumulate, weighted
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                p0 = [p.detach().clone() for p in actor_params]
                opt.step()
                for _ in range(3):
                    kl = _kl_now()
                    if kl <= args.target_kl or kl <= 1e-12:
                        break
                    sc = (args.target_kl / kl) ** 0.5
                    with torch.no_grad():
                        for p, q in zip(actor_params, p0):
                            p.copy_(q + (p - q) * sc)
                n_upd += 1
                if kl >= 0.999 * args.target_kl:               # the budget is spent
                    break
            for _ in range(args.epochs):                       # the critic, on every minibatch
                np.random.shuffle(idx)
                for s in range(0, batch, minibatch):
                    mb = idx[s:s + minibatch]
                    v = agent.value(b_ov[mb], b_om[mb], b_ot[mb], b_oh[mb]).squeeze(-1)
                    v_clip = b_v[mb] + (v - b_v[mb]).clamp(-args.clip, args.clip)
                    vloss = 0.5 * torch.max((v - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                    opt.zero_grad()
                    (args.vf * vloss).backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    opt.step()
                    n_vupd += 1
        for _ in range(0 if args.full_step else args.epochs):
            if stop and not args.critic_full:
                break
            np.random.shuffle(idx)
            for s in range(0, batch, minibatch):
                mb = idx[s:s + minibatch]
                _, lp, ent, v = agent.action_and_value(b_ov[mb], b_om[mb], b_ot[mb], b_oh[mb], b_a[mb])
                logratio = lp - b_lp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    kl_mb = ((ratio - 1) - logratio).mean().item()
                if not stop:
                    kl = kl_mb
                madv = b_adv[mb]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_clip = b_v[mb] + (v - b_v[mb]).clamp(-args.clip, args.clip)
                vloss = 0.5 * torch.max((v - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                n_vupd += 1
                if it - start_iter < args.critic_warmup or stop:
                    # value only: during the warm-up (a warm-started actor is not moved by an untrained
                    # critic) and, with --critic-full, after the KL brake, so the critic keeps
                    # training on every minibatch instead of the few before the brake fires
                    loss = args.vf * vloss
                else:
                    loss = pg + args.vf * vloss - args.ent * ent.mean()
                if demo_it is not None:
                    xd, md, td, hd, ad = next(demo_it)
                    pred = agent.mean(xd, md, td, hd)          # the whole actor (conv + transformer branch)
                    wd = torch.ones_like(pred); wd[:, :N_ARM] = 4.0      # as bc_train: arm steps integrate
                    bcl = (wd * (pred - ad) ** 2).mean()
                    loss = loss + args.bc_coef * bcl
                    bc_sum += bcl.item(); bc_n += 1
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                if not stop:
                    n_upd += 1
                # checked per minibatch, not per epoch: with a near-deterministic
                # policy a single epoch is already far past the trust region
                if not stop and args.target_kl is not None and kl > args.target_kl:
                    stop = True
                    if not args.critic_full:
                        break

        if hist:
            h = np.array(hist[-100:])
            sps = (gstep - start_iter * batch) / (time.time() - t0)
            print(f"it {it:>5d}  step {gstep:>10,d}  ret {h[:, 0].mean():9.1f}  len {h[:, 1].mean():6.0f}  "
                  f"prog {h[:, 2].mean():.2f}  drop {h[:, 3].mean():.2f}  finger {h[:, 4].mean():.2f}  "
                  f"done {h[:, 5].mean():.2f}  kl {kl:.4f}/{n_upd}v{n_vupd}  ev {ev:+.2f}  std {agent.actor_logstd.exp().mean():.3f}  "
                  + (f"bc {bc_sum / max(bc_n, 1):.4f}  " if demo_it is not None else "") + f"{sps:.0f} sps", flush=True)
            if len(uniq_tasks) > 1 and it % 10 == 0:
                h = np.array(hist[-40 * len(uniq_tasks):])
                print("       " + "  ".join(f"{t.split('_', 1)[1]} p{h[h[:, 6] == k, 2].mean():.2f}/d{h[h[:, 6] == k, 3].mean():.2f}"
                                            for k, t in enumerate(uniq_tasks) if (h[:, 6] == k).any()), flush=True)
        if args.save_freq and it % args.save_freq == 0 and it > start_iter:
            save(f"agent_{it:06d}.pt", it)
            save("agent.pt", it)
    save("agent.pt", n_iters)
    for e in envs:
        e.close()
    print(f"done: {gstep:,} steps in {(time.time() - t0) / 60:.1f} min -> {args.out}/agent.pt", flush=True)
    return agent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="L2_write_word", help="task id, comma list, level (L2), 'write' or 'all'")
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--vec", default="async", choices=["sync", "async"])
    ap.add_argument("--groups", type=int, default=2, help="vector-env groups stepped out of phase (double-buffering); 1 = lock-step")
    ap.add_argument("--obs-mode", default="state", help="kept for the env; the policy always sees all four groups")
    ap.add_argument("--control-mode", default="torque")
    ap.add_argument("--control-hz", type=int, default=50, help="50 Hz: half the physics of 100 Hz, still deployable")
    ap.add_argument("--reward", default="default", help="env.REWARD_PRESETS: default | fingers | fingers_hard")
    ap.add_argument("--perturb", default="none")
    ap.add_argument("--capture-radius", type=float, default=None,
                    help="curriculum: widen the waypoint capture radius [m]. The task default "
                         "is 0.0025; 0.008 is a good first rung, to get a non-zero completion "
                         "before tightening it back.")
    ap.add_argument("--grip-assist", type=float, default=0.0,
                    help="curriculum: fraction of a nominal grip torque the env supplies. "
                         "1.0 = the default grip is held for you, 0.0 = the benchmark contract. "
                         "At 0 a zero-torque hand loses the tool in 0.6 s, so stage 0 needs this.")
    ap.add_argument("--assist-steps", type=int, default=0,
                    help="curriculum: hold the tool in the hand for the first N control steps. "
                         "Under a torque action a random policy drops it within ~2 s at any "
                         "action scale, so stage 0 needs this to have anything to learn from.")
    ap.add_argument("--constraint", default="none")
    ap.add_argument("--reach", type=float, default=None, help="cap the target extent [m] (scaled down)")
    ap.add_argument("--start-height", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the arm places the tip over the target this high above the sheet at reset")
    ap.add_argument("--start-wrist", type=int, default=0, help="randomise the wrist start pose too")
    ap.add_argument("--force-jitter", type=float, default=0.0,
                    help="sim2real: per-episode gain error on the sensed pen-tip force (0.2 = +-20%%)")
    ap.add_argument("--wrist-teacher", type=float, default=0.0,
                    help="blend of the wrist-IK teacher into the wrist duty (0 = off)")
    ap.add_argument("--yaw-offset", type=float, default=None,
                    help="yaw the writing pose about world z [rad]; pi = pen tip forward (human pose)")
    ap.add_argument("--glyph-scale", type=float, default=1.0,
                    help="scale the sampled glyphs about their centre (before the reach cap)")
    ap.add_argument("--norm-reward", type=int, default=1, help="NormalizeReward + clip to [-10, 10]")
    ap.add_argument("--reward-scale", type=float, default=0.0, help="fixed reward scale (replaces the running normaliser when set)")
    ap.add_argument("--critic-full", type=int, default=0, help="the KL brake stops the actor only; the critic trains every minibatch")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--full-step", type=int, default=0,
                    help="actor takes N full-batch steps per iteration, each rescaled onto the KL "
                         "budget, instead of minibatch steps stopped by the brake (0 = minibatch steps)")
    ap.add_argument("--minibatch", type=int, default=512, help="minibatch SIZE (not count)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anneal-lr", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0, help="min-logstd is the anti-collapse guard, not this")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--min-logstd", type=float, default=-3.0, help="floor on the policy log-std")
    ap.add_argument("--init-logstd-arm", type=float, default=-1.6, help="initial log-std, 6 UR5e joints")
    ap.add_argument("--init-logstd-wrist", type=float, default=-1.6, help="initial log-std, 3 wrist joints")
    ap.add_argument("--init-logstd-hand", type=float, default=-1.6, help="initial log-std, 34 finger joints")
    ap.add_argument("--scale-arm", type=float, default=1.0, help="action scale, 6 UR5e joints")
    ap.add_argument("--scale-wrist", type=float, default=0.5, help="action scale, 3 wrist joints")
    ap.add_argument("--scale-hand", type=float, default=0.25, help="action scale, 34 finger joints")
    ap.add_argument("--scale-force", type=float, default=1.0, help="(unused: the action is one torque channel)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--cuda", type=int, default=1)
    ap.add_argument("--init", default=None, help="checkpoint to warm-start from")
    ap.add_argument("--end-mode", default="fixed", choices=["fixed", "policy"], help="policy: the policy declares the end (35th action), clock to 2x")
    ap.add_argument("--cursor-obs", type=int, default=1, help="0: the target's schedule is not in the observation")
    ap.add_argument("--base", default="none", choices=["none", "hybrid", "grip", "done"],
                    help="'hybrid' = the policy is a residual on IK arm + learned grip (policies/hybrid.py); "
                         "--scale-arm/wrist/hand are then the residual bounds")
    ap.add_argument("--grip-ckpt", default="checkpoints/grasp_v10_format.pt", help="the grip base for --base hybrid")
    ap.add_argument("--residual-cost", type=float, default=0.0, help="reward charge x mean(residual^2)")
    ap.add_argument("--residual-init", default="none", choices=["none", "zero"], help="zero the actor head at start")
    ap.add_argument("--zero-head", default="", help="comma list of output groups to silence at start: arm,wrist,hand,done")
    ap.add_argument("--reset-logstd", type=int, default=0, help="1: a warm-started policy restarts at the CLI exploration std on every row (RL from a BC clone)")
    ap.add_argument("--init-logstd-done", type=float, default=None, help="exploration log-std of the done entry (default: the hand's)")
    ap.add_argument("--freeze-grip", type=int, default=0, help="do not train the grip trunk / finger rows")
    ap.add_argument("--freeze-obs-stats", type=int, default=0, help="keep --init's observation statistics fixed (warm starts)")
    ap.add_argument("--critic-warmup", type=int, default=0, help="iterations trained on the value loss only before the actor moves (RL from a BC clone)")
    ap.add_argument("--arch", default="conv", choices=["conv", "xf", "mlp", "traj", "traj2"], help="xf: + transformer over a tip-centred map crop, the state and its history")
    ap.add_argument("--adv-mode", default="ink", choices=["ink", "tip"], help="how the timeline pointer advances - on ink within adv_tol, or on the tip being there and touching")
    ap.add_argument("--hist", type=int, default=0, help="xf: number of past observation vectors in the history tokens")
    ap.add_argument("--crop", type=int, default=96, help="xf: side of the tip-centred map crop [px = mm]")
    ap.add_argument("--bc-data", nargs="*", default=None, help="bc_collect shards: demo-anchored PPO (BC term on every minibatch)")
    ap.add_argument("--bc-coef", type=float, default=0.0, help="weight of the demo BC term")
    ap.add_argument("--bc-batch", type=int, default=256, help="demo frames per PPO minibatch")
    ap.add_argument("--resume", action="store_true", help="also restore the iteration counter from --init")
    ap.add_argument("--save-freq", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/ppo")
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
