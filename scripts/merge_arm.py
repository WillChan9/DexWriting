"""Merge the grasp network and the arm/wrist BC clone into one two-branch network.

    PYTHONPATH=. python scripts/merge_arm.py checkpoints/two_branch_init.pt runs/bc_s0.pt runs/bc_s0_merged.pt

The two-trunk transplant (grip trunk = stage 0) takes its arm trunk and arm head from the
single-trunk clone that was trained from scratch on the same demonstrations: enc_pi.vec ->
arm_trunk.0, actor_mean.0 -> arm_trunk.2, actor_mean.2's arm / wrist / done rows -> arm_head.
The clone's first layer is re-normalised to the transplant's statistics exactly (`renorm`),
so the merged network computes what the two networks compute side by side.
"""
import sys

import numpy as np
import torch

from dexcraft.env import N_ARM, N_WRIST, NU


def renorm(ckpt, mean, var, out=None):
    """Rewrite a checkpoint for new observation statistics, exactly."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    m0, v0 = np.asarray(ck["obs_mean"], np.float64), np.asarray(ck["obs_var"], np.float64)
    s0, s1 = np.sqrt(v0 + 1e-8), np.sqrt(np.asarray(var, np.float64) + 1e-8)
    r = torch.as_tensor(s1 / s0, dtype=torch.float32)
    d = torch.as_tensor((np.asarray(mean, np.float64) - m0) / s0, dtype=torch.float32)
    for enc in ("enc_pi", "enc_v"):
        W = sd[f"{enc}.vec.0.weight"]
        sd[f"{enc}.vec.0.bias"] = sd[f"{enc}.vec.0.bias"] + W @ d
        sd[f"{enc}.vec.0.weight"] = W * r
    ck["obs_mean"], ck["obs_var"] = np.asarray(mean, np.float64), np.asarray(var, np.float64)
    if out:
        torch.save(ck, out)
    return ck


INIT, SCRATCH, OUT = sys.argv[1:4]
a = torch.load(INIT, map_location="cpu", weights_only=False)
# the clone's statistics differ in the last digits (float32 vs float64 over the same frames):
# rewrite its first layer for the transplant's statistics exactly, then take it
b = renorm(SCRATCH, a["obs_mean"], a["obs_var"])
sa, sb = a["model"], b["model"]
sa["arm_trunk.0.weight"], sa["arm_trunk.0.bias"] = sb["enc_pi.vec.0.weight"], sb["enc_pi.vec.0.bias"]
sa["arm_trunk.2.weight"], sa["arm_trunk.2.bias"] = sb["actor_mean.0.weight"], sb["actor_mean.0.bias"]
rows = list(range(N_ARM + N_WRIST)) + list(range(NU, sb["actor_mean.2.bias"].shape[0]))
sa["arm_head.weight"][rows] = sb["actor_mean.2.weight"][rows]
sa["arm_head.bias"][rows] = sb["actor_mean.2.bias"][rows]
sa["arm_head.bias"][NU] = -1.0                                   # the done row stays off
a["iteration"] = f"merge({INIT}, {SCRATCH})"
torch.save(a, OUT)
print(f"saved {OUT}: arm trunk + head from {SCRATCH}, grip trunk from {INIT}")
