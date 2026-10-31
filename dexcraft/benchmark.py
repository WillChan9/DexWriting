"""What an official DexCraft number means.

Two things live here: the env every run is scored in, and the bar each level has to
clear. Keeping them in one place makes them part of the benchmark rather than per-run
settings, and makes a deviation something the result file has to admit to.

Some env options make the task much easier, for example:
  * `grip_assist=0.1` - a PD torque on the hand and wrist joints holding the captured grip
    pose. It is the difference between L0_scribble scoring 1.00 and scoring 0.62.
  * `reach=0.06` - shrinks any target longer than 60 mm to fit. L1_trace_line is sampled
    at 60-120 mm, so up to half the L1 set would be scaled down before the policy sees it.
"""
import numpy as np

# The env kwargs an official run is scored at. These are also the DexCraftEnv defaults -
# naming them here is what makes an override reportable instead of silent. `perturb` and
# `constraint` are absent on purpose: the TRACK sets those.
CONTRACT = dict(grip_assist=0.0, assist_steps=0, force_gain=1.0, force_jitter=0.0,
                reach=None, glyph_scale=1.0, capture_radius=None, wrist_teacher=0.0,
                brush_cfg="wolf", paper_cfg="shuxuan")     # L3: which brush / paper preset (brush.py)

# A policy may choose these: they are how a policy sets ITSELF up (where the hand
# starts, which way the pen points, how fast it is polled), not how hard the task is.
SETUP = ("start_height", "start_wrist", "yaw_offset", "roll_offset", "goal_on_paper",
         "control_mode", "control_hz", "obs_mode", "cameras", "image_size", "reward")

# Per-level bars. `success` and `completion` are means over the level's tasks; `drop_rate`
# is the fraction of episodes that lost the tool, and it is a CEILING.
#   L0 is a keep-the-tool rung, so it is graded on success and may not drop at all.
#   L1/L2 are graded on completion, the order-aware 3 mm metric - NOT on `coverage`, the
#   6 mm order-free variant the training log prints as `prog` (they differ by 2x).
GATES = {
    "L0": dict(metric="success",    bar=0.90, drop_rate=0.0),
    "L1": dict(metric="completion", bar=0.50, drop_rate=0.125),
    "L2": dict(metric="completion", bar=0.50, drop_rate=0.25),
    "L3": dict(metric="completion", bar=0.40, drop_rate=0.25),
    "L4": dict(metric="completion", bar=0.40, drop_rate=0.25),
    "L5": dict(metric="completion", bar=0.40, drop_rate=0.25),     # paint: colour, brushes, wells
    "L6": dict(metric="completion", bar=0.30, drop_rate=0.35),     # chopsticks
}


# `perturb` and `constraint` belong to the TRACK (evaluate.TRACKS). Passing either through
# --env-kwargs overrides the track for every episode of the run. For example the training
# preset `mini_arm_lock` adds a ring/pinky servo (idle_lock) the nominal track does not
# provide, which cuts drops on hold_moving / scribble from 14/24 and 13/24 to 2/24 and 0/24.
# An override is therefore a deviation the result file has to admit to, like the contract
# terms above.
TRACK_KEYS = ("perturb", "constraint")


def deviations(env_kwargs):
    """Which contract terms this run overrode, and to what. Empty dict == on contract.
    Track keys passed through env_kwargs are reported as {"contract": "<track>", ...}."""
    out = {}
    for k, v in CONTRACT.items():
        if env_kwargs and k in env_kwargs and env_kwargs[k] != v:
            out[k] = {"contract": v, "ran_with": env_kwargs[k]}
    for k in TRACK_KEYS:
        if env_kwargs and k in env_kwargs:
            out[k] = {"contract": "<set by the track>", "ran_with": env_kwargs[k]}
    return out


def gate(episodes, gates=None):
    """Per-level, per-task verdict over an evaluate.py episode list.

    Returns {level: {"pass": bool, "metric": str, "bar": float, "tasks": {task: {...}}}}.
    A level with no episodes is absent rather than passing by default."""
    gates = gates or GATES
    by = {}
    for e in episodes:
        by.setdefault(e["task_id"][:2], {}).setdefault(e["task_id"], []).append(e["metrics"])
    out = {}
    for lvl, tasks in sorted(by.items()):
        g = gates.get(lvl)
        if g is None:
            continue
        rows, ok = {}, True
        for t, ms in sorted(tasks.items()):
            vals = [m.get(g["metric"]) for m in ms if m.get(g["metric"]) is not None]
            score = float(np.mean(vals)) if vals else 0.0
            drops = sum(bool(m.get("dropped")) for m in ms)
            rate = drops / len(ms)
            passed = score >= g["bar"] and rate <= g["drop_rate"]
            rows[t] = dict(score=score, bar=g["bar"], drops=drops, n=len(ms),
                           drop_rate=rate, drop_ceiling=g["drop_rate"], passed=passed)
            ok = ok and passed
        out[lvl] = dict(**{"pass": ok}, metric=g["metric"], bar=g["bar"], tasks=rows)
    return out


def format_gate(verdict, deviation=None):
    """The verdict as the run logs should read it."""
    lines = []
    if deviation:
        lines.append("!! OFF CONTRACT - these numbers are not benchmark numbers:")
        for k, v in sorted(deviation.items()):
            lines.append(f"     {k}: contract {v['contract']!r}, ran with {v['ran_with']!r}")
    for lvl, v in sorted(verdict.items()):
        lines.append(f"  {lvl}  {'PASS' if v['pass'] else 'FAIL'}   "
                     f"({v['metric']} >= {v['bar']:.2f}, drops <= "
                     f"{GATES[lvl]['drop_rate']:.0%})")
        for t, r in v["tasks"].items():
            lines.append(f"     {'ok  ' if r['passed'] else 'FAIL'} {t:20s} "
                         f"{'succ' if v['metric'] == 'success' else 'compl'} {r['score']:.2f}"
                         f"  drops {r['drops']}/{r['n']}")
    return "\n".join(lines)
