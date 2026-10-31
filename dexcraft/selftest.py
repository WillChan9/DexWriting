"""Check the install end to end, so a failure points at one thing.

    python -m dexcraft.selftest          # ~30 s
    python -m dexcraft.selftest --full   # also a short PPO update and a recorded shard
"""
import argparse
import sys
import time

import numpy as np


def _ok(name, detail=""):
    print(f"  \033[32mok\033[0m   {name}" + (f"  ({detail})" if detail else ""), flush=True)


def _fail(name, err):
    print(f"  \033[31mFAIL\033[0m {name}: {err}", flush=True)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run a PPO update and record a shard")
    args = ap.parse_args(argv)
    bad = 0
    print("DexCraft selftest")

    try:
        import mujoco, gymnasium
        _ok("dependencies", f"mujoco {mujoco.__version__}, gymnasium {gymnasium.__version__}")
    except Exception as e:
        return _fail("dependencies", e)

    try:
        from .tasks import TASKS
        from .env import DexCraftEnv, GROUP_DIM, NU
        _ok("task suite", f"{len(TASKS)} families")
    except Exception as e:
        return _fail("import", e)

    try:
        import gymnasium as gym
        import dexcraft  # noqa: F401  (registers the ids)
        env = gym.make("DexCraft/L1_trace_line-v1")
        env.reset(seed=0)
        env.close()
        _ok("gym registration", "gym.make('DexCraft/<task>-v1')")
    except Exception as e:
        bad += _fail("gym registration", e)

    try:
        from . import brush
        n_bad = brush.selftest(out_dir=None, with_env=False)      # T1 T2 T4 T5; T3 via `python -m dexcraft.brush`
        if n_bad:
            raise AssertionError(f"{n_bad} brush acceptance test(s) failed")
        _ok("brush tip model", "spec T1 T2 T4 T5 (T3 needs `python -m dexcraft.brush`)")
    except Exception as e:
        bad += _fail("brush tip model", e)

    try:
        env = DexCraftEnv("L2_write_word", control_hz=25)
        obs, info = env.reset(seed=0)
        for g, n in GROUP_DIM.items():
            v = obs[g]["vec"] if isinstance(obs[g], dict) else obs[g]
            assert v.shape == (n,), f"{g}: {v.shape} != {(n,)}"
        from .env import INK_MAP_H, INK_MAP_W, INK_MAP_MM_PER_PX
        import dexcraft.scene as _S
        assert obs["task"]["map"].shape == obs["external"]["map"].shape == (INK_MAP_H, INK_MAP_W)
        # the map must BE the sheet: same rectangle, isotropic, one pixel per mm
        assert abs(INK_MAP_W * INK_MAP_MM_PER_PX / 1000 - 2 * _S.PAPER_HALF[0]) < 1e-9
        assert abs(INK_MAP_H * INK_MAP_MM_PER_PX / 1000 - 2 * _S.PAPER_HALF[1]) < 1e-9
        assert int((obs["task"]["map"] > 0).sum()) > 0, "the task map is empty"
        _ok("observation", " + ".join(f"{g} {n}" for g, n in GROUP_DIM.items())
            + f" + 2 maps {INK_MAP_W}x{INK_MAP_H} @ {INK_MAP_MM_PER_PX} mm/px")
    except Exception as e:
        bad += _fail("observation", e)

    try:
        t0 = time.time()
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(100):
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                env.reset()
        _ok("stepping", f"{100 / (time.time() - t0):.0f} control steps/s/core, action {env.action_space.shape[0]}")
        env.close()
    except Exception as e:
        bad += _fail("stepping", e)

    try:
        from .policies.scripted import make_policy
        from .metrics import episode_metrics
        env = DexCraftEnv("L1_trace_line", control_hz=25)
        pol = make_policy()
        obs, info = env.reset(seed=1)
        pol.reset({"task_id": info["task_id"], "level": info["level"],
                   "instruction": obs["instruction"], "dt": env.dt,
                   "control_hz": 25, "tool": env.tool})
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            pol.observe(info)
            done = term or trunc
        m = episode_metrics(env.episode_record())
        env.close()
        assert m["completion"] > 0.5, f"baseline only reached {m['completion']:.2f}"
        _ok("scripted baseline", f"completion {m['completion']:.2f}, force MAE "
                                 f"{m['force_mae_N']:.2f} N" if m["force_mae_N"] else "")
    except Exception as e:
        bad += _fail("scripted baseline", e)

    try:
        from .planning import map_to_strokes
        env = DexCraftEnv("L2_write_word", control_hz=25)
        obs, _ = env.reset(seed=0)
        st = map_to_strokes(obs["task"]["map"], tool="pen")
        env.close()
        assert st, "decoded no strokes"
        _ok("task-map decoder", f"{len(st)} strokes, {sum(len(s.xy) for s in st)} points")
    except Exception as e:
        bad += _fail("task-map decoder", e)

    try:
        import torch  # noqa: F401
        from .train_ppo import VEC_DIM, ACT_DIM, Agent, pack_obs
        env = DexCraftEnv("L1_trace_line", control_hz=25)
        obs, _ = env.reset(seed=0)
        vec, maps = pack_obs(obs)
        assert vec.shape == (VEC_DIM,) and vec.dtype == np.float32, (vec.shape, vec.dtype)
        assert maps.shape == (2, INK_MAP_H, INK_MAP_W) and maps.dtype == np.uint8, maps.dtype
        ag = Agent()
        a, lp, ent, v = ag.action_and_value(torch.as_tensor(vec).unsqueeze(0),
                                            torch.as_tensor(maps).unsqueeze(0))
        env.close()
        assert a.shape == (1, ACT_DIM)
        _ok("trainer", f"vec {VEC_DIM} + maps 2x{INK_MAP_H}x{INK_MAP_W} uint8 -> action {ACT_DIM}, "
                       f"{sum(p.numel() for p in ag.parameters()):,} params")
    except ImportError:
        print("  \033[33mskip\033[0m trainer (torch not installed: pip install 'dexcraft[train]')")
    except Exception as e:
        bad += _fail("trainer", e)

    if args.full:
        try:
            from .train_ppo import main as train_main
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                train_main(["--task", "L1_trace_line", "--steps", "2048", "--n-envs", "2",
                            "--rollout", "128", "--minibatch", "128", "--epochs", "1",
                            "--vec", "sync", "--cuda", "0", "--save-freq", "0", "--out", d])
            _ok("PPO update", "2k steps ran")
        except Exception as e:
            bad += _fail("PPO update", e)
        try:
            from .record import main as rec_main
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                rec_main(["--tasks", "L1_trace_line", "--episodes", "2", "--out", d,
                          "--shard-episodes", "2", "--verbose", "0"])
            _ok("dataset recording", "2 episodes")
        except Exception as e:
            bad += _fail("dataset recording", e)

    print("\nall good — you can train." if not bad else f"\n{bad} check(s) failed.")
    return bad


if __name__ == "__main__":
    sys.exit(main())
