"""DexCraft: a fine tool-use benchmark for high-DoF dexterous hands.

    import gymnasium as gym, dexcraft
    env = gym.make("DexCraft/L2_write_word-v1")          # every task id is registered
    obs, info = env.reset(seed=0)

or construct it directly for full control over the options:

    from dexcraft.env import DexCraftEnv
    env = DexCraftEnv("L2_write_word", control_hz=25, perturb="mild", constraint="arm_lock")

`python -m dexcraft.selftest` checks the install end to end.
"""
__version__ = "0.2.0"

from gymnasium.envs.registration import register as _register


def _register_tasks():
    from . import tasks as _tasks
    for _tid in _tasks.TASKS:
        _register(id=f"DexCraft/{_tid}-v1",
                  entry_point="dexcraft.env:DexCraftEnv",
                  kwargs={"task_id": _tid},
                  max_episode_steps=None,          # the env ends the episode itself
                  disable_env_checker=True)


try:                                               # importing dexcraft registers the tasks
    _register_tasks()
except Exception:                                  # a broken install should fail on use, not on import
    pass
