"""Per-task experts: route each task to one of several PPO-format checkpoints.

    DEXCRAFT_ROUTER="default=runs/a/agent.pt,L1_=runs/b/agent.pt,L2_write_word=runs/b/agent.pt" \\
        python -m dexcraft.evaluate --policy dexcraft.policies.router:make_policy ...

Keys are task-id prefixes (longest match wins); `default` catches the rest. The policy
discloses its architecture as "experts" (per-task policies) rather than "generalist".
"""
import os

from .base import Policy
from .ppo import PPOPolicy


class RouterPolicy(Policy):
    architecture = "experts"
    demos = "teacher clones"

    def __init__(self, spec):
        self.table = {}
        for kv in spec.split(","):
            k, _, path = kv.partition("=")
            self.table[k.strip()] = path.strip()
        self.experts = {p: PPOPolicy(p) for p in set(self.table.values())}
        first = next(iter(self.experts.values()))
        self.control_mode, self.control_hz, self.obs_mode = first.control_mode, first.control_hz, first.obs_mode
        self.active = first

    def reset(self, task_info):
        tid = task_info["task_id"]
        keys = [k for k in self.table if k != "default" and tid.startswith(k)]
        key = max(keys, key=len) if keys else "default"
        self.active = self.experts[self.table[key]]
        self.active.reset(task_info)

    def observe(self, info):
        self.active.observe(info)

    def act(self, obs):
        return self.active.act(obs)


def make_policy(spec=None):
    return RouterPolicy(spec or os.environ["DEXCRAFT_ROUTER"])
