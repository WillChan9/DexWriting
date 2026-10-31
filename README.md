# DexWriting

DexWriting is the code, checkpoints and paper source for **Grip, Trace, Write: force-aware imitation and reinforcement
learning for high-DoF robotic handwriting**: a 25-DoF anthropomorphic hand on a UR5e arm writes with a
freely held pen in MuJoCo. The simulation environment is the Python package `dexcraft`.

Project page: https://willchan9.github.io/DexWriting/ (source in `website/site/`).

## Contents

```
dexcraft/          the environment package (gymnasium): scene, tasks, ink and force model, metrics, PPO / BC trainers
dex_hand/     the hand model (MJCF, skin) mounted on a UR5e
scripts/           the training and evaluation pipeline used in the paper
checkpoints/       trained weights (see below)
results/           training curves and evaluation tables behind the paper figures
paper/             LaTeX source, figures and the scripts that draw them
website/           project page
```

## Install

```bash
conda env create -f environment.yml      # Python 3.10, MuJoCo 3.10, PyTorch
conda activate dexcraft
python -m dexcraft.selftest              # all checks should pass
```

or, in an existing Python 3.10 environment: `pip install -r requirements.txt && pip install -e .`.
Install in editable mode: the hand model is read from `dex_hand/` next to the package.

## Quick start

```python
import gymnasium as gym
import dexcraft  # registers the tasks

env = gym.make("DexCraft/L2_write_chinese-v1")
obs, info = env.reset(seed=0)
```

Watch a trained policy write (requires a display):

```bash
MUJOCO_GL=glfw python -m dexcraft.demo --help
```

## Checkpoints

| File | What it is |
|---|---|
| `checkpoints/grasp.pt` | learned pen grasp (BC on 240 teacher episodes, then 8 M PPO steps on the three hold tasks), trajectory-input format |
| `checkpoints/grasp_v10_format.pt` | the same grasp in its original input format; the IK-teacher baseline uses it |
| `checkpoints/two_branch_init.pt` | two-branch network template (grasp branch filled in) that BC writes the arm/wrist branch into |
| `checkpoints/bc.pt` | all-task behavior-cloning policy (initializes RL) |
| `checkpoints/rl.pt` | final policy after 40 M steps of PPO |
| `checkpoints/rl_snapshots/` | intermediate RL checkpoints used by the case-study figure |

## Evaluate

```bash
bash scripts/evaluate.sh rl            # final RL policy, 50 seeds per task, seven writing tasks
bash scripts/evaluate.sh bc            # BC initialization
bash scripts/evaluate.sh ik            # IK teacher with the learned grasp
bash scripts/evaluate.sh rl 50 zero-shot   # held-out words, idioms and Chinese characters
```

Each run writes one JSON per task to `eval/<name>/` and prints completion, precision, force error, time and
drops.

**Task suite.** The code ships the current seven writing tasks (line, circle, shape, digits, letters,
words, Chinese); cursive is sampled inside the word task. The paper's main eight-task table was measured
on an earlier task definition with a separate cursive task, so `evaluate.sh` reproduces the paper's
seven-task fresh-seed evaluation (appendix, font update) rather than that table exactly.

## Train

The pipeline has three stages; the first is provided as a checkpoint.

1. **Grasp**: `checkpoints/grasp.pt`.
2. **Demonstrations and BC**:
   ```bash
   bash scripts/collect_demos.sh          # IK teacher + learned grasp, clean and noise-perturbed episodes
   bash scripts/train_bc.sh 0             # one BC seed -> runs/bc_s0_final.pt
   ```
   BC is sensitive to the seed; the paper trained 18 seeds and kept the best by mean completion
   (`scripts/evaluate.sh runs/bc_s<seed>_final.pt 24`).
3. **RL**:
   ```bash
   bash scripts/train_rl.sh runs/bc_s0_final.pt   # 40 M steps; the paper used 512 parallel environments
   ```

Training was run on a 128-core CPU machine with one GPU; the 40 M-step RL run takes about 3 h 15 min there.

## Paper figures

Run from the repository root, `make_example_figs.py` first (the case study reuses its rollouts):

```bash
MUJOCO_GL=glfw python paper/figure_scripts/make_example_figs.py
MUJOCO_GL=glfw python paper/figure_scripts/make_case_study_fig.py
MUJOCO_GL=glfw python paper/figure_scripts/make_env_figs.py
MUJOCO_GL=glfw python paper/figure_scripts/make_task_suite_fig.py
python paper/figure_scripts/make_rl_curves.py
python paper/figure_scripts/make_pipeline_curves_fig.py
```

Rollout caches go to `cache/`. `paper/figs/model.pdf` and `training_pipeline_public.png` are drawn by hand.

## Website

The page is static HTML (`index.html`, `tasks.html`); open it directly or serve the folder. Its videos and
chart data are rendered from the shipped checkpoints:

```bash
MUJOCO_GL=glfw python website/make_media.py          # -> website/site/media/ (videos, data/*.json, data.js)
MUJOCO_GL=glfw python website/make_media.py --only tasks   # just the per-task clips
```

## Third-party material

See `THIRD_PARTY_NOTICES.md`.
