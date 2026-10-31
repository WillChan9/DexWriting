#!/usr/bin/env bash
# PPO refinement of the composed policy on all writing tasks with the outcome reward (preset v17_r1).
# The grasp branch's mean and the observation statistics stay frozen. ~40 M environment steps.
#   usage: bash scripts/train_rl.sh [INIT=checkpoints/bc.pt] [STEPS=40000000] [OUT=runs/rl]
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
INIT=${1:-checkpoints/bc.pt}; STEPS=${2:-40000000}; OUT=${3:-runs/rl}
ulimit -n 65536 || true
python -u -m dexcraft.train_ppo --task $WRITING_TASKS --steps $STEPS --glyph-scale 1.0 --reward v17_r1 \
  --reward-scale 0.001 --critic-full 1 --full-step 4 \
  --arch traj2 --adv-mode tip --base none \
  --scale-arm 1.0 --scale-wrist 1.0 --scale-hand 1.0 \
  --gamma 1.0 --lam 0.95 --end-mode policy --cursor-obs 0 --constraint idle_lock \
  --n-envs 512 --rollout 1024 --vec async --groups 2 --control-hz 25 --control-mode hybrid \
  --grip-assist 0.0 --start-height 0.002 0.006 --start-wrist 1 \
  --lr 1e-5 --minibatch 2048 --target-kl 0.03 --epochs 2 --min-logstd -6.0 --critic-warmup 3 \
  --freeze-obs-stats 1 --freeze-grip 1 \
  --init-logstd-arm -4.6 --init-logstd-wrist -4.6 --init-logstd-hand -3.9 --init-logstd-done -3.0 --reset-logstd 1 \
  --torch-threads 8 --cuda 1 --save-freq 3 \
  --init $INIT --out $OUT
