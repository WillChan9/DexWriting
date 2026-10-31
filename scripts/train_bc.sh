#!/usr/bin/env bash
# All-task behavior cloning of the arm and wrist on the pooled demonstrations, then assembly of the
# two-branch policy (grasp branch from checkpoints/two_branch_init.pt) and a fit of the stop output.
# BC is sensitive to the seed: train several seeds, evaluate each, and keep the best (the paper used
# 18 seeds and kept the best by mean completion on 24 evaluation seeds per task).
#   usage: bash scripts/train_bc.sh [SEED=0] [DEVICE=cuda]
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
SEED=${1:-0}; DEV=${2:-cuda}
mkdir -p runs logs
S=""; for T in ${WRITING_TASKS//,/ }; do S="$S data/${T}_clean.npz data/${T}_dart02.npz data/${T}_dart01.npz"; done
OMP_NUM_THREADS=4 python -u scripts/bc_train.py $S runs/bc_s$SEED.pt --arch traj --rows arm,wrist \
  --weight-by-std 1 --epochs 60 --lr 3e-4 --batch 256 --device $DEV --seed $SEED --workers 6 --allow-mixed reward
PYTHONPATH=. python scripts/merge_arm.py checkpoints/two_branch_init.pt runs/bc_s$SEED.pt runs/bc_s${SEED}_merged.pt
DP_TASKS=$WRITING_TASKS DP_REWARD=v16_dense1c25 OMP_NUM_THREADS=4 PYTHONPATH=. \
  python scripts/done_pretrain.py runs/bc_s${SEED}_merged.pt runs/bc_s${SEED}_final.pt 200
echo "wrote runs/bc_s${SEED}_final.pt"
