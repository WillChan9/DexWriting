#!/usr/bin/env bash
# Collect IK-teacher demonstrations for every writing task: 100 clean episodes plus 100 episodes each
# with action noise of sigma 0.02 and 0.01 (DART-style perturbation). The teacher drives the arm and
# wrist by inverse kinematics while the learned grasp holds the pen.
#   usage: bash scripts/collect_demos.sh [EPISODES=100] [WORKERS=24]
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
EP=${1:-100}; W=${2:-24}
mkdir -p data logs
for T in ${WRITING_TASKS//,/ }; do
  COMMON="--tasks $T --episodes $EP --workers $W --teacher hybrid:$GRASP --reach 0 --ga 0 \
    --reward v16_dense1c25 --constraint idle_lock --end-mode policy --cursor-obs 0 --traj-obs 16 --adv-mode tip \
    --start-height 0.002 0.006"
  python -u scripts/bc_collect.py data/${T}_clean.npz  $COMMON --seed-base 3000               > logs/collect_${T}_clean.log  2>&1 &
  python -u scripts/bc_collect.py data/${T}_dart02.npz $COMMON --seed-base 43000 --noise 0.02 > logs/collect_${T}_dart02.log 2>&1 &
  python -u scripts/bc_collect.py data/${T}_dart01.npz $COMMON --seed-base 83000 --noise 0.01 > logs/collect_${T}_dart01.log 2>&1 &
  wait
  tail -n1 logs/collect_${T}_*.log
done
