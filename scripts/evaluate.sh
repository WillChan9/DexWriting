#!/usr/bin/env bash
# Evaluate a policy on the seven writing tasks (or any comma-separated task list, e.g. $ZERO_SHOT_TASKS),
# one process per task.
#   usage: bash scripts/evaluate.sh <rl|bc|ik|CKPT.pt> [SEEDS=50] [TASKS|zero-shot] [OUT=eval/<name>]
#   rl / bc   the shipped checkpoints; ik = the IK teacher with the learned grasp; or any trained .pt
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
WHO=${1:?rl, bc, ik or a checkpoint}; SEEDS=${2:-50}; TASKS=${3:-$WRITING_TASKS}
[ "$TASKS" = zero-shot ] && TASKS=$ZERO_SHOT_TASKS
case $WHO in
  rl) CK=checkpoints/rl.pt ;; bc) CK=checkpoints/bc.pt ;; ik) CK="" ;; *) CK=$WHO ;;
esac
NAME=$(basename "${CK:-ik}" .pt); OUT=${4:-eval/$NAME}
mkdir -p "$OUT"
for T in ${TASKS//,/ }; do
  if [ -z "$CK" ]; then   # the IK teacher; the paper's baseline used the grasp in its original input format
    ENVV="DEXCRAFT_BASE=hybrid DEXCRAFT_GRIP_CKPT=checkpoints/grasp_v10_format.pt"; POLICY=dexcraft.policies.hybrid:make_policy
  else
    ENVV="DEXCRAFT_PPO_CKPT=$CK"; POLICY=dexcraft.policies.ppo:make_policy
  fi
  env $ENVV python -m dexcraft.evaluate --policy $POLICY --tasks $T --seeds $SEEDS --seed-base 0 \
    --tracks nominal --obs-mode state --control-hz 25 --env-kwargs "$EVAL_ENV" --out "$OUT/$T.json" > "$OUT/$T.log" 2>&1 &
done
wait
python scripts/summarize_eval.py "$OUT"
