#!/usr/bin/env bash
# Milestone check: one RoboTwin episode with the fine-tuned teacher through LeRobot's eval loop.
# Usage: scripts/run_teacher_smoke.sh [task] [n_episodes] [start_seed]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
TASK=${1:-beat_block_hammer}; N=${2:-1}; SEED=${3:-100000}
TEACHER_REV=6d128a33268a8abb558bfc12d0f2d4e0225d7d64                 # lerobot/lingbot_va_robotwin
WAN=$HF_HOME/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c
OUT=$PROJ/eval/results/smoke_teacher_${TASK}_${N}ep_seed${SEED}
cd "$ROBOTWIN_ROOT"
lerobot-eval \
  --policy.path=lerobot/lingbot_va_robotwin --policy.pretrained_revision=$TEACHER_REV \
  --policy.wan_pretrained_path="$WAN" --policy.device=cuda \
  --env.type=robotwin --env.task="$TASK" --env.action_mode=ee \
  --eval.n_episodes="$N" --eval.batch_size=1 --seed="$SEED" \
  --output_dir="$OUT"
echo "results: $OUT"
