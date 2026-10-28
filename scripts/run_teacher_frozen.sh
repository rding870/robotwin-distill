#!/usr/bin/env bash
# Teacher (as shipped) on the frozen seed list. Usage:
#   scripts/run_teacher_frozen.sh <run_name> [extra eval_frozen.py flags, e.g. --tasks X --max-seeds-per-task 2]
# Pass-through LeRobot flags are pinned here so every run uses the same policy/env settings.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
RUN=${1:?run name}; shift || true
TEACHER_REV=6d128a33268a8abb558bfc12d0f2d4e0225d7d64                 # lerobot/lingbot_va_robotwin
WAN=$HF_HOME/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c
cd "$ROBOTWIN_ROOT"
python "$PROJ/scripts/eval_frozen.py" --seeds "$PROJ/eval/seeds/v1/frozen_seeds.json" --run-name "$RUN" "$@" -- \
  --policy.path=lerobot/lingbot_va_robotwin --policy.pretrained_revision=$TEACHER_REV \
  --policy.wan_pretrained_path="$WAN" --policy.device=cuda \
  --env.type=robotwin --env.task=beat_block_hammer --env.action_mode=ee \
  --seed=1000 --output_dir="$PROJ/eval/results/$RUN/lerobot_cfg"
echo "results: $PROJ/eval/results/$RUN"
