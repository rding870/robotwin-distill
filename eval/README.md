# Evaluation harness

```
eval/seeds/<version>/frozen_seeds.json   the frozen scene set (see eval/seeds/README.md)
eval/results/<run_name>/                 one directory per run (git-ignored; copy summary.* into reports)
    manifest.json    everything needed to reproduce: argv, policy + env config, NFE, Hub revision,
                     seeds file + sha256, lerobot/RoboTwin SHAs, torch/CUDA, GPU, driver, hostname
    episodes.jsonl   one line per (task, seed): success, n_steps, error, wall_s, inference_s, env_s,
                     per-chunk inference latency, first-chunk latency, video path
    summary.json     per-task + overall: Wilson 95% CI on success, latency stats, mean steps
    summary.md       the results table
    videos/<task>/seed_<seed>.mp4   head-camera rollout, 10 fps
```

## Running

```bash
scripts/run_teacher_frozen.sh teacher_v1                       # all 10 tasks x 20 seeds
scripts/run_teacher_frozen.sh dev_x --tasks place_fan --max-seeds-per-task 2   # quick check
scripts/run_teacher_frozen.sh teacher_a16 --action-steps 16    # same checkpoint, fewer action denoising steps
```
`run_teacher_frozen.sh` pins the policy/env flags; `scripts/eval_frozen.py` is the generic driver
(wrapper flags first, then `--`, then any `lerobot-eval` flags, so a student is just a different
`--policy.path`). Runs are resumable: rerun the same command and finished episodes are skipped.

## What "latency" means here

LingBot-VA predicts a chunk of 32 actions (2 frames x 16) per denoising pass and then pops them one per
env step, so most `select_action` calls are free. `chunk_latency_s` is the time of each
`predict_action_chunk` call (CUDA-synchronised); the first chunk of an episode also pays VAE encoding of
the conditioning frame, so it is reported separately (`first_chunk_s`) from the steady-state chunks.
`inference_s` is the total time inside `select_action` for the episode (includes prompt encoding),
`env_s` is everything else (SAPIEN stepping, CuRobo IK, rendering).

Prompt embeddings are cached per prompt string across episodes (`--no-prompt-cache` disables this);
the embedding is a deterministic function of the text, so results are unchanged.

## Step limit

Episodes stop after RoboTwin's own per-task limit from `RoboTwin/task_config/_eval_step_limit.yml`
(400 for most of the frozen tasks, up to 1200 for `blocks_ranking_rgb`), the same count of
`take_action` calls RoboTwin's `eval_policy.py` uses, so success rates are comparable to published
RoboTwin numbers. Decided 2026-10-18 over LeRobot's flat 1200 default: a failed episode costs ~3x less
sim time. `--episode-length lerobot` or `--episode-length <int>` restores the other behaviour; the mode and
per-task limits are in `manifest.json`, and episode rows carry `step_limit` and `truncated`.
Note this is stricter: in the dev run, `place_mouse_pad` seed 100001 succeeded at step 621, which counts
as a failure under the 400-step limit.

## Running unattended

`run_teacher_frozen.sh` is launched with `setsid nohup ... < /dev/null &` so it survives SSH disconnects.
Progress: `tail -f logs/<run>.log` or `wc -l eval/results/<run>/episodes.jsonl`. If the host reboots or
the process dies, rerun the same command: finished episodes are skipped.

## NFE

The shipped teacher config runs **25 video / 50 action** denoising steps (`num_inference_steps` /
`action_num_inference_steps`); both are recorded in `manifest.json -> policy.nfe` and in `summary.md`.
