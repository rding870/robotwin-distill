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

Before a full run, smoke one seed per task so env/harness faults surface in minutes, not hours:
```bash
scripts/run_teacher_frozen.sh smoke_x --max-seeds-per-task 1 --no-video --abort-on-error
```

## Seeding

Each episode reseeds Python/NumPy/torch (CPU + CUDA) with `sha256(f"{eval_seed}:{task}:{seed}")` before its
rollout (`rng_seed` in episodes.jsonl). The policy's sampled noise is then a function of that episode alone,
so a resumed run, a `--tasks` subset and an uninterrupted run all see the same noise for the same
(task, seed). Remaining nondeterminism is GPU-level (bf16 kernels, `cudnn.benchmark`, CuRobo sampling);
the reproducibility check measures it rather than assuming it away. Runs before 2026-10-28 (`teacher_v1`)
seeded once per run and are not comparable episode-for-episode with later runs.

## Errors and validity

An episode that raises (`error` set in episodes.jsonl) is an env/harness fault, not a policy failure. It
still counts as a failure in `rate %`, and `rate % excl. errors` gives the rate over episodes that ran.
Any error marks the run **invalid**: `summary.json -> valid=false, tasks_with_errors`, a banner in
`summary.md`, and exit code 3. Do not report numbers from an invalid run.

## Harness version

`manifest.json -> harness` records this repo's commit, whether `scripts/` had uncommitted changes, and the
sha256 of `eval_frozen.py`. Only runs from a clean, committed harness are reference numbers.

## Expert-state patches

Some RoboTwin tasks set state in the expert's `play_once()` that `check_success()` reads. RoboTwin's
`eval_policy.py` runs the expert on the same task object before the policy episode, so it carries over;
LeRobot's env never runs the expert. `_patch_expert_state()` recomputes that state after `setup_demo()`
exactly as `play_once()` does (currently `open_laptop.arm_tag`) and lists what it patched in
`manifest.json -> expert_state_patches`. `put_object_cabinet` has the same pattern but is not in v1.

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
