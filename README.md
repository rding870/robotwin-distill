# robotwin-distill

Distilling the LingBot-VA robot manipulation policy into few-step students, evaluated head-to-head
against the teacher on frozen RoboTwin 2.0 scenes.

## Purpose — why distill the teacher?

The teacher is `lerobot/lingbot_va_robotwin`: a 5.09 B-parameter vision-language-action model that
generates robot actions by **flow-matching denoising** — it runs a video-prediction denoising loop and
then an action denoising loop for every chunk of ~16 actions it emits. Each loop takes N denoising
steps ("NFE N"), and each step is a full transformer forward pass (≈ 2N + 2 passes per chunk, with the
video passes doubled for classifier-free guidance).

That is the problem: **the policy is accurate but too slow for real-time control.** At its shipped
settings on an RTX A6000 it takes ~6.3 s of inference to produce each action chunk
(`eval/results/teacher_v1`), while the chunk covers well under a second of robot motion. The eventual
deployment target is a Jetson AGX Thor, which is slower still. Inference cost scales linearly with the
number of denoising steps, so the lever is NFE.

Simply running the teacher with fewer steps degrades it — a flow-matching sampler tuned for 30 steps
produces worse trajectories at 4. **Distillation (DMD2) trains a student to match the teacher's output
distribution while taking only 16, 8, or 4 steps.** The student has the *same architecture and size*
as the teacher; it is faster only because it runs the network fewer times per chunk. Two compounding
speedups:

1. **Fewer denoising steps:** NFE 30 → 16 / 8 / 4.
2. **Guidance distilled in:** the teacher needs a doubled (conditional + unconditional) batch for
   classifier-free guidance; the student is trained against the *guided* teacher and runs
   conditional-only (`guidance_scale=1`).

"Teacher at NFE 16/8/4 with no training" is kept as a control, so the final table shows exactly what
distillation buys over just turning the step count down.

## The two stages

### Stage 1 — reproducible teacher baseline (done)

Before training anything, build an environment and eval harness where "the student is better/worse
than the teacher" is a trustworthy statement:

- **Environment** (`SETUP.md`): one pinned Python 3.12 venv that runs both RoboTwin 2.0 (simulator)
  and LeRobot (policy loader + eval loop), with every git SHA / HF revision in `configs/pins.txt`
  and a full freeze in `requirements-lock.txt`. Includes headless-Vulkan and numpy-1.x ABI gotchas.
- **Frozen scene set** (`eval/seeds/v1/frozen_seeds.json`): 10 tasks × 20 expert-verified seeds.
  Every model — teacher, controls, students — is evaluated on exactly these scenes, so comparisons
  can be paired per seed (McNemar).
- **Harness** (`scripts/eval_frozen.py`, docs in `eval/README.md`): resumable runs, per-episode
  success/latency/error, action and observation fingerprints (sha256 per chunk), and a manifest
  capturing everything needed to reproduce a run. `scripts/compare_runs.py` diffs two runs from
  strictest (bit-identical actions) to loosest (summary table).
- **Teacher baseline** (`eval/results/teacher_v1`): 83.0 % success overall (166/200, Wilson 95 % CI
  77.2–87.6), steady-state chunk latency ~6.3 s. One task (`open_laptop`) errored across the board
  and was subsequently fixed in the harness.
- **Reproducibility check** (`eval/NOTES.md`): running the same eval twice initially flipped 2/50
  episodes; root cause was cuDNN benchmark mode. Deterministic mode is now the default (at a +27 %
  latency cost, applied uniformly to all runs). Residual noise floor: ~1 episode in 50 on success,
  so Stage 2 treats ±1 episode per task as noise.

### Stage 2 — DMD2 + LoRA students (in progress)

Plan: `distill/PLAN.md`. Trainer: `distill/dmd2_lora.py`.

- **Method:** DMD2 (Yin et al. 2024). A few-step generator (the student) is pushed toward the
  teacher's output distribution by the disagreement between two score models — the frozen teacher
  ("real score") and a "fake score" model trained online on the generator's outputs.
- **LoRA twist:** full-weight DMD2 would need ~173 GB (two trained 5 B copies + a frozen teacher);
  the single 46 GB A6000 can't hold that. Instead all three roles share **one frozen copy** of the
  teacher weights, with the generator and fake score each being a LoRA adapter on top. Both adapters
  start at zero, so all roles start exactly equal to the teacher.
- **Training matches inference:** teacher-forced, block-causal passes over whole episodes (the mode
  the teacher was trained in), on precomputed Wan2.2 video latents and UMT5 text embeddings from
  `robbyant/robotwin-clean-and-aug-lerobot` — no VAE or text encoder on the GPU. The student trains
  on its own N-step inference schedule, so train-time and test-time sampling are identical.
- **Eval:** the Stage 1 harness, unchanged — a student is just a different `--policy.path` with
  `--nfe N`. Exported students (e.g. `distill/exports/student_nfe16_s2000`) carry their guidance
  setting in `config.json`, so no extra flags.

**Status (2026-10-29):** the NFE-16 student is trained to 2000 generator steps
(`distill/runs/nfe16`, checkpoints every 250 steps) and exported; its eval on the frozen set is
underway (`eval/results/student_nfe16_s2000`) — early episodes show ~3.4 s steady-state chunk
latency, roughly half the teacher's. NFE 8 and NFE 4 students are next, then the full
teacher / control / student table and a latency-vs-success recommendation for the Jetson AGX Thor.

## Repo layout

```
SETUP.md                  environment build + Stage 1 milestones (start here to reproduce)
env.sh                    source before anything (venv, PYTHONPATH, Vulkan ICD, caches)
configs/                  pins.txt (every SHA/revision), uv overrides, Vulkan ICD, host snapshot
lerobot/                  huggingface/lerobot @ 146d847 — policy loader, RoboTwin gym env, eval loop
RoboTwin/                 RoboTwin-Platform/RoboTwin @ 0aeea2d — simulator, tasks, assets
lingbot-va/               upstream LingBot-VA reference @ 7c6ffa9 (read-only, not imported at runtime)
scripts/                  harness: eval_frozen.py, compare_runs.py, seed-list builder, smoke tests,
                          run_teacher_frozen.sh / run_repro_check.sh wrappers, setup patches
eval/
  README.md               how to run evals and repro checks; output formats
  NOTES.md                harness issue log (nondeterminism diagnosis and fixes)
  seeds/v1/               the frozen scene set (10 tasks × 20 seeds)
  results/                one directory per run (git-ignored): manifest, episodes.jsonl, summary, videos
distill/
  PLAN.md                 Stage 2 plan: DMD2 + LoRA, data, compute budget, milestones, risks
  dmd2_lora.py            fetch / train / export CLI for the students
  runs/                   LoRA checkpoints per student (git-ignored)
  exports/                merged student checkpoints the harness can load directly (git-ignored)
logs/                     run logs (git-ignored)
```

## Key invariants

- **Everything is pinned.** Git SHAs, HF revisions, and the package freeze live in `configs/pins.txt`
  and `requirements-lock.txt`; every eval manifest records them. Transitive deps drift on rebuild —
  `SETUP.md` has the one-liner that pulls them back to the lock.
- **Every eval reads seeds from `eval/seeds/v1/frozen_seeds.json`.** No ad-hoc seed choices.
- **Deterministic mode is on by default** in the harness; absolute latencies measured before
  2026-10-29 are not comparable with deterministic-mode runs (relative comparisons are unaffected).
- **Dataset license is CC BY-NC-SA 4.0** (non-commercial, share-alike) — check before deploying
  students trained on it.
