# Eval harness issue log

Running record of harness issues, their diagnosis, and fixes. Newest first within each entry's dates.

## Issue 1: Teacher eval runs not bit-reproducible

- **Found:** 2026-10-29 03:06 UTC, when the `repro_nfe30` check finished
  (`eval/results/repro_nfe30_report.md`) — all 50 episodes diverged at action chunk 0,
  19 differed in step count, 2 flipped success (48/50 vs 49/50 overall).
- **Diagnosed:** 2026-10-29. Ruled out seeding (per-episode seed is a pure hash of
  `(eval_seed, task, seed)`) and prompt embeddings (`prompt_embeds.json` sha256s
  bit-identical between runs A and B). Root cause: `torch.backends.cudnn.benchmark = True`
  in `scripts/eval_frozen.py` — cuDNN benchmark mode picks conv algorithms *by timing them*
  per process, so two runs can use bit-different kernels from the very first forward pass
  (the policy's VAE/image path is conv-heavy, hence divergence at chunk 0 everywhere).
- **Fixed:** 2026-10-29. Deterministic mode is now the default in `scripts/eval_frozen.py`:
  `cudnn.benchmark=False`, `cudnn.deterministic=True`,
  `torch.use_deterministic_algorithms(True, warn_only=True)`, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` set at module import (before CUDA init).
  `--no-deterministic` restores the old behavior; the settings are recorded under
  `determinism` in each run's `manifest.json`.
- **Verified:** 2026-10-29 04:08 UTC, `det_probe` (pick_dual_bottles × 3 seeds, run twice):
  all five comparison levels PASS, actions bit-identical
  (`eval/results/det_probe_report.md`).
- **Full recheck (5 tasks × 10 seeds, `repro_nfe30_det`, finished 2026-10-29 ~09:19 UTC,
  `eval/results/repro_nfe30_det_report.md`):** much improved but not fully bit-identical.
  Config PASS, latency PASS. Chunk 0 and all observations are now bit-identical in every
  episode (before the fix, chunk 0 diverged in 50/50); 7/50 episodes fully identical.
  The rest diverge at chunk 1+ **inside the policy forward with identical inputs**
  (`obs = none` in the report), typically the first two episodes per task match and later
  ones diverge — residual nondeterminism in the policy's history-conditioned path
  (suspects: Triton autotuning in compiled flex_attention, or flash-attn over growing
  KV lengths; no `warn_only` fallback warnings in the logs). Outcomes: 50/50 vs 49/50
  success (1 flip: place_fan 100008, a borderline episode), 11 episodes differ by ±1 step.
- **Practical noise floor (2026-10-29):** ~1 episode in 50 (~2%) on success, run-to-run,
  down from 2/50 flips + 19/50 step diffs before the fix. Stage 2 comparisons should be
  paired on the frozen seeds and treat ±1 episode per task as noise. Open decision:
  whether to chase bit-exactness (within-process double-forward probe of the policy) or
  declare outcome-level stability the acceptance bar for levels 1–3 in `compare_runs.py`.

## Issue 2: Config-level FAIL from committing mid-check

- **Found:** 2026-10-29 03:06 UTC, same report — runs A and B recorded different
  `harness.git_sha` (`f879a37` vs `d9a789b`) because the Stage 2 commit landed between
  the two legs of the check.
- **Fix:** procedural, no code change — **do not commit between the A and B legs of a
  repro run** (`scripts/run_repro_check.sh` runs them sequentially from the same tree).

## Issue 3: Divergence not attributable to env vs. policy

- **Found:** 2026-10-29 — the harness fingerprinted only actions (`chunk_sha`), so
  renderer nondeterminism (SAPIEN) could not be distinguished from policy-forward
  nondeterminism.
- **Fixed:** 2026-10-29 — `scripts/eval_frozen.py` now also hashes each chunk's policy
  input batch into `obs_sha` in `episodes.jsonl`; the `scripts/compare_runs.py` report
  gained a "first differing obs" column (`none` = inputs identical, divergence is in the
  policy forward; a number = divergence is upstream in the env/renderer).
- The 2026-10-29 probe showed observations are stable run-to-run, clearing the renderer.

## Issue 4 (open tradeoff): +27% inference latency under deterministic mode

- **Found:** 2026-10-29 04:08 UTC, probe results — steady-state chunk latency rose from
  ~5.5 s to ~7.0 s (deterministic kernels + heuristic instead of benchmarked cuDNN
  algorithm choice).
- **Impact:** relative teacher/student comparisons are unaffected (every run uses the same
  mode), but the absolute latency baseline shifts: NFE-30 teacher numbers measured before
  2026-10-29 are not comparable with deterministic-mode runs.
- **Possible follow-up:** probe whether `cudnn.benchmark=False` alone keeps bit-stability
  at lower cost (risky — heuristic choice can still pick atomics-based nondeterministic
  kernels), or accept the cost.
