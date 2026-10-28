# Stage 2 plan: DMD2 + LoRA students for LingBot-VA (NFE 16 / 8 / 4)

Status 2026-10-28: plan + first implementation (`distill/dmd2_lora.py`); nothing trained yet.
Stage 1 prerequisite still running: the NFE 30 teacher reproducibility check (`scripts/run_repro_check.sh`).

## 1. Goal

Train three students that run the fine-tuned teacher (`lerobot/lingbot_va_robotwin` @ `6d128a3`) with fewer
denoising steps, and compare them with the NFE 30 teacher on the frozen RoboTwin scenes for success rate
and latency. The students have the **same architecture and size** as the teacher (5.09 B params); they are
faster because they run the network fewer times per action chunk, not because they are smaller.

| model | NFE (video / action steps) | role |
|---|---|---|
| teacher | 30 / 30 | reference |
| student A | 16 / 16 | first compression target |
| student B | 8 / 8 | intermediate |
| student C | 4 / 4 | aggressive |
| *control* | teacher at 16, 8, 4 | **same checkpoint, fewer steps, no training**: shows what distillation buys |

**NFE definition (decision, 2026-10-28):** "NFE N" sets both denoising loops to N steps
(`--nfe N` in the harness). LingBot-VA runs a video loop and then an action loop per chunk, each with
its own flow-matching scheduler; each loop also makes one extra cache-writing pass, so NFE N costs
2N + 2 transformer calls per chunk (video calls are batched cond/uncond for guidance).

## 2. Method: DMD2, adapted to LingBot-VA

DMD2 (Yin et al. 2024, arXiv 2405.14867) trains a few-step **generator** so that the distribution of its
outputs matches the teacher's. Two score models judge a generated sample after re-noising it:
the frozen **teacher** ("real score") and a **fake score** model trained online on the generator's
outputs. Their disagreement is the gradient that moves the generator toward the teacher's distribution.

### 2.1 Three roles, one set of base weights (LoRA)

Full fine-tuning would need ~173 GB (two trained 5 B copies with AdamW + a frozen teacher) against
46 GB on the RTX A6000. Instead all three roles share **one frozen copy** of the teacher weights:

| role | weights | trained |
|---|---|---|
| teacher (real score) | base, adapters disabled | no |
| generator (the student) | base + LoRA adapter `gen` | yes |
| fake score | base + LoRA adapter `fake` | yes |

Both adapters start at zero, so both roles start exactly equal to the teacher (DMD2 initializes the fake
score from the teacher too). Only the adapters and their optimizer state are trained, so memory is
~10 GB of weights plus activations. Reported as **"DMD2 + LoRA"**, a weaker baseline than full-weight DMD2.

### 2.2 Training structure: teacher-forced, block-causal (as the teacher was trained)

The teacher was trained (upstream `wan_va/train.py`, LeRobot `training_loss_from_streams`) on **whole
episodes** in one block-causal pass. The sequence is `[noisy video | clean video | noisy actions | clean
actions]` over all latent frames, grouped into chunks:

* a noisy **video** chunk sees the clean video and clean actions of *earlier* chunks;
* a noisy **action** chunk sees the clean video of its *own* chunk (inverse dynamics) and earlier clean actions.

At inference the KV cache holds **real observations** of earlier chunks, so teacher forcing with real
context matches what the model sees at test time. The distillation uses the same structure: every role
is evaluated on the whole episode at once, with clean context taken from the data.

### 2.3 One generator step (student at N steps)

1. Load an episode: clean video latents `z`, actions `a` (normalized), text embedding `c`.
2. **Backward simulation** (DMD2 §4.3): pick a student step `k` uniformly from `0..N-1`. Start the noisy
   streams from pure noise and run `k` student Euler steps without gradients, using the student's own
   N-step schedule (video shift 5.0, action shift 1.0, as at inference). Run step `k` with gradients and
   convert the predicted velocity to a clean estimate, `x0_G = x_t - sigma_t * v`, for both streams.
3. **Distribution-matching gradient:** pick `tau`, re-noise `x0_G` to `x_tau`, and compute (no gradients)
   * the teacher's clean estimate `x0_real`, video with classifier-free guidance (scale 5.0, empty-prompt
     embedding), actions unguided (scale 1.0), exactly as the teacher samples;
   * the fake score's clean estimate `x0_fake`.

   `g = (x0_fake - x0_real) / mean|x0_G - x0_real|`; generator loss `0.5 * ||x0_G - sg(x0_G - g)||^2`,
   actions masked to the 16 used channels. Video and action losses are summed with equal weight.
4. **Fake-score steps:** `fake_updates_per_gen` (DMD2 uses 5) flow-matching steps for the `fake` adapter on
   `sg(x0_G)` with fresh noise and the same clean context, using the teacher's own training loss.

No GAN term and no regression loss in the baseline ("simple DMD2-style"). Both are listed in §7 as
follow-ups if the baseline underperforms.

### 2.4 Sampling the student at eval time

The student runs through the **unchanged** LeRobot inference loop with `--nfe N` and its LoRA adapter
loaded: same Euler flow-matching sampler, same KV-cache feedback, same harness. Training uses the same
N-step schedule, so train and test sampling match.

**Guidance is distilled in.** The real score is the teacher *with* classifier-free guidance (video 5.0), and
the generator is run conditional-only in training, so it learns to produce guided outputs from a single
pass. Students are therefore evaluated with `guidance_scale=1` (no cond/uncond batch), a second speed-up on
top of fewer steps. The teacher and the controls keep the shipped guidance 5.0. The export writes this
into the student's `config.json`, so the harness needs no extra flag.

## 3. Data

`robbyant/robotwin-clean-and-aug-lerobot` @ `595d4ac`, subset `lerobot_robotwin_eef_clean_50/`, the
10 v1 tasks (50 demos each, `demo_clean`, the same setting as the eval; ~7 GB). Robbyant prepared it
"for Lingbot-VA post-training" and it matches the teacher exactly:

* **Video latents precomputed** with the Wan2.2 VAE at the teacher's resolution (head 256x320 ->
  16x20 latent; wrists at half resolution), tiled in the `robotwin_tshape` layout. No VAE in the loop.
* **Text embeddings precomputed** (UMT5, 512x4096) per episode, plus `empty_emb.pt` for guidance.
  No 11 GB text encoder in the loop.
* **Actions:** 16-dim EEF pose + gripper per arm at 50 fps. Converted exactly as upstream
  `wan_va/dataset/lerobot_latent_dataset.py` does: poses relative to the first frame, 16 zero rows in front
  of latent frame 0, 16 actions per latent frame, scattered into the 30-channel layout, normalized with the
  quantiles q01/q99. **Verified 2026-10-28:** these quantiles are bit-identical to the teacher's
  un-normalizer, so data and teacher share one action space.

Caveats:
* **License CC BY-NC-SA 4.0** (non-commercial, share-alike). Check this is acceptable for how the
  students will be used before deploying them.
* Demo instructions are varied natural language; the eval prompt is the bare task name (LeRobot env
  default). The baseline trains on the dataset prompts only; the prompt-mismatch risk and its
  mitigation (train on both) are in §7.
* The demo seeds are not recorded in the metadata. RoboTwin collects demos from low seeds and the frozen
  eval seeds start at 100000, so overlap is unlikely but unverified.
* Latent files are torch pickles: load them with `weights_only=True` only (they contain only tensors, ints,
  lists and strings).

## 4. Compute

* Hardware: 1x RTX A6000 (46 GB), 96 cores, 503 GB RAM. One student trains at a time.
* Memory estimate per step: 10.2 GB bf16 base + adapters (~0.1-0.5 GB with AdamW state) + activations
  for one backward pass over ~2.7 k tokens (10 latent frames x 120 tokens x 2 + 160 action tokens x 2).
  Roughly 15-20 GB without gradient checkpointing, so it **should fit**; to be measured (§6 step 1).
* Time (estimate, to be measured): one generator step at NFE 16 is ~k/2 + 5 transformer passes plus the fake
  steps, so likely tens of seconds per generator step. A few thousand steps per student is then
  ~0.5-1 day each. Lower N is cheaper (fewer backward-simulation passes).

## 5. Evaluation (the Stage 1 harness, unchanged)

* Every model on the same frozen scenes: `eval/seeds/v1/frozen_seeds.json`. First the 5-task x 10-seed
  subset used for the reproducibility check (pick_dual_bottles, stamp_seal, place_mouse_pad, place_fan,
  place_burger_fries), then the full 10 x 20 set for the final table.
* Per model: success rate (Wilson 95% CI), errors reported separately, steady-state chunk latency (mean,
  p95) after the warm-up, plus manifest (NFE, adapter checkpoint hash, revisions, hardware).
* Comparison: per NFE, **student vs teacher-at-the-same-NFE (control) vs teacher at 30**. Paired by seed
  (McNemar), because every model sees identical scenes.
* Deliverable table: NFE, episodes, success rate, latency, hardware, runtime settings, failed runs.
* Recommendation for Jetson AGX Thor: from the success/latency trade-off on the A6000, then confirmed by
  re-measuring latency on the Thor (separate step, not in this plan).

## 6. Milestones

1. **Memory and speed test** (~15 min GPU, after the repro check frees the GPU): `train --nfe 16 --steps 2`
   on a real episode, watching peak GPU memory (nvidia-smi) and seconds per step. Go/no-go for the A6000.
2. **Sanity run:** ~200 generator steps on one task; the fake/teacher gap and losses must move, and the
   student at k = 0 must still produce valid actions (decode a few, compare with the teacher's).
3. **Student A (NFE 16):** train, eval on the 5 x 10 subset with the control (teacher at 16).
4. **Students B (8), C (4):** same.
5. **Full table** on 10 x 20 for the teacher (NFE 30), the controls and the students; write-up and Thor recommendation.

## 7. Risks and follow-ups

| risk | mitigation |
|---|---|
| LoRA too weak for 4 steps | raise rank; add the DMD2 GAN term on fake-score features; full fine-tune on multi-GPU |
| Teacher-forced training ≠ closed-loop rollout (actions see imagined, not real, video at test time) | condition the action stream on the generator's own video (`sg(x0_G)`) instead of the data video |
| Prompt mismatch (dataset prompts vs bare task name at eval) | train on both; or evaluate with RoboTwin's official instructions |
| Flex-attention mask recompiles per episode length | bucket episodes by latent length; cap at a fixed window |
| 50 demos/task too few | add `lerobot_robotwin_eef_aug_500` (500 randomized demos/task) |

## 8. Files

```
distill/PLAN.md            this plan
distill/dmd2_lora.py       data loader, DMD2 + LoRA trainer, adapter export
lingbot-va/                upstream reference @ 7c6ffa9 (read-only, not imported at run time)
```
