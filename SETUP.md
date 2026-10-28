# RoboTwin 2.0 × LingBot-VA — environment setup (Stage 1)

Scope: only what is needed to run the fine-tuned teacher (`lerobot/lingbot_va_robotwin`)
inside RoboTwin 2.0 and evaluate it reproducibly. No DMD2, no upstream `lingbot-va` repo,
no Jetson bits yet. Exact versions: `configs/pins.txt`; full freeze: `requirements-lock.txt`.

## Layout

```
robotwin-distill/
  env.sh                  source this before anything (venv, PYTHONPATH, Vulkan ICD, caches)
  lerobot/                huggingface/lerobot @ 146d847  (policy loader + RoboTwin gym env + eval loop)
  RoboTwin/               RoboTwin-Platform/RoboTwin @ 0aeea2d  (simulator, tasks, assets, envs/curobo)
  configs/pins.txt        every git SHA / HF revision / host detail
  configs/uv-overrides.txt resolver overrides (numpy 1.26.4, pyarrow 21) — pass to every `uv pip install`
  configs/vulkan/         EGL-based NVIDIA Vulkan ICD (see "Headless rendering")
  scripts/                smoke tests + harness
  eval/                   frozen seed list + results (results are git-ignored)
  logs/                   run logs (git-ignored)
```

Large weights live in `$HF_HOME` (`/workspace/.cache/huggingface`, ~20 GB).
`/workspace` is a FUSE mount (geesefs): no chmod/exec, so the venv, uv cache and repos
stay on local disk under `/root`.

## One environment, not two

RoboTwin's pins (torch 2.4, numpy 1.x) and LeRobot's (torch ≥2.7, numpy ≥2) conflict only in
metadata. In practice one Python 3.12 venv works:

* torch 2.11.0+cu128 runs RoboTwin fine (CuRobo and pytorch3d are built against it).
* **numpy must stay 1.26.4.** `mplib==0.2.1` (last release) is compiled against the NumPy 1 ABI
  and segfaults when handed a NumPy 2 array (RoboTwin planner → `mplib/sapien_utils/conversion.py:310`).
  LeRobot's own `docker/Dockerfile.benchmark.robotwin` ends up on numpy 1.x the same way.
  Consequences: `pyarrow` pinned to 21.0.0; `uv pip check` reports the two metadata violations
  (`lerobot`, `opencv-python-headless` want numpy ≥2) — expected.

## Build from scratch

```bash
export UV_CACHE_DIR=/root/.cache/uv HF_HOME=/workspace/.cache/huggingface
mkdir -p /root/robotwin-distill && cd /root/robotwin-distill
git clone https://github.com/huggingface/lerobot && git -C lerobot checkout 146d8474932c8b566717eb2738ba459337ab71c4
git clone https://github.com/RoboTwin-Platform/RoboTwin && git -C RoboTwin checkout 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
uv venv --python 3.12 .venv && source env.sh          # env.sh is in this repo
OV=configs/uv-overrides.txt
uv pip install --overrides $OV -e "./lerobot[lingbot_va,dataset,av-dep]"
uv pip install --overrides $OV "sapien==3.0.0b1" "mplib==0.2.1" "transforms3d==0.4.2" "trimesh==4.4.3" \
    "open3d==0.19.0" "imageio==2.34.2" termcolor zarr pydantic h5py "warp-lang==1.3.1" "setuptools<81" ninja
MAX_JOBS=48 uv pip install --overrides $OV --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git RoboTwin/envs/curobo
(cd RoboTwin/envs/curobo && MAX_JOBS=48 uv pip install --overrides ../../../$OV -e . --no-build-isolation)   # TORCH_CUDA_ARCH_LIST from env.sh
python scripts/patch_sim_deps.py                      # mplib `or collide` + sapien URDF/SRDF fixes (mirrors RoboTwin scripts/_install.sh)
python scripts/download_assets.py                     # embodiments.zip + objects.zip @ pinned dataset rev, then update_embodiment_config_path.py
hf download lerobot/lingbot_va_robotwin --revision 6d128a33268a8abb558bfc12d0f2d4e0225d7d64
for sub in vae text_encoder tokenizer; do hf download robbyant/lingbot-va-base --revision 68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c --include "$sub/*"; done
```

Notes
* `HF_HUB_ENABLE_HF_TRANSFER=1` is set globally on this host but `hf_transfer` is not installed; `unset` it before `hf download`.
* `background_texture.zip` (~11 GB) is not downloaded; `demo_clean` (the config LeRobot's env uses) has `random_background: false`.

## Headless rendering (Vulkan)

The container exposes the GPU with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. The stock ICD
(`/etc/vulkan/icd.d/nvidia_icd.json` → `libGLX_nvidia.so.0`) fails with
`vkCreateInstance: Found no drivers!`. Pointing the ICD at `libEGL_nvidia.so.0` works;
`env.sh` sets `VK_ICD_FILENAMES=$PROJ/configs/vulkan/nvidia_icd_egl.json`.

## Verify

```bash
source env.sh
(cd RoboTwin && python ../scripts/smoke_expert.py --task beat_block_hammer --start-seed 100000 --n 3)
#  seed=100000 accepted=True 16.7s / seed=100001 accepted=False / seed=100002 accepted=True
```
`accepted=False` is RoboTwin's normal expert-filter rejection (planner fails or task not solved by the
scripted expert); such seeds are excluded from the frozen set exactly as `RoboTwin/script/eval_policy.py` does.

## One teacher episode (milestone) — done 2026-10-17

```bash
scripts/run_teacher_smoke.sh beat_block_hammer 1 100000     # exact lerobot-eval call, pinned revisions, --env.action_mode=ee
```
Result: success 1/1, 109 env steps, 480 s wall (includes ~1 min model load, UMT5 on CPU).
Output: `eval/results/smoke_teacher_beat_block_hammer_1ep_seed100000/{eval_info.json,videos/…/eval_episode_0.mp4}`.
GPU peak ≈ 10–12 GB of 48 GB.

Gotcha: `hf download --include "a/*" "b/*" "c/*"` silently skipped the first pattern; download each
subfolder with its own `--include` and check the snapshot dir before running.

## Frozen seed list v1 — done 2026-10-18

```bash
python scripts/make_seed_list.py --version v1 --n-tasks 10 --task-rng-seed 0 --seeds-per-task 20 --start-seed 100000
```
10 tasks drawn once with `random.Random(0)` from the 50; for each, seeds 100000+ were swept with the scripted
expert and the first 20 accepted were kept (RoboTwin's own accept rule). 200 accepted seeds, 33 rejected
(reasons recorded). ~30 min wall. Output and rules: `eval/seeds/README.md`, `eval/seeds/v1/frozen_seeds.json`.
Every teacher/student eval must read seeds from that file.

## Teacher on the frozen set — started 2026-10-18 (Stage 1 validation run)

```bash
setsid nohup scripts/run_teacher_frozen.sh teacher_v1 > logs/teacher_v1.log 2>&1 < /dev/null &
```
`scripts/eval_frozen.py` wraps `lerobot_eval.rollout` (harness docs: `eval/README.md`). 10 tasks x 20 seeds,
RoboTwin per-task step limits, per-chunk latency, manifest with NFE (shipped teacher = 25 video / 50 action
denoising steps). Dev check (`dev_teacher_2ep`, place_mouse_pad x2): 2/2 success; steady chunk ~7 s, sim ~0.4 s/step;
UMT5 prompt encoding on CPU is ~6.5 min per unique prompt (cached per prompt string by the wrapper).
Expected wall time ~12-15 h; the run is resumable.
