"""DMD2 + LoRA distillation of LingBot-VA into an N-step student (see distill/PLAN.md).

    python distill/dmd2_lora.py fetch                                  # dataset subset -> $DMD_DATA
    python distill/dmd2_lora.py train  --nfe 16 --out distill/runs/nfe16
    python distill/dmd2_lora.py export --run distill/runs/nfe16 --step 2000 --nfe 16 --dest <dir>

Roles share one frozen copy of the teacher transformer: teacher = adapters disabled, generator = LoRA
adapter "default", fake score = LoRA adapter "fake". Training is LingBot-VA's teacher-forced, block-causal
mode over whole episodes, on the precomputed Wan2.2 latents and UMT5 text embeddings from Robbyant's
dataset (no VAE or text encoder on the GPU).
"""

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation

TEACHER = "lerobot/lingbot_va_robotwin"
TEACHER_REV = "6d128a33268a8abb558bfc12d0f2d4e0225d7d64"  # configs/pins.txt
DATA = Path(os.environ.get("DMD_DATA", "/workspace/data/robotwin-clean-and-aug-lerobot"))
DATA_REPO = "robbyant/robotwin-clean-and-aug-lerobot"
DATA_REV = "595d4acc7e026aa57afba218f4170c6d79e39103"
DATA_SUBSET = "lerobot_robotwin_eef_clean_50"
CAMS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]

# Action convention of upstream lingbot-va @ 7c6ffa9 (wan_va/configs/va_robotwin_cfg.py and
# wan_va/dataset/lerobot_latent_dataset.py). q01/q99 verified identical to the teacher's un-normalizer.
USED_IDS = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
INV_USED = [len(USED_IDS)] * 30
for _i, _j in enumerate(USED_IDS):
    INV_USED[_j] = _i
Q01 = np.array([-0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456, -1, -1, -1, -1, -0.3547105032205582,
                -1.3113021850585938e-06, -0.11975435614585876, -1, -1, -1, -1] + [0.] * 16)
Q99 = np.array([0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1, 1, 0.034201726913452024,
                0.39142737388610793, 0.1792279863357542, 1, 1, 1, 1] + [0.] * 14 + [1.0, 1.0])
LORA_TARGETS = r".*transformer\.blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)|.*transformer\.blocks\.\d+\.ffn\.net\.(0\.proj|2)"
TAU_RANGE = (0.02, 0.98)  # re-noising level for the distribution-matching step
WINDOW = 64               # block-causal attention window (frame ids)


# ---------------------------------------------------------------------------------------------- data
def fetch(tasks):
    from huggingface_hub import HfApi, snapshot_download

    subdirs = sorted(e.path.split("/")[-1] for e in HfApi().list_repo_tree(
        DATA_REPO, path_in_repo=DATA_SUBSET, repo_type="dataset", revision=DATA_REV))
    want = [d for d in subdirs if d.split("-")[0] in tasks]
    assert len(want) == len(tasks), f"missing tasks: {set(tasks) - {d.split('-')[0] for d in want}}"
    patterns = ["empty_emb.pt"] + [f"{DATA_SUBSET}/{d}/{sub}" for d in want for sub in ("meta/*", "data/**", "latents/**")]
    snapshot_download(DATA_REPO, repo_type="dataset", revision=DATA_REV, local_dir=DATA, allow_patterns=patterns)
    print("fetched", len(want), "tasks into", DATA)


def relative_pose(pose):
    """Port of upstream get_relative_pose: translation and rotation relative to the first frame."""
    rot = Rotation.from_quat(pose[:, 3:7])
    first = Rotation.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    return np.concatenate([pose[:, :3] - pose[:1, :3], (first.inv() * rot).as_quat()], axis=1)


class Episodes:
    """Whole episodes as LingBot-VA training streams, mirroring upstream LatentLeRobotDataset."""

    def __init__(self, tasks):
        self.items = []
        for d in sorted((DATA / DATA_SUBSET).iterdir()):
            task = d.name.split("-")[0]
            if task not in tasks:
                continue
            info = json.loads((d / "meta/info.json").read_text())
            for line in (d / "meta/episodes.jsonl").read_text().splitlines():
                ep = json.loads(line)
                chunk = ep["episode_index"] // info["chunks_size"]
                parquet = d / info["data_path"].format(episode_chunk=chunk, episode_index=ep["episode_index"])
                for acfg in ep["action_config"]:
                    s, e = acfg["start_frame"], acfg["end_frame"]
                    lat = [d / f"latents/chunk-{chunk:03d}/{c}/episode_{ep['episode_index']:06d}_{s}_{e}.pth" for c in CAMS]
                    if all(p.exists() for p in lat) and parquet.exists():
                        self.items.append({"task": task, "start": s, "end": e, "latents": lat, "parquet": parquet})
        assert self.items, f"no episodes found under {DATA} for {tasks}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        cams = [torch.load(p, map_location="cpu", weights_only=True) for p in it["latents"]]
        grids = [rearrange(c["latent"], "(f h w) c -> f h w c", f=c["latent_num_frames"], h=c["latent_height"],
                           w=c["latent_width"]) for c in cams]
        latents = torch.cat([torch.cat(grids[1:], dim=2), grids[0]], dim=1).permute(3, 0, 1, 2)  # robotwin_tshape [48,F,h,w]

        action = np.stack(pq.read_table(it["parquet"], columns=["action"]).column("action").to_numpy(zero_copy_only=False))
        action = action[it["start"]:it["end"]].astype("float64")
        frame_ids = cams[0]["frame_ids"]
        action = action[int(frame_ids[0] - it["start"]):]
        stride = frame_ids[1] - frame_ids[0]
        action = np.concatenate([relative_pose(action[:, :7]), action[:, 7:8], relative_pose(action[:, 8:15]), action[:, 15:16]], axis=1)
        action = np.pad(action, ((stride * 4, 0), (0, 0)))
        n_lat = (len(frame_ids) - 1) // 4 + 1
        action = action[: n_lat * stride * 4]
        assert action.shape[0] == n_lat * stride * 4 and n_lat == latents.shape[1], (it, action.shape, latents.shape)
        mask = np.pad(np.ones_like(action, dtype=bool), ((0, 0), (0, 1)))[:, INV_USED]
        action = np.pad(action, ((0, 0), (0, 1)))[:, INV_USED]
        action = np.clip((action - Q01[None]) / (Q99[None] - Q01[None] + 1e-6) * 2.0 - 1.0, -1.5, 1.5)
        mask = rearrange(mask, "(f n) c -> c f n 1", f=n_lat)
        action = rearrange(action, "(f n) c -> c f n 1", f=n_lat) * mask
        return {"latents": latents, "text_emb": cams[0]["text_emb"],
                "actions": torch.from_numpy(action).float(), "actions_mask": torch.from_numpy(mask)}


# --------------------------------------------------------------------------------------------- model
class Roles:
    """Teacher / generator / fake score on one frozen transformer with two LoRA adapters."""

    def __init__(self, rank, grad_ckpt):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
        from peft import LoraConfig, get_peft_model

        cfg = PreTrainedConfig.from_pretrained(TEACHER, revision=TEACHER_REV)
        cfg.attn_mode, cfg.device = "flex", "cuda"
        policy = LingBotVAPolicy.from_pretrained(TEACHER, config=cfg, revision=TEACHER_REV).to("cuda")
        policy.name_or_path = TEACHER
        lcfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=LORA_TARGETS,
                          revision=TEACHER_REV, init_lora_weights=True)  # zero-init: both roles start as the teacher
        self.peft = get_peft_model(policy, lcfg)  # adapter "default" = generator
        self.peft.add_adapter("fake", lcfg)
        self.policy, self.cfg, self.tr = policy, policy.config, policy.transformer
        self.policy._ensure_train_schedulers()
        if grad_ckpt:
            from torch.utils.checkpoint import checkpoint

            for blk in self.tr.blocks:
                def wrapped(*a, _f=blk.forward, **k):
                    return checkpoint(_f, *a, use_reentrant=False, **k) if torch.is_grad_enabled() else _f(*a, **k)
                blk.forward = wrapped
        self.params = {n: [p for k, p in self.tr.named_parameters() if "lora_" in k and f".{n}." in k] for n in ("default", "fake")}
        assert all(self.params.values()), "LoRA adapters not found on the transformer"

    def use(self, name):
        """Activate one adapter ('default' = generator, 'fake') and make only its weights trainable.
        Gradient checkpointing recomputes under the adapter active at backward time, so call this before backward."""
        self.peft.set_adapter(name)
        for n, ps in self.params.items():
            for p in ps:
                p.requires_grad_(n == name)

    def velocity(self, x_lat, x_act, ctx_lat, ctx_act, t_lat, t_act, text):
        """One teacher-forced pass. t_*: per-frame timesteps [F] (0..1000). Returns (v_lat, v_act)."""
        from lerobot.policies.lingbot_va.utils import data_seq_to_patch, get_mesh_id

        b, _, f, h, w = x_lat.shape
        p = self.cfg.patch_size
        dev = x_lat.device

        def stream(noisy, clean, t, action):
            pf, ph, pw = (1, 1, 1) if action else p
            grid = get_mesh_id(noisy.shape[-3] // pf, noisy.shape[-2] // ph, noisy.shape[-1] // pw, t=1 if action else 0,
                               f_w=1, f_shift=0, action=action).to(dev)[None].repeat(b, 1, 1)
            return {"noisy_latents": noisy, "latent": clean, "timesteps": t.to(dev)[None].repeat(b, 1).float(),
                    "cond_timesteps": torch.zeros(b, f, device=dev), "grid_id": grid, "text_emb": text}

        inp = {"latent_dict": stream(x_lat, ctx_lat, t_lat, False), "action_dict": stream(x_act, ctx_act, t_act, True),
               "chunk_size": self.cfg.frame_chunk_size, "window_size": WINDOW}
        v_lat, v_act = self.tr(inp, train_mode=True)
        v_act = rearrange(v_act, "b (f n) c -> b c f n 1", f=f)
        v_lat = data_seq_to_patch(p, v_lat, f, h, w, batch_size=b)
        return v_lat.float(), v_act.float()


# ------------------------------------------------------------------------------------------ training
def write_manifest(args, out: Path, n_episodes: int, trainable: int, seed: int):
    """Everything needed to reproduce this training run, next to its checkpoints (cf. eval manifests).
    Training is seeded but not bit-deterministic (no deterministic-kernel flags); the checkpoint is
    the reproducible artifact."""
    proj = Path(__file__).resolve().parents[1]

    def git(*cmd):
        try:
            return subprocess.check_output(["git", "-C", str(proj), *cmd], text=True).strip()
        except Exception:
            return None

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    import peft
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv[1:],
        "args": {k: v for k, v in vars(args).items() if k != "cmd"},
        "seed": seed,
        "code": {
            "git_sha": git("rev-parse", "HEAD"),
            "distill_dirty": bool(git("status", "--porcelain", "--", "distill")),
            "dmd2_lora_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "pins": {"teacher": TEACHER, "teacher_revision": TEACHER_REV, "data_repo": DATA_REPO,
                 "data_revision": DATA_REV, "data_subset": DATA_SUBSET, "data_path": str(DATA)},
        "lora": {"rank": args.rank, "alpha": args.rank, "targets": LORA_TARGETS,
                 "adapters": ["default (generator)", "fake (fake score)"]},
        "dataset": {"n_episodes": n_episodes, "tasks": args.tasks},
        "trainable_params_per_adapter": trainable,
        "tau_range": TAU_RANGE,
        "window": WINDOW,
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF")},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "peft": peft.__version__, "numpy": np.__version__},
        "hardware": {
            "hostname": socket.gethostname(),
            "gpu": gpu.name if gpu else None,
            "gpu_mem_gb": round(gpu.total_memory / 2**30, 1) if gpu else None,
            "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip() or None,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def masked_mse(a, b, m=None):  # frame 0 is the conditioning frame at inference: never a target
    a, b = a[:, :, 1:], b[:, :, 1:]
    if m is None:
        return F.mse_loss(a, b)
    m = m[:, :, 1:].float()
    return ((a - b) ** 2 * m).sum() / m.sum().clamp(min=1)


def train(args):
    from lerobot.policies.lingbot_va.utils import FlowMatchScheduler, _sample_timestep_id

    random.seed(0); torch.manual_seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Episodes(args.tasks)
    empty = torch.load(DATA / "empty_emb.pt", map_location="cpu", weights_only=True).cuda()[None].to(torch.bfloat16)

    R = Roles(args.rank, args.grad_ckpt)
    cfg = R.cfg
    g_v, g_a = float(cfg.guidance_scale), float(cfg.action_guidance_scale)
    # The student's own N-step schedule, exactly as inference builds it (reset()/predict_action_chunk).
    s_lat = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True); s_lat.set_timesteps(args.nfe)
    s_act = FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True); s_act.set_timesteps(args.nfe)
    tl, ta = R.policy._train_sched_latent, R.policy._train_sched_action  # 1000-step training grids
    opt = {n: torch.optim.AdamW(R.params[n], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0) for n in ("default", "fake")}
    n_train = sum(p.numel() for p in R.params["default"])
    write_manifest(args, out, len(data), n_train, seed=0)
    print(f"{len(data)} episodes | trainable/adapter {n_train / 1e6:.1f} M")

    for step in range(1, args.steps + 1):
        ep = data[random.randrange(len(data))]
        z = ep["latents"].cuda()[None].float()                       # [1,48,F,h,w] clean video (context + frame 0)
        a = ep["actions"].cuda()[None]                               # [1,30,F,16,1]
        am = ep["actions_mask"].cuda()[None]
        c = ep["text_emb"].cuda()[None].to(torch.bfloat16)
        f = z.shape[2]

        # ---- generator: backward simulation to student step k, then one step with gradients
        k = random.randrange(args.nfe)
        R.use("default")
        x_l = torch.randn_like(z); x_a = torch.randn_like(a)       # unmasked noise, as at inference
        x_l[:, :, :1] = z[:, :, :1]; x_a[:, :, :1] = 0             # inference: clean first frame, zero first actions
        with torch.no_grad():
            for i in range(k):
                v_l, v_a = R.velocity(x_l, x_a, z, a, s_lat.timesteps[i].repeat(f), s_act.timesteps[i].repeat(f), c)
                x_l = x_l + v_l * (s_lat.sigmas[i + 1] - s_lat.sigmas[i]).item()  # Euler, as FlowMatchScheduler.step
                x_a = x_a + v_a * (s_act.sigmas[i + 1] - s_act.sigmas[i]).item()
                x_l[:, :, :1] = z[:, :, :1]; x_a[:, :, :1] = 0
        v_l, v_a = R.velocity(x_l, x_a, z, a, s_lat.timesteps[k].repeat(f), s_act.timesteps[k].repeat(f), c)
        g_l = x_l - s_lat.sigmas[k].item() * v_l                       # clean estimates x0_G
        g_a_ = (x_a - s_act.sigmas[k].item() * v_a) * am
        g_l = torch.cat([z[:, :, :1], g_l[:, :, 1:]], dim=2)        # frame 0 stays the clean conditioning frame

        # ---- distribution matching: teacher (guided) vs fake score at a random noise level tau
        with torch.no_grad():
            il = _sample_timestep_id(1, *TAU_RANGE, tl.num_train_timesteps)
            ia = _sample_timestep_id(1, *TAU_RANGE, ta.num_train_timesteps)
            tau_l, tau_a = tl.timesteps[il].repeat(f), ta.timesteps[ia].repeat(f)
            sg_l, sg_a = tl.sigmas[il].item(), ta.sigmas[ia].item()
            xt_l = (1 - sg_l) * g_l + sg_l * torch.randn_like(z)
            xt_a = ((1 - sg_a) * g_a_ + sg_a * torch.randn_like(a)) * am   # masked, as the teacher was trained
            with R.peft.disable_adapter():
                rc_l, rc_a = R.velocity(xt_l, xt_a, z, a, tau_l, tau_a, c)
                if g_v > 1 or g_a > 1:
                    ru_l, ru_a = R.velocity(xt_l, xt_a, z, a, tau_l, tau_a, empty)
                    rc_l = ru_l + g_v * (rc_l - ru_l)
                    rc_a = ru_a + g_a * (rc_a - ru_a)
            R.use("fake")
            fk_l, fk_a = R.velocity(xt_l, xt_a, z, a, tau_l, tau_a, c)
            real_l, real_a = xt_l - sg_l * rc_l, xt_a - sg_a * rc_a
            fake_l, fake_a = xt_l - sg_l * fk_l, xt_a - sg_a * fk_a
            nrm_l = (g_l - real_l)[:, :, 1:].abs().mean().clamp(min=1e-4)
            nrm_a = (((g_a_ - real_a).abs() * am)[:, :, 1:].sum() / am[:, :, 1:].sum().clamp(min=1)).clamp(min=1e-4)
            grad_l = (fake_l - real_l) / nrm_l
            grad_a = (fake_a - real_a) / nrm_a
        R.use("default")
        loss_g = 0.5 * masked_mse(g_l, (g_l - grad_l).detach()) + 0.5 * masked_mse(g_a_, (g_a_ - grad_a).detach(), am)
        opt["default"].zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(R.params["default"], 1.0)
        opt["default"].step()

        # ---- fake score: flow matching on the (detached) generator sample, teacher's own training loss
        R.use("fake")
        s_l, s_a = g_l.detach(), g_a_.detach()
        for _ in range(args.fake_per_gen):
            ids_l = _sample_timestep_id(f, num_train_timesteps=tl.num_train_timesteps)
            ids_a = _sample_timestep_id(f, num_train_timesteps=ta.num_train_timesteps)
            e_l, e_a = torch.randn_like(s_l), torch.randn_like(s_a)
            sl = tl.sigmas[ids_l].to(s_l).view(1, 1, f, 1, 1); sa = ta.sigmas[ids_a].to(s_a).view(1, 1, f, 1, 1)
            n_l = (1 - sl) * s_l + sl * e_l; n_a = ((1 - sa) * s_a + sa * e_a) * am
            p_l, p_a = R.velocity(n_l, n_a, z, a, tl.timesteps[ids_l], ta.timesteps[ids_a], c)
            w_l = tl.training_weight(tl.timesteps[ids_l].cuda()).view(1, 1, f, 1, 1)
            w_a = ta.training_weight(ta.timesteps[ids_a].cuda()).view(1, 1, f, 1, 1)
            loss_f = ((p_l - (e_l - s_l)) ** 2 * w_l).mean() \
                + ((p_a - (e_a - s_a) * am) ** 2 * w_a * am).sum() / am.sum().clamp(min=1)
            opt["fake"].zero_grad(set_to_none=True)
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(R.params["fake"], 1.0)
            opt["fake"].step()

        if step % 10 == 0:
            print(f"step {step} loss_gen {loss_g.item():.4f} loss_fake {loss_f.item():.4f} "
                  f"gap_v {nrm_l.item():.4f} gap_a {nrm_a.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            R.peft.save_pretrained(out / f"step_{step:06d}", selected_adapters=["default", "fake"])


# -------------------------------------------------------------------------------------------- export
def export(args):
    """Merge the generator adapter into the teacher weights and write a plain LeRobot policy dir
    (config set to the student's NFE and guidance 1), loadable by the eval harness as --policy.path."""
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    from peft import PeftModel

    cfg = PreTrainedConfig.from_pretrained(TEACHER, revision=TEACHER_REV)
    policy = LingBotVAPolicy.from_pretrained(TEACHER, config=cfg, revision=TEACHER_REV)
    merged = PeftModel.from_pretrained(policy, Path(args.run) / f"step_{args.step:06d}").merge_and_unload()
    merged.config.num_inference_steps = merged.config.action_num_inference_steps = args.nfe
    merged.config.guidance_scale = merged.config.action_guidance_scale = 1.0  # guidance is distilled in
    dest = Path(args.dest); dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest)
    for p in Path(snapshot_download(TEACHER, revision=TEACHER_REV)).glob("policy_*processor*"):
        shutil.copy(p, dest / p.name)
    print("exported", dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tasks = "pick_dual_bottles,stamp_seal,place_mouse_pad,place_fan,place_burger_fries"
    sub.add_parser("fetch").add_argument("--tasks", default=tasks)
    t = sub.add_parser("train")
    t.add_argument("--nfe", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--tasks", default=tasks)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--fake-per-gen", type=int, default=5, help="fake-score updates per generator update (DMD2 TTUR)")
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--rank", type=int, default=64)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--save-every", type=int, default=250)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True); e.add_argument("--step", type=int, required=True)
    e.add_argument("--nfe", type=int, required=True); e.add_argument("--dest", required=True)
    args = ap.parse_args()
    if args.cmd == "export":
        return export(args)
    args.tasks = args.tasks.split(",")
    fetch(args.tasks) if args.cmd == "fetch" else train(args)


if __name__ == "__main__":
    main()
