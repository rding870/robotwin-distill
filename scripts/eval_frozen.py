"""Evaluate a LeRobot policy on the frozen RoboTwin seed list (Stage 1 harness).

Thin driver around lerobot_eval.rollout(): one rollout per (task, seed) from
eval/seeds/<version>/frozen_seeds.json, with per-chunk inference latency, per-episode wall
time, a run manifest (policy/env config, NFE, revisions, hardware) and a results table.
Resumable: episodes already in episodes.jsonl are skipped.

Wrapper flags (parsed here) come first; everything else is passed to LeRobot's draccus CLI
exactly as for `lerobot-eval` (--policy.path=..., --env.type=robotwin, --env.action_mode=ee, ...).

    python scripts/eval_frozen.py --seeds eval/seeds/v1/frozen_seeds.json --run-name teacher_nfe_default \
        -- --policy.path=lerobot/lingbot_va_robotwin --policy.pretrained_revision=<sha> \
           --policy.wan_pretrained_path=<dir> --policy.device=cuda \
           --env.type=robotwin --env.task=beat_block_hammer --env.action_mode=ee
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_seed(eval_seed: int, task: str, seed: int) -> int:
    """RNG seed for one episode: a pure function of (eval seed, task, scene seed), so the policy's
    sampled noise does not depend on which episodes ran before it (resume, --tasks subsets)."""
    h = hashlib.sha256(f"{eval_seed}:{task}:{seed}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def _harness_version() -> dict:
    """Identify the harness code itself; results change when it does (e.g. expert-state patches)."""
    try:
        dirty = bool(subprocess.check_output(["git", "-C", str(PROJ), "status", "--porcelain", "--", "scripts"],
                                             text=True).strip())
    except Exception:
        dirty = None
    return {"git_sha": _git_sha(PROJ), "scripts_dirty": dirty, "eval_frozen_sha256": _sha256(Path(__file__))}


def _jsonable(o):
    if dataclasses.is_dataclass(o):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _lat_stats(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_s": sum(xs) / len(xs),
        "median_s": _pct(xs, 0.5),
        "p95_s": _pct(xs, 0.95),
        "max_s": max(xs),
    }


def _patch_expert_state(tasks) -> dict:
    """Some RoboTwin tasks set state in the expert's play_once() that check_success() reads.
    RoboTwin's eval_policy.py runs play_once() on the same task object before the policy episode,
    so that state carries over; LeRobot's env never calls play_once(). Recompute it after
    setup_demo() exactly as play_once() does (a pure function of the seeded scene)."""
    import importlib

    patched = {}
    if "open_laptop" in tasks:
        mod = importlib.import_module("envs.open_laptop")
        cls = mod.open_laptop
        if not getattr(cls, "_arm_tag_patched", False):
            orig_setup = cls.setup_demo

            def setup_demo(self, *a, **k):
                r = orig_setup(self, *a, **k)
                face_prod = mod.get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
                self.arm_tag = mod.ArmTag("left" if face_prod > 0 else "right")
                return r

            cls.setup_demo = setup_demo
            cls._arm_tag_patched = True
        patched["open_laptop"] = "arm_tag set after setup_demo as in play_once (face_prod of laptop pose)"
    return patched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True, help="eval/seeds/<version>/frozen_seeds.json")
    ap.add_argument("--run-name", required=True, help="results go to eval/results/<run-name>/")
    ap.add_argument("--tasks", default=None, help="comma-separated subset of the frozen tasks (default: all)")
    ap.add_argument("--max-seeds-per-task", type=int, default=None, help="dev runs: only the first N seeds per task")
    ap.add_argument("--no-video", action="store_true", help="skip per-episode head-camera mp4s")
    ap.add_argument("--abort-on-error", action="store_true",
                    help="stop the run at the first episode that raises (smoke runs before a full run)")
    ap.add_argument("--no-prompt-cache", action="store_true",
                    help="re-encode the text prompt with UMT5 on every episode (LeRobot default) instead of caching per prompt string")
    ap.add_argument("--episode-length", default="robotwin",
                    help="max env steps per episode: 'robotwin' = RoboTwin's per-task limit from "
                         "RoboTwin/task_config/_eval_step_limit.yml (default); an int applies to every task; "
                         "'lerobot' = LeRobot's RoboTwinEnvConfig default (1200)")
    ap.add_argument("--video-steps", type=int, default=None, help="override policy.num_inference_steps (video denoising NFE)")
    ap.add_argument("--action-steps", type=int, default=None, help="override policy.action_num_inference_steps (action denoising NFE)")
    wrapper, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    # Hand the remaining args to LeRobot's draccus parser (it reads sys.argv).
    sys.argv = [sys.argv[0]] + rest

    import torch
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.eval_stats import success_summary
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import init_logging

    @parser.wrap()
    def run(cfg: EvalPipelineConfig) -> None:
        init_logging()
        log = logging.getLogger("eval_frozen")
        if cfg.policy is None:
            raise ValueError("pass --policy.path=...")
        if wrapper.video_steps is not None:
            cfg.policy.num_inference_steps = wrapper.video_steps
        if wrapper.action_steps is not None:
            cfg.policy.action_num_inference_steps = wrapper.action_steps

        seeds_path = Path(wrapper.seeds).resolve()
        frozen = json.loads(seeds_path.read_text())
        tasks = list(frozen["tasks"])
        if wrapper.tasks:
            want = [t.strip() for t in wrapper.tasks.split(",") if t.strip()]
            missing = [t for t in want if t not in tasks]
            if missing:
                raise ValueError(f"tasks not in frozen list: {missing}")
            tasks = want
        plan = {t: frozen["tasks"][t]["accepted_seeds"][: wrapper.max_seeds_per_task] for t in tasks}
        n_total = sum(len(v) for v in plan.values())

        # Per-task step limit. RoboTwin's own evaluator stops an episode after
        # _eval_step_limit.yml[task] calls to take_action(); LeRobot's env counts the same unit
        # (one take_action per env.step), so the limit maps 1:1 onto episode_length.
        import yaml
        if wrapper.episode_length == "robotwin":
            limits = yaml.safe_load((PROJ / "RoboTwin" / "task_config" / "_eval_step_limit.yml").read_text())
            step_limit = {t: int(limits[t]) for t in tasks}
        elif wrapper.episode_length == "lerobot":
            step_limit = {t: int(cfg.env.episode_length) for t in tasks}
        else:
            step_limit = {t: int(wrapper.episode_length) for t in tasks}

        out = PROJ / "eval" / "results" / wrapper.run_name
        out.mkdir(parents=True, exist_ok=True)
        ep_file = out / "episodes.jsonl"
        done = set()
        if ep_file.exists():
            for line in ep_file.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["task"], r["seed"]))
        log.info("run=%s tasks=%d episodes=%d already_done=%d", wrapper.run_name, len(tasks), n_total, len(done))

        device = get_safe_torch_device(cfg.policy.device, log=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_seed(cfg.seed)  # policy load; every episode is reseeded with _episode_seed()

        # ---- manifest -------------------------------------------------------------------------
        gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        manifest = {
            "run_name": wrapper.run_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv[1:],
            "wrapper_args": vars(wrapper),
            "seeds_file": str(seeds_path.relative_to(PROJ)) if seeds_path.is_relative_to(PROJ) else str(seeds_path),
            "seeds_sha256": _sha256(seeds_path),
            "seeds_version": frozen.get("version"),
            "plan": plan,
            "policy": {
                "type": cfg.policy.type,
                "pretrained_path": str(cfg.policy.pretrained_path),
                "pretrained_revision": cfg.policy.pretrained_revision,
                "nfe": {
                    "video_num_inference_steps": getattr(cfg.policy, "num_inference_steps", None),
                    "action_num_inference_steps": getattr(cfg.policy, "action_num_inference_steps", None),
                },
                "config": _jsonable(cfg.policy),
            },
            "env": _jsonable(cfg.env),
            "episode_length_mode": wrapper.episode_length,
            "step_limit_per_task": step_limit,
            "eval_seed": cfg.seed,
            "rng_seeding": "per_episode: set_seed(sha256(f'{eval_seed}:{task}:{seed}')[:4] & 0x7fffffff)",
            "prompt_cache": not wrapper.no_prompt_cache,
            "expert_state_patches": _patch_expert_state(tasks),
            "harness": _harness_version(),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "lerobot_sha": _git_sha(PROJ / "lerobot"),
                "robotwin_sha": _git_sha(PROJ / "RoboTwin"),
            },
            "hardware": {
                "hostname": socket.gethostname(),
                "gpu": gpu.name if gpu else None,
                "gpu_mem_gb": round(gpu.total_memory / 2**30, 1) if gpu else None,
                "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip() or None,
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

        # ---- policy ---------------------------------------------------------------------------
        log.info("Loading policy")
        t0 = time.perf_counter()
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        load_s = time.perf_counter() - t0
        log.info("policy loaded in %.0fs", load_s)

        # ---- instrumentation ------------------------------------------------------------------
        timing = {"chunk": [], "select": []}

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        orig_chunk = policy.predict_action_chunk

        def timed_chunk(*a, **k):
            _sync(); t = time.perf_counter()
            r = orig_chunk(*a, **k)
            _sync(); timing["chunk"].append(time.perf_counter() - t)
            return r

        policy.predict_action_chunk = timed_chunk

        orig_select = policy.select_action

        def timed_select(*a, **k):
            _sync(); t = time.perf_counter()
            r = orig_select(*a, **k)
            _sync(); timing["select"].append(time.perf_counter() - t)
            return r

        policy.select_action = timed_select

        if not wrapper.no_prompt_cache and hasattr(policy, "_encode_prompt"):
            # policy.reset() drops the prompt embeddings every episode and UMT5 runs on CPU; the
            # embedding is a pure function of the prompt string, so cache it. Identical outputs.
            cache: dict[str, tuple] = {}
            orig_encode = policy._encode_prompt

            def cached_encode(prompt):
                if prompt not in cache:
                    cache[prompt] = orig_encode(prompt)
                return cache[prompt]

            policy._encode_prompt = cached_encode

        # ---- episodes -------------------------------------------------------------------------
        fps = 10
        n_done_now = 0
        run_t0 = time.perf_counter()
        for task in tasks:
            todo = [s for s in plan[task] if (task, s) not in done]
            if not todo:
                log.info("[%s] all %d seeds already done", task, len(plan[task]))
                continue
            env_cfg = replace(cfg.env, task=task, episode_length=step_limit[task])
            log.info("[%s] episode_length=%d", task, step_limit[task])
            envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
            vec = envs[task][0]
            try:
                for seed in todo:
                    frames = []
                    cb = None
                    if not wrapper.no_video:
                        def cb(e, _frames=frames):
                            _frames.append(e.envs[0].render())
                    timing["chunk"].clear(); timing["select"].clear()
                    err = None
                    rng_seed = _episode_seed(cfg.seed, task, seed)
                    set_seed(rng_seed)
                    t_ep = time.perf_counter()
                    try:
                        with torch.no_grad():
                            ret = rollout(vec, policy, env_pre, env_post, preprocessor, postprocessor,
                                          seeds=[seed], render_callback=cb)
                        success = bool(ret["success"][0].any().item())
                        n_steps = int(ret["done"].shape[1])
                    except Exception as e:  # keep going; the failure is a result, not a crash
                        log.exception("[%s] seed=%d rollout failed", task, seed)
                        success, n_steps, err = False, None, f"{type(e).__name__}: {e}"[:500]
                    wall = time.perf_counter() - t_ep
                    chunk = list(timing["chunk"]); select = list(timing["select"])
                    infer_s = sum(select)
                    rec = {
                        "task": task,
                        "seed": seed,
                        "rng_seed": rng_seed,
                        "success": success,
                        "n_steps": n_steps,
                        "step_limit": step_limit[task],
                        "truncated": (n_steps is not None and not success and n_steps >= step_limit[task]),
                        "error": err,
                        "wall_s": round(wall, 3),
                        "inference_s": round(infer_s, 3),          # all select_action time (incl. prompt encode + chunks)
                        "env_s": round(wall - infer_s, 3),         # sim stepping + rendering + IK
                        "n_chunks": len(chunk),
                        "chunk_latency_s": [round(x, 4) for x in chunk],
                        "first_chunk_s": round(chunk[0], 4) if chunk else None,
                        "steady_chunk_mean_s": round(sum(chunk[1:]) / len(chunk[1:]), 4) if len(chunk) > 1 else None,
                        "first_select_s": round(select[0], 4) if select else None,
                        "video": None,
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    if frames and not wrapper.no_video:
                        vp = out / "videos" / task / f"seed_{seed}.mp4"
                        vp.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            write_video(str(vp), frames, fps)
                            rec["video"] = str(vp.relative_to(PROJ))
                        except Exception as e:
                            log.warning("video write failed: %s", e)
                    with ep_file.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done.add((task, seed)); n_done_now += 1
                    elapsed = time.perf_counter() - run_t0
                    remaining = n_total - len(done)
                    log.info("[%s] seed=%d success=%s steps=%s wall=%.0fs infer=%.0fs chunks=%d steady_chunk=%.2fs | %d/%d done, ETA %.1fh",
                             task, seed, success, n_steps, wall, infer_s, len(chunk),
                             rec["steady_chunk_mean_s"] or 0.0, len(done), n_total,
                             remaining * (elapsed / max(n_done_now, 1)) / 3600)
                    if err and wrapper.abort_on_error:
                        raise SystemExit(f"--abort-on-error: [{task}] seed={seed}: {err}")
            finally:
                close_envs(envs)

        summary = write_summary(out, manifest, plan, success_summary)
        log.info("done: %s", out)
        if not summary["valid"]:
            log.error("INVALID run: episodes raised in %s (see episodes.jsonl 'error')", summary["tasks_with_errors"])
            sys.exit(3)

    def write_summary(out: Path, manifest: dict, plan: dict, success_summary) -> None:
        recs = [json.loads(l) for l in (out / "episodes.jsonl").read_text().splitlines() if l.strip()]
        by_task = {t: [r for r in recs if r["task"] == t] for t in plan}
        nfe = manifest["policy"]["nfe"]

        def agg(rs):
            s = success_summary([r["success"] for r in rs])
            steady = [x for r in rs for x in r["chunk_latency_s"][1:]]
            first = [r["first_chunk_s"] for r in rs if r["first_chunk_s"] is not None]
            # An episode that raised is a harness/env fault, not a policy failure: count it in
            # pc_success (conservative) but also report the rate over episodes that ran.
            ran = success_summary([r["success"] for r in rs if not r["error"]])
            s.update({
                "n_planned": None,
                "n_errors": sum(1 for r in rs if r["error"]),
                "n_ran": ran["n_episodes"],
                "pc_success_ran": ran["pc_success"],
                "pc_success_ran_ci95": ran["pc_success_ci95"],
                "mean_steps": (sum(r["n_steps"] for r in rs if r["n_steps"]) / max(1, sum(1 for r in rs if r["n_steps"]))) if rs else None,
                "mean_wall_s": sum(r["wall_s"] for r in rs) / len(rs) if rs else None,
                "mean_inference_s": sum(r["inference_s"] for r in rs) / len(rs) if rs else None,
                "mean_env_s": sum(r["env_s"] for r in rs) / len(rs) if rs else None,
                "chunk_latency_steady": _lat_stats(steady),
                "chunk_latency_first": _lat_stats(first),
                "mean_chunks_per_episode": sum(r["n_chunks"] for r in rs) / len(rs) if rs else None,
            })
            return s

        per_task = {}
        for t, rs in by_task.items():
            a = agg(rs); a["n_planned"] = len(plan[t]); per_task[t] = a
        overall = agg(recs); overall["n_planned"] = sum(len(v) for v in plan.values())
        tasks_with_errors = [t for t, a in per_task.items() if a["n_errors"]]
        summary = {"run_name": manifest["run_name"], "nfe": nfe, "seeds_file": manifest["seeds_file"],
                   "seeds_sha256": manifest["seeds_sha256"], "hardware": manifest["hardware"],
                   "harness": manifest.get("harness"),
                   "valid": not tasks_with_errors, "tasks_with_errors": tasks_with_errors,
                   "per_task": per_task, "overall": overall}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

        def f(x, d=1):
            return "-" if x is None or x != x else f"{x:.{d}f}"
        hw = manifest["hardware"]; pol = manifest["policy"]; env = manifest["env"]
        lines = [
            f"# {manifest['run_name']}", "",
            f"policy `{pol['pretrained_path']}` @ `{pol['pretrained_revision']}` | NFE video/action = "
            f"{nfe['video_num_inference_steps']}/{nfe['action_num_inference_steps']} | "
            f"guidance {pol['config'].get('guidance_scale')}/{pol['config'].get('action_guidance_scale')} | dtype {pol['config'].get('dtype')}",
            f"env robotwin action_mode={env.get('action_mode')} step_limit={manifest.get('episode_length_mode')} "
            f"({min(manifest['step_limit_per_task'].values())}-{max(manifest['step_limit_per_task'].values())}) | "
            f"seeds `{manifest['seeds_file']}` sha256 {manifest['seeds_sha256'][:12]} | {hw['gpu']} driver {hw['driver']} | "
            f"torch {manifest['software']['torch']} lerobot {manifest['software']['lerobot_sha'][:8]} robotwin {manifest['software']['robotwin_sha'][:8]}"
            f" | harness {(manifest.get('harness') or {}).get('git_sha') or '-'}"
            f"{' (scripts dirty)' if (manifest.get('harness') or {}).get('scripts_dirty') else ''}",
            "",
        ]
        if tasks_with_errors:
            lines += [f"**INVALID RUN:** episodes raised in {', '.join(tasks_with_errors)} "
                      "(counted as failures in `rate %`; see `error` in episodes.jsonl).", ""]
        lines += [
            "| task | episodes | success | rate % | 95% CI | errors | rate % excl. errors | steady chunk s (mean/p95) | first chunk s | infer s/ep | env s/ep | wall s/ep |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t, a in list(per_task.items()) + [("**overall**", overall)]:
            ci = a["pc_success_ci95"]; st = a["chunk_latency_steady"]; fc = a["chunk_latency_first"]
            lines.append(
                f"| {t} | {a['n_episodes']}/{a['n_planned']} | {a['n_success']} | {f(a['pc_success'])} | "
                f"{f(ci[0])}–{f(ci[1])} | {a['n_errors']} | {f(a['pc_success_ran'])} | {f(st.get('mean_s'), 2)}/{f(st.get('p95_s'), 2)} | "
                f"{f(fc.get('mean_s'), 2)} | {f(a['mean_inference_s'], 0)} | {f(a['mean_env_s'], 0)} | {f(a['mean_wall_s'], 0)} |")
        (out / "summary.md").write_text("\n".join(lines) + "\n")
        return summary

    run()


if __name__ == "__main__":
    main()
