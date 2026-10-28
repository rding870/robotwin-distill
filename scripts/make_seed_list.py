"""Build the frozen evaluation seed list (Stage 1 artifact).

1. Pick N tasks from RoboTwin's 50 with a fixed RNG seed -> the task set never changes.
2. For each task, walk seeds upward from --start-seed and run RoboTwin's scripted expert
   (same accept rule as RoboTwin/script/eval_policy.py: no exception, plan_success and
   check_success). Keep the first --seeds-per-task accepted seeds; log every rejection.
3. Write eval/seeds/<version>/{tasks.json, <task>.json, frozen_seeds.json}.

Each task runs in its own subprocess so a simulator crash in one task cannot take the
whole sweep down, and per-task files make the sweep resumable.

Usage (from the project root, after `source env.sh`):
    python scripts/make_seed_list.py --version v1
    python scripts/make_seed_list.py --version v1 --task beat_block_hammer   # single task (worker mode)
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
ROBOTWIN = PROJ / "RoboTwin"
ENVS = ROBOTWIN / "envs"


def all_tasks() -> list[str]:
    """Every envs/<name>.py that defines `class <name>(` — 50 tasks at the pinned RoboTwin SHA."""
    names = []
    for f in sorted(ENVS.glob("*.py")):
        n = f.stem
        if n.startswith("_"):
            continue
        if f"class {n}(" in f.read_text(encoding="utf-8"):
            names.append(n)
    return names


def pick_tasks(n: int, rng_seed: int) -> list[str]:
    tasks = all_tasks()
    return sorted(random.Random(rng_seed).sample(tasks, n))


def sweep_task(task: str, start_seed: int, want: int, max_candidates: int, out_file: Path) -> dict:
    """Worker: run the expert on consecutive seeds until `want` are accepted. Writes out_file after every seed."""
    from lerobot.envs.robotwin import _load_robotwin_setup_kwargs, _load_robotwin_task

    try:
        from envs.utils.create_actor import UnStableError  # RoboTwin's own physics-instability signal
    except Exception:  # pragma: no cover
        UnStableError = ()  # type: ignore[assignment]

    kwargs = _load_robotwin_setup_kwargs(task)
    kwargs["eval_mode"] = True
    env = _load_robotwin_task(task)()

    rec = {"task": task, "start_seed": start_seed, "want": want, "accepted": [], "rejected": [], "done": False}
    if out_file.exists():  # resume
        prev = json.loads(out_file.read_text())
        if prev.get("start_seed") == start_seed:
            rec = prev
    tried = {r["seed"] for r in rec["rejected"]} | {r["seed"] for r in rec["accepted"]}
    seed = max(tried) + 1 if tried else start_seed

    while len(rec["accepted"]) < want and seed < start_seed + max_candidates:
        if seed in tried:
            seed += 1
            continue
        t0 = time.perf_counter()
        ok, reason = False, ""
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **kwargs)
            env.play_once()
            if not env.plan_success:
                reason = "plan_failed"
            elif not env.check_success():
                reason = "check_success_false"
            else:
                ok = True
        except UnStableError as e:
            reason = f"UnStableError: {e}"
        except Exception as e:  # mirrors eval_policy.py: any expert error rejects the seed
            reason = f"{type(e).__name__}: {e}"
        finally:
            try:
                env.close_env()
            except Exception:
                pass
        dt = round(time.perf_counter() - t0, 1)
        entry = {"seed": seed, "seconds": dt}
        if ok:
            rec["accepted"].append(entry)
        else:
            entry["reason"] = reason[:300]
            rec["rejected"].append(entry)
        print(f"[{task}] seed={seed} accepted={ok} {dt}s {reason} ({len(rec['accepted'])}/{want})", flush=True)
        out_file.write_text(json.dumps(rec, indent=1))
        seed += 1

    rec["done"] = len(rec["accepted"]) >= want
    rec["candidates_tried"] = len(rec["accepted"]) + len(rec["rejected"])
    out_file.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="eval/seeds/<version>/")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--task-rng-seed", type=int, default=0, help="fixes WHICH tasks are chosen")
    ap.add_argument("--seeds-per-task", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=100000)
    ap.add_argument("--max-candidates", type=int, default=300, help="give up on a task after this many seeds")
    ap.add_argument("--task", default=None, help="worker mode: sweep only this task")
    a = ap.parse_args()

    out_dir = PROJ / "eval" / "seeds" / a.version
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.task:  # worker
        sweep_task(a.task, a.start_seed, a.seeds_per_task, a.max_candidates, out_dir / f"{a.task}.json")
        return

    tasks_file = out_dir / "tasks.json"
    if tasks_file.exists():
        tasks = json.loads(tasks_file.read_text())["tasks"]
    else:
        tasks = pick_tasks(a.n_tasks, a.task_rng_seed)
        tasks_file.write_text(json.dumps({
            "tasks": tasks,
            "n_tasks": a.n_tasks,
            "task_rng_seed": a.task_rng_seed,
            "pool": all_tasks(),
            "rule": "sorted(random.Random(task_rng_seed).sample(sorted(pool), n_tasks))",
        }, indent=1))
    print("tasks:", tasks, flush=True)

    env = dict(os.environ, PYTHONPATH=f"{ROBOTWIN}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    for t in tasks:
        f = out_dir / f"{t}.json"
        if f.exists() and json.loads(f.read_text()).get("done"):
            print(f"[{t}] already done, skipping", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--version", a.version, "--task", t,
               "--start-seed", str(a.start_seed), "--seeds-per-task", str(a.seeds_per_task),
               "--max-candidates", str(a.max_candidates)]
        r = subprocess.run(cmd, cwd=ROBOTWIN, env=env)
        if r.returncode != 0:
            print(f"[{t}] worker exited {r.returncode}; partial results kept in {f}", flush=True)

    frozen = {"version": a.version, "start_seed": a.start_seed, "seeds_per_task": a.seeds_per_task,
              "task_rng_seed": a.task_rng_seed, "config": "demo_clean", "embodiment": "aloha-agilex",
              "robotwin_sha": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4", "tasks": {}}
    for t in tasks:
        f = out_dir / f"{t}.json"
        if not f.exists():
            frozen["tasks"][t] = {"accepted_seeds": [], "n_rejected": None, "complete": False}
            continue
        rec = json.loads(f.read_text())
        frozen["tasks"][t] = {
            "accepted_seeds": [e["seed"] for e in rec["accepted"]],
            "n_rejected": len(rec["rejected"]),
            "complete": bool(rec.get("done")),
        }
    (out_dir / "frozen_seeds.json").write_text(json.dumps(frozen, indent=1))
    n_ok = sum(len(v["accepted_seeds"]) for v in frozen["tasks"].values())
    print(f"wrote {out_dir / 'frozen_seeds.json'}: {n_ok} accepted seeds over {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
