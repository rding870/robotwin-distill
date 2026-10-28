"""Simulator-only smoke test: run RoboTwin's expert planner on a few seeds.

Exercises SAPIEN rendering, mplib/CuRobo planning and the success check, using the
same acceptance rule as RoboTwin's script/eval_policy.py (plan_success and check_success).

Usage: python scripts/smoke_expert.py --task beat_block_hammer --start-seed 100000 --n 3
"""

import argparse
import time

from lerobot.envs.robotwin import _load_robotwin_setup_kwargs, _load_robotwin_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="beat_block_hammer")
    ap.add_argument("--start-seed", type=int, default=100000)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    kwargs = _load_robotwin_setup_kwargs(a.task)
    kwargs["eval_mode"] = True
    env = _load_robotwin_task(a.task)()
    for seed in range(a.start_seed, a.start_seed + a.n):
        t0 = time.perf_counter()
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **kwargs)
            env.play_once()
            ok = bool(env.plan_success and env.check_success())
            err = ""
        except Exception as e:  # mirrors eval_policy.py: any expert error rejects the seed
            ok, err = False, f"{type(e).__name__}: {e}"
        finally:
            env.close_env()
        print(f"seed={seed} accepted={ok} {time.perf_counter() - t0:.1f}s {err}", flush=True)


if __name__ == "__main__":
    main()
