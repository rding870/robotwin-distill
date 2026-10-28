# Frozen evaluation seeds (Stage 1 artifact)

RoboTwin 2.0 has no fixed test set: scenes are generated from integer seeds and RoboTwin's own
evaluator (`RoboTwin/script/eval_policy.py`) filters seeds at run time by whether the scripted expert
can solve them. To compare the teacher and every student on identical scenes we run that filter once
and freeze the result here. **Every eval run reads from `<version>/frozen_seeds.json`; nothing picks
seeds on the fly.**

## v1

Generated with `scripts/make_seed_list.py --version v1 --n-tasks 10 --task-rng-seed 0 --seeds-per-task 20 --start-seed 100000`.

* **Tasks (10 of 50):** `sorted(random.Random(0).sample(sorted(all_50_tasks), 10))` -> `v1/tasks.json`.
  Chosen once; never re-rolled.
* **Seeds per task (20):** walk seeds upward from 100000; run the expert (`setup_demo` + `play_once`
  under `demo_clean`, embodiment `aloha-agilex`, `eval_mode=True`); accept iff no exception,
  `plan_success` and `check_success()` - the same rule as `eval_policy.py`. Keep the first 20 accepted.
* **Rejections are kept**, with reason (`plan_failed`, `check_success_false`, `UnStableError`, other
  exception), in `v1/<task>.json`, as the brief asks for excluded episodes to be recorded.
* Sim pinned to RoboTwin `0aeea2d` (see `configs/pins.txt`). The expert's CuRobo planner is GPU-sampled
  and not bit-reproducible on borderline seeds, so the saved file is the artifact, not the procedure.

Files
```
v1/tasks.json          the 10 task names + the pool and rule they were drawn from
v1/<task>.json         accepted + rejected seeds for that task, with timing and reasons
v1/frozen_seeds.json   the list consumers read: {task: {accepted_seeds: [...], n_rejected, complete}}
```
Do not edit v1 files by hand. A different task set or count is a new version (v2, ...).
