"""Compare two eval_frozen.py runs of the same configuration (reproducibility check).

    python scripts/compare_runs.py eval/results/repro_a eval/results/repro_b --out eval/results/repro_report.md

Levels, strictest first:
  0. config     manifests identical except run name / start time (else the comparison is meaningless)
  1. bit-level  per-episode action fingerprints (actions_sha) identical
  2. outcomes   success, n_steps, error identical for every (task, seed)
  3. table      per-task and overall success counts identical (implied by 2)
  latency       steady-state chunk latency mean within --mean-tol, p95 within --p95-tol (relative)
Exit code 0 iff every level passes.
"""

import argparse
import json
import sys
from pathlib import Path

IGNORED = {"run_name", "started_at", "wrapper_args.run_name"}


def _load(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    eps = {}
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            eps[(r["task"], r["seed"])] = r
    return manifest, eps


def _flatten(o, prefix=""):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: o}


def _config_diff(ma, mb):
    # The run name also appears inside paths (argv --output_dir, env/policy config); normalise it away.
    fa = _flatten(json.loads(json.dumps(ma).replace(ma["run_name"], "<RUN>")))
    fb = _flatten(json.loads(json.dumps(mb).replace(mb["run_name"], "<RUN>")))
    return [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if k not in IGNORED and fa.get(k) != fb.get(k)]


def _pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _steady(rs):
    return [x for r in rs for x in r["chunk_latency_s"][1:]]  # first chunk also VAE-encodes the conditioning frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--mean-tol", type=float, default=0.05, help="max relative difference of steady chunk mean")
    ap.add_argument("--p95-tol", type=float, default=0.10, help="max relative difference of steady chunk p95")
    ap.add_argument("--out", type=Path, default=None, help="also write the report here (markdown)")
    args = ap.parse_args()

    ma, ea = _load(args.run_a)
    mb, eb = _load(args.run_b)
    na, nb = ma["run_name"], mb["run_name"]
    lines = [f"# Reproducibility: `{na}` vs `{nb}`", ""]
    verdict = {}

    # 0. config
    diff = _config_diff(ma, mb)
    verdict["config"] = not diff
    lines += ["## 0. Config", "", "identical (except run name / start time)" if not diff else "**differs:**", ""]
    lines += [f"- `{k}`: `{a}` vs `{b}`" for k, a, b in diff[:40]]
    if harness := ma.get("harness"):
        lines += [f"harness `{harness.get('git_sha')}` dirty={harness.get('scripts_dirty')}", ""]

    keys = sorted(set(ea) | set(eb))
    missing = [k for k in keys if k not in ea or k not in eb]
    both = [k for k in keys if k in ea and k in eb]

    # 1. bit-level + 2. outcomes, per episode
    rows, bit_ok, out_ok = [], True, True
    for k in both:
        a, b = ea[k], eb[k]
        sha_a, sha_b = a.get("actions_sha"), b.get("actions_sha")
        if sha_a is None or sha_b is None:
            bit = None
            bit_ok = False
        else:
            bit = sha_a == sha_b
            bit_ok &= bit
        first_div = obs_div = None
        if bit is False:
            ca, cb = a.get("chunk_sha", []), b.get("chunk_sha", [])
            first_div = next((i for i, (x, y) in enumerate(zip(ca, cb)) if x != y), min(len(ca), len(cb)))
            oa, ob = a.get("obs_sha", []), b.get("obs_sha", [])
            if oa and ob:  # older runs lack obs fingerprints
                obs_div = next((i for i, (x, y) in enumerate(zip(oa, ob)) if x != y), None)
        same = all(a[f] == b[f] for f in ("success", "n_steps", "error"))
        out_ok &= same
        rows.append((k, bit, first_div, obs_div, a, b, same))
    verdict["bit_level"] = bit_ok and not missing
    verdict["outcomes"] = out_ok and not missing

    lines += ["## 1-2. Episodes", "",
              f"{len(both)} episodes in both runs" + (f"; **{len(missing)} only in one run**: {missing}" if missing else ""), "",
              "| task | seed | actions identical | first differing chunk | first differing obs | success A/B | steps A/B | error A/B |",
              "|---|---|---|---|---|---|---|---|"]
    for (t, s), bit, fd, od, a, b, same in rows:
        mark = "" if same else " **≠**"
        # obs column: '-' = actions identical or no obs fingerprints; 'none' = inputs identical where
        # both recorded them (divergence is in the policy forward, not the env/renderer).
        obs = "-" if fd is None or not (a.get("obs_sha") and b.get("obs_sha")) else ("none" if od is None else od)
        lines.append(f"| {t} | {s} | {'-' if bit is None else ('yes' if bit else '**no**')} | {'-' if fd is None else fd} | {obs} | "
                     f"{a['success']}/{b['success']}{mark} | {a['n_steps']}/{b['n_steps']} | "
                     f"{bool(a['error'])}/{bool(b['error'])} |")
    lines.append("")

    # 3. table + latency, per task
    tasks = sorted({t for t, _ in both})
    table_ok, lat_ok = True, True
    lines += ["## 3. Success table and latency (steady-state chunks)", "",
              "| task | success A | success B | mean A s | mean B s | Δ mean | p95 A s | p95 B s | Δ p95 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for t in tasks + ["overall"]:
        ks = [k for k in both if t == "overall" or k[0] == t]
        sa = sum(ea[k]["success"] for k in ks); sb = sum(eb[k]["success"] for k in ks)
        table_ok &= sa == sb
        la, lb = _steady([ea[k] for k in ks]), _steady([eb[k] for k in ks])
        if la and lb:
            ma_, mb_, pa, pb = sum(la) / len(la), sum(lb) / len(lb), _pct(la, 0.95), _pct(lb, 0.95)
            dm, dp = (mb_ - ma_) / ma_, (pb - pa) / pa
            ok = abs(dm) <= args.mean_tol and abs(dp) <= args.p95_tol
            lat_ok &= ok
            lat = f"{ma_:.3f} | {mb_:.3f} | {dm:+.1%}{'' if abs(dm) <= args.mean_tol else ' **✗**'} | " \
                  f"{pa:.3f} | {pb:.3f} | {dp:+.1%}{'' if abs(dp) <= args.p95_tol else ' **✗**'}"
        else:
            lat = "- | - | - | - | - | -"
        name = f"**{t}**" if t == "overall" else t
        lines.append(f"| {name} | {sa}/{len(ks)} | {sb}/{len(ks)} | {lat} |")
    verdict["table"] = table_ok and not missing
    verdict["latency"] = lat_ok
    lines.append("")

    lines += ["## Verdict", "", "| level | result |", "|---|---|"]
    names = {"config": "0. config identical", "bit_level": "1. actions bit-identical", "outcomes": "2. episode outcomes identical",
             "table": "3. success table identical",
             "latency": f"latency within ±{args.mean_tol:.0%} mean / ±{args.p95_tol:.0%} p95"}
    lines += [f"| {names[k]} | {'PASS' if v else '**FAIL**'} |" for k, v in verdict.items()]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report)
    sys.exit(0 if all(verdict.values()) else 1)


if __name__ == "__main__":
    main()
