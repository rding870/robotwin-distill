"""Record DMD2 training curves from a dmd2_lora.py train log.

Parses the `step N loss_gen X loss_fake Y gap_v Z gap_a W` lines and writes, per log,
a CSV of the raw series and a PNG with one panel per metric (raw points + rolling mean).
Re-run any time; outputs are overwritten in place.

    python scripts/plot_training.py logs/dmd2_nfe16.log logs/dmd2_nfe8.log --out distill/curves
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["loss_gen", "loss_fake", "gap_v", "gap_a"]
TITLES = {"loss_gen": "generator loss", "loss_fake": "fake-score loss",
          "gap_v": "gap_v (teacher disagreement, video)", "gap_a": "gap_a (teacher disagreement, actions)"}
LINE = re.compile(r"^step (\d+) loss_gen ([\d.]+) loss_fake ([\d.]+) gap_v ([\d.]+) gap_a ([\d.]+)")

# chart chrome (dataviz reference palette, light mode)
BLUE, INK, MUTED, GRID, BASE, SURFACE = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def parse(log: Path):
    rows = []
    for line in log.read_text(errors="replace").replace("\r", "\n").splitlines():
        if m := LINE.match(line.strip()):
            rows.append([int(m[1])] + [float(g) for g in m.groups()[1:]])
    return rows


def rolling(xs, n=15):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - n + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "distill" / "curves")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for log in args.logs:
        rows = parse(log)
        if not rows:
            print(f"{log}: no step lines found, skipped")
            continue
        name = log.stem.removeprefix("dmd2_")

        with (args.out / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + METRICS)
            w.writerows(rows)

        steps = [r[0] for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for ax, metric in zip(axes.flat, METRICS):
            i = 1 + METRICS.index(metric)
            ys = [r[i] for r in rows]
            ax.set_facecolor(SURFACE)
            ax.plot(steps, ys, ".", ms=2.5, color=BLUE, alpha=0.25)
            ax.plot(steps, rolling(ys), lw=2, color=BLUE, solid_capstyle="round")
            ax.set_title(TITLES[metric], fontsize=10, color=INK, loc="left")
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(BASE)
            ax.set_xlabel("generator step", fontsize=8, color=MUTED)
            ax.set_ylim(bottom=0)
        fig.suptitle(f"DMD2 training — {name}  (dots: per-10-step values, line: rolling mean of 15)",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.out / f"{name}.png", facecolor=SURFACE)
        plt.close(fig)
        print(f"{log} -> {args.out / name}.csv/.png ({len(rows)} points, last step {steps[-1]})")


if __name__ == "__main__":
    main()
