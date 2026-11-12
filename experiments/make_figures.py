#!/usr/bin/env python3
"""Result figures from the 100 Hz record (offline, no ROS; needs matplotlib).

Usage:
    python3 experiments/make_figures.py data/clock_100hz results/aggregate_100hz.json <figure dir>

  fig5_perception.png  (a) vision-stage center error per shape (arm parked,
                       latency-compensated): silhouette centroid (runs
                       h_cvn_r1..3) vs top-face recovery (h_cvc_r1..3);
                       (b) estimator error vs truth, detecting / coasting,
                       P0 vs P1
  fig6_tracking.png    (a) one P0 run (clk100_p0_r1): error vs the true target
                       and distance to the true contour over time; (b)
                       per-shape distance to the true contour, GT / P0 / P1
                       (mean +/- std of runs)
  fig7_paths.png       tool path in the TRUE part frame vs the outline (the
                       same P0 run)
  fig8_speed.png       belt-speed sweep: (a) accuracy, (b) perimeter coverage
                       (fraction of the outline visited within tolerance) and
                       cycle success

The aggregate file is the output of aggregate.py --clock100.
"""

import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

SHAPES = ["square", "circle", "triangle", "pentagon", "irregular"]
COL = {"gt": "#2c3e50", "p0": "#c0392b", "p1": "#2980b9"}
LAB = {"gt": "ground truth", "p0": "perception P0", "p1": "perception P1"}
plt.rcParams.update({"font.family": "serif", "font.size": 9})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# Run labels behind the data-driven panels.
CV_RAW = "h_cvn_r"          # vision stage without the corrections, repetitions 1..3
CV_COR = "h_cvc_r"          # vision stage with both corrections, repetitions 1..3
P0_RUN = "clk100_p0_r1"     # the single P0 run shown in fig6 (a) and fig7


def cv_errors(d, prefix):
    data = {s: [] for s in SHAPES}
    for i in (1, 2, 3):
        with open(os.path.join(d, f"{prefix}{i}_perception_log.csv")) as f:
            for r in csv.DictReader(f):
                gx, gy, ga = fnum(r["gt_x"]), fnum(r["gt_y"]), fnum(r["gt_age_s"])
                dx, dy, da = fnum(r["det_x"]), fnum(r["det_y"]), fnum(r["det_age_s"])
                if not all(map(math.isfinite, (gx, ga, dx, da))) or da > 0.3 or ga > 0.3:
                    continue
                if not -0.18 <= gy <= 0.18 or r["det_shape"] != r["gt_shape"]:
                    continue
                data[r["gt_shape"]].append(
                    1000 * math.hypot(dx - gx, dy - (gy - 0.03 * (da - ga))))
    return data


def fig5(d, agg, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    raw = cv_errors(d, CV_RAW)
    cor = cv_errors(d, CV_COR)
    for k, (data, color, lab) in enumerate(((raw, "#7f8c8d", "silhouette centroid"),
                                            (cor, "#c0392b", "top-face recovery"))):
        bp = ax1.boxplot([data[s] for s in SHAPES], showfliers=False, whis=(5, 95),
                         positions=[i + (k - 0.5) * 0.36 for i in range(5)],
                         widths=0.3, patch_artist=True,
                         medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set_facecolor(color)
        ax1.plot([], [], color=color, lw=6, label=lab)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(SHAPES)
    ax1.legend(fontsize=7)
    ax1.set_ylabel("center error [mm]")
    ax1.set_title("(a) vision stage, arm parked (box: quartiles, whiskers: 5-95 %)",
                  fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    regimes = [("est_detecting", "detecting"), ("est_coasting", "coasting"),
               ("est_err", "all")]
    w = 0.36
    for k, cond in enumerate(("p0", "p1")):
        g = agg["main"][cond]
        gu = agg["uncorrected"]["main"][cond]
        ax2.plot([i + (k - 0.5) * w for i in range(3)],
                 [gu[m]["run_mean"] for m, _ in regimes], "x", color="#7f8c8d", ms=7,
                 mew=1.5, label="without top-face recovery" if k == 0 else None)
        means = [g[m]["run_mean"] for m, _ in regimes]
        stds = [g[m]["run_std"] for m, _ in regimes]
        p95 = [g[m]["p95"] for m, _ in regimes]
        xs = [i + (k - 0.5) * w for i in range(3)]
        ax2.bar(xs, means, w, yerr=stds, color=COL[cond], label=LAB[cond], capsize=3)
        ax2.plot(xs, p95, "k_", ms=12, mew=1.5,
                 label="95th percentile" if k == 0 else None)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([n for _, n in regimes])
    ax2.set_ylabel("estimator error vs truth [mm]")
    ax2.set_title("(b) estimator (mean ± std of runs)", fontsize=9)
    # head room so that the legend never covers a marker
    ax2.set_ylim(0, 1.45 * ax2.get_ylim()[1])
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig6(d, agg, shapes, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    run = A.analyze_run(os.path.join(d, P0_RUN), shapes, 0.03)
    t0 = run["cycles"][0]["t0"]
    first = True
    for c in run["cycles"][:6]:
        if not c["tracked"]:
            continue
        ts = [s["t"] - t0 for s in c["win"]]
        ax1.plot(ts, [s["e_true"] for s in c["win"]], color="#c0392b", lw=0.8,
                 label="error vs true target" if first else None)
        ax1.plot(ts, [s["d_contour"] for s in c["win"]], color="#2c3e50", lw=0.8,
                 label="distance to true contour" if first else None)
        ax1.axvline(c["t0"] - t0, color="#999", lw=0.5, ls=":")
        ax1.text(c["t0"] - t0 + 0.5, 11.2, c["shape"], fontsize=6.5, color="#555")
        first = False
    ax1.axhline(5.0, color="#27ae60", lw=0.8, ls="--", label="5 mm tolerance")
    ax1.set_ylim(0, 15)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("error [mm]")
    ax1.set_title("(a) one P0 run, processing windows of six parts", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right", ncol=3)
    ax1.grid(True, alpha=0.3)

    w = 0.26
    for k, cond in enumerate(("gt", "p0", "p1")):
        means, stds = [], []
        for sh in SHAPES:
            g = agg["per_shape"][cond][sh]
            means.append(g["d_contour"]["run_mean"])
            stds.append(g["d_contour"]["run_std"])
        ax2.bar([i + (k - 1) * w for i in range(5)], means, w, yerr=stds,
                color=COL[cond], label=LAB[cond], capsize=2)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(SHAPES, fontsize=7.5)
    ax2.set_ylabel("distance to true contour [mm]")
    ax2.set_title("(b) per shape (mean ± std of runs)", fontsize=9)
    # head room so that the legend never covers a marker
    ax2.set_ylim(0, 1.45 * ax2.get_ylim()[1])
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig7(d, shapes, out):
    run = A.analyze_run(os.path.join(d, P0_RUN), shapes, 0.03)
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.6))
    for ax, name in zip(axes, SHAPES):
        kind, data = shapes[name]
        if kind == "circle":
            verts = [(data * math.cos(2 * math.pi * i / 72),
                      data * math.sin(2 * math.pi * i / 72)) for i in range(72)]
        else:
            verts = data
        vx = [v[1] * 1000 for v in verts] + [verts[0][1] * 1000]
        vy = [v[0] * 1000 for v in verts] + [verts[0][0] * 1000]
        ax.plot(vx, vy, color="#2c3e50", lw=1.6, label="true outline")
        first = True
        for c in run["cycles"]:
            if c["shape"] != name or not c["tracked"]:
                continue
            ax.plot([s["py"] * 1000 for s in c["win"]],
                    [s["px"] * 1000 for s in c["win"]], color="#c0392b", lw=0.7,
                    label="tool path" if first else None)
            first = False
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-125, 125)
        ax.set_ylim(-125, 125)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("part-frame x [mm]")
    axes[2].set_xlabel("part-frame y (belt direction) [mm]")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig8(agg, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 2.9))
    tags = [t for t in ("v020", "v030", "v045", "v060") if agg["speed"][t]["p0"]]
    vs = [agg["speed"][t]["v"] * 1000 for t in tags]
    for cond in ("gt", "p0"):
        gs = [agg["speed"][t][cond] for t in tags]
        if not all(gs):
            continue
        ax1.errorbar(vs, [g["d_contour"]["run_mean"] for g in gs],
                     yerr=[g["d_contour"]["run_std"] for g in gs], marker="o",
                     color=COL[cond], capsize=3, label=LAB[cond] + ", contour distance")
        ax1.errorbar(vs, [g["e_true"]["run_mean"] for g in gs],
                     yerr=[g["e_true"]["run_std"] for g in gs], marker="s", ls="--",
                     color=COL[cond], capsize=3, label=LAB[cond] + ", true-target error")
        ax2.plot(vs, [g["coverage"][0] for g in gs], marker="o", color=COL[cond],
                 label=LAB[cond] + ", perimeter coverage")
        ax2.plot(vs, [g["cycles_success"] / max(1, g["cycles_tracked"]) for g in gs],
                 marker="^", ls=":", color=COL[cond], label=LAB[cond] + ", success ratio")
    ax1.set_xlabel("belt speed [mm/s]")
    ax1.set_ylabel("mean error [mm]")
    ax1.set_title("(a) accuracy in the processing window", fontsize=9)
    ax1.legend(fontsize=6.5)
    ax1.grid(True, alpha=0.3)
    ax2.axhline(1.0, color="#27ae60", lw=0.8, ls="--")
    ax2.set_ylim(0.0, 1.08)
    ax2.set_xlabel("belt speed [mm/s]")
    ax2.set_ylabel("fraction")
    ax2.set_title("(b) perimeter coverage per pass and cycle success", fontsize=9)
    ax2.legend(fontsize=6.5)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    d, agg_path, fig_dir = sys.argv[1:4]
    with open(agg_path) as f:
        agg = json.load(f)
    shapes = A.load_catalog(A.DEFAULT_SHAPES)
    os.makedirs(fig_dir, exist_ok=True)
    fig5(d, agg, os.path.join(fig_dir, "fig5_perception.png"))
    fig6(d, agg, shapes, os.path.join(fig_dir, "fig6_tracking.png"))
    fig7(d, shapes, os.path.join(fig_dir, "fig7_paths.png"))
    fig8(agg, os.path.join(fig_dir, "fig8_speed.png"))
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
