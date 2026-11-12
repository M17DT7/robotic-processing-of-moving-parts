#!/usr/bin/env python3
"""Belt velocity fitted by the estimator at lock time, with and without
top-face recovery (side-wall parallax correction).

Offline tool (no ROS). Reads the "Confident: ... velocity (vx, vy)" line
that perception_target_estimator logs once per part when its first
constant-velocity fit becomes confident (15 detections at the belt entry),
from <data dir>/logs/<label>_estimator.log, and compares the fitted speed
with the true belt speed. Two groups (run labels of the 100 Hz record):

  tracking     h_p0t_r1..3 (arm tracking, without the perception corrections)
               vs clk100_p0_r1..3 (top-face recovery + principal-point
               correction)
  arm parked   h_occ_2s/5s/8s (detections are suppressed only AFTER the lock,
               so the lock itself is clean) vs h_occ_2s_c/5s_c/8s_c

Usage:
    python3 experiments/fitted_velocity.py data/clock_100hz --json results/fitted_velocity_100hz.json
"""
import argparse
import json
import os
import re
import statistics

LINE = re.compile(r"velocity \((-?[\d.]+), (-?[\d.]+)\) m/s from (\d+) detections")
TRUE_V = 0.03                                   # m/s, nominal belt speed
GROUPS = {
    "tracking": ([f"h_p0t_r{i}" for i in (1, 2, 3)],
                 [f"clk100_p0_r{i}" for i in (1, 2, 3)]),
    "arm_parked": (["h_occ_2s", "h_occ_5s", "h_occ_8s"],
                   ["h_occ_2s_c", "h_occ_5s_c", "h_occ_8s_c"]),
}


def locks(log_dir, label):
    """Fitted (vx, vy) of every lock in the estimator log of one run."""
    path = os.path.join(log_dir, f"{label}_estimator.log")
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            mt = LINE.search(line)
            if mt and "Confident" in line:
                out.append((float(mt.group(1)), float(mt.group(2))))
    return out


def summarize(vy):
    mean = statistics.fmean(vy)
    std = statistics.stdev(vy) if len(vy) > 1 else 0.0
    return dict(n=len(vy), mean_mm_s=1000 * mean, std_mm_s=1000 * std,
                bias_pct=100.0 * (mean / TRUE_V - 1.0),
                min_mm_s=1000 * min(vy), max_mm_s=1000 * max(vy))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", help="data/clock_100hz (logs are in <data_dir>/logs)")
    ap.add_argument("--json", default="", help="write the result to this file")
    args = ap.parse_args()
    log_dir = os.path.join(args.data_dir, "logs")

    result = dict(true_mm_s=1000 * TRUE_V)
    for name, (plain, corrected) in GROUPS.items():
        result[name] = {}
        for key, labels in (("uncorrected", plain), ("corrected", corrected)):
            vy = [v[1] for lb in labels for v in locks(log_dir, lb)]
            vx = [v[0] for lb in labels for v in locks(log_dir, lb)]
            s = summarize(vy)
            s["labels"] = labels
            s["vx_mean_mm_s"] = 1000 * statistics.fmean(vx)
            result[name][key] = s
            print(f"{name:11s} {key:12s} n={s['n']:3d}  vy {s['mean_mm_s']:.2f} "
                  f"± {s['std_mm_s']:.2f} mm/s  ({s['bias_pct']:+.1f} %)  "
                  f"range {s['min_mm_s']:.1f}–{s['max_mm_s']:.1f}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
