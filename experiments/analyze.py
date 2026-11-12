#!/usr/bin/env python3
"""Analysis of one or more recorded runs: tracking error against the TRUE part.

Offline tool (no ROS). Reads <base>_tracking_log.csv, <base>_perception_log.csv
(optional) and <base>_pipeline_audit.json written by the experiment recorder
(<base> = data directory + run label) and computes, per cycle / shape / run:

  e_ctrl_rec   tool vs commanded target AS RECORDED. The recorder samples on
               a simulation-clock tick; the latest target message is then one
               tick old while the TF pose is current, so this compares
               tool(t) with target(t - one tick).
  e_ctrl       tool vs commanded target, TIME-ALIGNED: the target whose
               stamp equals the tool sample time (taken from the following
               log row).
  e_true       tool vs the true target at the same path phase and time:
               commanded target - estimated part origin + ground-truth part
               origin (parts do not rotate). Equals e_ctrl for ground-truth
               runs.
  d_contour    shortest planar distance from the tool point to the TRUE
               part outline (ground-truth origin + shapes.yaml outline).
  est_err      estimator part origin vs ground truth (perception runs).

Windows:
  steady state:        t >= 5 s after the plant cycle start
  processing window:   from the first time the tool stays within TOL of the
                       true contour AND within ON_PATH_MM of its target for
                       HOLD consecutive samples, until the target is
                       released.
Cycle success: at least SUCCESS_RATIO of the processing-window samples
within TOL of the true contour. Also reported: approach time (spawn ->
window start), perimeter laps travelled inside tolerance and the perimeter
COVERAGE: the fraction of COVER_BINS equal arc-length bins of the true
outline that the tool visited within TOL during the processing window
(1.0 = the whole perimeter was processed at least once).

Estimator samples: the single recorder row at which the estimator's label
changes from 'pending' to a resolved shape is flagged `transition` and kept
out of the detecting / coasting / lifetime statistics: at that row the
published state is still the raw centroid while the label already names the
part-frame origin, so the row measures the catalog offset of the shape
(13.3 mm triangle, 17.8 mm irregular), not an estimation error. It is
reported separately as `est_transition`.

Usage:
    python3 experiments/analyze.py <base> [<base> ...] [--v 0.03]
                                   [--json out.json] [--shapes shapes.yaml]
                                   [--quiet]
    e.g. python3 experiments/analyze.py data/clock_100hz/clk100_p0_r1
"""

import argparse
import csv
import json
import math
import os

TOOL_Z_OFFSET = -0.120
STEADY_OFFSET = 5.0
TOL_MM = 5.0
HOLD = 5
ON_PATH_MM = 20.0       # tracker's own path_start_tolerance: tool has reached the path
SUCCESS_RATIO = 0.90
COVER_BINS = 100        # arc-length bins of the outline for the coverage metric
PATH_SPEED = 0.06
FRESH_TARGET = 0.25     # s; a target older than this = tracker not publishing
FRESH_EST = 0.25
FRESH_DET = 0.3
DEFAULT_SHAPES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "robotic_tracking_system", "config", "shapes.yaml")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_catalog(path):
    import yaml
    with open(path) as f:
        cat = yaml.safe_load(f)
    shapes = {}
    for name, spec in cat["shapes"].items():
        if spec["type"] == "polygon":
            shapes[name] = ("polygon",
                            [(float(x), float(y)) for x, y in spec["vertices"]])
        else:
            shapes[name] = ("circle", float(spec["radius"]))
    return shapes


def perimeter(shape):
    kind, data = shape
    if kind == "circle":
        return 2.0 * math.pi * data
    n = len(data)
    return sum(math.hypot(data[(i + 1) % n][0] - data[i][0],
                          data[(i + 1) % n][1] - data[i][1]) for i in range(n))


def dist_to_outline(px, py, shape):
    """Planar distance (m) from a part-frame point to the shape outline."""
    kind, data = shape
    if kind == "circle":
        return abs(math.hypot(px, py) - data)
    best = float("inf")
    n = len(data)
    for i in range(n):
        ax, ay = data[i]
        bx, by = data[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def outline_param(px, py, shape):
    """Normalized arc-length position in [0, 1) of the outline point
    nearest to the part-frame point (px, py)."""
    kind, data = shape
    if kind == "circle":
        return (math.atan2(py, px) / (2.0 * math.pi)) % 1.0
    best, best_s, acc = float("inf"), 0.0, 0.0
    n = len(data)
    for i in range(n):
        ax, ay = data[i]
        bx, by = data[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        L = math.sqrt(L2)
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        dist = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        if dist < best:
            best, best_s = dist, acc + t * L
        acc += L
    return (best_s / acc) % 1.0 if acc > 0 else 0.0


def coverage(samples, shape, tol_mm=TOL_MM, bins=COVER_BINS):
    """Fraction of outline arc-length bins visited within tol_mm."""
    hit = set()
    for s in samples:
        if math.isfinite(s["d_contour"]) and s["d_contour"] <= tol_mm:
            hit.add(min(bins - 1, int(outline_param(s["px"], s["py"], shape) * bins)))
    return len(hit) / bins


def stats(values):
    v = sorted(x for x in values if math.isfinite(x))
    if not v:
        return None
    n = len(v)
    return dict(n=n, mean=sum(v) / n,
                rmse=math.sqrt(sum(x * x for x in v) / n),
                p95=v[min(n - 1, int(math.ceil(0.95 * n)) - 1)], max=v[-1])


def mean_std(vals):
    vals = [x for x in vals if x is not None and math.isfinite(x)]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, s


def load_run(base):
    with open(base + "_tracking_log.csv") as f:
        trk = list(csv.DictReader(f))
    per = None
    if os.path.exists(base + "_perception_log.csv"):
        with open(base + "_perception_log.csv") as f:
            per = list(csv.DictReader(f))
        if len(per) != len(trk):
            # written in the same sample() call; tolerate a truncated tail
            n = min(len(per), len(trk))
            trk, per = trk[:n], per[:n]
    with open(base + "_pipeline_audit.json") as f:
        audit = json.load(f)
    return trk, per, audit


def analyze_run(base, shapes, v_belt, reconstruct_gt=False):
    trk, per, audit = load_run(base)
    n = len(trk)
    t = [fnum(r["sim_t"]) for r in trk]
    cycles = [(c["sim_t"], c["shape"]) for c in audit.get("cycle_starts", [])]

    # Per-cycle belt speed from the ground-truth stamps (handles speed
    # changes); falls back to v_belt.
    def gt_at(i):
        if per is None:
            return None
        gx, gy, ga = fnum(per[i]["gt_x"]), fnum(per[i]["gt_y"]), fnum(per[i]["gt_age_s"])
        if not math.isfinite(gx) or not math.isfinite(ga) or ga > 0.5:
            return None
        return gx, gy, ga

    rows = []
    for i in range(n):
        r = trk[i]
        tool = (fnum(r["ee_x"]), fnum(r["ee_y"]), fnum(r["ee_z"]) + TOOL_Z_OFFSET)
        tgt = (fnum(r["tgt_x"]), fnum(r["tgt_y"]), fnum(r["tgt_z"]))
        age = fnum(r["tgt_age_s"])
        rows.append(dict(t=t[i], tool=tool, tgt=tgt, age=age))

    out_cycles = []
    for ci, (c0, shape_name) in enumerate(cycles):
        c1 = cycles[ci + 1][0] if ci + 1 < len(cycles) else None
        if c1 is None:
            continue                       # last cycle is cut by shutdown
        idx = [i for i in range(n) if c0 <= t[i] < c1]
        if not idx or shape_name not in shapes:
            continue
        shape = shapes[shape_name]

        # belt speed of this cycle
        v = v_belt
        if per is not None:
            pts = [(t[i] - gt_at(i)[2], gt_at(i)[1]) for i in idx
                   if gt_at(i) and per[i]["gt_shape"] == shape_name]
            if len(pts) > 20:
                tm = sum(p[0] for p in pts) / len(pts)
                ym = sum(p[1] for p in pts) / len(pts)
                den = sum((p[0] - tm) ** 2 for p in pts)
                if den > 1e-9:
                    v = sum((p[0] - tm) * (p[1] - ym) for p in pts) / den

        samples = []
        prev_est_shape = None
        for i in idx:
            row = rows[i]
            s = dict(t=row["t"], rel=row["t"] - c0)
            tool = row["tool"]
            # ---- as recorded
            fresh = (math.isfinite(row["tgt"][0]) and math.isfinite(row["age"])
                     and row["age"] <= FRESH_TARGET)
            s["tracked"] = fresh
            s["e_ctrl_rec"] = (1000.0 * math.dist(tool, row["tgt"])
                               if fresh else float("nan"))
            # ---- time-aligned target: the message stamped at this row's time
            tgt_al = None
            for j in (i + 1, i + 2, i):
                if j < n and rows[j]["t"] < c1 - 0.05 and \
                        math.isfinite(rows[j]["tgt"][0]) and \
                        abs((rows[j]["t"] - rows[j]["age"]) - row["t"]) < 0.02:
                    tgt_al = rows[j]["tgt"]
                    break
            s["e_ctrl"] = (1000.0 * math.dist(tool, tgt_al)
                           if (fresh and tgt_al) else float("nan"))
            # ---- ground-truth origin at this row's time
            g = gt_at(i)
            if g and per[i]["gt_shape"] == shape_name:
                gx, gy = g[0], g[1] + v * g[2]
            elif reconstruct_gt or per is None:
                gx, gy = 0.50, -0.25 + v * (row["t"] - c0)
            else:
                gx = gy = float("nan")
            s["d_contour"] = (1000.0 * dist_to_outline(tool[0] - gx, tool[1] - gy, shape)
                              if math.isfinite(gx) else float("nan"))
            s["px"], s["py"] = tool[0] - gx, tool[1] - gy   # true part frame
            # ---- true target (perception runs): shift by the estimator error
            s["e_true"] = s["e_ctrl"]
            s["est_err"] = float("nan")
            s["detecting"] = None
            s["transition"] = False
            if per is not None and math.isfinite(fnum(per[i]["est_x"])):
                ex, ey, ea = (fnum(per[i]["est_x"]), fnum(per[i]["est_y"]),
                              fnum(per[i]["est_age_s"]))
                # 'pending' = shape not resolved yet: the state is the raw
                # centroid by design, not an estimate of the part origin.
                est_shape = per[i].get("est_shape", "")
                resolved = est_shape not in ("", "pending")
                # First resolved row after 'pending': the label already
                # names the origin while the published state is still the
                # raw centroid of the previous tick (see module docstring).
                s["transition"] = resolved and prev_est_shape == "pending"
                prev_est_shape = est_shape
                if (resolved and math.isfinite(ea) and ea <= FRESH_EST
                        and math.isfinite(gx)):
                    ex_now, ey_now = ex, ey + v * ea
                    s["est_err"] = 1000.0 * math.hypot(ex_now - gx, ey_now - gy)
                    if fresh and tgt_al:
                        true_tgt = (tgt_al[0] + gx - ex_now,
                                    tgt_al[1] + gy - ey_now, tgt_al[2])
                        s["e_true"] = 1000.0 * math.dist(tool, true_tgt)
                    da = fnum(per[i]["det_age_s"])
                    s["detecting"] = math.isfinite(da) and da <= FRESH_DET
                elif fresh:
                    s["e_true"] = float("nan")
            samples.append(s)

        tracked = [s for s in samples if s["tracked"]]
        cyc = dict(index=ci, shape=shape_name, t0=c0, period=c1 - c0, v=v,
                   n_tracked=len(tracked), tracked=len(tracked) >= 20)
        if cyc["tracked"]:
            # processing window
            start = None
            for k in range(len(tracked) - HOLD + 1):
                if all(math.isfinite(tracked[k + m]["d_contour"])
                       and tracked[k + m]["d_contour"] <= TOL_MM
                       and tracked[k + m]["e_ctrl_rec"] <= ON_PATH_MM
                       for m in range(HOLD)):
                    start = k
                    break
            cyc["resolved"] = True
            if start is None:
                cyc.update(approach_s=None, window_n=0, in_tol=0.0, laps=0.0,
                           coverage=0.0, success=False)
                win = []
            else:
                win = tracked[start:]
                inside = [s for s in win if s["d_contour"] <= TOL_MM]
                cyc["approach_s"] = win[0]["rel"]
                cyc["window_n"] = len(win)
                cyc["window_s"] = win[-1]["rel"] - win[0]["rel"]
                cyc["in_tol"] = len(inside) / len(win)
                cyc["laps"] = PATH_SPEED * 0.1 * len(inside) / perimeter(shape)
                cyc["coverage"] = coverage(win, shape)
                cyc["success"] = cyc["in_tol"] >= SUCCESS_RATIO
            cyc["win"] = win
            cyc["steady"] = [s for s in tracked if s["rel"] >= STEADY_OFFSET]
        cyc["est"] = [s for s in samples if math.isfinite(s["est_err"]) and not s["transition"]]
        cyc["est_transition"] = [s for s in samples if math.isfinite(s["est_err"]) and s["transition"]]
        out_cycles.append(cyc)
    return dict(base=base, cycles=out_cycles, audit=audit)


METRICS = ("e_ctrl_rec", "e_ctrl", "e_true", "d_contour")


def summarize(run, window="win", shape=None):
    """Pooled sample statistics of one run for the chosen window."""
    def want(c):
        if shape is None:
            return True
        if isinstance(shape, (list, tuple, set)):
            return c["shape"] in shape
        return c["shape"] == shape

    cyc = [c for c in run["cycles"] if c["tracked"] and want(c)]
    res = dict(cycles_total=len([c for c in run["cycles"] if want(c)]),
               cycles_tracked=len(cyc),
               cycles_success=sum(1 for c in cyc if c.get("success")))
    for m in METRICS:
        res[m] = stats([s[m] for c in cyc for s in c.get(window, [])])
    for regime, flag in (("detecting", True), ("coasting", False)):
        res["e_true_" + regime] = stats(
            [s["e_true"] for c in cyc for s in c.get(window, [])
             if s["detecting"] is flag])
    est_all = [s for c in run["cycles"] if shape is None or c["shape"] == shape
               for s in c["est"]]
    res["est_err"] = stats([s["est_err"] for s in est_all])
    res["est_detecting"] = stats([s["est_err"] for s in est_all if s["detecting"]])
    res["est_coasting"] = stats([s["est_err"] for s in est_all if s["detecting"] is False])
    res["est_transition"] = stats([s["est_err"] for c in run["cycles"]
                                   if shape is None or c["shape"] == shape
                                   for s in c["est_transition"]])
    res["approach_s"] = mean_std([c.get("approach_s") for c in cyc])
    res["in_tol"] = mean_std([c.get("in_tol") for c in cyc])
    res["laps"] = mean_std([c.get("laps") for c in cyc])
    res["coverage"] = mean_std([c.get("coverage") for c in cyc])
    res["window_s"] = mean_std([c.get("window_s") for c in cyc])
    return res


def fmt(s):
    if not s:
        return "      (no data)"
    return (f"mean {s['mean']:5.2f}  rmse {s['rmse']:5.2f}  p95 {s['p95']:5.2f}  "
            f"max {s['max']:6.2f}  (n={s['n']})")


def strip(run):
    """JSON-serialisable copy (drops the per-sample lists)."""
    return dict(base=run["base"], cycles=[
        {k: v for k, v in c.items() if k not in ("win", "steady", "est", "est_transition")}
        for c in run["cycles"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bases", nargs="+")
    ap.add_argument("--v", type=float, default=0.03,
                    help="nominal belt speed in m/s")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--json", default="")
    ap.add_argument("--reconstruct-gt", action="store_true",
                    help="logs without ground-truth columns: rebuild the "
                         "part origin from the cycle starts")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    shapes = load_catalog(args.shapes)

    result = {}
    for base in args.bases:
        run = analyze_run(base, shapes, args.v, args.reconstruct_gt)
        name = os.path.basename(base)
        entry = dict(run=strip(run))
        for window in ("win", "steady"):
            entry[window] = dict(ALL=summarize(run, window))
            for sh in shapes:
                entry[window][sh] = summarize(run, window, sh)
        events = run["audit"].get("servo_status_events", [])
        entry["servo_status"] = sorted({e["status"] for e in events})
        entry["twist_nan"] = run["audit"].get("twist_nan")
        result[name] = entry
        if args.quiet:
            continue
        a = entry["win"]["ALL"]
        print(f"\n=== {name}: cycles {a['cycles_total']} total / "
              f"{a['cycles_tracked']} tracked / {a['cycles_success']} successful "
              f"(tolerance {TOL_MM:.0f} mm, >= {SUCCESS_RATIO:.0%} in tolerance); "
              f"servo status {entry['servo_status']}, NaN {entry['twist_nan']}")
        for window, title in (("steady", "steady state (t >= 5 s)"),
                              ("win", "processing window")):
            w = entry[window]["ALL"]
            print(f"  -- {title}")
            print(f"     tool vs target, as recorded : {fmt(w['e_ctrl_rec'])}")
            print(f"     tool vs target, time-aligned: {fmt(w['e_ctrl'])}")
            print(f"     tool vs TRUE target         : {fmt(w['e_true'])}")
            print(f"     distance to TRUE contour    : {fmt(w['d_contour'])}")
        w = entry["win"]["ALL"]
        if w["est_err"]:
            print(f"     estimator vs truth          : {fmt(w['est_err'])}")
            print(f"       detecting                 : {fmt(w['est_detecting'])}")
            print(f"       coasting                  : {fmt(w['est_coasting'])}")
            print(f"       transition rows (excluded): {fmt(w['est_transition'])}")
        for key, unit in (("approach_s", "s"), ("window_s", "s"),
                          ("in_tol", ""), ("laps", "laps"), ("coverage", "of perimeter")):
            if w[key]:
                print(f"     {key:11s}: {w[key][0]:.2f} +/- {w[key][1]:.2f} {unit}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
