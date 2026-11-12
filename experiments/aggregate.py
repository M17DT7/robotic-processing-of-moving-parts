#!/usr/bin/env python3
"""Aggregate the recorded runs into the tables of results (offline, no ROS).

Usage:
    python3 experiments/aggregate.py data/clock_100hz --clock100 --json results/aggregate_100hz.json
    python3 experiments/aggregate.py data/clock_10hz --main-only --json results/aggregate_10hz.json

--clock100   every table from the record with the simulation clock at 100 Hz
             (main conditions, vision stage, belt speed, robustness, forced
             occlusion, component ablations, long run, lead sweep); the text
             report is printed to standard output.
--main-only  only the three main conditions (GT, P0, P1) and the lead sweep.
             This is what the 10 Hz record contains, so it is also the
             behaviour when --clock100 is not given.

The tables are written in terms of condition labels (gt_v030_r1, p0t_r1_c,
p0t_noise02_c, ...; the naming scheme is explained in campaign.py). With
--clock100 a condition label is mapped to the run labels of the 100 Hz
record: clk100_* for the three main conditions and the lead sweep, otherwise
h_<label> or the three repetitions h_<label>_r1..3.

Run-level numbers come from analyze.analyze_run (processing window,
time-aligned errors, distance to the true contour, perimeter coverage). A
group of runs is reported as: mean +/- std ACROSS RUNS of the run means, plus
RMSE / 95th percentile / maximum over all samples of the group, and exact
counts of runs, cycles (total / tracked / successful) and samples.
"""

import argparse
import csv
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

SHAPES = None
PANDA_LIMITS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
                (-2.8973, 2.8973)]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def exists(d, label):
    return os.path.exists(os.path.join(d, label + "_pipeline_audit.json"))


# --clock100: every table is built from the 100 Hz record. A condition label
# maps to the run labels of the 100 Hz record: the three main conditions and
# the lead sweep are clk100_*, everything else is "h_<label>" or the three
# repetitions "h_<label>_r1..3".
CLOCK100 = False
_H_MAIN = ((r"gt_v030_r(\d)", "clk100_gt_r{}"), (r"p0t_r(\d)_c", "clk100_p0_r{}"),
           (r"p1_r(\d)_c", "clk100_p1_r{}"), (r"lead_gt_(\d+)", "clk100_gt_{}"))


def expand(d, labels):
    """Run labels to read for the requested condition labels."""
    if not CLOCK100:
        return list(labels)
    out = []
    for lb in labels:
        if lb.startswith(("clk100_", "h_")):
            out.append(lb)
            continue
        for pat, fmt in _H_MAIN:
            m = re.fullmatch(pat, lb)
            if m:
                out.append(fmt.format(m.group(1)))
                break
        else:
            out += [c for c in ["h_" + lb] + [f"h_{lb}_r{i}" for i in (1, 2, 3)]
                    if exists(d, c)]
    return out


def single(d, label):
    """First run of a condition (per-run diagnostics: joints, orientation...)."""
    got = expand(d, [label])
    return got[0] if got else label


def group(d, labels, v=0.03, shape=None, window="win"):
    """Aggregate a list of run labels."""
    labels = expand(d, labels)
    runs = [A.analyze_run(os.path.join(d, lb), SHAPES, v)
            for lb in labels if exists(d, lb)]
    if not runs:
        return None
    out = dict(runs=len(runs), labels=[os.path.basename(r["base"]) for r in runs])
    per_run = [A.summarize(r, window, shape) for r in runs]
    out["cycles_total"] = sum(p["cycles_total"] for p in per_run)
    out["cycles_tracked"] = sum(p["cycles_tracked"] for p in per_run)
    out["cycles_success"] = sum(p["cycles_success"] for p in per_run)
    for metric in A.METRICS + ("est_err", "est_detecting", "est_coasting",
                               "est_transition", "e_true_detecting", "e_true_coasting"):
        means = [p[metric]["mean"] for p in per_run if p[metric]]
        pooled = []
        for r in runs:
            for c in r["cycles"]:
                if shape is not None and (c["shape"] not in shape if isinstance(shape, (list, tuple, set))
                                          else c["shape"] != shape):
                    continue
                if metric in A.METRICS and c["tracked"]:
                    pooled += [s[metric] for s in c.get(window, [])]
                elif metric == "est_err":
                    pooled += [s["est_err"] for s in c["est"]]
                elif metric == "est_detecting":
                    pooled += [s["est_err"] for s in c["est"] if s["detecting"]]
                elif metric == "est_coasting":
                    pooled += [s["est_err"] for s in c["est"] if s["detecting"] is False]
                elif metric == "est_transition":
                    pooled += [s["est_err"] for s in c["est_transition"]]
                elif metric == "e_true_detecting" and c["tracked"]:
                    pooled += [s["e_true"] for s in c.get(window, []) if s["detecting"] is True]
                elif metric == "e_true_coasting" and c["tracked"]:
                    pooled += [s["e_true"] for s in c.get(window, []) if s["detecting"] is False]
        st = A.stats(pooled)
        if st:
            ms = A.mean_std(means)
            st["run_mean"], st["run_std"] = ms if ms else (None, None)
        out[metric] = st
    cyc = [c for r in runs for c in r["cycles"]
           if c["tracked"] and (shape is None or c["shape"] == shape)]
    for key in ("approach_s", "window_s", "in_tol", "laps", "coverage"):
        out[key] = A.mean_std([c.get(key) for c in cyc])
    out["servo_status"] = sorted({e["status"] for r in runs
                                  for e in r["audit"].get("servo_status_events", [])})
    out["twist_nan"] = sum(r["audit"].get("twist_nan") or 0 for r in runs)
    return out


def cv_stats(d, labels, v=0.03):
    """Arm-free perception accuracy (latency-compensated, correct sign):
    the ground truth is moved to the IMAGE stamp, gt - v * (det_age - gt_age).
    Evaluation zone: part fully visible inside the label trust band
    (|gt_y| <= 0.18 m); 'pending' frames carry the raw centroid by design
    and are counted separately."""
    out = dict(runs=0, per_shape={})
    run_means, run_det, run_cls = [], [], []
    pooled = {}
    n_frames = n_det = n_pending = n_labeled = n_correct = 0
    cycles = 0
    for lb in expand(d, labels):
        path = os.path.join(d, lb + "_perception_log.csv")
        if not os.path.exists(path):
            continue
        out["runs"] += 1
        with open(os.path.join(d, lb + "_pipeline_audit.json")) as f:
            cycles += max(0, len(json.load(f).get("cycle_starts", [])) - 1)
        errs = []
        r_frames = r_det = r_lab = r_ok = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                gx, gy, ga = fnum(row["gt_x"]), fnum(row["gt_y"]), fnum(row["gt_age_s"])
                if not math.isfinite(gx) or not math.isfinite(ga) or ga > 0.3:
                    continue
                if not -0.18 <= gy <= 0.18:
                    continue
                r_frames += 1
                dx, dy, da = fnum(row["det_x"]), fnum(row["det_y"]), fnum(row["det_age_s"])
                if not math.isfinite(dx) or not math.isfinite(da) or da > 0.3:
                    continue
                r_det += 1
                label = row["det_shape"]
                if label == "pending":
                    n_pending += 1
                    continue
                if label in ("", "unknown"):
                    continue
                r_lab += 1
                r_ok += label == row["gt_shape"]
                e = 1000.0 * math.hypot(dx - gx, dy - (gy - v * (da - ga)))
                errs.append(e)
                pooled.setdefault(row["gt_shape"], []).append(e)
        n_frames += r_frames
        n_det += r_det
        n_labeled += r_lab
        n_correct += r_ok
        if errs:
            run_means.append(sum(errs) / len(errs))
        if r_frames:
            run_det.append(r_det / r_frames)
        if r_lab:
            run_cls.append(r_ok / r_lab)
    allv = [e for v_ in pooled.values() for e in v_]
    out.update(cycles=cycles, frames=n_frames, detected=n_det, pending=n_pending,
               labeled=n_labeled, correct=n_correct,
               center=A.stats(allv), run_mean=A.mean_std(run_means),
               detection_rate=A.mean_std(run_det), classification=A.mean_std(run_cls))
    for shape, vals in pooled.items():
        out["per_shape"][shape] = A.stats(vals)
    return out


def latch_accuracy(d, labels):
    """Latched shape vs ground truth, per tracked cycle (perception runs)."""
    total = ok = 0
    for lb in expand(d, labels):
        path = os.path.join(d, lb + "_perception_log.csv")
        if not os.path.exists(path):
            continue
        seen = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                es, gs = row.get("est_shape", ""), row.get("gt_shape", "")
                ea = fnum(row.get("est_age_s"))
                if es in ("", "pending") or not gs or not math.isfinite(ea) or ea > 0.25:
                    continue
                key = (lb, gs, int(fnum(row["sim_t"]) // 5))
                seen.setdefault((lb, gs, round(fnum(row["gt_y"]), 0)), []).append(es == gs)
                seen[key] = seen.get(key, []) + [es == gs]
        vals = [x for v_ in seen.values() for x in v_]
        total += len(vals)
        ok += sum(vals)
    return dict(samples=total, correct=ok)


def occlusion(d, label, band, v=0.03):
    """Estimator error at the end of a forced detection blackout (arm parked)."""
    label = single(d, label)
    if not exists(d, label):
        return None
    run = A.analyze_run(os.path.join(d, label), SHAPES, v)
    y0, y1 = band
    path = os.path.join(d, label + "_perception_log.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    end_err, after_err, during = [], [], []
    for row in rows:
        gx, gy, ga = fnum(row["gt_x"]), fnum(row["gt_y"]), fnum(row["gt_age_s"])
        ex, ey, ea = fnum(row["est_x"]), fnum(row["est_y"]), fnum(row["est_age_s"])
        if not all(map(math.isfinite, (gx, gy, ga, ex, ey, ea))) or ea > 0.25 or ga > 0.25:
            continue
        gy_now, ey_now = gy + v * ga, ey + v * ea
        e = 1000.0 * math.hypot(ex - gx, ey_now - gy_now)
        if y0 <= gy_now <= y1:
            during.append(e)
        if y1 - 0.012 <= gy_now <= y1:
            end_err.append(e)
        if y1 + 0.03 <= gy_now <= y1 + 0.06:
            after_err.append(e)
    return dict(duration_s=(y1 - y0) / v, during=A.stats(during),
                at_end=A.stats(end_err), after=A.stats(after_err),
                cycles=len(run["cycles"]))


def orientation_drift(d, label):
    """Orientation error slope (deg/min) over tracked samples."""
    label = single(d, label)
    path = os.path.join(d, label + "_tracking_log.csv")
    if not os.path.exists(path):
        return None
    ts, es = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            a = fnum(row["tgt_age_s"])
            if math.isfinite(a) and a <= 0.25:
                ts.append(fnum(row["sim_t"]))
                es.append(fnum(row["ori_err_deg"]))
    if len(ts) < 10:
        return None
    tm, em = sum(ts) / len(ts), sum(es) / len(es)
    den = sum((t - tm) ** 2 for t in ts)
    slope = sum((t - tm) * (e - em) for t, e in zip(ts, es)) / den if den else 0.0
    return dict(deg_per_min=slope * 60.0, max_deg=max(es), mean_deg=em)


def joint_margin(d, label):
    """Minimum distance to the Panda joint limits (rad) and joint1 range."""
    label = single(d, label)
    path = os.path.join(d, label + "_joint_log.csv")
    if not os.path.exists(path):
        return None
    worst = float("inf")
    worst_joint = None
    j1 = []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) < 8:
                continue
            q = [fnum(x) for x in row[1:8]]
            j1.append(q[0])
            for k, (lo, hi) in enumerate(PANDA_LIMITS):
                mrg = min(q[k] - lo, hi - q[k])
                if mrg < worst:
                    worst, worst_joint = mrg, k + 1
    return dict(min_margin_rad=worst, joint=worst_joint,
                joint1_min=min(j1), joint1_max=max(j1))


def run_info(d, label):
    label = single(d, label)
    path = os.path.join(d, label + "_run_meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    with open(os.path.join(d, label + "_pipeline_audit.json")) as f:
        audit = json.load(f)
    ev = audit.get("servo_status_events", [])
    t0 = audit.get("sim_t_start") or 0.0
    first_warn = next((e["sim_t"] - t0 for e in ev if e["status"] in (1, 2, 6)), None)
    first_halt = next((e["sim_t"] - t0 for e in ev if e["status"] == 2), None)
    return dict(ok=meta.get("ok"), notes=meta.get("notes"),
                sim_s=meta.get("sim_recorded_s"), wall_s=meta.get("wall_s"),
                servo_status=meta.get("servo_status"),
                first_warning_s=first_warn, first_halt_s=first_halt,
                twist_nan=audit.get("twist_nan"))


def time_trend(d, label, v=0.03):
    """Contour distance in the first vs last third of a long run."""
    label = single(d, label)
    if not exists(d, label):
        return None
    run = A.analyze_run(os.path.join(d, label), SHAPES, v)
    cyc = [c for c in run["cycles"] if c["tracked"]]
    k = max(1, len(cyc) // 3)

    def pooled(cs, metric):
        return A.stats([s[metric] for c in cs for s in c.get("win", [])])
    return dict(first_third=dict(d_contour=pooled(cyc[:k], "d_contour"),
                                 e_true=pooled(cyc[:k], "e_true")),
                last_third=dict(d_contour=pooled(cyc[-k:], "d_contour"),
                                e_true=pooled(cyc[-k:], "e_true")))


LEADS_10HZ = (0.14, 0.19, 0.22, 0.24, 0.26, 0.28, 0.34)
LEADS_100HZ = (0.04, 0.08, 0.12, 0.16, 0.20, 0.22, 0.24, 0.26, 0.28)


def lead_sweep(d):
    """Ground-truth runs of the lead-time sweep, keyed by the lead in s."""
    R = {f"{lead:.2f}": group(d, [f"lead_gt_{round(lead * 100):03d}"])
         for lead in (LEADS_100HZ if CLOCK100 else LEADS_10HZ)}
    return {k: g for k, g in R.items() if g}


def build(d, sfx, minimal=False):
    """All tables for one perception variant: sfx '_c' = with top-face
    recovery and principal-point correction, '' = without both. Ground-truth
    runs have no perception in the loop and carry no suffix. minimal: only
    the groups that were recorded for the variant without the corrections
    (main conditions, vision stage, forced occlusion)."""
    R = {}
    P = lambda label: label + sfx          # noqa: E731
    main_groups = dict(
        gt=[f"gt_v030_r{i}" for i in (1, 2, 3)],
        p0=[P(f"p0t_r{i}") for i in (1, 2, 3)],
        p1=[P(f"p1_r{i}") for i in (1, 2, 3)],
        p0c=[P("p0c_r1")])
    if minimal:
        del main_groups["p0c"]
    R["main"] = {k: group(d, v) for k, v in main_groups.items()}
    R["main_steady"] = {k: group(d, v, window="steady") for k, v in main_groups.items()}
    R["per_shape"] = {k: {sh: group(d, v, shape=sh) for sh in SHAPES}
                      for k, v in main_groups.items() if k != "p0c"}
    R["latch"] = {k: latch_accuracy(d, main_groups[k]) for k in ("p0", "p1")}
    # vision stage alone: both corrections / none
    R["cv"] = cv_stats(d, [f"cvc_r{i}" if sfx else f"cvn_r{i}" for i in (1, 2, 3)])
    R["occlusion"] = {lb: occlusion(d, P(lb), band) for lb, band in
                      (("occ_2s", (-0.10, -0.04)), ("occ_5s", (-0.10, 0.05)),
                       ("occ_8s", (-0.10, 0.14)))}
    if minimal:
        return R
    R["speed"] = {}
    for tag, v in (("v020", 0.02), ("v030", 0.03), ("v045", 0.045), ("v060", 0.06)):
        gt_l = [f"gt_{tag}_r{i}" for i in (1, 2, 3)]
        p0_l = ([P(f"p0t_r{i}") for i in (1, 2, 3)] if tag == "v030"
                else [P(f"p0t_{tag}_r{i}") for i in (1, 2, 3)])
        R["speed"][tag] = dict(v=v, gt=group(d, gt_l, v=v), p0=group(d, p0_l, v=v),
                               # the non-convex part alone
                               p0_irregular=group(d, p0_l, v=v, shape="irregular"))
    _vc = single(d, P("p0t_vchange"))
    if exists(d, _vc):
        run = A.analyze_run(os.path.join(d, _vc), SHAPES, 0.03)
        seg = {}
        for c in run["cycles"]:
            if not c["tracked"]:
                continue
            key = min((0.02, 0.03, 0.045), key=lambda s_: abs(s_ - c["v"]))
            seg.setdefault(key, []).append(c)
        R["vchange"] = {str(k): dict(
            cycles=len(cs), success=sum(1 for c in cs if c.get("success")),
            d_contour=A.stats([s_["d_contour"] for c in cs for s_ in c["win"]]),
            e_true=A.stats([s_["e_true"] for c in cs for s_ in c["win"]]),
            est_err=A.stats([s_["est_err"] for c in cs for s_ in c["est"]]))
            for k, cs in sorted(seg.items())}
        R["vchange_info"] = run_info(d, P("p0t_vchange"))
    R["prior"] = {tag: group(d, [P(f"p1_prior_{tag}")])
                  for tag in ("m20", "m10", "p10", "p20")}
    R["robust"] = {lb: group(d, [P(f"p0t_{lb}")]) for lb in
                   ("noise02", "noise05", "calib05", "calib10",
                    "delay050", "delay100j")}
    R["ablation"] = {}
    for lb in ("abl_no_labelfit", "abl_no_rebase", "abl_no_both", "abl_no_ff",
               "abl_no_lead", "abl_lead014", "abl_no_orihold", "abl_no_rehome"):
        R["ablation"][lb] = dict(stats=group(d, [P(lb)]), info=run_info(d, P(lb)),
                                 orientation=orientation_drift(d, P(lb)),
                                 joints=joint_margin(d, P(lb)))
    R["baseline_orientation"] = orientation_drift(d, P("p0t_r1"))
    R["baseline_joints"] = joint_margin(d, P("p0t_r1"))
    R["baseline_info"] = run_info(d, P("p0t_r1"))
    R["long"] = dict(stats=group(d, [P("p0t_long")]), info=run_info(d, P("p0t_long")),
                     joints=joint_margin(d, P("p0t_long")),
                     orientation=orientation_drift(d, P("p0t_long")),
                     trend=time_trend(d, P("p0t_long")))
    return R


def main_only(d):
    """The three main conditions and the lead sweep."""
    main_groups = dict(gt=[f"gt_v030_r{i}" for i in (1, 2, 3)],
                       p0=[f"p0t_r{i}_c" for i in (1, 2, 3)],
                       p1=[f"p1_r{i}_c" for i in (1, 2, 3)])
    R = {"main": {k: group(d, v) for k, v in main_groups.items()},
         "lead": lead_sweep(d), "clock_hz": 100 if CLOCK100 else 10}
    for k, g in R["main"].items():
        if not g:
            sys.exit(f"no runs of condition '{k}' in {d}"
                     + ("" if CLOCK100 else " (a 100 Hz record needs --clock100)"))
        print(f"{k}: contour {g['d_contour']['run_mean']:.2f} ± {g['d_contour']['run_std']:.2f} mm, "
              f"true target {g['e_true']['run_mean']:.2f} ± {g['e_true']['run_std']:.2f} mm, "
              f"cycles {g['cycles_success']}/{g['cycles_tracked']}")
    return R


def full(d):
    """Every table of the 100 Hz record, with the text report."""
    R = build(d, "_c")
    R["uncorrected"] = build(d, "", minimal=True)
    R["lead"] = lead_sweep(d)
    R["clock_hz"] = 100
    # 2 x 2 split of the two perception corrections (arm parked)
    R["cv_split"] = {
        "neither": cv_stats(d, [f"cvn_r{i}" for i in (1, 2, 3)]),
        "principal_point": cv_stats(d, [f"cvpp_r{i}" for i in (1, 2, 3)]),
        "top_face": cv_stats(d, [f"cvtf_r{i}" for i in (1, 2, 3)]),
        "both": cv_stats(d, [f"cvc_r{i}" for i in (1, 2, 3)]),
    }
    # randomized P0 runs (shuffled shape order, entry-x jitter, pixel noise)
    R["random"] = group(d, [f"rnd_p0_r{i}" for i in (1, 2, 3)])
    # estimator fit in the label-independent raw-centroid frame
    raw_l = [f"abl_rawfit_r{i}" for i in (1, 2, 3)]
    CONVEX = tuple(sh for sh in SHAPES if sh != "irregular")
    R["ablation"]["abl_rawfit"] = dict(
        stats=group(d, raw_l), info=run_info(d, "abl_rawfit_r1"),
        orientation=orientation_drift(d, "abl_rawfit_r1"), joints=joint_margin(d, "abl_rawfit_r1"),
        per_shape={sh: group(d, raw_l, shape=sh) for sh in SHAPES},
        convex=group(d, raw_l, shape=CONVEX))
    # the four convex shapes pooled, for the comparison with that variant
    # (same labels as the main groups built in build())
    R["per_shape_convex"] = {
        "gt": group(d, [f"gt_v030_r{i}" for i in (1, 2, 3)], shape=CONVEX),
        "p0": group(d, [f"p0t_r{i}_c" for i in (1, 2, 3)], shape=CONVEX),
        "p1": group(d, [f"p1_r{i}_c" for i in (1, 2, 3)], shape=CONVEX)}

    def line(name, g):
        if not g:
            print(f"  {name:16s} (missing)")
            return

        def f(m):
            s = g.get(m)
            if not s:
                return "      -      "
            rm = f"{s['run_mean']:.2f}±{s['run_std']:.2f}" if s.get("run_mean") is not None else "-"
            return f"{rm} r{s['rmse']:.2f} p{s['p95']:.2f} x{s['max']:.1f}"
        it = g["in_tol"][0] if g["in_tol"] else float("nan")
        lp = g["laps"][0] if g["laps"] else float("nan")
        cv_ = g.get("coverage")
        cv_ = cv_[0] if cv_ else float("nan")
        ap_ = g["approach_s"][0] if g["approach_s"] else float("nan")
        print(f"  {name:16s} cyc {g['cycles_success']}/{g['cycles_tracked']}/{g['cycles_total']} "
              f"| ctrl {f('e_ctrl')} | true {f('e_true')} | contour {f('d_contour')} "
              f"| est {f('est_err')} | appr {ap_:.1f}s intol {it:.3f} laps {lp:.2f} "
              f"cover {cv_:.2f} | servo {g['servo_status']}")

    print("legend: cyc success/tracked/total; metric = runmean±std rRMSE pP95 xMax (mm)")
    for title, table in (("MAIN (processing window)", R["main"]),
                         ("MAIN, without the perception corrections", R["uncorrected"]["main"]),
                         ("PRIOR ERROR (P1)", R["prior"]), ("ROBUSTNESS (P0)", R["robust"]),
                         ("LEAD SWEEP (GT)", R["lead"]),
                         ("RANDOMIZED P0", {"rnd_p0": R["random"]})):
        print(f"\n== {title}")
        for k, g in table.items():
            line(k, g)
    print("\n== SPEED")
    for tag, e in R["speed"].items():
        line(tag + " gt", e["gt"])
        line(tag + " p0", e["p0"])
    print("\n== ABLATION")
    for k, e in R["ablation"].items():
        line(k, e["stats"])
        print(f"       info {e['info']}  ori {e['orientation']}  joints {e['joints']}")
    print("\n== LONG", R["long"]["info"], R["long"]["joints"], R["long"]["orientation"])
    line("p0t_long", R["long"]["stats"])
    print("   trend", R["long"]["trend"])
    print("\n== CV without the corrections",
          {k: v for k, v in R["uncorrected"]["cv"].items() if k != "per_shape"})
    print("\n== CV 2x2 SPLIT")
    for k, v in R["cv_split"].items():
        print(f"  {k:16s}", {kk: vv for kk, vv in v.items() if kk != "per_shape"} if v else None)
    print("\n== CV", {k: v for k, v in R["cv"].items() if k != "per_shape"})
    print("   per shape", R["cv"]["per_shape"])
    print("\n== OCCLUSION", R["occlusion"])
    print("\n== VCHANGE", R.get("vchange"), R.get("vchange_info"))
    print("\n== LATCH", R["latch"])
    print("\n== ESTIMATOR regimes")
    for k in ("p0", "p1", "p0c"):
        g = R["main"].get(k)
        if g:
            print(f"  {k}: detecting {g['est_detecting']}\n      coasting {g['est_coasting']}"
                  f"\n      transition rows (excluded) {g['est_transition']}")
    return R


def main():
    global SHAPES, CLOCK100
    ap = argparse.ArgumentParser(
        description="Aggregate the recorded runs of one data directory.")
    ap.add_argument("data_dir", help="data/clock_100hz or data/clock_10hz")
    ap.add_argument("--json", default="", help="write the aggregate to this file")
    ap.add_argument("--main-only", action="store_true",
                    help="only the three main conditions and the lead sweep "
                         "(always the case without --clock100)")
    ap.add_argument("--clock100", action="store_true",
                    help="the directory holds the 100 Hz record: build every table")
    args = ap.parse_args()
    CLOCK100 = args.clock100
    SHAPES = A.load_catalog(A.DEFAULT_SHAPES)
    if args.main_only or not CLOCK100:
        R = main_only(args.data_dir)
    else:
        if not exists(args.data_dir, "clk100_gt_r1"):
            sys.exit(f"no 100 Hz record in {args.data_dir}")
        R = full(args.data_dir)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(R, f, indent=1, default=str)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
