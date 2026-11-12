#!/usr/bin/env python3
"""Headless orchestrator that recorded the runs in data/ (one simulation per run).

For every run label it starts Gazebo (server only) + MoveIt, MoveIt Servo,
the experiment recorder and the nodes of the selected mode, lets them run
for a fixed duration and then tears everything down by process group and by
an environment tag (no name-pattern kills, so several workers can run side
by side on different ROS domains). It is not a ROS node itself.

Usage:
    python3 experiments/campaign.py --list
    python3 experiments/campaign.py <out_dir> <worker_id> <label> [<label> ...]

--list prints the run definitions and needs no ROS installation. Recording
needs a built ROS 2 workspace that contains the robotic_tracking_system
package: its path is taken from the environment variable ROS_WS
(default ~/ros2_ws). A label whose <label>_run_meta.json already exists in
<out_dir> is skipped. Per run the recorder writes <label>_tracking_log.csv,
_perception_log.csv, _joint_log.csv and _pipeline_audit.json; this script
adds <label>_run_meta.json (the configuration used and the outcome) and the
node logs in <out_dir>/logs.

Modes:
    percep  part stream + perception + estimator + tracker (camera targets)
            + supervisor (re-home in the gap after every part)
    gt      part stream + tracker (ground-truth targets)
            + supervisor (re-home every 3 parts, skipping one part)
    cvest   part stream + perception + estimator, arm parked (no Servo
            commands): the vision stage and the estimator alone
Every node runs on simulation time.

Run labels (kept exactly as in the data):
    gt        ground-truth targets
    p0t       camera only (belt velocity fitted from the detections)
    p1        camera with the belt-speed prior
    p0c       fitted velocity, belt-speed prior only on long coasts
    _c        suffix: with top-face recovery and principal-point correction
              (no suffix: without both)
    h_ / clk100_   simulation clock at 100 Hz (all other runs: 10 Hz)
    v020 / v045 / v060   belt speed in mm/s (unmarked runs: 30 mm/s)
    vchange   belt speed changed between parts during the run
    p1_prior_m20 .. p20   belt-speed prior wrong by -20 .. +20 %
    noise / calib / delay   image noise, extrinsic calibration error,
              perception publishing delay (with jitter: suffix j)
    rnd       randomized runs (shape order, entry position, image noise)
    abl_*     one component switched off
    occ_*     forced occlusion of 2 / 5 / 8 s with the arm parked
    cv*       vision stage alone with the arm parked: cvc both corrections,
              cvn none, cvpp principal point only, cvtf top-face recovery only
    lead_gt_0xx / clk100_gt_0xx   lead-time sweep with ground-truth targets
              (value in hundredths of a second)
    _rN       repetition N
"""

import json
import os
import random
import signal
import subprocess
import sys
import time

WS = os.path.expanduser(os.environ.get("ROS_WS") or "~/ros2_ws")
PKG = "robotic_tracking_system"
SETUP = f"source /opt/ros/humble/setup.bash && source {WS}/install/setup.bash"

NOMINAL_V = 0.03        # m/s, nominal belt speed
LEAD = 0.24             # s, tracker prediction lead unless a run overrides it
ZONE_TRAVEL = 0.45      # m of belt travel inside the working zone
GAP_FOR_REHOME = 9.0    # s of empty belt after each part (idle re-home)
# Rate of the simulation clock passed to the launch file when a run
# definition has no "clock_rate" entry (the 10 Hz comparison runs).
DEFAULT_CLOCK_RATE = 10.0

# Perception settings: with / without the two corrections.
CORR = dict(parallax_correction=True, principal_point_shift_px=-1.0)
UNCORR = dict(parallax_correction=False, principal_point_shift_px=0.0)
# Estimator settings of condition P1 (belt-speed prior everywhere).
PRIOR = dict(velocity_prior_blend=1.0, coast_prior_blend=1.0)
REPS = (1, 2, 3)


def percep(duration, v=NOMINAL_V, **kw):
    """Camera-in-the-loop run definition; kw overrides node parameters."""
    d = dict(mode="percep", duration=duration, v=v,
             spawn_period=round(ZONE_TRAVEL / v + GAP_FOR_REHOME, 1),
             stream={}, perception={}, estimator={}, tracker={},
             supervisor=True, world="conveyor_world.world", events=[])
    # Camera only by default: both belt-prior blends off.
    d["estimator"].update(velocity_prior_blend=0.0, coast_prior_blend=0.0,
                          velocity_prior=[0.0, v],
                          coast_time_limit=round(ZONE_TRAVEL / v, 1))
    for key, val in kw.items():
        if isinstance(d.get(key), dict):
            d[key].update(val)
        else:
            d[key] = val
    return d


def gt(duration, v=NOMINAL_V, **tracker):
    """Ground-truth run definition; keyword arguments are tracker parameters."""
    return dict(mode="gt", duration=duration, v=v,
                spawn_period=round(0.48 / v, 1), stream={}, tracker=tracker,
                supervisor=True, world="conveyor_world.world", events=[])


def cvest(duration, blackout, flags):
    """Arm-parked run: detections are suppressed while the part centre is
    inside the belt interval `blackout` (y range in m; empty = never)."""
    d = percep(duration, perception=dict(blackout_y=blackout, **flags))
    d["mode"] = "cvest"
    return d


def clock100(cfg):
    cfg["clock_rate"] = 100.0
    return cfg


RUNS = {}

# ======================================================================
# Simulation clock at 10 Hz: the comparison runs (data/clock_10hz)
# ======================================================================
for i in REPS:
    RUNS[f"gt_v030_r{i}"] = gt(270)
    RUNS[f"p0t_r{i}_c"] = percep(400, perception=CORR)
    RUNS[f"p1_r{i}_c"] = percep(400, perception=CORR, estimator=PRIOR)
# lead-time sweep with ground-truth targets
for lead in (0.14, 0.19, 0.22, 0.24, 0.26, 0.28, 0.34):
    RUNS[f"lead_gt_{round(lead * 100):03d}"] = gt(150, target_lead_time=lead)

# ======================================================================
# Simulation clock at 100 Hz: the main record (data/clock_100hz)
# ======================================================================
# --- main conditions: GT, P0, P1, with and without the two perception
#     corrections, and the variant with the prior only on long coasts
for i in REPS:
    RUNS[f"clk100_gt_r{i}"] = clock100(gt(400))
    RUNS[f"clk100_p0_r{i}"] = clock100(percep(400, perception=CORR))
    RUNS[f"clk100_p1_r{i}"] = clock100(percep(400, perception=CORR, estimator=PRIOR))
    RUNS[f"h_p0t_r{i}"] = clock100(percep(400, perception=UNCORR))
    RUNS[f"h_p1_r{i}"] = clock100(percep(400, perception=UNCORR, estimator=PRIOR))
    RUNS[f"h_p0c_r1_c_r{i}"] = clock100(percep(
        400, perception=CORR, estimator=dict(coast_prior_blend=1.0)))

# --- vision stage alone (arm parked): 2 x 2 split of the two corrections
for i in REPS:
    RUNS[f"h_cvc_r{i}"] = clock100(cvest(300, [0.0, 0.0], CORR))
    RUNS[f"h_cvn_r{i}"] = clock100(cvest(300, [0.0, 0.0], UNCORR))
    RUNS[f"h_cvpp_r{i}"] = clock100(cvest(300, [0.0, 0.0], dict(
        parallax_correction=False, principal_point_shift_px=-1.0)))
    RUNS[f"h_cvtf_r{i}"] = clock100(cvest(300, [0.0, 0.0], dict(
        parallax_correction=True, principal_point_shift_px=0.0)))

# --- belt speed: GT and P0 at 20 / 45 / 60 mm/s (30 mm/s = main conditions);
#     the duration covers the same number of parts at every speed
for v, tag in ((0.02, "v020"), (0.045, "v045"), (0.06, "v060")):
    for i in REPS:
        RUNS[f"h_gt_{tag}_r{i}"] = clock100(gt(int(30 + 15 * 0.48 / v), v=v))
        RUNS[f"h_p0t_{tag}_r{i}_c"] = clock100(percep(
            int(30 + 12 * (ZONE_TRAVEL / v + 9)), v=v, perception=CORR))
# belt speed changed between parts (30 -> 45 -> 20 mm/s); an event is
# (time in s, node, parameter, value)
RUNS["h_p0t_vchange_c"] = clock100(percep(420, perception=CORR, events=[
    (120, "gazebo_shape_stream_node", "target_vy", 0.045),
    (120, "gazebo_shape_stream_node", "spawn_period", 19.0),
    (250, "gazebo_shape_stream_node", "target_vy", 0.02),
    (250, "gazebo_shape_stream_node", "spawn_period", 31.5),
], estimator=dict(coast_time_limit=22.5)))

# --- robustness
for i in REPS:
    # P1 with a wrong belt-speed prior
    for pct, tag in ((-20, "m20"), (-10, "m10"), (10, "p10"), (20, "p20")):
        RUNS[f"h_p1_prior_{tag}_c_r{i}"] = clock100(percep(
            300, perception=CORR, estimator=dict(
                PRIOR, velocity_prior=[0.0, round(NOMINAL_V * (1 + pct / 100.0), 5)])))
    # P0 with image noise, calibration error, publishing delay / jitter
    RUNS[f"h_p0t_noise02_c_r{i}"] = clock100(percep(
        300, perception=CORR, world="conveyor_world_noise02.world"))
    RUNS[f"h_p0t_noise05_c_r{i}"] = clock100(percep(
        300, perception=CORR, world="conveyor_world_noise05.world"))
    RUNS[f"h_p0t_calib05_c_r{i}"] = clock100(percep(
        300, perception=dict(CORR, extrinsic_offset=[0.005, 0.005, 0.0])))
    RUNS[f"h_p0t_calib10_c_r{i}"] = clock100(percep(
        300, perception=dict(CORR, extrinsic_offset=[0.010, 0.010, 0.0])))
    RUNS[f"h_p0t_delay050_c_r{i}"] = clock100(percep(
        300, perception=dict(CORR, publish_delay=0.05)))
    RUNS[f"h_p0t_delay100j_c_r{i}"] = clock100(percep(
        300, perception=dict(CORR, publish_delay=0.10, publish_delay_jitter=0.05)))
# randomized P0 runs: shuffled shape order (seeded), shifted entry position,
# image-noise world (its noise generator is seeded by Gazebo per run)
_SHAPES = ["square", "circle", "triangle", "pentagon", "irregular"]
for i, x0 in zip(REPS, (0.49, 0.505, 0.51)):
    seq = list(_SHAPES)
    random.Random(2026 + i).shuffle(seq)
    RUNS[f"h_rnd_p0_r{i}"] = clock100(percep(
        400, perception=CORR, stream=dict(shape_sequence=seq, part_start_x=x0),
        world="conveyor_world_noise02.world"))

# --- forced occlusion (arm parked): detections suppressed for 2 / 5 / 8 s
for tag, band in (("2s", [-0.10, -0.04]), ("5s", [-0.10, 0.05]), ("8s", [-0.10, 0.14])):
    RUNS[f"h_occ_{tag}"] = clock100(cvest(200, band, UNCORR))
    RUNS[f"h_occ_{tag}_c"] = clock100(cvest(200, band, CORR))

# --- component ablations (P0, both corrections): one component switched off
for i in REPS:
    RUNS[f"h_abl_no_labelfit_c_r{i}"] = clock100(percep(
        300, perception=CORR, estimator=dict(label_consistent_fit=False)))
    RUNS[f"h_abl_no_rebase_c_r{i}"] = clock100(percep(
        300, perception=CORR, estimator=dict(rebase_pending=False)))
    RUNS[f"h_abl_no_both_c_r{i}"] = clock100(percep(
        300, perception=CORR, estimator=dict(label_consistent_fit=False,
                                             rebase_pending=False)))
    RUNS[f"h_abl_no_ff_c_r{i}"] = clock100(percep(
        300, perception=CORR, tracker=dict(use_feedforward=False)))
    RUNS[f"h_abl_no_lead_c_r{i}"] = clock100(percep(
        300, perception=CORR, tracker=dict(target_lead_time=0.0)))
    RUNS[f"h_abl_lead014_c_r{i}"] = clock100(percep(
        300, perception=CORR, tracker=dict(target_lead_time=0.14)))
    RUNS[f"h_abl_no_orihold_c_r{i}"] = clock100(percep(
        300, perception=CORR, tracker=dict(enable_orientation_hold=False)))
    RUNS[f"h_abl_no_rehome_c_r{i}"] = clock100(percep(
        400, perception=CORR, supervisor=False))
    # estimator fit in the label-independent raw-centroid frame
    RUNS[f"h_abl_rawfit_r{i}"] = clock100(percep(
        300, perception=CORR, estimator=dict(fit_frame="centroid")))

# --- long run (one hour of simulated time)
RUNS["h_p0t_long_c"] = clock100(percep(3630, perception=CORR))

# --- lead-time sweep with ground-truth targets
for lead in (0.04, 0.08, 0.12, 0.16, 0.20, 0.22, 0.24, 0.26, 0.28):
    RUNS[f"clk100_gt_{round(lead * 100):03d}"] = clock100(gt(150, target_lead_time=lead))


def fmt_param(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            return "[" + ",".join(value) + "]"
        return "[" + ",".join(repr(float(v)) for v in value) + "]"
    return repr(value) if isinstance(value, float) else str(value)


class Worker:
    def __init__(self, out_dir, worker_id):
        self.out_dir = os.path.abspath(out_dir)
        self.log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.tag = f"CAMPAIGN_WORKER={worker_id}"
        self.env = dict(os.environ)
        self.env["CAMPAIGN_WORKER"] = str(worker_id)
        self.env["ROS_DOMAIN_ID"] = str(40 + worker_id)
        self.env["GAZEBO_MASTER_URI"] = f"http://localhost:{11350 + worker_id}"
        self.procs = []

    def spawn(self, name, command, label):
        log = open(os.path.join(self.log_dir, f"{label}_{name}.log"), "w")
        proc = subprocess.Popen(
            ["bash", "-c", f"{SETUP} && exec {command}"],
            stdout=log, stderr=subprocess.STDOUT, env=self.env,
            start_new_session=True)
        self.procs.append((name, proc, log))
        return proc

    def node(self, name, executable, params, label):
        args = " ".join(f"-p {k}:={fmt_param(v)}" for k, v in params.items())
        return self.spawn(
            name, f"ros2 run {PKG} {executable} --ros-args "
                  f"-p use_sim_time:=true {args}", label)

    def call(self, command, timeout=30):
        return subprocess.run(
            ["bash", "-c", f"{SETUP} && {command}"], env=self.env,
            capture_output=True, text=True, timeout=timeout)

    def wait_log(self, label, name, text, timeout):
        path = os.path.join(self.log_dir, f"{label}_{name}.log")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with open(path, errors="replace") as f:
                if text in f.read():
                    return True
            time.sleep(1.0)
        return False

    def stop(self, names, sig=signal.SIGINT, wait=6.0):
        for name, proc, _ in self.procs:
            if name in names and proc.poll() is None:
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + wait
        while time.monotonic() < end and any(
                p.poll() is None for n, p, _ in self.procs if n in names):
            time.sleep(0.2)

    def kill_tagged(self):
        """SIGKILL every process carrying this worker's environment tag."""
        killed = 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    env = f.read().split(b"\0")
            except OSError:
                continue
            if self.tag.encode() in env:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed += 1
                except OSError:
                    pass
        return killed

    def teardown(self):
        app = [n for n, _, _ in self.procs
               if n not in ("sim", "servo", "recorder")]
        self.stop(app)
        time.sleep(3.0)               # recorder captures the stop
        self.stop(["recorder"], wait=10.0)
        self.stop(["servo"])
        self.stop(["sim"], wait=10.0)
        time.sleep(1.0)
        leftover = self.kill_tagged()
        for _, _, log in self.procs:
            log.close()
        self.procs = []
        time.sleep(2.0)
        return leftover

    def run(self, label, cfg):
        t_wall0 = time.time()
        meta = dict(label=label, config=cfg, ok=False, notes=[])
        try:
            self.spawn("sim", f"ros2 launch {PKG} panda_gazebo_moveit.launch.py "
                              f"gui:=false world:={cfg['world']} "
                              f"clock_rate:={cfg.get('clock_rate', DEFAULT_CLOCK_RATE)}", label)
            if not self.wait_log(label, "sim", "Arm at ready pose", 120):
                meta["notes"].append("homing not confirmed within 120 s")
                return meta
            time.sleep(4.0)           # move_group (t = 8 s) settles
            self.spawn("servo", f"ros2 launch {PKG} panda_gazebo_servo.launch.py",
                       label)
            time.sleep(8.0)
            started = False
            for _ in range(5):
                r = self.call("ros2 service call /servo_node/start_servo "
                              "std_srvs/srv/Trigger", timeout=30)
                if "success=True" in r.stdout:
                    started = True
                    break
                time.sleep(3.0)
            if not started:
                meta["notes"].append("start_servo failed")
                return meta

            self.node("recorder", "experiment_recorder",
                      dict(output_dir=self.out_dir, label=label,
                           record_perception=True), label)
            time.sleep(3.0)

            v = cfg["v"]
            stream = dict(spawn_period=cfg["spawn_period"], target_vy=v)
            stream.update(cfg["stream"])
            self.node("stream", "gazebo_shape_stream_node", stream, label)
            if cfg["mode"] in ("percep", "cvest"):
                self.node("perception", "part_perception_node",
                          cfg["perception"], label)
                self.node("estimator", "perception_target_estimator",
                          cfg["estimator"], label)
            if cfg["mode"] == "percep":
                tracker = dict(target_source="perception", sync_with_stream=False,
                               target_lead_time=LEAD)
                tracker.update(cfg["tracker"])
                self.node("tracker", "moving_shape_perimeter_tracker", tracker, label)
                if cfg["supervisor"]:
                    self.node("supervisor", "tracking_supervisor",
                              dict(rehome_mode="idle", rehome_every_n_cycles=1),
                              label)
            elif cfg["mode"] == "gt":
                tracker = dict(target_vy=v, target_lead_time=LEAD)
                tracker.update(cfg["tracker"])
                self.node("tracker", "moving_shape_perimeter_tracker", tracker, label)
                if cfg["supervisor"]:
                    self.node("supervisor", "tracking_supervisor",
                              dict(rehome_mode="skip", rehome_every_n_cycles=3),
                              label)

            t_start = time.monotonic()
            events = sorted(cfg.get("events", []))
            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= cfg["duration"]:
                    break
                while events and events[0][0] <= elapsed:
                    _, node, pname, value = events.pop(0)
                    r = self.call(f"ros2 param set /{node} {pname} {value}")
                    meta["notes"].append(
                        f"t={elapsed:.0f}s set {node}.{pname}={value}: "
                        f"{r.stdout.strip()}")
                dead = [n for n, p, _ in self.procs if p.poll() is not None]
                if dead:
                    meta["notes"].append(f"process exited early: {dead}")
                    break
                time.sleep(1.0)
            meta["ok"] = not any("exited early" in n for n in meta["notes"])
        finally:
            meta["leftover_killed"] = self.teardown()
            meta["wall_s"] = round(time.time() - t_wall0, 1)
            audit = os.path.join(self.out_dir, f"{label}_pipeline_audit.json")
            if os.path.exists(audit):
                with open(audit) as f:
                    a = json.load(f)
                if a.get("sim_t_start") is not None:
                    meta["sim_recorded_s"] = round(
                        a["sim_t_end"] - a["sim_t_start"], 1)
                meta["twist_nan"] = a.get("twist_nan")
                meta["servo_status"] = sorted(
                    {e["status"] for e in a.get("servo_status_events", [])})
            else:
                meta["ok"] = False
                meta["notes"].append("no audit written")
            with open(os.path.join(self.out_dir, f"{label}_run_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            print(f"[{time.strftime('%H:%M:%S')}] {label}: ok={meta['ok']} "
                  f"wall={meta['wall_s']} sim={meta.get('sim_recorded_s')} "
                  f"servo_status={meta.get('servo_status')} "
                  f"notes={meta['notes']}", flush=True)
        return meta


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        for label, cfg in RUNS.items():
            print(f"{label:24s} {cfg['mode']:6s} {cfg['duration']:5d} s  "
                  f"v={cfg['v']}  period={cfg['spawn_period']}  "
                  f"clock={cfg.get('clock_rate', DEFAULT_CLOCK_RATE):.0f} Hz")
        return
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    worker = Worker(sys.argv[1], int(sys.argv[2]))
    worker.kill_tagged()
    for label in sys.argv[3:]:
        if os.path.exists(os.path.join(worker.out_dir, f"{label}_run_meta.json")):
            print(f"{label}: already done, skipped", flush=True)
            continue
        if label not in RUNS:
            sys.exit(f"unknown run label: {label} (see --list)")
        worker.run(label, RUNS[label])


if __name__ == "__main__":
    main()
