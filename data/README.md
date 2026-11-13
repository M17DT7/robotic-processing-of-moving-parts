# Recorded runs

Every number in the article is computed from the runs in this directory by
`experiments/aggregate.py` and `experiments/fitted_velocity.py`; the outputs
are stored in `results/`.

- `clock_100hz/` – the experiments of the article (simulation clock published
  at 100 Hz).
- `clock_10hz/` – the three main conditions and the lead-time sweep with the
  simulation clock at the simulator's default of 10 Hz (comparison rows).

## Files of one run

| File | Content |
|---|---|
| `<label>_tracking_log.csv` | simulation time, commanded target, tool point, age of the target message, tool orientation and orientation error (10 Hz) |
| `<label>_perception_log.csv` | true part pose and shape, detection, estimator state and their ages (10 Hz) |
| `<label>_joint_log.csv` | joint positions (10 Hz) |
| `<label>_pipeline_audit.json` | message counts, command statistics and MoveIt Servo status events of the run |
| `<label>_run_meta.json` | configuration of the run (mode, duration, belt speed, node parameters) and its outcome |
| `logs/<label>_estimator.log` | estimator log; holds the belt velocity fitted at lock time (only for the runs used by `fitted_velocity.py`) |

Positions are in metres in the robot base frame (`panda_link0`), times in
seconds of simulation time.

## Run labels

| Part of the label | Meaning |
|---|---|
| `gt` | targets from the true part pose (ground truth) |
| `p0t` | camera only (P0) |
| `p1` | camera with the belt-speed prior (P1) |
| `p0c` | fitted velocity, belt prior only on long coasts |
| suffix `_c` | with top-face recovery and principal-point correction; without the suffix both are off |
| prefix `h_`, `clk100_` | simulation clock at 100 Hz |
| `v020`, `v045`, `v060` | belt speed 0.02, 0.045, 0.06 m/s (otherwise 0.03 m/s) |
| `vchange` | belt speed changed between parts |
| `prior_m20` … `prior_p20` | belt prior wrong by −20 … +20 % |
| `noise02`, `noise05` | Gaussian pixel noise 0.02, 0.05 |
| `calib05`, `calib10` | camera position error 5, 10 mm in x and y |
| `delay050`, `delay100j` | extra detection delay 50 ms; 100 ms with 0–50 ms jitter |
| `occ_2s`, `occ_5s`, `occ_8s` | detections suppressed for 2, 5, 8 s (arm parked) |
| `abl_*` | one component switched off (see `experiments/campaign.py`) |
| `cvc`, `cvn`, `cvpp`, `cvtf` | vision stage alone, arm parked: both corrections, none, principal point only, top-face recovery only |
| `rnd_p0` | randomized shape order, entry position and pixel noise |
| `long` | one-hour run |
| `lead_gt_0xx`, `clk100_gt_0xx` | lead-time sweep, value in hundredths of a second |
| `_r1`, `_r2`, `_r3` | repetition |

## Licence

The data in this directory are released under the same BSD 3-Clause licence
as the code (`LICENSE` in the repository root).
