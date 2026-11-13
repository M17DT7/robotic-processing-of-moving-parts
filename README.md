# Robotic Processing of Moving Parts

Software and recorded data for the article

> **Perception-Driven Continuous Path Tracking of Moving Conveyor Parts Using
> ROS 2 and MoveIt Servo**
> Mahmoud Tafran, Furkan Korkmaz, Ergun Nart —
> Sakarya University of Applied Sciences

A Franka Emika Panda follows the perimeter of parts that move on a conveyor.
One overhead camera is the only online source of the part motion: an
image-processing stage detects the part and its shape, a target-state
estimator carries the part motion through the seconds in which the arm hides
the part from the camera, and a velocity controller drives the arm through
MoveIt Servo. Everything runs in Gazebo.

```
overhead camera ─► part_perception_node ─► perception_target_estimator ─►
moving_shape_perimeter_tracker ─► MoveIt Servo ─► joint trajectory controller ─► Panda
                         tracking_supervisor (re-homing between parts)
```

## Contents

| Directory | Content |
|---|---|
| `robotic_tracking_system/` | the ROS 2 package: nodes, launch files, world, robot description, parameters |
| `experiments/` | scripts that record the experiment campaign and compute the tables and figures |
| `data/` | the recorded runs behind the article (see `data/README.md`) |
| `results/` | aggregated results computed from `data/` |

## Requirements

- Ubuntu 22.04, ROS 2 Humble
- MoveIt 2 (with `moveit_servo` and the Panda resources), Gazebo Classic 11,
  `gazebo_ros2_control`, `ros2_controllers`
- Python 3.10 with OpenCV, NumPy, PyYAML, PyQt5; Matplotlib for the figures

The analysis scripts in `experiments/` need only Python, NumPy and Matplotlib.

## Reproducing the tables without a simulator

```bash
python3 experiments/aggregate.py data/clock_100hz --clock100 --json results/aggregate_100hz.json
python3 experiments/aggregate.py data/clock_10hz --main-only --json results/aggregate_10hz.json
python3 experiments/fitted_velocity.py data/clock_100hz --json results/fitted_velocity_100hz.json
python3 experiments/make_figures.py data/clock_100hz results/aggregate_100hz.json figures
```

`experiments/analyze.py` prints the per-cycle analysis of a single run.

## Building the package

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone <this repository>
ln -s robotic-processing-of-moving-parts/robotic_tracking_system .
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select robotic_tracking_system
source install/setup.bash
```

## Running the system

Every node runs on simulation time.

```bash
# terminal 1: Gazebo, the robot, MoveIt; the arm moves to its ready pose
ros2 launch robotic_tracking_system panda_gazebo_moveit.launch.py

# terminal 2: MoveIt Servo, then start it
ros2 launch robotic_tracking_system panda_gazebo_servo.launch.py
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger

# terminal 3: part stream, perception, estimator, tracker and supervisor
ros2 launch robotic_tracking_system perception_tracking.launch.py

# optional: operator dashboard
ros2 run robotic_tracking_system operator_dashboard --ros-args -p use_sim_time:=true
```

`shape_perimeter_tracking.launch.py` runs the same task with targets from the
true part pose instead of the camera. The simulation clock is published at
100 Hz; `clock_rate:=10.0` on the first launch file selects the simulator's
default.

## Repeating the experiments

`experiments/campaign.py` records the runs headless, one simulation per run
label, and can run several workers side by side:

```bash
python3 experiments/campaign.py --list
ROS_WS=~/ros2_ws python3 experiments/campaign.py <output dir> <worker id> <label> [<label> ...]
```

## Licence

Code and recorded data: BSD 3-Clause (`LICENSE`).
The Panda meshes are third-party files, see
`robotic_tracking_system/meshes/NOTICE`.

## Citation

The article is under review; its reference will be added here when it is
published.
