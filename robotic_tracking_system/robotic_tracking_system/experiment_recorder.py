"""Record tracking-experiment data to CSV/JSON.

Node name:      experiment_recorder
Purpose:        Sample the end-effector pose (TF, position AND orientation),
                the active target, and the command pipeline on a fixed
                sim-time grid, and write the tracking log, joint log and
                pipeline audit files that the experiment analysis reads.
Subscriptions:  /active_target_point (geometry_msgs/PointStamped)
                /servo_node/delta_twist_cmds (geometry_msgs/TwistStamped)
                /panda_arm_controller/joint_trajectory
                    (trajectory_msgs/JointTrajectory)
                /joint_states (sensor_msgs/JointState)
                /part_stream/cycle_start (std_msgs/String)
                /servo_node/status (std_msgs/Int8)
                /tracking_supervisor/state (std_msgs/String, latched;
                    only published when tracking_supervisor runs)
Publications:   none
Parameters:     output_dir (required, absolute path), label ('run'),
                sample_period (0.1 s), planning_frame, ee_frame,
                record_joints (True), record_perception (False; also
                subscribes /perception/part_pose, /perception/part_corners,
                /perception/part_shape, /part_stream/ground_truth_pose and
                /perception/part_state, /perception/cycle_start and
                writes <label>_perception_log.csv)
Frames:         planning_frame -> ee_frame via TF (panda_link0 -> panda_link8)
Update rate:    1 / sample_period (10 Hz default), sim time
Dependencies:   run with use_sim_time:=true so the sampling grid matches the
                sim clock the trackers and Servo run on
Outputs:        <output_dir>/<label>_tracking_log.csv
                    sim_t,tgt_x,tgt_y,tgt_z,ee_x,ee_y,ee_z,tgt_age_s,
                    qx,qy,qz,qw,ori_err_deg
                    (tgt_age_s = sample time minus the target stamp;
                    ori_err_deg = shortest-path angle to the first valid TF
                    sample of the run)
                <output_dir>/<label>_joint_log.csv (if record_joints)
                    sim_t,panda_joint1..panda_joint7
                <output_dir>/<label>_pipeline_audit.json (on shutdown)
"""

import json
import math
import os

import rclpy
import rclpy.time
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import PointStamped, PolygonStamped, PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8, String
from trajectory_msgs.msg import JointTrajectory

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

# Commands with any linear axis beyond this magnitude count as saturated
# (Servo clips unitless commands to [-1, 1]; the trackers clamp at 0.999).
SATURATION_THRESHOLD = 0.99
# Below this magnitude a command axis is treated as zero for sign-reversal
# counting, so numerical noise around zero is not counted as chatter.
SIGN_EPSILON = 1e-4


def _finite(*values):
    return all(math.isfinite(v) for v in values)


class ChannelStats:
    """Online per-axis stats for one 3-vector command channel."""

    def __init__(self):
        self.prev = None
        self.prev_sign = [0, 0, 0]
        self.sign_reversals = [0, 0, 0]
        self.step_sq_sum = [0.0, 0.0, 0.0]
        self.step_count = 0
        self.saturated_msgs = 0
        self.max_abs = [0.0, 0.0, 0.0]

    def update(self, vec):
        if max(abs(v) for v in vec) > SATURATION_THRESHOLD:
            self.saturated_msgs += 1
        for i in range(3):
            self.max_abs[i] = max(self.max_abs[i], abs(vec[i]))
            sign = 0
            if vec[i] > SIGN_EPSILON:
                sign = 1
            elif vec[i] < -SIGN_EPSILON:
                sign = -1
            if sign != 0 and self.prev_sign[i] != 0 and sign != self.prev_sign[i]:
                self.sign_reversals[i] += 1
            if sign != 0:
                self.prev_sign[i] = sign
        if self.prev is not None:
            for i in range(3):
                self.step_sq_sum[i] += (vec[i] - self.prev[i]) ** 2
            self.step_count += 1
        self.prev = list(vec)

    def summary(self, msg_count):
        step_std = [
            math.sqrt(s / self.step_count) if self.step_count else 0.0
            for s in self.step_sq_sum
        ]
        return {
            "sign_reversals_xyz": self.sign_reversals,
            "step_std_xyz": [round(v, 5) for v in step_std],
            "saturated_msgs": self.saturated_msgs,
            "saturation_fraction": (
                round(self.saturated_msgs / msg_count, 4) if msg_count else 0.0
            ),
            "max_abs_xyz": [round(v, 5) for v in self.max_abs],
        }


def quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_angle_deg(a, b):
    """Shortest-path rotation angle between two unit quaternions, degrees."""
    q = quat_mul(a, quat_conj(b))
    w = min(1.0, max(-1.0, abs(q[3])))
    return math.degrees(2.0 * math.acos(w))


class ExperimentRecorder(Node):
    def __init__(self):
        super().__init__("experiment_recorder")

        self.declare_parameter("output_dir", "")
        self.declare_parameter("label", "run")
        self.declare_parameter("sample_period", 0.1)
        self.declare_parameter("planning_frame", "panda_link0")
        self.declare_parameter("ee_frame", "panda_link8")
        self.declare_parameter("record_joints", True)
        self.declare_parameter("record_perception", False)

        self.output_dir = self.get_parameter("output_dir").value
        self.label = self.get_parameter("label").value
        self.sample_period = self.get_parameter("sample_period").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.record_joints = self.get_parameter("record_joints").value
        self.record_perception = self.get_parameter("record_perception").value

        if not self.output_dir:
            raise RuntimeError("Parameter 'output_dir' is required")
        os.makedirs(self.output_dir, exist_ok=True)

        base = os.path.join(self.output_dir, self.label)
        self.tracking_csv = open(base + "_tracking_log.csv", "w")
        self.tracking_csv.write(
            "sim_t,tgt_x,tgt_y,tgt_z,ee_x,ee_y,ee_z,tgt_age_s,"
            "qx,qy,qz,qw,ori_err_deg\n"
        )
        self.joint_csv = None
        if self.record_joints:
            self.joint_csv = open(base + "_joint_log.csv", "w")
            self.joint_csv.write("sim_t," + ",".join(PANDA_JOINTS) + "\n")
        self.audit_path = base + "_pipeline_audit.json"
        self.perception_csv = None
        if self.record_perception:
            self.perception_csv = open(base + "_perception_log.csv", "w")
            self.perception_csv.write(
                "sim_t,det_x,det_y,det_z,det_age_s,det_shape,n_corners,"
                "corners,gt_x,gt_y,gt_z,gt_age_s,gt_shape,"
                "est_x,est_y,est_age_s,est_shape\n"
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Latest-value caches filled by subscriptions
        self.latest_target = None
        self.latest_joint_positions = None

        # Audit accumulators (written to <label>_pipeline_audit.json)
        self.twist_count = 0
        self.twist_nan = 0
        self.twist_nonfinite_examples = []
        self.twist_first_stamp = None
        self.twist_last_stamp = None
        self.twist_last_nonzero_stamp = None
        self.linear_stats = ChannelStats()
        self.angular_stats = ChannelStats()
        self.angular_nonzero_msgs = 0

        self.traj_count = 0
        self.traj_nan = 0
        self.traj_first_stamp = None
        self.traj_last_recv_sim = None

        self.js_count = 0
        self.samples = 0
        self.sim_t_start = None
        self.sim_t_end = None

        self.cycle_starts = []
        self.servo_status_events = []
        self.last_servo_status = None
        self.supervisor_state_events = []

        # Perception caches (only filled when record_perception)
        self.latest_detection = None
        self.latest_corners = None
        self.latest_det_shape = ""
        self.latest_ground_truth = None
        self.detection_msgs = 0
        self.latest_estimate = None       # /perception/part_state
        self.latest_est_shape = ""

        self.reference_quat = None
        self.ori_err_max_deg = 0.0
        self.audit_written = False

        self.create_subscription(
            PointStamped, "/active_target_point", self.target_callback, 10
        )
        self.create_subscription(
            TwistStamped, "/servo_node/delta_twist_cmds", self.twist_callback, 10
        )
        self.create_subscription(
            JointTrajectory,
            "/panda_arm_controller/joint_trajectory",
            self.trajectory_callback,
            10,
        )
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.create_subscription(
            String, "/part_stream/cycle_start", self.cycle_start_callback, 10
        )
        self.create_subscription(
            Int8, "/servo_node/status", self.servo_status_callback, 10
        )
        # Latched to match the supervisor's transient_local publisher (also
        # captures the current state when the recorder starts late).
        self.create_subscription(
            String,
            "/tracking_supervisor/state",
            self.supervisor_state_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        if self.record_perception:
            self.create_subscription(
                PoseStamped, "/perception/part_pose", self.detection_callback, 10
            )
            self.create_subscription(
                PolygonStamped, "/perception/part_corners", self.det_corners_callback, 10
            )
            self.create_subscription(
                String, "/perception/part_shape", self.det_shape_callback, 10
            )
            self.create_subscription(
                PoseStamped,
                "/part_stream/ground_truth_pose",
                self.ground_truth_callback,
                10,
            )
            self.create_subscription(
                PoseStamped, "/perception/part_state", self.estimate_callback, 10
            )
            self.create_subscription(
                String, "/perception/cycle_start", self.est_shape_callback, 10
            )

        self.timer = self.create_timer(self.sample_period, self.sample)

        self.get_logger().info(
            f"Recording to {base}_* (sample_period {self.sample_period}s, "
            f"label '{self.label}')"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def target_callback(self, msg):
        self.latest_target = msg

    def twist_callback(self, msg):
        self.twist_count += 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.twist_first_stamp is None:
            self.twist_first_stamp = stamp
        self.twist_last_stamp = stamp
        lin = (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z)
        ang = (msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z)
        if not _finite(*lin, *ang):
            self.twist_nan += 1
            if len(self.twist_nonfinite_examples) < 5:
                self.twist_nonfinite_examples.append(
                    {"stamp": stamp, "linear": list(lin), "angular": list(ang)}
                )
            return
        if any(abs(v) > SIGN_EPSILON for v in lin + ang):
            self.twist_last_nonzero_stamp = stamp
        if any(abs(v) > SIGN_EPSILON for v in ang):
            self.angular_nonzero_msgs += 1
        self.linear_stats.update(lin)
        self.angular_stats.update(ang)

    def trajectory_callback(self, msg):
        self.traj_count += 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.traj_first_stamp is None:
            self.traj_first_stamp = stamp
        self.traj_last_recv_sim = self.now_sim()
        for point in msg.points:
            if not _finite(*point.positions):
                self.traj_nan += 1
                break

    def joint_state_callback(self, msg):
        self.js_count += 1
        positions = {}
        for name, pos in zip(msg.name, msg.position):
            positions[name] = pos
        if all(j in positions for j in PANDA_JOINTS):
            self.latest_joint_positions = [positions[j] for j in PANDA_JOINTS]

    def cycle_start_callback(self, msg):
        self.cycle_starts.append({"sim_t": self.now_sim(), "shape": msg.data})

    def detection_callback(self, msg):
        self.latest_detection = msg
        self.detection_msgs += 1

    def det_corners_callback(self, msg):
        self.latest_corners = msg

    def det_shape_callback(self, msg):
        self.latest_det_shape = msg.data

    def ground_truth_callback(self, msg):
        self.latest_ground_truth = msg

    def estimate_callback(self, msg):
        self.latest_estimate = msg

    def est_shape_callback(self, msg):
        self.latest_est_shape = msg.data

    def supervisor_state_callback(self, msg):
        self.supervisor_state_events.append(
            {"sim_t": self.now_sim(), "state": msg.data}
        )

    def servo_status_callback(self, msg):
        if msg.data != self.last_servo_status:
            self.servo_status_events.append(
                {"sim_t": self.now_sim(), "status": int(msg.data)}
            )
            self.last_servo_status = msg.data

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def now_sim(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sample(self):
        now = self.now_sim()
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.001),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF {self.planning_frame} -> {self.ee_frame} unavailable: {ex}",
                throttle_duration_sec=5.0,
            )
            return

        t = transform.transform.translation
        r = transform.transform.rotation
        quat = (r.x, r.y, r.z, r.w)
        if self.reference_quat is None:
            self.reference_quat = quat
            self.get_logger().info(
                f"Reference orientation captured at sim_t {now:.2f}: "
                f"({r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f})"
            )
        ori_err_deg = quat_angle_deg(self.reference_quat, quat)
        self.ori_err_max_deg = max(self.ori_err_max_deg, ori_err_deg)

        if self.latest_target is not None:
            tgt = self.latest_target.point
            tgt_stamp = (
                self.latest_target.header.stamp.sec
                + self.latest_target.header.stamp.nanosec * 1e-9
            )
            tgt_age = now - tgt_stamp
            tgt_row = f"{tgt.x:.6f},{tgt.y:.6f},{tgt.z:.6f}"
        else:
            tgt_age = float("nan")
            tgt_row = "nan,nan,nan"

        self.tracking_csv.write(
            f"{now:.1f},{tgt_row},{t.x:.6f},{t.y:.6f},{t.z:.6f},"
            f"{tgt_age:.3f},{r.x:.6f},{r.y:.6f},{r.z:.6f},{r.w:.6f},"
            f"{ori_err_deg:.4f}\n"
        )
        self.tracking_csv.flush()

        if self.joint_csv is not None and self.latest_joint_positions is not None:
            joints = ",".join(f"{p:.5f}" for p in self.latest_joint_positions)
            self.joint_csv.write(f"{now:.1f},{joints}\n")
            self.joint_csv.flush()

        if self.perception_csv is not None:
            self.write_perception_row(now)

        self.samples += 1
        if self.sim_t_start is None:
            self.sim_t_start = now
        self.sim_t_end = now

    @staticmethod
    def _stamp_to_sec(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def write_perception_row(self, now):
        if self.latest_detection is not None:
            d = self.latest_detection.pose.position
            det_age = now - self._stamp_to_sec(self.latest_detection.header)
            det_row = f"{d.x:.6f},{d.y:.6f},{d.z:.6f},{det_age:.3f}"
        else:
            det_row = "nan,nan,nan,nan"
        if self.latest_corners is not None:
            pts = self.latest_corners.polygon.points
            corners = ";".join(f"{p.x:.6f}:{p.y:.6f}" for p in pts)
            n_corners = len(pts)
        else:
            corners = ""
            n_corners = 0
        if self.latest_ground_truth is not None:
            g = self.latest_ground_truth.pose.position
            gt_age = now - self._stamp_to_sec(self.latest_ground_truth.header)
            gt_row = f"{g.x:.6f},{g.y:.6f},{g.z:.6f},{gt_age:.3f}"
        else:
            gt_row = "nan,nan,nan,nan"
        gt_shape = self.cycle_starts[-1]["shape"] if self.cycle_starts else ""
        if self.latest_estimate is not None:
            e = self.latest_estimate.pose.position
            est_age = now - self._stamp_to_sec(self.latest_estimate.header)
            est_row = f"{e.x:.6f},{e.y:.6f},{est_age:.3f}"
        else:
            est_row = "nan,nan,nan"
        self.perception_csv.write(
            f"{now:.1f},{det_row},{self.latest_det_shape},{n_corners},"
            f"{corners},{gt_row},{gt_shape},{est_row},{self.latest_est_shape}\n"
        )
        self.perception_csv.flush()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def write_audit(self):
        if self.audit_written:
            return
        self.audit_written = True
        audit = {
            "twist_count": self.twist_count,
            "twist_nan": self.twist_nan,
            "twist_nonfinite_examples": self.twist_nonfinite_examples,
            "twist_first_stamp": self.twist_first_stamp,
            "twist_last_stamp": self.twist_last_stamp,
            "twist_last_nonzero_stamp": self.twist_last_nonzero_stamp,
            "traj_count": self.traj_count,
            "traj_nan": self.traj_nan,
            "traj_first_stamp": self.traj_first_stamp,
            "traj_last_recv_sim": self.traj_last_recv_sim,
            "js_count": self.js_count,
            "samples": self.samples,
            "sim_t_start": self.sim_t_start,
            "sim_t_end": self.sim_t_end,
            "twist_linear": self.linear_stats.summary(self.twist_count),
            "twist_angular": self.angular_stats.summary(self.twist_count),
            "twist_angular_nonzero_msgs": self.angular_nonzero_msgs,
            "ori_err_max_deg": round(self.ori_err_max_deg, 4),
            "reference_quat_xyzw": (
                [round(v, 6) for v in self.reference_quat]
                if self.reference_quat
                else None
            ),
            "cycle_starts": self.cycle_starts,
            "servo_status_events": self.servo_status_events,
            "supervisor_state_events": self.supervisor_state_events,
            "detection_msgs": self.detection_msgs,
        }
        with open(self.audit_path, "w") as f:
            json.dump(audit, f, indent=2)
        self.get_logger().info(
            f"Audit written to {self.audit_path} "
            f"({self.samples} samples, {self.twist_count} twists)"
        )

    def close(self):
        self.write_audit()
        self.tracking_csv.close()
        if self.joint_csv is not None:
            self.joint_csv.close()
        if self.perception_csv is not None:
            self.perception_csv.close()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
