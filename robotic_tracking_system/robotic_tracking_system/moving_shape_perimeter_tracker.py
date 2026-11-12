"""Generic moving-shape perimeter tracking using MoveIt Servo.

Node name:      moving_shape_perimeter_tracker
Purpose:        Follow the PERIMETER of parts of varying shape (polygons
                with any number of corners, circles) while the part moves
                along the conveyor. Every shape is an ordered CCW vertex
                list in the part frame (circles are discretized), the path
                is the closed polyline through the vertices, and corners
                are exact path points.
Subscriptions:  /part_stream/cycle_start (std_msgs/String) - optional sync:
                each message starts a new part cycle with the named shape
                (published by gazebo_shape_stream_node; model mode only)
                target_source == 'perception' instead subscribes:
                /perception/part_state (geometry_msgs/PoseStamped),
                /perception/part_state_velocity (Vector3Stamped),
                /perception/cycle_start (std_msgs/String) - the part pose,
                velocity and cycle boundaries then come entirely from the
                camera pipeline (perception_target_estimator); the
                mathematical part model and ground-truth stream sync are
                not used
                /tracking_enabled (std_msgs/Bool, latched) - optional gate
                from tracking_supervisor: while False the control loop
                publishes nothing (no twists, no target) so the supervisor
                can re-home the arm; without a publisher the tracker runs
                unconditionally
                /drawn_paths (std_msgs/String JSON, latched) - named paths
                in the part frame,
                {"paths": [{"name", "closed", "points": [[x, y], ...]}]}
                /mission (std_msgs/String JSON, latched) - segment
                sequence executed per matching part: modes perimeter
                (laps), drawn (name, reverse, passes, dwell), corner /
                edge (index, dwell), center (dwell). Open drawn paths
                ping-pong (passes = one-way traversals), closed ones lap.
Publications:   /servo_node/delta_twist_cmds (geometry_msgs/TwistStamped)
                /mission_state (std_msgs/String) - segment transitions
                /active_target_point (geometry_msgs/PointStamped)
                /part_stream/active_shape (std_msgs/String, on cycle change)
                /shape_outline_marker, /shape_corners_marker,
                /active_target_marker (visualization_msgs/Marker)
Parameters:     shapes_file ('' = package share config/shapes.yaml),
                shape_sequence, circle_discretization, perimeter_path_speed,
                start_path_when_near, path_start_tolerance, sync_with_stream,
                part_start_x/y, part_z, target_vx/vy, part_yaw,
                part_angular_speed, target_z_offset, follow_duration,
                control_rate, kp/ki/kd_linear, integral_limit,
                max_linear_speed, use_feedforward, tool_z_offset,
                servo_linear_scale, target_lead_time, marker_decimation,
                enable_orientation_hold, kp_angular, servo_angular_scale,
                max_angular_speed, reference_orientation,
                planning_frame, ee_frame, servo_topic
Frames:         planning_frame (panda_link0) == Gazebo world frame
Update rate:    control_rate (100 Hz nominal)
Dependencies:   MoveIt Servo running with use_sim_time; run this node with
                use_sim_time:=true (Servo judges command staleness by
                comparing its sim clock with the command header stamp)
"""

import json
import math
import os

import yaml

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import (Point, PoseStamped, TwistStamped,
                               PointStamped, Vector3Stamped)
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

import tf2_ros
from tf2_ros import TransformException


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


def quat_to_axis_angle(q):
    """Rotation vector (axis * angle, rad) of a unit quaternion, shortest path."""
    x, y, z, w = q
    if w < 0.0:  # q and -q are the same rotation; take the short way round
        x, y, z, w = -x, -y, -z, -w
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(norm, w)
    return (x / norm * angle, y / norm * angle, z / norm * angle)


class ShapePath:
    """Closed polyline through the shape vertices, parameterized by arc length."""

    def __init__(self, name, vertices):
        self.name = name
        self.vertices = vertices  # list of (x, y) in part frame, CCW
        self.edge_lengths = []
        n = len(vertices)
        for i in range(n):
            ax, ay = vertices[i]
            bx, by = vertices[(i + 1) % n]
            self.edge_lengths.append(math.hypot(bx - ax, by - ay))
        self.perimeter = sum(self.edge_lengths)

    def point_and_direction(self, s):
        """Position and unit direction on the closed path at arc length s."""
        if self.perimeter < 1e-9:
            # Degenerate single-point path (the 'pending' entry-region
            # placeholder): hold the part origin.
            return self.vertices[0], (0.0, 0.0)
        s = s % self.perimeter
        n = len(self.vertices)
        for i in range(n):
            if s <= self.edge_lengths[i] or i == n - 1:
                ax, ay = self.vertices[i]
                bx, by = self.vertices[(i + 1) % n]
                length = self.edge_lengths[i]
                frac = s / length if length > 1e-9 else 0.0
                ux = (bx - ax) / length if length > 1e-9 else 0.0
                uy = (by - ay) / length if length > 1e-9 else 0.0
                return (ax + frac * (bx - ax), ay + frac * (by - ay)), (ux, uy)
            s -= self.edge_lengths[i]
        return self.vertices[0], (0.0, 0.0)


class DrawnPath:
    """Named path in the part frame, received on /drawn_paths.
    Open paths PING-PONG: pass 1 runs start->end, pass 2 end->start,
    ... (like the edge mode); closed paths wrap around (laps). 'reverse'
    starts from the last point instead of the first."""

    def __init__(self, name, points, closed):
        self.name = name
        self.closed = closed
        self.points = list(points)    # path order, start first
        pts = self.points + ([self.points[0]] if closed else [])
        self.seg_lengths = [
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        ]
        self._pts = pts
        self.length = sum(self.seg_lengths)

    def _at(self, s):
        """Point and unit direction at arc length s in [0, length]."""
        if self.length < 1e-9:
            return self._pts[0], (0.0, 0.0)
        s = min(max(s, 0.0), self.length)
        for i, seg_len in enumerate(self.seg_lengths):
            if s <= seg_len + 1e-9 and seg_len > 1e-9:
                ax, ay = self._pts[i]
                bx, by = self._pts[i + 1]
                frac = s / seg_len
                return ((ax + frac * (bx - ax), ay + frac * (by - ay)),
                        ((bx - ax) / seg_len, (by - ay) / seg_len))
            s -= seg_len
        return self._pts[-1], (0.0, 0.0)

    def point_and_direction(self, s, reverse=False):
        """s is the distance travelled since the segment started; beyond
        passes * length the caller ends the segment."""
        if self.length < 1e-9:
            return self._pts[0], (0.0, 0.0)
        if self.closed:
            u = s % self.length
            if reverse:
                u = self.length - u
            (x, y), (dx, dy) = self._at(u)
            return (x, y), ((-dx, -dy) if reverse else (dx, dy))
        u = s % (2.0 * self.length)
        sign = 1.0
        if u > self.length:
            u, sign = 2.0 * self.length - u, -1.0
        if reverse:
            u, sign = self.length - u, -sign
        (x, y), (dx, dy) = self._at(u)
        return (x, y), (sign * dx, sign * dy)

    def end_point(self, passes, reverse=False):
        """Where the path is after 'passes' traversals (dwell point)."""
        if self.closed:
            return self.points[0]
        at_start = (int(passes) % 2 == 0)
        if reverse:
            at_start = not at_start
        return self.points[0] if at_start else self.points[-1]


class MovingShapePerimeterTracker(Node):
    def __init__(self):
        super().__init__("moving_shape_perimeter_tracker")

        self.declare_parameter("planning_frame", "panda_link0")
        self.declare_parameter("ee_frame", "panda_link8")
        self.declare_parameter("servo_topic", "/servo_node/delta_twist_cmds")

        # 'model' = mathematical part model + ground-truth stream sync;
        # 'perception' = part pose, velocity and cycle boundaries from
        # the camera pipeline.
        self.declare_parameter("target_source", "model")
        self.declare_parameter("perception_state_timeout", 1.0)

        self.declare_parameter("shapes_file", "")
        self.declare_parameter(
            "shape_sequence",
            ["square", "circle", "triangle", "pentagon", "irregular"],
        )
        self.declare_parameter("circle_discretization", 36)

        self.declare_parameter("part_start_x", 0.50)
        self.declare_parameter("part_start_y", -0.25)
        self.declare_parameter("part_z", 0.41)
        self.declare_parameter("target_vx", 0.00)
        self.declare_parameter("target_vy", 0.03)
        self.declare_parameter("part_yaw", 0.0)
        self.declare_parameter("part_angular_speed", 0.0)
        self.declare_parameter("target_z_offset", 0.06)
        self.declare_parameter("follow_duration", 16.0)

        self.declare_parameter("perimeter_path_speed", 0.06)
        self.declare_parameter("start_path_when_near", True)
        self.declare_parameter("path_start_tolerance", 0.02)
        self.declare_parameter("sync_with_stream", True)
        # Tracking behavior on the part (GUI-selectable):
        #   perimeter    - closed polyline through the corners (default)
        #   corners      - jump corner to corner, dwelling
        #                  corner_dwell_time at each
        #   edge_pingpong- triangle-wave along edge edge_index
        #   single_lap   - one perimeter circuit, then hold the part
        #                  center until the part leaves
        #   center       - follow the part origin itself
        #   corner       - follow (ride on) the single corner
        #                  corner_index; out-of-range clamps to the last
        # Changes latch at the NEXT cycle (pending_path_mode) so a
        # mid-part switch never jerks the arm.
        self.declare_parameter("path_mode", "perimeter")
        self.declare_parameter("corner_dwell_time", 1.5)
        self.declare_parameter("edge_index", 0)
        self.declare_parameter("corner_index", 0)

        self.declare_parameter("control_rate", 100.0)
        # kp is in 1/s: velocity correction (m/s) per metre of error.
        # Commands are computed in m/s and converted to Servo's unitless
        # range by dividing by servo_linear_scale (= scale.linear in the
        # Servo YAML). Without that conversion the feed-forward is far too
        # weak and the feedback must saturate to compensate, which produces
        # a 1.5-2.3 Hz end-effector oscillation.
        self.declare_parameter("kp_linear", 8.0)
        self.declare_parameter("ki_linear", 0.0)
        self.declare_parameter("kd_linear", 0.0)
        self.declare_parameter("integral_limit", 0.05)
        self.declare_parameter("max_linear_speed", 0.999)
        self.declare_parameter("use_feedforward", True)
        self.declare_parameter("tool_z_offset", -0.120)
        self.declare_parameter("servo_linear_scale", 0.4)
        # Compensates the transport delay of the control chain (tracker
        # period + Servo publish period + trajectory interpolation + TF
        # feedback): the position error is computed against the target
        # predicted target_lead_time seconds ahead along its known motion.
        # Measured against the TRUE part position, a lead-time sweep has
        # its lowest tracking error at 0.24 s (at 0.24-0.26 s with the
        # simulation clock at 10 Hz).
        self.declare_parameter("target_lead_time", 0.24)
        # Publish RViz markers every Nth control tick (marker serialization
        # is a large share of the control-loop budget).
        self.declare_parameter("marker_decimation", 5)
        # Orientation hold: with angular commands always zero the
        # EE orientation is uncontrolled and drifts at ~10 deg/min during
        # continuous tracking. A proportional angular velocity steers the
        # EE back to a reference orientation captured at the first path
        # start (persisting across part cycles - re-capturing each cycle
        # would silently re-baseline the accumulated drift). The hold
        # reduces the drift from 10.08 to 0.04 deg/min and leaves position
        # tracking unchanged.
        self.declare_parameter("enable_orientation_hold", True)
        self.declare_parameter("kp_angular", 2.0)
        # = scale.rotational in the Servo YAML: converts rad/s to Servo's
        # unitless range, mirroring servo_linear_scale.
        self.declare_parameter("servo_angular_scale", 0.8)
        self.declare_parameter("max_angular_speed", 0.5)
        # Optional fixed reference quaternion [x, y, z, w]; all-zero (the
        # default, not a valid rotation) = capture from TF at the first
        # path start.
        self.declare_parameter("reference_orientation", [0.0, 0.0, 0.0, 0.0])

        self.load_parameters()
        self.load_shapes()

        # Runtime tuning (operator dashboard): these parameters take
        # effect immediately when set via the ROS 2 parameter services
        # (without this callback, parameters would only be read once at
        # init). path_mode is latched at the next cycle so a mid-part
        # switch never jerks the arm.
        self.RUNTIME_TUNABLE = {
            "kp_linear", "ki_linear", "kd_linear", "integral_limit",
            "max_linear_speed", "target_lead_time", "use_feedforward",
            "kp_angular", "max_angular_speed", "enable_orientation_hold",
            "perimeter_path_speed", "corner_dwell_time", "edge_index",
            "corner_index", "path_mode",
        }
        self.add_on_set_parameters_callback(self.on_parameters_changed)

        self.twist_pub = self.create_publisher(TwistStamped, self.servo_topic, 10)
        self.target_point_pub = self.create_publisher(PointStamped, "/active_target_point", 10)
        self.active_shape_pub = self.create_publisher(String, "/part_stream/active_shape", 10)
        self.outline_marker_pub = self.create_publisher(Marker, "/shape_outline_marker", 10)
        self.corners_marker_pub = self.create_publisher(Marker, "/shape_corners_marker", 10)
        self.target_marker_pub = self.create_publisher(Marker, "/active_target_marker", 10)

        if self.target_source == "perception":
            # Cycle boundaries and the part state come from the camera
            # pipeline; the ground-truth stream topic is deliberately NOT
            # subscribed in this mode.
            self.percep_state = None       # (stamp_s, x, y)
            self.percep_vel = (0.0, 0.0)
            self.create_subscription(
                String, "/perception/cycle_start", self.cycle_start_callback, 10
            )
            self.create_subscription(
                PoseStamped, "/perception/part_state", self.percep_state_callback, 10
            )
            self.create_subscription(
                Vector3Stamped,
                "/perception/part_state_velocity",
                self.percep_vel_callback,
                10,
            )
        elif self.sync_with_stream:
            self.create_subscription(String, "/part_stream/cycle_start", self.cycle_start_callback, 10)

        # Missions: named paths in the part frame received on
        # /drawn_paths and a segment sequence received as JSON on
        # /mission, both latched so a tracker restart keeps the active
        # mission.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.drawn_paths = {}          # name -> DrawnPath
        self.mission_spec = None       # parsed /mission JSON
        self.active_mission = None     # armed per cycle if shape matches
        self.mission_seg_idx = 0
        self.mission_seg_t0 = 0.0
        self.create_subscription(
            String, "/drawn_paths", self.drawn_paths_callback, latched
        )
        self.create_subscription(
            String, "/mission", self.mission_callback, latched
        )
        self.mission_state_pub = self.create_publisher(
            String, "/mission_state", 10
        )

        # Supervisor gate. Latched QoS so a late-starting tracker
        # receives the supervisor's current state; True without a supervisor.
        self.tracking_enabled = True
        self.create_subscription(
            Bool,
            "/tracking_enabled",
            self.tracking_enabled_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Set on the first control tick (with use_sim_time the sim clock is
        # not received until the node spins).
        self.part_start_time = None
        self.last_control_time = None
        self.stream_synced = False

        self.current_part_index = 0
        self.pending_cycle = False     # entry-region 'pending' cycle active
        self.path_started = False
        self.path_start_time = None

        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]

        # Orientation-hold reference (x, y, z, w). Captured once per RUN at
        # the first path start unless a valid reference_orientation
        # parameter was given; never reset by cycle changes.
        self.hold_ref_quat = None
        if math.sqrt(sum(v * v for v in self.reference_orientation)) > 0.5:
            n = math.sqrt(sum(v * v for v in self.reference_orientation))
            self.hold_ref_quat = tuple(v / n for v in self.reference_orientation)
            self.get_logger().info(
                f"Orientation hold reference from parameter: {self.hold_ref_quat}"
            )

        timer_period = 1.0 / self.control_rate
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            "Moving shape perimeter tracker started. "
            f"Shapes: {list(self.paths.keys())}, sequence: {self.shape_sequence}, "
            f"sync_with_stream: {self.sync_with_stream}"
        )

    # ------------------------------------------------------------------
    # Parameters and shape catalog
    # ------------------------------------------------------------------

    def load_parameters(self):
        gp = lambda name: self.get_parameter(name).value
        self.planning_frame = str(gp("planning_frame"))
        self.ee_frame = str(gp("ee_frame"))
        self.servo_topic = str(gp("servo_topic"))
        self.target_source = str(gp("target_source"))
        if self.target_source not in ("model", "perception"):
            raise RuntimeError(f"Unknown target_source '{self.target_source}'")
        self.perception_state_timeout = float(gp("perception_state_timeout"))

        self.shapes_file = str(gp("shapes_file"))
        self.shape_sequence = [str(s) for s in gp("shape_sequence")]
        self.circle_discretization = int(gp("circle_discretization"))

        self.part_start_x = float(gp("part_start_x"))
        self.part_start_y = float(gp("part_start_y"))
        self.part_z = float(gp("part_z"))
        self.target_vx = float(gp("target_vx"))
        self.target_vy = float(gp("target_vy"))
        self.part_yaw = float(gp("part_yaw"))
        self.part_angular_speed = float(gp("part_angular_speed"))
        self.target_z_offset = float(gp("target_z_offset"))
        self.follow_duration = float(gp("follow_duration"))

        self.perimeter_path_speed = float(gp("perimeter_path_speed"))
        self.start_path_when_near = bool(gp("start_path_when_near"))
        self.path_start_tolerance = float(gp("path_start_tolerance"))
        self.sync_with_stream = bool(gp("sync_with_stream"))
        self.path_mode = str(gp("path_mode"))
        self.pending_path_mode = self.path_mode
        self.corner_dwell_time = float(gp("corner_dwell_time"))
        self.edge_index = int(gp("edge_index"))
        self.corner_index = int(gp("corner_index"))

        self.control_rate = float(gp("control_rate"))
        self.kp_linear = float(gp("kp_linear"))
        self.ki_linear = float(gp("ki_linear"))
        self.kd_linear = float(gp("kd_linear"))
        self.integral_limit = float(gp("integral_limit"))
        self.max_linear_speed = float(gp("max_linear_speed"))
        self.use_feedforward = bool(gp("use_feedforward"))
        self.tool_z_offset = float(gp("tool_z_offset"))
        self.servo_linear_scale = float(gp("servo_linear_scale"))
        self.target_lead_time = float(gp("target_lead_time"))
        self.marker_decimation = max(1, int(gp("marker_decimation")))
        self.enable_orientation_hold = bool(gp("enable_orientation_hold"))
        self.kp_angular = float(gp("kp_angular"))
        self.servo_angular_scale = float(gp("servo_angular_scale"))
        self.max_angular_speed = float(gp("max_angular_speed"))
        self.reference_orientation = [float(v) for v in gp("reference_orientation")]

    def on_parameters_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name not in self.RUNTIME_TUNABLE:
                continue
            if p.name == "path_mode":
                if p.value not in ("perimeter", "corners", "edge_pingpong",
                                   "single_lap", "center", "corner"):
                    return SetParametersResult(
                        successful=False,
                        reason=f"unknown path_mode '{p.value}'",
                    )
                self.pending_path_mode = str(p.value)
                self.get_logger().info(
                    f"path_mode '{p.value}' latched for the next part"
                )
            elif p.name in ("edge_index", "corner_index"):
                setattr(self, p.name, int(p.value))
            elif p.name in ("use_feedforward", "enable_orientation_hold"):
                setattr(self, p.name, bool(p.value))
            else:
                setattr(self, p.name, float(p.value))
            if p.name != "path_mode":
                self.get_logger().info(f"Parameter {p.name} = {p.value}")
        return SetParametersResult(successful=True)

    def load_shapes(self):
        path = self.shapes_file
        if not path:
            share = get_package_share_directory("robotic_tracking_system")
            path = os.path.join(share, "config", "shapes.yaml")

        with open(path, "r") as f:
            catalog = yaml.safe_load(f)

        self.part_height = float(catalog.get("height", 0.08))
        self.paths = {}
        for name, spec in catalog["shapes"].items():
            if spec["type"] == "polygon":
                vertices = [(float(x), float(y)) for x, y in spec["vertices"]]
            elif spec["type"] == "circle":
                r = float(spec["radius"])
                n = self.circle_discretization
                vertices = [
                    (r * math.cos(2.0 * math.pi * i / n), r * math.sin(2.0 * math.pi * i / n))
                    for i in range(n)
                ]
            else:
                self.get_logger().error(f"Unknown shape type for '{name}': {spec['type']}")
                continue
            self.paths[name] = ShapePath(name, vertices)

        missing = [s for s in self.shape_sequence if s not in self.paths]
        if missing:
            raise RuntimeError(f"shape_sequence contains unknown shapes: {missing}")

        # Entry-region placeholder (perception mode): while the label is
        # not yet vote-trusted the estimator announces the cycle as
        # 'pending'. The EE must NOT hover over the part (it would sit
        # in the camera's view cone and block the very detections the
        # label vote needs: detections would stop at y -0.21, the trust
        # band starts at -0.18, so the vote would never fill). Instead it
        # rides ALONGSIDE the part at a stand-off toward the robot,
        # outside the silhouette cone (part extent <= 0.10 m projects
        # to +-0.08 m at EE height), until the resolved cycle
        # re-announcement restarts the path with the real perimeter.
        self.paths["pending"] = ShapePath("pending", [(-0.16, 0.0)])

    # ------------------------------------------------------------------
    # Cycle handling
    # ------------------------------------------------------------------

    def current_shape(self) -> ShapePath:
        if self.pending_cycle:
            return self.paths["pending"]
        name = self.shape_sequence[self.current_part_index % len(self.shape_sequence)]
        return self.paths[name]

    def start_new_cycle(self, now, shape_name=None):
        self.pending_cycle = shape_name == "pending"
        if shape_name == "pending":
            # Shape not yet trusted: follow the part origin and do not
            # advance the sequence - the resolved re-announcement will
            # align it.
            pass
        elif shape_name is not None and shape_name in self.paths:
            # Align the sequence position to the named shape.
            seq = self.shape_sequence
            if shape_name in seq:
                base = self.current_part_index % len(seq)
                offset = (seq.index(shape_name) - base) % len(seq)
                self.current_part_index += offset if offset else len(seq)
            else:
                self.get_logger().warn(f"Shape '{shape_name}' not in shape_sequence; keeping sequence order")
                self.current_part_index += 1
        else:
            self.current_part_index += 1

        self.part_start_time = now
        self.path_started = False
        self.path_start_time = None
        self.reset_pid()
        if self.pending_path_mode != self.path_mode:
            self.get_logger().info(
                f"Path mode: {self.path_mode} -> {self.pending_path_mode}"
            )
            self.path_mode = self.pending_path_mode

        shape = self.current_shape()
        msg = String()
        msg.data = shape.name
        self.active_shape_pub.publish(msg)
        self.get_logger().info(f"Part #{self.current_part_index}: shape '{shape.name}' ({len(shape.vertices)} path points)")

        # Arm the mission for this part if the shape matches; otherwise
        # this part runs the normal path_mode. A 'pending' cycle never
        # arms a mission (the EE would move over the part and occlude
        # the classification); the resolved cycle arms it.
        self.active_mission = None
        if self.mission_spec is not None and not self.pending_cycle:
            want = self.mission_spec.get("shape", "any")
            if want in ("any", shape.name):
                self.active_mission = self.mission_spec
                self.mission_seg_idx = 0
                self.mission_seg_t0 = 0.0
                segs = self.active_mission["segments"]
                self.publish_mission_state(
                    f"segment 1/{len(segs)}: {self.describe_segment(segs[0])}")

    def cycle_start_callback(self, msg: String):
        self.stream_synced = True
        self.start_new_cycle(self.get_clock().now(), shape_name=msg.data)

    def reset_pid(self):
        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]

    def drawn_paths_callback(self, msg: String):
        """/drawn_paths JSON: {"paths": [{"name", "closed", "points"}]}."""
        try:
            spec = json.loads(msg.data) if msg.data.strip() else {}
        except ValueError:
            self.get_logger().error("Invalid /drawn_paths JSON ignored")
            return
        paths = {}
        for p in spec.get("paths", []) if isinstance(spec, dict) else []:
            pts = [(float(x), float(y)) for x, y in p.get("points", [])]
            if len(pts) >= 2 and p.get("name"):
                paths[str(p["name"])] = DrawnPath(
                    str(p["name"]), pts, bool(p.get("closed", False)))
        self.drawn_paths = paths
        if paths:
            self.get_logger().info(
                "Drawn paths received: " + ", ".join(
                    f"{d.name} ({'closed' if d.closed else 'open'}, "
                    f"{d.length * 1000:.0f} mm)" for d in paths.values())
            )
        else:
            self.get_logger().info("Drawn paths cleared")

    def mission_callback(self, msg: String):
        try:
            spec = json.loads(msg.data) if msg.data.strip() else {}
        except ValueError:
            self.get_logger().error("Invalid /mission JSON ignored")
            return
        if spec.get("segments"):
            self.mission_spec = spec
            self.get_logger().info(
                f"Mission set for shape '{spec.get('shape', 'any')}': "
                + " -> ".join(s.get("mode", "?") for s in spec["segments"])
            )
        else:
            self.mission_spec = None
            self.active_mission = None
            self.get_logger().info("Mission cleared")

    @staticmethod
    def describe_segment(seg):
        mode = seg.get("mode", "?")
        if mode == "drawn":
            text = f"drawn {seg.get('name', '')}".rstrip()
            if seg.get("reverse"):
                text += " reverse"
            passes = int(seg.get("passes", 1))
            if passes > 1:
                text += f" x{passes}"
            return text
        if mode in ("corner", "edge"):
            return f"{mode} {seg.get('index', 0)}"
        if mode == "perimeter":
            return f"perimeter x{seg.get('laps', 1)}"
        return mode

    def publish_mission_state(self, text):
        msg = String()
        msg.data = text
        self.mission_state_pub.publish(msg)
        self.get_logger().info(f"Mission: {text}")

    def percep_state_callback(self, msg: PoseStamped):
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.percep_state = (stamp_s, msg.pose.position.x, msg.pose.position.y)

    def percep_vel_callback(self, msg: Vector3Stamped):
        self.percep_vel = (msg.vector.x, msg.vector.y)

    def tracking_enabled_callback(self, msg: Bool):
        if msg.data == self.tracking_enabled:
            return
        self.tracking_enabled = msg.data
        if not msg.data:
            # The supervisor is taking the arm home: forget the current
            # approach so the next enabled cycle re-approaches cleanly.
            # The orientation-hold reference deliberately survives
            # (re-capturing would re-baseline accumulated drift).
            self.reset_pid()
            self.path_started = False
            self.path_start_time = None
            self.last_control_time = None
            self.get_logger().info("Tracking disabled by supervisor")
        else:
            self.get_logger().info("Tracking enabled by supervisor")

    def compute_local_target(self, shape, path_elapsed):
        """Target point and unit direction in the part frame per path_mode."""
        if shape.name == "pending":
            # The stand-off wait pose applies in EVERY path mode:
            # 'center' targets the part origin directly, and
            # 'single_lap' treats the zero-perimeter pending "lap" as
            # done immediately and holds the center - either parks the
            # EE inside the camera's view cone and blocks the very
            # detections the label vote needs, so the shape would never
            # resolve (an occlusion self-deadlock).
            return shape.vertices[0], (0.0, 0.0)
        if self.path_mode == "corners":
            # Jump corner to corner, dwelling at each; the target is
            # static in the part frame, so feed-forward is belt motion
            # only.
            n = len(shape.vertices)
            idx = int(path_elapsed / max(0.1, self.corner_dwell_time)) % n
            return shape.vertices[idx], (0.0, 0.0)
        if self.path_mode == "center":
            # Follow the part origin itself.
            return (0.0, 0.0), (0.0, 0.0)
        if self.path_mode == "corner":
            # Ride on a single selected corner; out-of-range clamps to
            # the shape's last corner (dashboard convention, same as
            # edge_pingpong). Corner i is where edge i starts.
            n = len(shape.vertices)
            c = max(0, min(self.corner_index, n - 1))
            return shape.vertices[c], (0.0, 0.0)
        if self.path_mode == "edge_pingpong":
            # Triangle wave along one edge, direction flipped on
            # the return leg. An out-of-range edge_index clamps to the
            # shape's LAST edge (a triangle asked for edge 4 follows
            # edge 3), matching the operator-dashboard convention.
            n = len(shape.vertices)
            e = max(0, min(self.edge_index, n - 1))
            ax, ay = shape.vertices[e]
            bx, by = shape.vertices[(e + 1) % n]
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-9:
                return (ax, ay), (0.0, 0.0)
            u = (self.perimeter_path_speed * path_elapsed / length) % 2.0
            sign = 1.0
            if u > 1.0:
                u = 2.0 - u
                sign = -1.0
            ux, uy = (bx - ax) / length, (by - ay) / length
            return (ax + u * (bx - ax), ay + u * (by - ay)), (sign * ux, sign * uy)
        s = self.perimeter_path_speed * path_elapsed
        if self.path_mode == "single_lap" and s >= shape.perimeter:
            # Lap done: hold the part center until the part leaves.
            return (0.0, 0.0), (0.0, 0.0)
        return shape.point_and_direction(s)

    def compute_mission_target(self, shape, path_elapsed):
        """Sequential mission segments; returns the part-frame target and
        unit direction, advancing segments on completion."""
        segs = self.active_mission["segments"]
        while self.mission_seg_idx < len(segs):
            seg = segs[self.mission_seg_idx]
            t_local = path_elapsed - self.mission_seg_t0
            mode = seg.get("mode", "")
            n = len(shape.vertices)
            done = False
            if mode == "perimeter":
                laps = max(0.1, float(seg.get("laps", 1)))
                s = self.perimeter_path_speed * t_local
                if s < laps * shape.perimeter:
                    return shape.point_and_direction(s)
                done = True
            elif mode == "drawn":
                name = str(seg.get("name", ""))
                path = self.drawn_paths.get(name)
                if path is None and len(self.drawn_paths) == 1 and not name:
                    path = next(iter(self.drawn_paths.values()))
                if path is None:
                    self.get_logger().warn(
                        f"Mission segment 'drawn' refers to unknown path "
                        f"'{name}' (have: {sorted(self.drawn_paths)}); skipped",
                        throttle_duration_sec=5.0,
                    )
                    done = True
                else:
                    reverse = bool(seg.get("reverse", False))
                    passes = max(1, int(seg.get("passes", 1)))
                    dwell = max(0.0, float(seg.get("dwell", 0.0)))
                    s = self.perimeter_path_speed * t_local
                    travel = passes * path.length
                    if s < travel:
                        return path.point_and_direction(s, reverse)
                    if t_local < travel / self.perimeter_path_speed + dwell:
                        return path.end_point(passes, reverse), (0.0, 0.0)
                    done = True
            elif mode == "corner":
                dwell = float(seg.get("dwell", 3.0))
                if t_local < dwell:
                    c = max(0, min(int(seg.get("index", 0)), n - 1))
                    return shape.vertices[c], (0.0, 0.0)
                done = True
            elif mode == "edge":
                dwell = float(seg.get("dwell", 6.0))
                if t_local < dwell:
                    e = max(0, min(int(seg.get("index", 0)), n - 1))
                    ax, ay = shape.vertices[e]
                    bx, by = shape.vertices[(e + 1) % n]
                    length = math.hypot(bx - ax, by - ay)
                    if length < 1e-9:
                        return (ax, ay), (0.0, 0.0)
                    u = (self.perimeter_path_speed * t_local / length) % 2.0
                    sign = 1.0
                    if u > 1.0:
                        u, sign = 2.0 - u, -1.0
                    return ((ax + u * (bx - ax), ay + u * (by - ay)),
                            (sign * (bx - ax) / length,
                             sign * (by - ay) / length))
                done = True
            elif mode == "center":
                dwell = float(seg.get("dwell", 3.0))
                if t_local < dwell:
                    return (0.0, 0.0), (0.0, 0.0)
                done = True
            else:
                self.get_logger().warn(
                    f"Unknown mission segment mode '{mode}'; skipped")
                done = True
            if done:
                self.mission_seg_idx += 1
                self.mission_seg_t0 = path_elapsed
                if self.mission_seg_idx < len(segs):
                    self.publish_mission_state(
                        f"segment {self.mission_seg_idx + 1}/{len(segs)}: "
                        + self.describe_segment(segs[self.mission_seg_idx]))
                else:
                    self.publish_mission_state(
                        "mission complete - holding part center")
        return (0.0, 0.0), (0.0, 0.0)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def control_loop(self):
        now = self.get_clock().now()
        if self.part_start_time is None:
            self.part_start_time = now
            msg = String()
            msg.data = self.current_shape().name
            self.active_shape_pub.publish(msg)

        # Supervisor gate: publish nothing while disabled. Servo halts by
        # command timeout; /active_target_point going stale marks the skip
        # window for the offline analyzer. Cycle bookkeeping continues via
        # cycle_start_callback so the shape sequence stays aligned.
        if not self.tracking_enabled:
            return

        elapsed_on_part = (now - self.part_start_time).nanoseconds * 1e-9

        # When a Gazebo stream drives the cycles we never roll over on our
        # own timer - the stream's cycle_start messages do it. In
        # perception mode cycles come only from the estimator.
        if (self.target_source == "model" and not self.stream_synced
                and elapsed_on_part >= self.follow_duration):
            self.start_new_cycle(now)
            elapsed_on_part = 0.0

        shape = self.current_shape()

        # Part pose and velocity: mathematical model, or the perception
        # estimator's state extrapolated to now with its velocity.
        if self.target_source == "perception":
            if self.percep_state is None:
                self.publish_zero_command(now)
                return
            stamp_s, sx, sy = self.percep_state
            state_age = now.nanoseconds * 1e-9 - stamp_s
            if state_age > self.perception_state_timeout:
                self.get_logger().warn(
                    f"Perception part state stale ({state_age:.2f} s); halting",
                    throttle_duration_sec=2.0,
                )
                self.publish_zero_command(now)
                return
            part_vx, part_vy = self.percep_vel
            part_wz = 0.0
            cx = sx + part_vx * state_age
            cy = sy + part_vy * state_age
            yaw = 0.0
        else:
            part_vx = self.target_vx
            part_vy = self.target_vy
            part_wz = self.part_angular_speed
            cx = self.part_start_x + self.target_vx * elapsed_on_part
            cy = self.part_start_y + self.target_vy * elapsed_on_part
            yaw = self.part_yaw + self.part_angular_speed * elapsed_on_part
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # Path progress
        if self.path_started:
            path_elapsed = (now - self.path_start_time).nanoseconds * 1e-9
        else:
            path_elapsed = 0.0
        if self.active_mission is not None:
            (lx, ly), (dx_local, dy_local) = self.compute_mission_target(
                shape, path_elapsed
            )
        else:
            (lx, ly), (dx_local, dy_local) = self.compute_local_target(
                shape, path_elapsed
            )

        # Target in planning frame
        tx = cx + cos_y * lx - sin_y * ly
        ty = cy + sin_y * lx + cos_y * ly
        tz = self.part_z + self.target_z_offset

        self.publish_target_point(tx, ty, tz, now)
        self._marker_tick = getattr(self, "_marker_tick", 0) + 1
        if self._marker_tick % self.marker_decimation == 0:
            self.publish_markers(shape, cx, cy, cos_y, sin_y, tx, ty, tz, now)

        ee_pose = self.lookup_end_effector_position()
        if ee_pose is None:
            self.publish_zero_command(now)
            return
        ee, ee_rot = ee_pose
        ee_quat = (ee_rot.x, ee_rot.y, ee_rot.z, ee_rot.w)

        # Target velocity (part motion + rotation + path traversal), m/s.
        # Used both as feed-forward and to lead-predict the target.
        tv_x = tv_y = 0.0
        r_x = tx - cx
        r_y = ty - cy
        tv_x = part_vx - part_wz * r_y
        tv_y = part_vy + part_wz * r_x
        if self.path_started:
            tv_x += self.perimeter_path_speed * (cos_y * dx_local - sin_y * dy_local)
            tv_y += self.perimeter_path_speed * (sin_y * dx_local + cos_y * dy_local)

        lead = self.target_lead_time
        error_x = (tx + tv_x * lead) - ee.x
        error_y = (ty + tv_y * lead) - ee.y
        error_z = tz - (ee.z + self.tool_z_offset)
        error_norm = math.sqrt(error_x**2 + error_y**2 + error_z**2)

        if (self.start_path_when_near and not self.path_started
                and error_norm <= self.path_start_tolerance):
            self.path_started = True
            self.path_start_time = now
            self.reset_pid()
        elif not self.start_path_when_near and not self.path_started:
            self.path_started = True
            self.path_start_time = self.part_start_time

        # Capture the orientation-hold reference once per run, at the first
        # path start (the approach transient is over, the EE is in its
        # working orientation). Deliberately NOT per cycle: re-capturing
        # would re-baseline the accumulated drift each cycle.
        if (self.enable_orientation_hold and self.hold_ref_quat is None
                and self.path_started):
            self.hold_ref_quat = ee_quat
            self.get_logger().info(
                "Orientation hold reference captured: "
                f"({ee_quat[0]:.4f}, {ee_quat[1]:.4f}, "
                f"{ee_quat[2]:.4f}, {ee_quat[3]:.4f})"
            )

        # Feed-forward = the target velocity computed above
        ff_x = tv_x if self.use_feedforward else 0.0
        ff_y = tv_y if self.use_feedforward else 0.0

        # PID feedback
        if self.last_control_time is None:
            dt = 1.0 / self.control_rate
        else:
            dt = (now - self.last_control_time).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = 1.0 / self.control_rate
        self.last_control_time = now

        cmd_v = [0.0, 0.0, 0.0]
        errors = [error_x, error_y, error_z]
        feedforward = [ff_x, ff_y, 0.0]
        for i in range(3):
            self.integral[i] = self.clamp(
                self.integral[i] + errors[i] * dt, -self.integral_limit, self.integral_limit
            )
            derivative = (errors[i] - self.prev_error[i]) / dt
            self.prev_error[i] = errors[i]
            feedback = (
                self.kp_linear * errors[i]
                + self.ki_linear * self.integral[i]
                + self.kd_linear * derivative
            )
            # feedforward + feedback are in m/s; Servo expects unitless
            # commands in [-1, 1] that it multiplies by scale.linear.
            velocity_mps = feedforward[i] + feedback
            cmd_v[i] = self.clamp(
                velocity_mps / self.servo_linear_scale,
                -self.max_linear_speed,
                self.max_linear_speed,
            )

        cmd = TwistStamped()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = self.planning_frame
        cmd.twist.linear.x = cmd_v[0]
        cmd.twist.linear.y = cmd_v[1]
        cmd.twist.linear.z = cmd_v[2]
        if self.enable_orientation_hold and self.hold_ref_quat is not None:
            # Rotation carrying the current EE orientation back to the
            # reference, expressed in the planning frame (both quaternions
            # are planning<-ee, and Servo's robot_link_command_frame is the
            # planning frame). Proportional angular velocity, converted to
            # Servo's unitless range like the linear channel.
            rot_err = quat_to_axis_angle(
                quat_mul(self.hold_ref_quat, quat_conj(ee_quat))
            )
            cmd.twist.angular.x = self.clamp(
                self.kp_angular * rot_err[0] / self.servo_angular_scale,
                -self.max_angular_speed, self.max_angular_speed,
            )
            cmd.twist.angular.y = self.clamp(
                self.kp_angular * rot_err[1] / self.servo_angular_scale,
                -self.max_angular_speed, self.max_angular_speed,
            )
            cmd.twist.angular.z = self.clamp(
                self.kp_angular * rot_err[2] / self.servo_angular_scale,
                -self.max_angular_speed, self.max_angular_speed,
            )
        self.twist_pub.publish(cmd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def lookup_end_effector_position(self):
        """EE translation and rotation in the planning frame, or None."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.001),
            )
            return transform.transform.translation, transform.transform.rotation
        except TransformException as ex:
            self.get_logger().warn(
                f"Could not get TF {self.planning_frame} -> {self.ee_frame}: {ex}",
                throttle_duration_sec=2.0,
            )
            return None

    def publish_zero_command(self, now):
        cmd = TwistStamped()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = self.planning_frame
        self.twist_pub.publish(cmd)

    def publish_target_point(self, x, y, z, now):
        msg = PointStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.planning_frame
        msg.point.x = x
        msg.point.y = y
        msg.point.z = z
        self.target_point_pub.publish(msg)

    def publish_markers(self, shape, cx, cy, cos_y, sin_y, tx, ty, tz, now):
        outline = Marker()
        outline.header.stamp = now.to_msg()
        outline.header.frame_id = self.planning_frame
        outline.ns = "shape_outline"
        outline.id = 0
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.004
        outline.color.r = 0.1
        outline.color.g = 1.0
        outline.color.b = 0.3
        outline.color.a = 1.0
        outline.pose.orientation.w = 1.0
        z = self.part_z + self.target_z_offset
        for vx, vy in shape.vertices + [shape.vertices[0]]:
            p = Point()
            p.x = cx + cos_y * vx - sin_y * vy
            p.y = cy + sin_y * vx + cos_y * vy
            p.z = z
            outline.points.append(p)
        self.outline_marker_pub.publish(outline)

        corners = Marker()
        corners.header = outline.header
        corners.ns = "shape_corners"
        corners.id = 0
        corners.type = Marker.SPHERE_LIST
        corners.action = Marker.ADD
        corners.scale.x = corners.scale.y = corners.scale.z = 0.015
        corners.color.r = 1.0
        corners.color.g = 0.6
        corners.color.b = 0.0
        corners.color.a = 1.0
        corners.pose.orientation.w = 1.0
        corners.points = outline.points[:-1]
        self.corners_marker_pub.publish(corners)

        target = Marker()
        target.header = outline.header
        target.ns = "active_target"
        target.id = 0
        target.type = Marker.SPHERE
        target.action = Marker.ADD
        target.pose.position.x = tx
        target.pose.position.y = ty
        target.pose.position.z = tz
        target.pose.orientation.w = 1.0
        target.scale.x = target.scale.y = target.scale.z = 0.03
        target.color.r = 1.0
        target.color.g = 0.2
        target.color.b = 0.2
        target.color.a = 1.0
        self.target_marker_pub.publish(target)

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(value, high))


def main(args=None):
    rclpy.init(args=args)
    node = MovingShapePerimeterTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
