"""Operator dashboard GUI for the perception-based tracking system.

Node name:      operator_dashboard (GUI process; rclpy node spun in a
                background thread, PyQt5 in the main thread)
Purpose:        Single-window operator HMI: camera view with the CV
                overlay, a to-scale 2D top-down workcell canvas with
                toggleable layers (EE path, target path, perceived vs
                ground-truth part, error line, zones), a color-coded
                robot status feed with event log, live tracking-precision
                and joint-angle graphs, and a control panel (GO HOME /
                START / STOP, live-tunable parameters with
                experimentally-validated "optimal" presets, the
                tracking-behavior selector and a manual joint mode).
                Pure PyQt5 + a small custom plot widget - no dependencies
                beyond what ROS ships.
Subscriptions:  /perception/debug_image, /camera/conveyor_camera/image_raw
                (sensor_msgs/Image), /active_target_point (PointStamped),
                /perception/part_state (PoseStamped),
                /perception/part_corners (PolygonStamped),
                /perception/cycle_start (String),
                /part_stream/ground_truth_pose (PoseStamped),
                /part_stream/cycle_start (String),
                /joint_states (JointState),
                /system_status (String JSON, latched)
Publications:   /panda_arm_controller/joint_trajectory
                (trajectory_msgs/JointTrajectory) - manual-mode joint
                target, sent only while Servo is paused
Services used:  /tracking_supervisor/go_home, ~/pause_tracking,
                ~/resume_tracking (std_srvs/Trigger);
                /servo_node/pause_servo, /servo_node/start_servo
                (std_srvs/Trigger) for manual mode;
                <node>/set_parameters (rcl_interfaces/SetParameters) for
                live tuning of the tracker / estimator / supervisor and
                for the tracking-behavior selector (path_mode,
                edge_index, corner_index of the tracker)
Parameters:     use_sim_time (run with true). Environment:
                DASHBOARD_SCREENSHOT="<png path>,<seconds>" grabs the
                main window after the delay and quits.
Frames:         panda_link0 (TF panda_link0 -> panda_link8 for the EE)
Update rate:    ROS collection ~20 Hz, GUI refresh 25 Hz, sim time
Dependencies:   PyQt5, cv_bridge, numpy; run with use_sim_time:=true.
                The GUI is OPTIONAL and read-mostly: the system runs
                headless without it, and a GUI crash cannot affect
                tracking (commands go through supervisor services and
                the official parameter interface only).
"""

import json
import math
import os
import queue
import sys
import threading
from collections import deque

import numpy as np
import yaml

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros
from tf2_ros import TransformException

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PolygonStamped, PoseStamped
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from builtin_interfaces.msg import Duration as DurationMsg
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from PyQt5.QtCore import QPointF, QSettings, Qt, QTimer
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF,
)
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPlainTextEdit, QPushButton, QRadioButton,
    QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
TOOL_Z_OFFSET = -0.120  # recorder/analyzer convention
# The precision graph and per-part stats exclude the approach flight
# (home -> part) by CONVERGENCE, not by a fixed time: recording starts
# the first moment the error drops below this threshold (the tracker's
# own path_start_tolerance, 0.02 m) and from then on captures every
# sample of the part - no data is lost after settling, and the
# exclusion adapts to fast or slow approaches.
SETTLE_THRESHOLD_MM = 20.0

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Live-tunable parameters. "optimal" values are the ones used in the
# experiments; the tooltips say what each parameter does.
PARAM_SPECS = [
    dict(name="kp_linear", node="moving_shape_perimeter_tracker",
         lo=0.0, hi=30.0, step=0.5, dec=1, optimal=8.0,
         tip="Proportional gain [1/s]; a moderate gain with the target "
             "lead tracks better than a high gain that saturates Servo"),
    dict(name="ki_linear", node="moving_shape_perimeter_tracker",
         lo=0.0, hi=2.0, step=0.05, dec=2, optimal=0.0,
         tip="The validated configuration uses no integral term"),
    dict(name="kd_linear", node="moving_shape_perimeter_tracker",
         lo=0.0, hi=2.0, step=0.05, dec=2, optimal=0.0,
         tip="The validated configuration uses no derivative term"),
    dict(name="target_lead_time", node="moving_shape_perimeter_tracker",
         lo=0.0, hi=0.5, step=0.01, dec=2, optimal=0.24,
         tip="Prediction lead [s] that compensates the delay of the "
             "command chain; the error to the true part is lowest at "
             "0.24 s"),
    dict(name="max_linear_speed", node="moving_shape_perimeter_tracker",
         lo=0.1, hi=1.0, step=0.05, dec=3, optimal=0.999,
         tip="Unitless Servo command clamp (x0.4 m/s)"),
    dict(name="kp_angular", node="moving_shape_perimeter_tracker",
         lo=0.0, hi=10.0, step=0.1, dec=1, optimal=2.0,
         tip="Gain of the tool-orientation hold [1/s]"),
    dict(name="perimeter_path_speed", node="moving_shape_perimeter_tracker",
         lo=0.0, hi=0.15, step=0.005, dec=3, optimal=0.06,
         tip="Path traversal speed along the shape [m/s] "
             "(baseline value)"),
    dict(name="velocity_prior_blend", node="perception_target_estimator",
         lo=0.0, hi=1.0, step=0.05, dec=2, optimal=0.0,
         tip="0 = belt velocity fitted from the camera (P0); "
             "1 = calibrated belt velocity (P1)"),
    dict(name="rehome_every_n_cycles", node="tracking_supervisor",
         lo=0, hi=10, step=1, dec=0, optimal=1,
         tip="Gap re-home period; 1 is free in idle mode and keeps the "
             "posture always fresh"),
]

PATH_MODES = [
    ("single_lap", "Perimeter Follow (one lap)"),
    ("center", "Part Center Follow"),
    ("edge_pingpong", "Edge Follow"),
    ("corner", "Corner Follow"),
]
# Modes with a 1-based index selector: (row label, tracker parameter).
MODE_SELECTORS = {
    "edge_pingpong": ("Edge #", "edge_index"),
    "corner": ("Corner #", "corner_index"),
}

# Real Panda joint limits, degrees (manual-mode slider bounds).
JOINT_LIMITS_DEG = [
    (-166.0, 166.0), (-101.0, 101.0), (-166.0, 166.0), (-176.0, -4.0),
    (-166.0, 166.0), (-1.0, 215.0), (-166.0, 166.0),
]

STATUS_COLORS = {
    "TRACKING": "#2e7d32",
    "WAITING": "#1565c0",
    "READY": "#1565c0",
    "HOMING": "#ef6c00",
    "PAUSED": "#616161",
    "ERROR": "#c62828",
    "STARTING": "#616161",
}


def load_shape_catalog():
    share = get_package_share_directory("robotic_tracking_system")
    with open(os.path.join(share, "config", "shapes.yaml")) as f:
        catalog = yaml.safe_load(f)
    shapes = {}
    for name, spec in catalog["shapes"].items():
        if spec["type"] == "polygon":
            shapes[name] = [(float(x), float(y)) for x, y in spec["vertices"]]
        elif spec["type"] == "circle":
            r = float(spec["radius"])
            shapes[name] = [
                (r * math.cos(2 * math.pi * i / 36),
                 r * math.sin(2 * math.pi * i / 36))
                for i in range(36)
            ]
    return shapes


class RosBridge(Node):
    """Collects everything the GUI shows; owns all ROS communication."""

    def __init__(self):
        super().__init__("operator_dashboard")
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.shapes = load_shape_catalog()

        # GUI-facing state (guarded by self.lock)
        self.images = {"overlay": None, "raw": None}
        self.ee_trace = deque(maxlen=6000)       # (t, x, y)
        self.target_trace = deque(maxlen=6000)   # (t, x, y)
        self.latest_target = None                # (t, x, y, z)
        self.latest_ee = None                    # (t, x, y, z)
        self.percep_pose = None                  # (t, x, y)
        self.percep_corners = []
        self.percep_shape = ""
        self.gt_pose = None                      # (t, x, y, yaw)
        self.gt_shape = ""
        self.joints = deque(maxlen=4000)         # (t, [7])
        self.err = deque(maxlen=4000)            # (t, mm)
        self.part_err = []
        # True before the first perception cycle (model-mode fallback:
        # plot everything); reset per part, set on convergence.
        self.part_settled = True
        self.status = {"status": "STARTING", "shape": "", "event": None}
        self.events = deque(maxlen=500)
        self.sim_t = 0.0
        self._last_joint_t = -1.0

        self.commands = queue.Queue()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Image, "/perception/debug_image",
            lambda m: self.on_image(m, "overlay"), 2)
        self.create_subscription(
            Image, "/camera/conveyor_camera/image_raw",
            lambda m: self.on_image(m, "raw"), 2)
        self.create_subscription(
            PointStamped, "/active_target_point", self.on_target, 10)
        self.create_subscription(
            PoseStamped, "/perception/part_state", self.on_part_state, 10)
        self.create_subscription(
            PolygonStamped, "/perception/part_corners", self.on_corners, 10)
        self.create_subscription(
            String, "/perception/cycle_start", self.on_percep_cycle, 10)
        self.create_subscription(
            PoseStamped, "/part_stream/ground_truth_pose", self.on_gt, 10)
        self.create_subscription(
            String, "/part_stream/cycle_start", self.on_gt_cycle, 10)
        self.create_subscription(
            JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(
            String, "/system_status", self.on_status, LATCHED_QOS)

        self.srv_clients = {
            "go_home": self.create_client(
                Trigger, "/tracking_supervisor/go_home"),
            "pause": self.create_client(
                Trigger, "/tracking_supervisor/pause_tracking"),
            "resume": self.create_client(
                Trigger, "/tracking_supervisor/resume_tracking"),
            "servo_pause": self.create_client(
                Trigger, "/servo_node/pause_servo"),
            "servo_start": self.create_client(
                Trigger, "/servo_node/start_servo"),
        }
        self.param_clients = {}
        # Manual mode: single 4 s trajectory to the arm controller (the
        # home_arm_node mechanism); only sent while Servo is paused.
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)

        self.create_timer(0.05, self.sample_tf)
        self.create_timer(0.1, self.process_commands)

    # -- subscriptions -------------------------------------------------

    @staticmethod
    def stamp_to_sec(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def on_image(self, msg, key):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self.lock:
            self.images[key] = frame

    def on_target(self, msg):
        t = self.stamp_to_sec(msg.header)
        with self.lock:
            self.latest_target = (t, msg.point.x, msg.point.y, msg.point.z)
            self.target_trace.append((t, msg.point.x, msg.point.y))

    def on_part_state(self, msg):
        with self.lock:
            self.percep_pose = (self.stamp_to_sec(msg.header),
                                msg.pose.position.x, msg.pose.position.y)

    def on_corners(self, msg):
        with self.lock:
            self.percep_corners = [(p.x, p.y) for p in msg.polygon.points]

    def on_percep_cycle(self, msg):
        with self.lock:
            self.percep_shape = msg.data
            self.part_err = []          # per-part precision stats reset
            self.part_settled = False   # new approach begins

    def on_gt(self, msg):
        q = msg.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        with self.lock:
            self.gt_pose = (self.stamp_to_sec(msg.header),
                            msg.pose.position.x, msg.pose.position.y, yaw)

    def on_gt_cycle(self, msg):
        with self.lock:
            self.gt_shape = msg.data

    def on_joints(self, msg):
        t = self.stamp_to_sec(msg.header)
        if t - self._last_joint_t < 0.05:
            return
        positions = dict(zip(msg.name, msg.position))
        if not all(j in positions for j in PANDA_JOINTS):
            return
        self._last_joint_t = t
        with self.lock:
            self.joints.append((t, [positions[j] for j in PANDA_JOINTS]))

    def on_status(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.status = payload
            if payload.get("event"):
                self.events.append((payload.get("t", 0.0), payload["event"]))

    def sample_tf(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        try:
            tr = self.tf_buffer.lookup_transform(
                "panda_link0", "panda_link8", rclpy.time.Time())
        except TransformException:
            return
        p = tr.transform.translation
        with self.lock:
            self.sim_t = t
            self.latest_ee = (t, p.x, p.y, p.z)
            self.ee_trace.append((t, p.x, p.y))
            if self.latest_target is not None and t - self.latest_target[0] < 0.4:
                _, tx, ty, tz = self.latest_target
                ex = tx - p.x
                ey = ty - p.y
                ez = tz - (p.z + TOOL_Z_OFFSET)
                mm = 1000.0 * math.sqrt(ex * ex + ey * ey + ez * ez)
                # Exclude only the approach flight, by convergence: once
                # the error first dips under the tracker's approach
                # tolerance, every subsequent sample of this part is
                # recorded (gaps in the curve = approaches).
                if not self.part_settled and mm <= SETTLE_THRESHOLD_MM:
                    self.part_settled = True
                if self.part_settled:
                    self.err.append((t, mm))
                    self.part_err.append(mm)

    # -- commands (queued from the GUI thread, executed here) ----------

    def process_commands(self):
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            if cmd[0] == "service":
                self.call_service(cmd[1])
            elif cmd[0] == "param":
                self.set_remote_param(*cmd[1:])
            elif cmd[0] == "manual_move":
                self.send_manual(cmd[1])

    def add_event(self, text):
        with self.lock:
            self.events.append((self.sim_t, text))

    def call_service(self, key):
        client = self.srv_clients[key]
        if not client.service_is_ready():
            self.add_event(f"[GUI] {key}: service unavailable")
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.add_event(
                f"[GUI] {key}: "
                + (f.result().message if f.result() else "no response")))

    def send_manual(self, positions_rad):
        traj = JointTrajectory()
        traj.joint_names = PANDA_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions_rad]
        point.velocities = [0.0] * 7
        point.time_from_start = DurationMsg(sec=4)
        traj.points = [point]
        self.traj_pub.publish(traj)
        self.add_event("[GUI] manual move sent ("
                       + ", ".join(f"{math.degrees(v):.0f}"
                                   for v in positions_rad) + " deg)")

    def set_remote_param(self, node_name, pname, value, is_int):
        if node_name not in self.param_clients:
            self.param_clients[node_name] = self.create_client(
                SetParameters, f"/{node_name}/set_parameters")
        client = self.param_clients[node_name]
        if not client.service_is_ready():
            self.add_event(f"[GUI] {node_name}/{pname}: node unavailable")
            return
        pv = ParameterValue()
        if is_int:
            pv.type = ParameterType.PARAMETER_INTEGER
            pv.integer_value = int(value)
        elif pname == "path_mode":
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = str(value)
        else:
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = float(value)
        req = SetParameters.Request()
        req.parameters = [ParameterMsg(name=pname, value=pv)]
        future = client.call_async(req)

        def done(f):
            res = f.result()
            ok = res and res.results and res.results[0].successful
            self.add_event(
                f"[GUI] set {node_name}.{pname} = {value}"
                + ("" if ok else " FAILED"))

        future.add_done_callback(done)

    def snapshot(self):
        with self.lock:
            part = self.part_err[-600:]
            mean = sum(part) / len(part) if part else 0.0
            rms = math.sqrt(sum(v * v for v in part) / len(part)) if part else 0.0
            return {
                "images": dict(self.images),
                "ee_trace": list(self.ee_trace),
                "target_trace": list(self.target_trace),
                "latest_target": self.latest_target,
                "latest_ee": self.latest_ee,
                "percep_pose": self.percep_pose,
                "percep_corners": list(self.percep_corners),
                "percep_shape": self.percep_shape,
                "gt_pose": self.gt_pose,
                "gt_shape": self.gt_shape,
                "joints": list(self.joints),
                "err": list(self.err),
                "err_mean": mean,
                "err_rms": rms,
                "status": dict(self.status),
                "events": list(self.events),
                "sim_t": self.sim_t,
            }


# ----------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------

class WorkcellCanvas(QWidget):
    """To-scale top view: horizontal = world +Y (belt travel), vertical =
    world +X (belt away from the robot). Layers toggled by checkboxes."""

    WORLD_X = (-0.15, 0.90)   # across-belt extent shown
    WORLD_Y = (-0.55, 0.75)   # along-belt extent shown

    def __init__(self, shapes):
        super().__init__()
        self.shapes = shapes
        self.snap = None
        self.layers = {}
        self.trace_window = 30.0
        self.setMinimumSize(420, 320)

    def to_px(self, x, y):
        # Free scaling: the view fills the widget; a fixed aspect ratio
        # would letterbox at full screen.
        w, h = self.width(), self.height()
        py = (y - self.WORLD_Y[0]) / (self.WORLD_Y[1] - self.WORLD_Y[0]) * w
        px = h - (x - self.WORLD_X[0]) / (self.WORLD_X[1] - self.WORLD_X[0]) * h
        return QPointF(py, px)

    def shape_outline(self, name, cx, cy, yaw=0.0):
        verts = self.shapes.get(name)
        if not verts:
            return None
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        poly = QPolygonF()
        for vx, vy in verts:
            poly.append(self.to_px(cx + cos_y * vx - sin_y * vy,
                                   cy + sin_y * vx + cos_y * vy))
        return poly

    def draw_trace(self, qp, trace, color, now, width=2):
        pts = [(t, x, y) for t, x, y in trace if now - t <= self.trace_window]
        if len(pts) < 2:
            return
        n = len(pts)
        for i in range(1, n):
            alpha = int(40 + 215 * i / n)
            c = QColor(color)
            c.setAlpha(alpha)
            qp.setPen(QPen(c, width))
            qp.drawLine(self.to_px(pts[i - 1][1], pts[i - 1][2]),
                        self.to_px(pts[i][1], pts[i][2]))

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor("#1c1c22"))
        s = self.snap
        now = s["sim_t"] if s else 0.0

        # Belt
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor("#33343c"))
        tl = self.to_px(0.675, -0.45)
        br = self.to_px(0.325, 0.75)
        qp.drawRect(int(tl.x()), int(tl.y()),
                    int(br.x() - tl.x()), int(br.y() - tl.y()))

        if self.layers.get("Zones", True):
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(QColor("#00bcd4"), 1, Qt.DashLine))
            tl = self.to_px(0.64, -0.40)
            br = self.to_px(0.36, 0.30)
            qp.drawRect(int(tl.x()), int(tl.y()),
                        int(br.x() - tl.x()), int(br.y() - tl.y()))
            qp.setPen(QPen(QColor("#ff7043"), 1, Qt.DashLine))
            a = self.to_px(0.675, 0.20)
            b = self.to_px(0.325, 0.20)
            qp.drawLine(a, b)

        # Robot base and camera nadir
        qp.setPen(QPen(QColor("#9e9e9e"), 2))
        qp.setBrush(QColor("#4a4a55"))
        base = self.to_px(0.0, 0.0)
        qp.drawEllipse(base, 14, 14)
        qp.setPen(QPen(QColor("#eeeeee"), 1))
        qp.drawText(base + QPointF(-18, 28), "robot")
        cam = self.to_px(0.5, 0.0)
        qp.setPen(QPen(QColor("#80cbc4"), 1))
        qp.drawLine(cam + QPointF(-6, 0), cam + QPointF(6, 0))
        qp.drawLine(cam + QPointF(0, -6), cam + QPointF(0, 6))
        qp.drawText(cam + QPointF(8, -6), "cam")

        if self.layers.get("Home marker", True):
            qp.setPen(QPen(QColor("#66bb6a"), 2))
            hp = self.to_px(0.307, 0.0)
            qp.drawEllipse(hp, 6, 6)
            qp.drawText(hp + QPointF(8, 4), "home")

        if s is None:
            return

        if self.layers.get("Ground-truth part", True) and s["gt_pose"]:
            t, gx, gy, gyaw = s["gt_pose"]
            if now - t < 1.0:
                poly = self.shape_outline(s["gt_shape"], gx, gy, gyaw)
                if poly:
                    qp.setBrush(Qt.NoBrush)
                    qp.setPen(QPen(QColor("#e0e0e0"), 1, Qt.DashLine))
                    qp.drawPolygon(poly)

        if self.layers.get("Perceived part", True) and s["percep_pose"]:
            t, px_, py_ = s["percep_pose"]
            if now - t < 1.0:
                if s["percep_shape"] == "pending":
                    # Entry region: the label is not vote-trusted yet -
                    # show a neutral marker, never a guessed outline.
                    c = self.to_px(px_, py_)
                    qp.setBrush(Qt.NoBrush)
                    qp.setPen(QPen(QColor("#e040fb"), 2, Qt.DashLine))
                    qp.drawEllipse(c, 10, 10)
                    qp.setPen(QColor("#e040fb"))
                    qp.setFont(QFont("Sans", 8))
                    qp.drawText(c + QPointF(14, 4), "detecting…")
                poly = self.shape_outline(s["percep_shape"], px_, py_)
                if poly:
                    qp.setBrush(Qt.NoBrush)
                    qp.setPen(QPen(QColor("#e040fb"), 2))
                    qp.drawPolygon(poly)
                qp.setPen(Qt.NoPen)
                qp.setBrush(QColor("#ff5252"))
                for cx, cy in s["percep_corners"]:
                    qp.drawEllipse(self.to_px(cx, cy), 3, 3)
                # Numbered edge badges (edge i = catalog vertex i -> i+1,
                # the same numbering the tracker's edge_index uses).
                verts = self.shapes.get(s["percep_shape"])
                if verts and len(verts) <= 8:
                    qp.setFont(QFont("Sans", 7, QFont.Bold))
                    n = len(verts)
                    for i in range(n):
                        ax, ay = verts[i]
                        bx, by = verts[(i + 1) % n]
                        mp = self.to_px(px_ + (ax + bx) / 2.0,
                                        py_ + (ay + by) / 2.0)
                        qp.setPen(QPen(QColor("#ffffff"), 1))
                        qp.setBrush(QColor("#37474f"))
                        qp.drawEllipse(mp, 8, 8)
                        qp.drawText(
                            int(mp.x()) - 3 - (2 if i >= 9 else 0),
                            int(mp.y()) + 4, str(i + 1))
                    qp.setBrush(Qt.NoBrush)

        if self.layers.get("Target path", True):
            self.draw_trace(qp, s["target_trace"], "#ffb300", now)
        if self.layers.get("EE path", True):
            self.draw_trace(qp, s["ee_trace"], "#4fc3f7", now)

        if (self.layers.get("Error line", True) and s["latest_ee"]
                and s["latest_target"] and now - s["latest_target"][0] < 0.4):
            _, ex, ey, _ = s["latest_ee"]
            _, tx, ty, _ = s["latest_target"]
            qp.setPen(QPen(QColor("#ff1744"), 2))
            qp.drawLine(self.to_px(ex, ey), self.to_px(tx, ty))

        if s["latest_ee"]:
            _, ex, ey, _ = s["latest_ee"]
            qp.setPen(QPen(QColor("#4fc3f7"), 2))
            qp.setBrush(QColor("#0d47a1"))
            qp.drawEllipse(self.to_px(ex, ey), 5, 5)


class ScrollingPlot(QWidget):
    """Minimal scrolling multi-series line plot (no external deps)."""

    def __init__(self, title, y_label):
        super().__init__()
        self.title = title
        self.y_label = y_label
        self.series = []          # list of (name, color, [(t, v), ...])
        self.window_s = 60.0
        self.paused = False
        self.frozen_t = None
        self.setMinimumHeight(170)

    def set_series(self, series, now):
        if not self.paused:
            self.series = series
            self.now = now
        self.update()

    def toggle_pause(self):
        self.paused = not self.paused

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor("#17171c"))
        margin_l, margin_r, margin_t, margin_b = 52, 10, 22, 18
        w = self.width() - margin_l - margin_r
        h = self.height() - margin_t - margin_b
        if w <= 10 or h <= 10 or not getattr(self, "now", None):
            qp.setPen(QColor("#888"))
            qp.drawText(10, 20, f"{self.title} (waiting for data)")
            return
        t1 = self.now
        t0 = t1 - self.window_s

        vals = [v for _, _, pts in self.series
                for t, v in pts if t0 <= t <= t1]
        if not vals:
            qp.setPen(QColor("#888"))
            qp.drawText(10, 20, f"{self.title} (no samples in window)")
            return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def to_px(t, v):
            return QPointF(
                margin_l + (t - t0) / (t1 - t0) * w,
                margin_t + (1.0 - (v - lo) / (hi - lo)) * h)

        qp.setPen(QPen(QColor("#33333c"), 1))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = margin_t + frac * h
            qp.drawLine(int(margin_l), int(y), int(margin_l + w), int(y))
            qp.setPen(QColor("#9e9e9e"))
            qp.drawText(2, int(y) + 4, f"{hi - frac * (hi - lo):7.2f}")
            qp.setPen(QPen(QColor("#33333c"), 1))

        for name, color, pts in self.series:
            visible = [(t, v) for t, v in pts if t0 <= t <= t1]
            if len(visible) < 2:
                continue
            qp.setPen(QPen(QColor(color), 2))
            last = to_px(*visible[0])
            for i in range(1, len(visible)):
                t, v = visible[i]
                p = to_px(t, v)
                # Break the line across gaps (e.g. no active target)
                if t - visible[i - 1][0] < 2.0:
                    qp.drawLine(last, p)
                last = p

        qp.setPen(QColor("#e0e0e0"))
        qp.setFont(QFont("Sans", 9, QFont.Bold))
        qp.drawText(margin_l, 15, self.title)
        x = margin_l + 160
        qp.setFont(QFont("Sans", 8))
        for name, color, _ in self.series:
            qp.setPen(QColor(color))
            qp.drawText(x, 15, name)
            x += 10 + 7 * len(name)
        if self.paused:
            qp.setPen(QColor("#ffb300"))
            qp.drawText(self.width() - 70, 15, "PAUSED")


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------

class Dashboard(QMainWindow):
    def __init__(self, ros):
        super().__init__()
        self.ros = ros
        self.settings = QSettings("robotic_tracking_system", "operator_dashboard")
        self.setWindowTitle("Robotic Tracking System — Operator Dashboard")
        self.events_seen = 0

        self.canvas = WorkcellCanvas(ros.shapes)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self.canvas, stretch=1)
        lay.addLayout(self.build_layer_column())
        self.setCentralWidget(central)


        self.build_camera_dock()
        self.build_status_dock()
        self.build_graphs_dock()
        self.build_control_dock()

        geo = self.settings.value("geometry")
        if geo is not None and not os.environ.get("DASHBOARD_SCREENSHOT"):
            self.restoreGeometry(geo)
        state = self.settings.value("windowState")
        if state is not None:
            self.restoreState(state)
        else:
            self.resize(1500, 950)
        if os.environ.get("DASHBOARD_SCREENSHOT"):
            # Headless figure capture: fixed size, independent of any
            # saved geometry.
            self.resize(1850, 1180)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(40)

    # -- panels --------------------------------------------------------

    def build_layer_column(self):
        col = QVBoxLayout()
        col.addWidget(QLabel("<b>Trace Layers</b>"))
        self.layer_boxes = {}
        for name in ("EE path", "Target path", "Perceived part",
                     "Ground-truth part", "Error line", "Zones",
                     "Home marker"):
            cb = QCheckBox(name)
            cb.setChecked(self.settings.value(
                f"layer/{name}", True, type=bool))
            cb.stateChanged.connect(self.layers_changed)
            self.layer_boxes[name] = cb
            col.addWidget(cb)
        col.addWidget(QLabel("Trace length [s]"))
        self.trace_spin = QSpinBox()
        self.trace_spin.setRange(5, 300)
        self.trace_spin.setValue(self.settings.value(
            "layer/trace_window", 30, type=int))
        col.addWidget(self.trace_spin)
        col.addStretch(1)
        self.layers_changed()
        return col

    def layers_changed(self):
        for name, cb in self.layer_boxes.items():
            self.canvas.layers[name] = cb.isChecked()
            self.settings.setValue(f"layer/{name}", cb.isChecked())
        self.settings.setValue("layer/trace_window", self.trace_spin.value())

    def build_camera_dock(self):
        dock = QDockWidget("Camera", self)
        dock.setObjectName("camera_dock")
        w = QWidget()
        lay = QVBoxLayout(w)
        self.cam_select = QComboBox()
        self.cam_select.addItems(["Detection overlay", "Raw camera"])
        lay.addWidget(self.cam_select)
        self.cam_label = QLabel("waiting for images...")
        self.cam_label.setMinimumSize(480, 360)
        self.cam_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.cam_label, stretch=1)
        self.cam_caption = QLabel("")
        lay.addWidget(self.cam_caption)
        dock.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def build_status_dock(self):
        dock = QDockWidget("Robot status", self)
        dock.setObjectName("status_dock")
        w = QWidget()
        lay = QVBoxLayout(w)
        self.status_label = QLabel("STARTING")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(56)
        self.status_label.setFont(QFont("Sans", 15, QFont.Bold))
        lay.addWidget(self.status_label)
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(500)
        lay.addWidget(self.event_log, stretch=1)
        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def build_graphs_dock(self):
        dock = QDockWidget("Live graphs", self)
        dock.setObjectName("graphs_dock")
        w = QWidget()
        lay = QVBoxLayout(w)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Window [s]"))
        self.win_spin = QSpinBox()
        self.win_spin.setRange(10, 600)
        self.win_spin.setValue(60)
        controls.addWidget(self.win_spin)
        pause_btn = QPushButton("Pause/resume graphs")
        pause_btn.clicked.connect(self.toggle_graphs)
        controls.addWidget(pause_btn)
        export_btn = QPushButton("Export PNGs")
        export_btn.clicked.connect(self.export_graphs)
        controls.addWidget(export_btn)
        self.err_readout = QLabel("")
        controls.addWidget(self.err_readout)
        controls.addStretch(1)
        lay.addLayout(controls)

        self.err_plot = ScrollingPlot("Tracking precision — error (mm)", "mm")
        self.joint_plot = ScrollingPlot("Joint angles (degree)", "deg")
        lay.addWidget(self.err_plot)
        lay.addWidget(self.joint_plot)
        dock.setWidget(w)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def toggle_graphs(self):
        self.err_plot.toggle_pause()
        self.joint_plot.toggle_pause()

    def export_graphs(self):
        path = QFileDialog.getExistingDirectory(self, "Export plots to...")
        if not path:
            return
        self.err_plot.grab().save(os.path.join(path, "tracking_precision.png"))
        self.joint_plot.grab().save(os.path.join(path, "joint_angles.png"))
        self.log(f"[GUI] plots exported to {path}")

    def build_control_dock(self):
        dock = QDockWidget("Control panel", self)
        dock.setObjectName("control_dock")
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)

        btns = QHBoxLayout()
        for label, key, style in (
                ("GO HOME", "go_home", "background:#ef6c00;color:white"),
                ("START TRACKING", "resume", "background:#2e7d32;color:white"),
                ("STOP TRACKING", "pause", "background:#c62828;color:white")):
            b = QPushButton(label)
            b.setStyleSheet(style + ";font-weight:bold;padding:6px")
            b.clicked.connect(lambda _, k=key: self.ros.commands.put(("service", k)))
            btns.addWidget(b)
        outer_lay.addLayout(btns)

        mode_box = QGroupBox("Tracking behavior (applies at the next part)")
        mode_lay = QVBoxLayout(mode_box)
        self.mode_group = QButtonGroup(self)
        self.mode_radios = {}
        self.mode_spins = {}
        for value, label in PATH_MODES:
            row = QHBoxLayout()
            rb = QRadioButton(label)
            rb.toggled.connect(
                lambda on, v=value: on and self.apply_mode(v))
            self.mode_group.addButton(rb)
            self.mode_radios[value] = rb
            row.addWidget(rb)
            if value in MODE_SELECTORS:
                sel_label, _ = MODE_SELECTORS[value]
                row.addWidget(QLabel(sel_label))
                spin = QSpinBox()
                spin.setRange(1, 8)
                spin.setToolTip(
                    "1-based number; edge N starts at corner N (see the "
                    "numbered badges on the part in the trace view). A "
                    "number beyond the shape's count selects its last one.")
                spin.valueChanged.connect(
                    lambda v, m=value: self.apply_mode_index(m, v))
                self.mode_spins[value] = spin
                row.addWidget(spin)
            row.addStretch(1)
            mode_lay.addLayout(row)

        grid_box = QGroupBox("Parameters (live)")
        grid = QGridLayout(grid_box)
        grid.addWidget(QLabel("<b>parameter</b>"), 0, 0)
        grid.addWidget(QLabel("<b>value</b>"), 0, 1)
        self.param_spins = {}
        for row, spec in enumerate(PARAM_SPECS, start=1):
            lbl = QLabel(spec["name"])
            lbl.setToolTip(spec["tip"])
            grid.addWidget(lbl, row, 0)
            spin = QDoubleSpinBox()
            spin.setDecimals(spec["dec"])
            spin.setRange(spec["lo"], spec["hi"])
            spin.setSingleStep(spec["step"])
            spin.setValue(spec["optimal"])
            spin.setToolTip(spec["tip"])
            grid.addWidget(spin, row, 1)
            slider = QSlider(Qt.Horizontal)
            steps = int(round((spec["hi"] - spec["lo"]) / spec["step"]))
            slider.setRange(0, steps)
            slider.setValue(int(round(
                (spec["optimal"] - spec["lo"]) / spec["step"])))
            grid.addWidget(slider, row, 2)

            def sync_from_slider(v, sp=spin, s=spec):
                sp.blockSignals(True)
                sp.setValue(s["lo"] + v * s["step"])
                sp.blockSignals(False)

            def sync_from_spin(v, sl=slider, s=spec):
                sl.blockSignals(True)
                sl.setValue(int(round((v - s["lo"]) / s["step"])))
                sl.blockSignals(False)

            slider.valueChanged.connect(sync_from_slider)
            spin.valueChanged.connect(sync_from_spin)

            def apply(s=spec, sp=spin):
                self.ros.commands.put(
                    ("param", s["node"], s["name"], sp.value(),
                     s["dec"] == 0))

            slider.sliderReleased.connect(apply)
            spin.editingFinished.connect(apply)

            opt = QPushButton("optimal")
            opt.setToolTip(spec["tip"])

            def set_opt(_, s=spec, sp=spin):
                sp.setValue(s["optimal"])
                self.ros.commands.put(
                    ("param", s["node"], s["name"], s["optimal"],
                     s["dec"] == 0))

            opt.clicked.connect(set_opt)
            grid.addWidget(opt, row, 3)
            self.param_spins[spec["name"]] = (spin, spec)

        all_opt = QPushButton("ALL OPTIMAL (validated set)")
        all_opt.clicked.connect(self.apply_all_optimal)
        grid.addWidget(all_opt, len(PARAM_SPECS) + 1, 0, 1, 4)
        scroll = QScrollArea()
        scroll.setWidget(grid_box)
        scroll.setWidgetResizable(True)

        # Compact manual mode: one slider + an axis selector
        # (space-saving single-axis editor over a 7-slider stack);
        # SEND sends the full 7-axis target.
        manual_box = QGroupBox("MANUAL MODE (pauses tracking and Servo)")
        man_lay = QGridLayout(manual_box)
        self.manual_btn = QPushButton("MANUAL MODE")
        self.manual_btn.setCheckable(True)
        self.manual_btn.setStyleSheet(
            "QPushButton:checked{background:#6a1b9a;color:white;"
            "font-weight:bold}")
        self.manual_btn.toggled.connect(self.manual_toggled)
        man_lay.addWidget(self.manual_btn, 0, 0, 1, 2)
        self.send_btn = QPushButton("SEND")
        self.send_btn.setEnabled(False)
        self.send_btn.setStyleSheet("font-weight:bold")
        self.send_btn.clicked.connect(self.manual_send)
        man_lay.addWidget(self.send_btn, 0, 2, 1, 2)

        self.manual_targets = [0.0] * 7   # radians, all 7 axes
        self._loading_axis = False
        self.axis_combo = QComboBox()
        self.axis_combo.addItems([f"Axis {i + 1}" for i in range(7)])
        self.axis_combo.setEnabled(False)
        self.axis_combo.currentIndexChanged.connect(self.load_axis)
        man_lay.addWidget(self.axis_combo, 1, 0)
        self.axis_spin = QDoubleSpinBox()
        self.axis_spin.setDecimals(1)
        self.axis_spin.setSuffix(" deg")
        self.axis_spin.setEnabled(False)
        self.axis_spin.valueChanged.connect(self.axis_value_changed)
        man_lay.addWidget(self.axis_spin, 1, 1)
        self.axis_slider = QSlider(Qt.Horizontal)
        self.axis_slider.setEnabled(False)
        self.axis_slider.valueChanged.connect(
            lambda v: self.axis_spin.setValue(float(v)))
        man_lay.addWidget(self.axis_slider, 1, 2, 1, 2)

        # Panel order: command buttons, MANUAL MODE directly under
        # them, then behaviors, then parameters.
        outer_lay.addWidget(manual_box)
        outer_lay.addWidget(mode_box)
        outer_lay.addWidget(scroll, stretch=1)

        dock.setWidget(outer)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def log(self, text):
        """Append to the event log and keep the newest line visible."""
        self.event_log.appendPlainText(text)
        sb = self.event_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_mode(self, value):
        if value in self.mode_spins:
            self.apply_mode_index(value, self.mode_spins[value].value())
        self.ros.commands.put(
            ("param", "moving_shape_perimeter_tracker", "path_mode",
             value, False))

    def apply_mode_index(self, mode, value):
        if self.mode_radios[mode].isChecked():
            _, pname = MODE_SELECTORS[mode]
            self.ros.commands.put(
                ("param", "moving_shape_perimeter_tracker", pname,
                 int(value) - 1, True))

    def load_axis(self, idx):
        lo, hi = JOINT_LIMITS_DEG[idx]
        self._loading_axis = True
        self.axis_spin.setRange(lo, hi)
        self.axis_slider.setRange(int(lo), int(hi))
        deg = math.degrees(self.manual_targets[idx])
        self.axis_spin.setValue(deg)
        self.axis_slider.setValue(int(round(deg)))
        self._loading_axis = False

    def axis_value_changed(self, deg):
        self.axis_slider.blockSignals(True)
        self.axis_slider.setValue(int(round(deg)))
        self.axis_slider.blockSignals(False)
        if not self._loading_axis:
            self.manual_targets[self.axis_combo.currentIndex()] = \
                math.radians(deg)

    def manual_toggled(self, on):
        for w in (self.axis_combo, self.axis_spin, self.axis_slider,
                  self.send_btn):
            w.setEnabled(on)
        if on:
            # Single command source: gate the tracker off via the
            # supervisor AND pause Servo before any manual trajectory.
            self.ros.commands.put(("service", "pause"))
            self.ros.commands.put(("service", "servo_pause"))
            s = self.ros.snapshot()
            if s["joints"]:
                _, vals = s["joints"][-1]
                self.manual_targets = list(vals)
            self.load_axis(self.axis_combo.currentIndex())
            self.log(
                "[GUI] MANUAL MODE on - tracking paused, Servo paused")
        else:
            self.ros.commands.put(("service", "servo_start"))
            self.log(
                "[GUI] MANUAL MODE off - Servo restarted; press START "
                "TRACKING to resume")

    def manual_send(self):
        if not self.manual_btn.isChecked():
            return
        self.ros.commands.put(("manual_move", list(self.manual_targets)))

    def apply_all_optimal(self):
        for name, (spin, spec) in self.param_spins.items():
            spin.setValue(spec["optimal"])
            self.ros.commands.put(
                ("param", spec["node"], spec["name"], spec["optimal"],
                 spec["dec"] == 0))

    # -- refresh -------------------------------------------------------

    def refresh(self):
        s = self.ros.snapshot()
        self.canvas.trace_window = float(self.trace_spin.value())
        self.canvas.snap = s
        self.canvas.update()

        # Camera
        key = "overlay" if self.cam_select.currentIndex() == 0 else "raw"
        frame = s["images"].get(key)
        if frame is not None and key == "raw":
            # The overlay arrives pre-flipped from the perception node;
            # flip the raw feed here so both match the canvas
            # orientation (robot at the bottom).
            frame = np.ascontiguousarray(np.flipud(frame))
        if frame is not None:
            h, w, _ = frame.shape
            img = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img).scaled(
                self.cam_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            self.cam_label.setPixmap(pix)
        self.cam_caption.setText(
            f"detected: {s['percep_shape'] or '—'}   "
            f"ground truth: {s['gt_shape'] or '—'}   "
            f"sim t: {s['sim_t']:.1f} s")

        # Status
        status = s["status"].get("status", "STARTING")
        color = "#616161"
        for k, c in STATUS_COLORS.items():
            if status.startswith(k):
                color = c
                break
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"background:{color};color:white;border-radius:6px")
        events = s["events"]
        while self.events_seen < len(events):
            t, text = events[self.events_seen]
            self.log(f"[{t:8.1f}] {text}")
            self.events_seen += 1

        # Graphs
        window = float(self.win_spin.value())
        self.err_plot.window_s = window
        self.joint_plot.window_s = window
        self.err_plot.set_series(
            [("error", "#ff5252", s["err"])], s["sim_t"])
        joint_series = []
        colors = ["#e57373", "#ffb74d", "#fff176", "#81c784",
                  "#4fc3f7", "#ba68c8", "#f06292"]
        for j in range(7):
            pts = [(t, math.degrees(vals[j])) for t, vals in s["joints"]]
            joint_series.append((f"j{j + 1}", colors[j], pts))
        self.joint_plot.set_series(joint_series, s["sim_t"])
        if s["err_mean"] > 0.0:
            self.err_readout.setText(
                f"  part mean {s['err_mean']:.2f} mm   "
                f"RMS {s['err_rms']:.2f} mm  (approach excluded)")
        else:
            self.err_readout.setText("  approaching part...")

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)


def main(args=None):
    rclpy.init(args=args)
    ros = RosBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    win = Dashboard(ros)
    win.show()
    # DASHBOARD_SCREENSHOT="<png path>,<seconds>": grab the window after
    # the delay and quit (works with QT_QPA_PLATFORM=offscreen, so a
    # reproducible screenshot can be produced headless instead of a
    # hand-captured frame).
    shot = os.environ.get("DASHBOARD_SCREENSHOT")
    if shot:
        shot_path, shot_after = shot.rsplit(",", 1)

        def _capture():
            win.grab().save(shot_path)
            print(f"dashboard screenshot saved to {shot_path}", flush=True)
            app.quit()

        QTimer.singleShot(int(float(shot_after) * 1000), _capture)
    code = app.exec_()

    executor.shutdown()
    ros.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
