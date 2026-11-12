"""Supervise continuous tracking: periodic RETURN_HOME between part cycles.

Node name:      tracking_supervisor
Purpose:        Mitigate the null-space posture drift of Cartesian-only
                servoing (in continuous tracking joint1 walks
                ~+0.16 rad/cycle until Servo emergency-stops after
                ~6-7 cycles). Servo's twist path has no null-space posture
                control, so the arm is periodically re-homed instead: every
                rehome_every_n_cycles tracked parts the supervisor skips one
                full part cycle, disables the tracker, pauses Servo, drives
                the arm back to the SRDF 'ready' pose and resumes.
States:         WAIT_READY -> TRACKING -> RETURN_HOME -> WAIT_FOR_PART
                -> TRACKING -> ... (ERROR if homing fails repeatedly; the
                tracker then stays disabled rather than drifting into a
                singularity)
Subscriptions:  /part_stream/cycle_start (std_msgs/String) - cycle
                boundaries from gazebo_shape_stream_node
                /joint_states (sensor_msgs/JointState) - homing verification
                /servo_node/status (std_msgs/Int8) - singularity warnings
                trigger an early re-home at the next cycle boundary
Publications:   /tracking_enabled (std_msgs/Bool, latched) - gate consumed
                by moving_shape_perimeter_tracker
                /tracking_supervisor/state (std_msgs/String, latched) - on
                every state transition, recorded by experiment_recorder
                /system_status (std_msgs/String, latched JSON) - the
                consolidated operator feed: display status, raw state,
                current shape, last servo status, optional event line
                (part detected, shape corrected, tracking finished,
                servo warning, operator commands)
                /panda_arm_controller/joint_trajectory
                (trajectory_msgs/JointTrajectory) - the homing move
Services:       ~/go_home, ~/pause_tracking, ~/resume_tracking
                (std_srvs/Trigger) - operator commands (GUI), all routed
                through this state machine; pause enters a PAUSED state
                and resume re-arms at the next part
Services used:  /servo_node/pause_servo, /servo_node/start_servo
                (std_srvs/Trigger). Resume MUST use start_servo, not
                unpause_servo: start_servo re-primes Servo's internal state
                and filters from the homed configuration and clears any
                latched singularity condition.
Parameters:     rehome_mode (default 'skip'):
                'skip' - when due, sacrifice one whole part cycle to
                re-home (needed when parts stream back-to-back with no
                idle time).
                'idle' - re-home in the gap BETWEEN parts, skipping
                none. Requires an actual gap: the
                perception pipeline releases the part at coast_limit_y
                and the next part must arrive late enough for the
                ~6.7 s re-home (spawn_period >= ~23 s at 0.03 m/s).
                Idle is detected via /perception/part_state going quiet
                for idle_detect_time, so 'idle' mode requires the
                perception stack.
                rehome_every_n_cycles (int, default 3; 0 = never periodic).
                NOTE (skip mode): with the default 5-shape sequence the
                skip period is N+1 cycles; N+1 must not divide the
                sequence length or the same shape is skipped every time
                (N=4 would always skip the same shape; N=3 rotates the
                skip). In idle mode N=1 is free (the gap is idle anyway)
                and keeps the posture permanently fresh.
                idle_detect_time (s, default 1.0).
                rehome_on_servo_warning (bool, default True) - re-home at
                the next cycle boundary after any Servo singularity status
                move_duration (s, 4.0), position_tolerance (rad, 0.05),
                max_home_attempts (3), settle_before_home (s, 0.5) - lets
                Servo's num_outgoing_halt_msgs_to_publish trajectories
                flush before pausing, home_positions (rad[7], SRDF ready)
Frames:         none (joint space + cycle timing only)
Update rate:    20 Hz state machine timer, sim time (run with
                use_sim_time:=true like every node in the Gazebo setup)
Dependencies:   gazebo_shape_stream_node (cycle starts), servo_node,
                panda_arm_controller active
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int8, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robotic_tracking_system.home_arm_node import ARM_JOINTS, READY_POSE

# moveit_servo StatusCode values that indicate the arm is degrading toward
# a singular configuration (status_codes.h: 1 = DECELERATE_FOR_APPROACHING_
# SINGULARITY, 2 = HALT_FOR_SINGULARITY, 6 = DECELERATE_FOR_LEAVING_
# SINGULARITY).
SINGULARITY_STATUSES = {1, 2, 6}

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class TrackingSupervisor(Node):
    def __init__(self):
        super().__init__("tracking_supervisor")

        self.declare_parameter("rehome_mode", "skip")
        self.declare_parameter("rehome_every_n_cycles", 3)
        self.declare_parameter("rehome_on_servo_warning", True)
        self.declare_parameter("idle_detect_time", 1.0)
        self.declare_parameter("move_duration", 4.0)
        self.declare_parameter("position_tolerance", 0.05)
        self.declare_parameter("max_home_attempts", 3)
        self.declare_parameter("settle_before_home", 0.5)
        self.declare_parameter("home_positions", READY_POSE)

        gp = lambda name: self.get_parameter(name).value
        self.rehome_mode = str(gp("rehome_mode"))
        if self.rehome_mode not in ("skip", "idle"):
            raise RuntimeError(f"Unknown rehome_mode '{self.rehome_mode}'")
        self.rehome_every_n_cycles = int(gp("rehome_every_n_cycles"))
        self.rehome_on_servo_warning = bool(gp("rehome_on_servo_warning"))
        self.idle_detect_time = float(gp("idle_detect_time"))
        self.move_duration = float(gp("move_duration"))
        self.position_tolerance = float(gp("position_tolerance"))
        self.max_home_attempts = int(gp("max_home_attempts"))
        self.settle_before_home = float(gp("settle_before_home"))
        self.home_positions = [float(v) for v in gp("home_positions")]
        if len(self.home_positions) != len(ARM_JOINTS):
            raise RuntimeError(
                f"home_positions must have {len(ARM_JOINTS)} values"
            )

        self.enabled_pub = self.create_publisher(
            Bool, "/tracking_enabled", LATCHED_QOS
        )
        self.state_pub = self.create_publisher(
            String, "/tracking_supervisor/state", LATCHED_QOS
        )
        # Consolidated operator status feed (GUI): JSON with the display
        # status, raw state, current shape, last servo status, and an
        # optional event string. Latched so a late-starting GUI sees the
        # current status immediately.
        self.status_pub = self.create_publisher(
            String, "/system_status", LATCHED_QOS
        )
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/panda_arm_controller/joint_trajectory", 10
        )

        self.create_subscription(
            String, "/part_stream/cycle_start", self.cycle_start_callback, 10
        )
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.create_subscription(
            Int8, "/servo_node/status", self.servo_status_callback, 10
        )
        # Part-state liveness: idle-mode gap detection AND the operator
        # status feed both use it, so subscribe in every mode.
        self.create_subscription(
            PoseStamped, "/perception/part_state", self.part_state_callback, 10
        )
        self.create_subscription(
            String, "/perception/cycle_start", self.percep_cycle_callback, 10
        )

        self.pause_client = self.create_client(Trigger, "/servo_node/pause_servo")
        self.start_client = self.create_client(Trigger, "/servo_node/start_servo")

        # Operator commands (GUI): all routed through this state machine
        # so external commands can never fight it.
        self.create_service(Trigger, "~/go_home", self.srv_go_home)
        self.create_service(Trigger, "~/pause_tracking", self.srv_pause)
        self.create_service(Trigger, "~/resume_tracking", self.srv_resume)

        self.joint_positions = None
        self.cycles_since_home = 0
        self.rehome_pending = False
        self.skipped_parts = 0
        self.last_state_time = None      # /perception/part_state liveness
        self.state_seen_since_resume = False
        self.paused = False
        self.current_shape = ""
        self.part_active = False
        self.last_servo_status_val = 0
        self.last_status_json = ""
        self.state = None
        self.phase = None  # RETURN_HOME sub-phase
        self.phase_start = None
        self.pending_future = None
        self.home_attempt = 0

        self.set_tracking(True)
        self.set_state("WAIT_READY")

        self.timer = self.create_timer(0.05, self.tick)
        self.status_timer = self.create_timer(0.2, self.status_tick)
        self.add_on_set_parameters_callback(self.on_parameters_changed)

        self.get_logger().info(
            f"Tracking supervisor started ({self.rehome_mode} mode): "
            f"re-home every {self.rehome_every_n_cycles} tracked cycles"
            + (", and after Servo singularity warnings"
               if self.rehome_on_servo_warning else "")
            + (" - re-homing happens in the part gap, no parts skipped"
               if self.rehome_mode == "idle" else "")
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def set_state(self, state):
        if state == self.state:
            return
        self.get_logger().info(f"State: {self.state} -> {state}")
        self.state = state
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.publish_status(event=f"State: {state}")

    # ------------------------------------------------------------------
    # Operator status feed and commands
    # ------------------------------------------------------------------

    def display_status(self):
        if self.state == "WAIT_READY":
            return "STARTING"
        if self.state == "TRACKING":
            if self.part_active:
                return "TRACKING PART"
            return "WAITING FOR PART"
        if self.state == "RETURN_HOME":
            return "HOMING"
        if self.state == "WAIT_FOR_PART":
            return "READY - WAITING FOR PART"
        if self.state == "PAUSED":
            return "PAUSED"
        return self.state or "STARTING"

    def publish_status(self, event=None):
        payload = {
            "t": round(self.get_clock().now().nanoseconds * 1e-9, 2),
            "status": self.display_status(),
            "state": self.state or "",
            "shape": self.current_shape,
            "servo_status": self.last_servo_status_val,
            "event": event,
        }
        msg = String()
        msg.data = json.dumps(payload)
        if event is not None or msg.data != self.last_status_json:
            self.status_pub.publish(msg)
            self.last_status_json = json.dumps({**payload, "event": None})

    def status_tick(self):
        fresh = (
            self.last_state_time is not None
            and (self.get_clock().now() - self.last_state_time)
            .nanoseconds * 1e-9 < 1.0
        )
        if fresh != self.part_active:
            self.part_active = fresh
            if not fresh and self.state == "TRACKING":
                self.publish_status(
                    event=f"TRACKING FINISHED - '{self.current_shape}' "
                          "left the working zone"
                )
                return
        self.publish_status()

    def percep_cycle_callback(self, msg):
        if self.part_active and msg.data != self.current_shape:
            self.current_shape = msg.data
            self.publish_status(event=f"SHAPE CORRECTED -> '{msg.data}'")
        else:
            self.current_shape = msg.data
            self.publish_status(event=f"PART DETECTED: '{msg.data}'")

    def begin_return_home(self, reason):
        self.get_logger().info(f"Re-homing ({reason})")
        self.set_tracking(False)
        self.home_attempt = 0
        self.set_state("RETURN_HOME")
        self.enter_phase("SETTLE")

    def srv_go_home(self, request, response):
        if self.state in ("TRACKING", "WAIT_FOR_PART", "PAUSED"):
            self.publish_status(event="Operator command: GO HOME")
            self.begin_return_home("operator command")
            response.success = True
            response.message = "homing"
        else:
            response.success = False
            response.message = f"cannot home from state {self.state}"
        return response

    def srv_pause(self, request, response):
        self.paused = True
        if self.state in ("TRACKING", "WAIT_FOR_PART"):
            self.publish_status(event="Operator command: STOP TRACKING")
            self.set_tracking(False)
            self.set_state("PAUSED")
            response.message = "paused"
        elif self.state == "RETURN_HOME":
            response.message = "will pause after homing"
        else:
            response.message = f"paused (state {self.state})"
        response.success = True
        return response

    def srv_resume(self, request, response):
        self.paused = False
        if self.state == "PAUSED":
            self.publish_status(event="Operator command: START TRACKING")
            self.set_state("WAIT_FOR_PART")
            response.message = "resuming at the next part"
        else:
            response.message = f"not paused (state {self.state})"
        response.success = True
        return response

    def on_parameters_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "rehome_every_n_cycles":
                self.rehome_every_n_cycles = int(p.value)
            elif p.name == "rehome_on_servo_warning":
                self.rehome_on_servo_warning = bool(p.value)
            elif p.name == "idle_detect_time":
                self.idle_detect_time = float(p.value)
            else:
                continue
            self.get_logger().info(f"Parameter {p.name} = {p.value}")
        return SetParametersResult(successful=True)

    def set_tracking(self, enabled):
        msg = Bool()
        msg.data = enabled
        self.enabled_pub.publish(msg)

    def enter_phase(self, phase):
        self.phase = phase
        self.phase_start = self.get_clock().now()

    def phase_elapsed(self):
        return (self.get_clock().now() - self.phase_start).nanoseconds * 1e-9

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def joint_state_callback(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(j in positions for j in ARM_JOINTS):
            self.joint_positions = positions

    def part_state_callback(self, msg):
        self.last_state_time = self.get_clock().now()
        self.state_seen_since_resume = True

    def rehome_due(self):
        return (
            self.rehome_pending
            or (self.rehome_every_n_cycles > 0
                and self.cycles_since_home >= self.rehome_every_n_cycles)
        )

    def servo_status_callback(self, msg):
        if msg.data != self.last_servo_status_val:
            self.last_servo_status_val = int(msg.data)
            if msg.data in SINGULARITY_STATUSES:
                self.publish_status(
                    event=f"SERVO WARNING: singularity status {msg.data}"
                )
        if not self.rehome_on_servo_warning:
            return
        if msg.data in SINGULARITY_STATUSES and not self.rehome_pending:
            if self.state == "TRACKING":
                self.rehome_pending = True
                self.get_logger().warn(
                    f"Servo singularity status {msg.data}: re-homing at the "
                    "next cycle boundary"
                )

    def cycle_start_callback(self, msg):
        if self.state == "PAUSED":
            return
        if self.state == "TRACKING":
            if self.rehome_mode == "skip" and self.rehome_due():
                self.skipped_parts += 1
                self.get_logger().info(
                    f"Skipping part '{msg.data}' to re-home "
                    f"(tracked cycles since home: {self.cycles_since_home}, "
                    f"servo warning pending: {self.rehome_pending})"
                )
                self.begin_return_home("skip mode, due")
            else:
                # idle mode never skips: when due, the re-home happens in
                # the gap after this part (see tick()).
                self.cycles_since_home += 1
        elif self.state == "WAIT_FOR_PART":
            self.cycles_since_home = 1
            self.state_seen_since_resume = False
            self.set_tracking(True)
            self.set_state("TRACKING")
            self.get_logger().info(f"Tracking resumed on part '{msg.data}'")
        elif self.state == "RETURN_HOME":
            # Homing overran into the next cycle (only after retries):
            # keep homing, count this part as skipped too.
            self.skipped_parts += 1
            self.get_logger().warn(
                f"Still homing when part '{msg.data}' entered; skipping it too"
            )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def tick(self):
        if self.state == "WAIT_READY":
            if (self.joint_positions is not None
                    and self.pause_client.service_is_ready()
                    and self.start_client.service_is_ready()):
                self.set_state("TRACKING")
        elif self.state == "TRACKING" and self.rehome_mode == "idle":
            # Re-home in the gap between parts: the estimator stops
            # publishing when the part leaves the working zone; once the
            # state has been quiet for idle_detect_time and a re-home is
            # due, home now - no part is skipped.
            if (self.rehome_due()
                    and self.state_seen_since_resume
                    and self.last_state_time is not None
                    and (self.get_clock().now() - self.last_state_time)
                    .nanoseconds * 1e-9 > self.idle_detect_time):
                self.get_logger().info(
                    "Part gap detected: re-homing between parts "
                    f"(tracked cycles since home: {self.cycles_since_home}, "
                    f"servo warning pending: {self.rehome_pending})"
                )
                self.begin_return_home("part gap, due")
        elif self.state == "RETURN_HOME":
            self.tick_return_home()

    def tick_return_home(self):
        if self.phase == "SETTLE":
            # Tracker is disabled; wait for Servo's halt trajectories
            # (num_outgoing_halt_msgs_to_publish) to flush so they cannot
            # preempt the homing trajectory.
            if self.phase_elapsed() >= self.settle_before_home:
                self.pending_future = self.pause_client.call_async(
                    Trigger.Request()
                )
                self.enter_phase("PAUSING")

        elif self.phase == "PAUSING":
            if self.pending_future.done():
                self.pending_future = None
                self.publish_home_trajectory()
                self.enter_phase("MOVING")
            elif self.phase_elapsed() > 3.0:
                self.get_logger().warn(
                    "pause_servo did not respond within 3 s; homing anyway"
                )
                self.pending_future = None
                self.publish_home_trajectory()
                self.enter_phase("MOVING")

        elif self.phase == "MOVING":
            # Same verify-and-retry pattern as home_arm_node: the
            # subscription can exist before the controller truly accepts
            # trajectories, so check arrival via /joint_states.
            if self.phase_elapsed() < self.move_duration + 2.0:
                return
            error = self.max_home_error()
            if error <= self.position_tolerance:
                self.get_logger().info(
                    f"Arm re-homed (max error {error:.3f} rad, "
                    f"attempt {self.home_attempt}/{self.max_home_attempts})"
                )
                self.pending_future = self.start_client.call_async(
                    Trigger.Request()
                )
                self.enter_phase("RESUMING")
            elif self.home_attempt < self.max_home_attempts:
                self.get_logger().warn(
                    f"Arm not at home after attempt {self.home_attempt} "
                    f"(max error {error:.3f} rad), retrying"
                )
                self.publish_home_trajectory()
                self.enter_phase("MOVING")
            else:
                self.get_logger().error(
                    f"Homing failed after {self.max_home_attempts} attempts "
                    f"(max error {error:.3f} rad); tracking stays disabled"
                )
                self.set_state("ERROR")

        elif self.phase == "RESUMING":
            if self.pending_future.done():
                result = self.pending_future.result()
                self.pending_future = None
                if result is not None and result.success:
                    self.finish_return_home()
                else:
                    self.get_logger().error(
                        "start_servo failed after homing; tracking stays "
                        "disabled"
                    )
                    self.set_state("ERROR")
            elif self.phase_elapsed() > 3.0:
                self.get_logger().warn(
                    "start_servo did not respond within 3 s; resuming anyway"
                )
                self.pending_future = None
                self.finish_return_home()

    def finish_return_home(self):
        self.cycles_since_home = 0
        self.rehome_pending = False
        self.phase = None
        self.set_state("PAUSED" if self.paused else "WAIT_FOR_PART")

    def publish_home_trajectory(self):
        self.home_attempt += 1
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = self.home_positions
        point.velocities = [0.0] * len(ARM_JOINTS)
        point.time_from_start = Duration(
            sec=int(self.move_duration),
            nanosec=int((self.move_duration % 1.0) * 1e9),
        )
        traj.points = [point]
        self.traj_pub.publish(traj)
        self.get_logger().info(
            f"Homing trajectory published ({self.move_duration:.1f} s, "
            f"attempt {self.home_attempt}/{self.max_home_attempts})"
        )

    def max_home_error(self):
        if self.joint_positions is None:
            return math.inf
        return max(
            abs(self.joint_positions.get(name, math.inf) - target)
            for name, target in zip(ARM_JOINTS, self.home_positions)
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrackingSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
