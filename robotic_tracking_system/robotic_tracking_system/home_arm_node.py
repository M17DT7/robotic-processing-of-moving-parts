"""Move the Panda arm to the 'ready' pose once at startup.

Node name:      home_arm_node
Purpose:        Drive the arm from the Gazebo spawn pose (all joints ~0,
                which is outside the real Panda joint limits and a singular
                configuration) to the SRDF 'ready' pose so that MoveIt
                Servo can operate. Publishes a single trajectory and exits.
Subscriptions:  /joint_states (sensor_msgs/JointState) - to verify arrival
Publications:   /panda_arm_controller/joint_trajectory
                (trajectory_msgs/JointTrajectory), published once
Parameters:     move_duration (double, s) - trajectory duration, default 4.0
                position_tolerance (double, rad) - arrival check, default 0.05
Frames:         none (joint space only)
Update rate:    one-shot
Dependencies:   panda_arm_controller must become active (waits for it)
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class HomeArmNode(Node):
    def __init__(self):
        super().__init__("home_arm_node")
        self.declare_parameter("move_duration", 4.0)
        self.declare_parameter("position_tolerance", 0.05)

        self._joint_positions = None
        self._pub = self.create_publisher(
            JointTrajectory, "/panda_arm_controller/joint_trajectory", 10
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

    def _on_joint_states(self, msg):
        self._joint_positions = dict(zip(msg.name, msg.position))

    def _spin_for(self, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _max_error(self):
        return max(
            abs(self._joint_positions.get(name, math.inf) - target)
            for name, target in zip(ARM_JOINTS, READY_POSE)
        )

    def run(self):
        # Wait until the trajectory controller subscribes (it does so when
        # it becomes active) and joint states are available.
        wait_start = time.monotonic()
        while rclpy.ok() and (
            self._pub.get_subscription_count() == 0 or self._joint_positions is None
        ):
            if time.monotonic() - wait_start > 30.0:
                self.get_logger().error(
                    "panda_arm_controller not available after 30 s, giving up"
                )
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        move_duration = self.get_parameter("move_duration").value
        tolerance = self.get_parameter("position_tolerance").value

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = READY_POSE
        point.velocities = [0.0] * len(ARM_JOINTS)
        point.time_from_start = Duration(
            sec=int(move_duration), nanosec=int((move_duration % 1.0) * 1e9)
        )
        traj.points = [point]

        # The subscription can exist before the controller is truly active
        # (under load the first trajectory is then silently dropped), so
        # publish, verify arrival via /joint_states, and retry if needed.
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            self._pub.publish(traj)
            self.get_logger().info(
                f"Moving arm to ready pose over {move_duration:.1f} s "
                f"(attempt {attempt}/{max_attempts})"
            )
            self._spin_for(move_duration + 2.0)

            error = self._max_error()
            if error <= tolerance:
                self.get_logger().info(
                    f"Arm at ready pose (max error {error:.3f} rad)"
                )
                return
            self.get_logger().warn(
                f"Arm not at ready pose after attempt {attempt} "
                f"(max error {error:.3f} rad)"
            )

        self.get_logger().error(
            f"Arm did not reach ready pose after {max_attempts} attempts "
            f"(max error {self._max_error():.3f} rad)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HomeArmNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
