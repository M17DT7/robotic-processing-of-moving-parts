"""Stream parts of varying shape along the conveyor in Gazebo.

Node name:      gazebo_shape_stream_node
Purpose:        Spawn one Gazebo model per catalog shape (box, cylinder,
                polyline-extruded polygons) at a parking row, then every
                spawn_period seconds place the next shape in shape_sequence
                at the conveyor start and drive it along the belt
                kinematically (/set_entity_state at update_rate), parking
                the previous one. Publishes the cycle starts so the
                perimeter tracker can phase-lock to the visual parts.
Publications:   /part_stream/cycle_start (std_msgs/String = shape name),
                published at the start of every part cycle
                /part_stream/ground_truth_pose (geometry_msgs/PoseStamped)
                - true center of the active part on the part-top plane
                (world == panda_link0), every update tick; ground truth
                for evaluating the perception output
Services used:  /spawn_entity (gazebo_msgs/SpawnEntity, at startup)
                /set_entity_state (gazebo_msgs/SetEntityState, update_rate)
Parameters:     shapes_file ('' = package share config/shapes.yaml),
                shape_sequence, part_start_x/y, part_z, target_vx/vy,
                part_yaw, part_angular_speed, spawn_period, update_rate,
                parking_x, parking_y0, parking_spacing
Frames:         world (== panda_link0, robot spawned at origin)
Update rate:    update_rate (50 Hz default)
Dependencies:   Gazebo with libgazebo_ros_factory.so (spawn) and
                libgazebo_ros_state.so (set state); run with use_sim_time
"""

import math
import os

import yaml

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

SHAPE_COLORS = {
    "square": (0.1, 0.4, 1.0),
    "triangle": (1.0, 0.5, 0.1),
    "pentagon": (0.6, 0.2, 0.8),
    "irregular": (0.9, 0.85, 0.1),
    "circle": (0.9, 0.15, 0.15),
}
DEFAULT_COLOR = (0.5, 0.5, 0.5)


class GazeboShapeStreamNode(Node):
    def __init__(self):
        super().__init__("gazebo_shape_stream_node")

        self.declare_parameter("shapes_file", "")
        self.declare_parameter(
            "shape_sequence",
            ["square", "circle", "triangle", "pentagon", "irregular"],
        )

        self.declare_parameter("part_start_x", 0.50)
        self.declare_parameter("part_start_y", -0.25)
        self.declare_parameter("part_z", 0.41)
        self.declare_parameter("target_vx", 0.00)
        self.declare_parameter("target_vy", 0.03)
        self.declare_parameter("part_yaw", 0.0)
        self.declare_parameter("part_angular_speed", 0.0)
        self.declare_parameter("spawn_period", 16.0)
        self.declare_parameter("update_rate", 50.0)

        # Parking row for inactive shapes (off the belt, out of camera view)
        self.declare_parameter("parking_x", 1.5)
        self.declare_parameter("parking_y0", -0.6)
        self.declare_parameter("parking_spacing", 0.4)

        self.load_parameters()
        self.load_shapes()

        self.cycle_pub = self.create_publisher(String, "/part_stream/cycle_start", 10)
        self.ground_truth_pub = self.create_publisher(
            PoseStamped, "/part_stream/ground_truth_pose", 10
        )

        self.state_client = self.create_client(SetEntityState, "/set_entity_state")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.get_logger().info("Waiting for /spawn_entity and /set_entity_state...")
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Still waiting for /spawn_entity...")
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Still waiting for /set_entity_state...")

        self.spawn_all_shapes()

        self.start_time = None  # set on first tick (sim clock)
        self.cycle_start = None
        self.current_cycle = -1

        # Belt speed and spawn period are runtime-settable but LATCH AT
        # THE NEXT CYCLE rollover - a mid-cycle change would teleport the
        # active part.
        self.pending_target_vy = None
        self.pending_spawn_period = None
        self.add_on_set_parameters_callback(self.on_parameters_changed)

        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        self.get_logger().info(
            f"Gazebo shape stream started. Sequence: {self.shape_sequence}, "
            f"spawn_period: {self.spawn_period} s"
        )

    def load_parameters(self):
        gp = lambda name: self.get_parameter(name).value
        self.shapes_file = str(gp("shapes_file"))
        self.shape_sequence = [str(s) for s in gp("shape_sequence")]
        self.part_start_x = float(gp("part_start_x"))
        self.part_start_y = float(gp("part_start_y"))
        self.part_z = float(gp("part_z"))
        self.target_vx = float(gp("target_vx"))
        self.target_vy = float(gp("target_vy"))
        self.part_yaw = float(gp("part_yaw"))
        self.part_angular_speed = float(gp("part_angular_speed"))
        self.spawn_period = float(gp("spawn_period"))
        self.update_rate = float(gp("update_rate"))
        self.parking_x = float(gp("parking_x"))
        self.parking_y0 = float(gp("parking_y0"))
        self.parking_spacing = float(gp("parking_spacing"))

    def load_shapes(self):
        path = self.shapes_file
        if not path:
            share = get_package_share_directory("robotic_tracking_system")
            path = os.path.join(share, "config", "shapes.yaml")
        with open(path, "r") as f:
            catalog = yaml.safe_load(f)
        self.part_height = float(catalog.get("height", 0.08))
        self.shapes = catalog["shapes"]
        missing = [s for s in self.shape_sequence if s not in self.shapes]
        if missing:
            raise RuntimeError(f"shape_sequence contains unknown shapes: {missing}")

    # ------------------------------------------------------------------
    # SDF generation and spawning
    # ------------------------------------------------------------------

    def shape_sdf(self, name):
        spec = self.shapes[name]
        h = self.part_height
        r, g, b = SHAPE_COLORS.get(name, DEFAULT_COLOR)

        if spec["type"] == "circle":
            geometry = (
                f"<geometry><cylinder><radius>{spec['radius']}</radius>"
                f"<length>{h}</length></cylinder></geometry>"
            )
        else:
            vertices = spec["vertices"]
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            if (max(xs) - min(xs)) > 0 and abs(max(xs) + min(xs)) < 1e-6 and \
               abs(max(ys) + min(ys)) < 1e-6 and len(vertices) == 4:
                # Axis-aligned rectangle centered on origin: use a box
                geometry = (
                    f"<geometry><box><size>{max(xs)-min(xs)} {max(ys)-min(ys)} {h}"
                    f"</size></box></geometry>"
                )
            else:
                # Polyline extrusion: extrudes from z=0 upward, so the link
                # pose below shifts it down by h/2 to center it like the
                # box/cylinder geometries.
                points = "".join(f"<point>{x} {y}</point>" for x, y in vertices)
                geometry = (
                    f"<geometry><polyline>{points}<height>{h}</height>"
                    f"</polyline></geometry>"
                )

        # Polyline geometry is extruded from the link origin upward; center it.
        link_pose = f"<pose>0 0 {-h/2} 0 0 0</pose>" if "polyline" in geometry else ""

        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="part_{name}">
    <static>false</static>
    <link name="link">
      {link_pose}
      <gravity>false</gravity>
      <inertial>
        <mass>0.1</mass>
        <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="visual">
        {geometry}
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        {geometry}
      </collision>
    </link>
  </model>
</sdf>"""

    def parking_pose(self, index):
        return (self.parking_x, self.parking_y0 + index * self.parking_spacing)

    def spawn_all_shapes(self):
        for i, name in enumerate(dict.fromkeys(self.shape_sequence)):
            px, py = self.parking_pose(i)
            req = SpawnEntity.Request()
            req.name = f"part_{name}"
            req.xml = self.shape_sdf(name)
            req.initial_pose.position.x = px
            req.initial_pose.position.y = py
            req.initial_pose.position.z = self.part_z
            req.initial_pose.orientation.w = 1.0
            future = self.spawn_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            result = future.result()
            if result is None or not result.success:
                msg = result.status_message if result else "timeout"
                # "already exists" is fine on restart
                if "exist" in msg.lower():
                    self.get_logger().info(f"Model part_{name} already in world")
                else:
                    self.get_logger().error(f"Failed to spawn part_{name}: {msg}")
            else:
                self.get_logger().info(f"Spawned part_{name} at parking ({px:.2f}, {py:.2f})")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def on_parameters_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "target_vy":
                self.pending_target_vy = float(p.value)
                self.get_logger().info(
                    f"Belt speed {p.value} m/s latched for the next cycle")
            elif p.name == "spawn_period":
                self.pending_spawn_period = float(p.value)
                self.get_logger().info(
                    f"spawn_period {p.value} s latched for the next cycle")
        return SetParametersResult(successful=True)

    def timer_callback(self):
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            self.cycle_start = now
        elapsed_on_part = (now - self.cycle_start).nanoseconds * 1e-9

        rollover = (self.current_cycle < 0
                    or elapsed_on_part >= self.spawn_period)
        if rollover:
            if self.current_cycle >= 0:
                self.cycle_start = now
                elapsed_on_part = 0.0
            previous_name = (
                self.shape_sequence[self.current_cycle % len(self.shape_sequence)]
                if self.current_cycle >= 0 else None
            )
            self.current_cycle += 1
            if self.pending_target_vy is not None:
                self.target_vy = self.pending_target_vy
                self.pending_target_vy = None
            if self.pending_spawn_period is not None:
                self.spawn_period = self.pending_spawn_period
                self.pending_spawn_period = None
            active_name = self.shape_sequence[
                self.current_cycle % len(self.shape_sequence)]
            if previous_name is not None and previous_name != active_name:
                idx = list(dict.fromkeys(self.shape_sequence)).index(previous_name)
                px, py = self.parking_pose(idx)
                self.set_entity(f"part_{previous_name}", px, py, self.part_z, 0.0, 0.0, 0.0)
            msg = String()
            msg.data = active_name
            self.cycle_pub.publish(msg)
            self.get_logger().info(
                f"Cycle #{self.current_cycle}: shape '{active_name}' enters "
                f"the conveyor (belt {self.target_vy} m/s, period "
                f"{self.spawn_period} s)")
        active_name = self.shape_sequence[
            self.current_cycle % len(self.shape_sequence)]

        x = self.part_start_x + self.target_vx * elapsed_on_part
        y = self.part_start_y + self.target_vy * elapsed_on_part
        yaw = self.part_yaw + self.part_angular_speed * elapsed_on_part
        self.set_entity(f"part_{active_name}", x, y, self.part_z, yaw,
                        self.target_vx, self.target_vy)

        # Perception ground truth: true part center on the part-top plane
        # (world == panda_link0, robot spawned at the origin).
        gt = PoseStamped()
        gt.header.stamp = now.to_msg()
        gt.header.frame_id = "panda_link0"
        gt.pose.position.x = x
        gt.pose.position.y = y
        gt.pose.position.z = self.part_z + self.part_height / 2.0
        gt.pose.orientation.z = math.sin(yaw / 2.0)
        gt.pose.orientation.w = math.cos(yaw / 2.0)
        self.ground_truth_pub.publish(gt)

    def set_entity(self, name, x, y, z, yaw, vx, vy):
        state = EntityState()
        state.name = name
        state.reference_frame = "world"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        state.twist.linear.x = vx
        state.twist.linear.y = vy
        state.twist.angular.z = self.part_angular_speed
        req = SetEntityState.Request()
        req.state = state
        future = self.state_client.call_async(req)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(
                    f"Failed to set entity state: {response.status_message}",
                    throttle_duration_sec=5.0,
                )
        except Exception as ex:
            self.get_logger().error(f"Service call failed: {ex}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboShapeStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
