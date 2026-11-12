#!/usr/bin/env python3
"""Save one /perception/debug_image frame per catalog shape (utility node).

Subscribes to the perception debug image, the detected part pose and the
shape label; for every shape, the first frame whose detection is resolved
(label not 'pending'/'unknown') and whose part centre lies within
|y| <= y_max of the camera axis is written to <out>/debug_<shape>.png.
Start it next to a running simulation with the part stream and the
perception node (for readable frames keep the arm parked). Run with
use_sim_time like every node of the Gazebo setup.

Usage:
    python3 experiments/save_debug_frames.py --out DIR [--y-max 0.05]
        --ros-args -p use_sim_time:=true
"""
import argparse
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

SHAPES = ("square", "circle", "triangle", "pentagon", "irregular")


class FrameSaver(Node):
    def __init__(self, out, y_max):
        super().__init__("save_debug_frames")
        os.makedirs(out, exist_ok=True)
        self.out, self.y_max = out, y_max
        self.bridge = CvBridge()
        self.label, self.y = "unknown", None
        self.saved = set()
        self.create_subscription(String, "/perception/part_shape", self.on_label, 10)
        self.create_subscription(PoseStamped, "/perception/part_pose", self.on_pose, 10)
        self.create_subscription(Image, "/perception/debug_image", self.on_image, 2)
        self.get_logger().info(f"saving debug frames to {out}")

    def on_label(self, msg):
        self.label = msg.data

    def on_pose(self, msg):
        self.y = msg.pose.position.y

    def on_image(self, msg):
        if (self.label in SHAPES and self.label not in self.saved
                and self.y is not None and abs(self.y) <= self.y_max):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            path = os.path.join(self.out, f"debug_{self.label}.png")
            cv2.imwrite(path, frame)
            self.saved.add(self.label)
            self.get_logger().info(f"saved {path} (y = {self.y:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory for the PNG files")
    ap.add_argument("--y-max", type=float, default=0.05,
                    help="largest |y| of the part centre in m")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = FrameSaver(args.out, args.y_max)
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
