"""Detect conveyor parts in the overhead camera image (classical CV).

Node name:      part_perception_node
Purpose:        Perception stage. Detects the part on the belt in the
                overhead camera image, estimates its pose
                and corners, classifies the shape from contour GEOMETRY
                (not color: segmentation is a color-agnostic foreground
                mask against the belt; classification is Hu-moment
                contour matching (cv2.matchShapes) against the
                shapes.yaml catalog, with a circularity test for circles
                and a physical-area sanity check), and projects image
                points onto the known part-top plane (z = 0.45 in
                panda_link0) through the camera intrinsics and the static
                panda_link0 -> camera_link_optical TF (flat-belt
                homography; the simulated camera is an ideal pinhole).
                The published pose is the PART-FRAME ORIGIN (the frame
                the shape catalog and the perimeter tracker use):
                detected contour centroid minus the classified shape's
                catalog centroid offset (valid for yaw 0). Contours that
                touch the ROI boundary (partially visible parts) are
                SUPPRESSED rather than published with a biased center.
Subscriptions:  /camera/conveyor_camera/image_raw (sensor_msgs/Image)
                /camera/conveyor_camera/camera_info (sensor_msgs/CameraInfo,
                    intrinsics latched once)
Publications:   /perception/part_pose (geometry_msgs/PoseStamped) - part
                    center on the part-top plane, planning frame; stamped
                    with the IMAGE stamp so downstream consumers can
                    compensate the true pipeline latency
                /perception/part_corners (geometry_msgs/PolygonStamped) -
                    detected corner points on the part-top plane (empty
                    for circles), image stamp
                /perception/part_shape (std_msgs/String) - classified
                    shape name ('unknown' if no catalog match)
                /perception/debug_image (sensor_msgs/Image) - annotated
                    frame (contour, corners, center, label), decimated
Parameters:     image_topic, camera_info_topic, planning_frame,
                camera_optical_frame, part_top_z (0.45),
                belt_roi [x_min, x_max, y_min, y_max] (m, planning frame,
                on the part-top plane; default covers the belt tracking
                zone and excludes the parked robot arm),
                color_distance_threshold (0-255, vs per-frame belt median),
                min_contour_area_px, approx_epsilon_frac (of perimeter,
                corner extraction only), circle_circularity_threshold,
                area_match_tolerance (relative, vs catalog area),
                match_score_threshold (max cv2.matchShapes score to
                accept a classification), morph_kernel_px,
                shapes_file ('' = package share config/shapes.yaml),
                circle_discretization (catalog area computation only),
                debug_decimation (publish every Nth debug frame, 0 = off),
                parallax_correction (default True; top-face recovery
                from the silhouette),
                principal_point_shift_px (default -1.0),
                extrinsic_offset, publish_delay, publish_delay_jitter,
                blackout_y (controlled-test hooks, default off)
Frames:         detections in planning_frame (panda_link0 == world)
Update rate:    per camera frame (30 Hz), single-threaded
Dependencies:   cv_bridge, OpenCV, numpy; camera + static TF from
                panda_gazebo_moveit.launch.py. Run with
                use_sim_time:=true like every node in the Gazebo setup.
Limitations:    - yaw is not estimated (parts stream with yaw 0; the pose
                  orientation is identity)
                - one part per frame (largest contour); the stream node
                  guarantees a single part in the tracking zone
                - the overhead camera cannot see a part occluded by the
                  arm; perception-only measurements need the detection
                  zone upstream of the EE or the tracker switched off
"""

import math
import os
import random
import time
from collections import Counter, deque

import numpy as np
import yaml

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point32, PolygonStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

import tf2_ros
from tf2_ros import TransformException


def quat_to_matrix(x, y, z, w):
    """Rotation matrix from a unit quaternion (x, y, z, w)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def polygon_area(points):
    """Shoelace area of an (N, 2) array."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(points):
    """Area-weighted centroid of an (N, 2) polygon."""
    a = cx = cy = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-12:
        return np.zeros(2)
    return np.array([cx / (6.0 * a), cy / (6.0 * a)])


class PartPerceptionNode(Node):
    def __init__(self):
        super().__init__("part_perception_node")

        self.declare_parameter("image_topic", "/camera/conveyor_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/conveyor_camera/camera_info")
        self.declare_parameter("planning_frame", "panda_link0")
        self.declare_parameter("camera_optical_frame", "camera_link_optical")
        self.declare_parameter("part_top_z", 0.45)
        # Belt tracking zone on the part-top plane. The parked arm's
        # silhouette projects onto the plane at x < 0.30, outside this
        # window. y_min = -0.40 extends the zone upstream of the spawn
        # point, so parts are fully visible the moment they spawn at
        # y = -0.25 (largest corner radius 0.115 m + boundary-clip
        # margin): the estimator locks in ~0.6 s and the arm engages as
        # the part appears. This requires the extended belt
        # (worlds/conveyor_world.world spans y >= -0.45) and stays
        # inside the camera FOV (+-0.433 m at the part-top plane).
        self.declare_parameter("belt_roi", [0.36, 0.64, -0.40, 0.30])
        self.declare_parameter("color_distance_threshold", 40.0)
        self.declare_parameter("min_contour_area_px", 1500)
        self.declare_parameter("approx_epsilon_frac", 0.02)
        self.declare_parameter("circle_circularity_threshold", 0.85)
        self.declare_parameter("area_match_tolerance", 0.35)
        self.declare_parameter("match_score_threshold", 0.5)
        # Temporal majority filter on the shape label. At oblique viewing
        # angles (part far from the camera nadir) the part's side walls
        # enter the silhouette and can fill concavities, flipping the
        # instantaneous classification (e.g. the irregular part reads as
        # a square at y > 0.09). The label is constant per part, so a
        # majority vote over the whole pass removes these flips (the window
        # must exceed the corrupted stretch, so it spans a full ~14 s
        # pass at 30 Hz; a detection gap longer than label_filter_gap
        # clears the window - new part). 0 = off.
        self.declare_parameter("label_filter_window", 450)
        self.declare_parameter("label_filter_gap", 0.5)
        # Only frames whose detected center lies within this |y| band
        # feed the label vote: at larger |y| the oblique view distorts
        # the contour enough to misclassify systematically (parts are
        # detectable from y = -0.285, where an irregular part reads as a
        # square for the first ~3 s; those frames would poison the vote
        # past the occlusion cutoff). 0.18 is the band inside which the
        # classification is 100 % correct. Until the vote has content the
        # reserved label 'pending' is published (see image_callback).
        self.declare_parameter("label_trust_band", 0.18)
        self.declare_parameter("morph_kernel_px", 5)
        self.declare_parameter("shapes_file", "")
        self.declare_parameter("circle_discretization", 36)
        self.declare_parameter("debug_decimation", 5)
        # Controlled-test hooks for robustness experiments; all default
        # off, so the normal behavior is unchanged.
        #   extrinsic_offset [dx, dy, dz] (m): error added to the camera
        #     position used for projection (calibration-error emulation)
        #   publish_delay / publish_delay_jitter (s): detections are held
        #     for delay + U(0, jitter) before publication (the image
        #     stamp is kept, so the pipeline sees real extra latency)
        #   blackout_y [y_min, y_max] (m): detections whose center lies
        #     in this band are dropped (controlled occlusion length);
        #     y_min >= y_max disables it
        # Side-wall parallax correction. The silhouette of an
        # extruded part seen by the straight-down camera is the sweep of
        # its top face scaled about the camera nadir by s in [1 - rho, 1],
        # rho = part_height / (camera height above the part base): the
        # nadir-facing side wall adds a band that biases the centroid
        # (and, because the band shrinks as the part approaches the
        # nadir, the fitted belt VELOCITY ~5 % low) and fills
        # concavities (irregular -> square). Along every ray from
        # the nadir the top face is exactly  S AND scale(S, 1/(1 - rho)),
        # so the top-face mask is recovered with one image scaling about
        # the principal point and a bitwise AND - no shape knowledge
        # needed (exact for shapes that are star-convex along nadir
        # rays). Enabled by default: together with the principal-point
        # shift it reduces the vision error from 2.54 to 0.18 mm; False
        # reproduces the uncorrected behavior.
        self.declare_parameter("parallax_correction", True)
        # gazebo_ros_camera reports the principal point as (w + 1) / 2
        # (1-indexed pixel convention); with OpenCV's 0-indexed pixel
        # centers the optical axis is at (w - 1) / 2, i.e. one pixel
        # less. -1.0 removes the resulting constant ~1.35 mm offset in x
        # and y; 0.0 uses the reported principal point unchanged.
        self.declare_parameter("principal_point_shift_px", -1.0)
        self.declare_parameter("extrinsic_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("publish_delay", 0.0)
        self.declare_parameter("publish_delay_jitter", 0.0)
        self.declare_parameter("blackout_y", [0.0, 0.0])

        gp = lambda name: self.get_parameter(name).value
        self.image_topic = str(gp("image_topic"))
        self.camera_info_topic = str(gp("camera_info_topic"))
        self.planning_frame = str(gp("planning_frame"))
        self.camera_optical_frame = str(gp("camera_optical_frame"))
        self.part_top_z = float(gp("part_top_z"))
        self.belt_roi = [float(v) for v in gp("belt_roi")]
        self.color_distance_threshold = float(gp("color_distance_threshold"))
        self.min_contour_area_px = int(gp("min_contour_area_px"))
        self.approx_epsilon_frac = float(gp("approx_epsilon_frac"))
        self.circle_circularity_threshold = float(gp("circle_circularity_threshold"))
        self.area_match_tolerance = float(gp("area_match_tolerance"))
        self.match_score_threshold = float(gp("match_score_threshold"))
        self.label_filter_window = int(gp("label_filter_window"))
        self.label_filter_gap = float(gp("label_filter_gap"))
        self.label_trust_band = float(gp("label_trust_band"))
        self.morph_kernel_px = int(gp("morph_kernel_px"))
        self.shapes_file = str(gp("shapes_file"))
        self.circle_discretization = int(gp("circle_discretization"))
        self.debug_decimation = int(gp("debug_decimation"))
        self.parallax_correction = bool(gp("parallax_correction"))
        self.principal_point_shift_px = float(gp("principal_point_shift_px"))
        self.extrinsic_offset = [float(v) for v in gp("extrinsic_offset")]
        self.publish_delay = float(gp("publish_delay"))
        self.publish_delay_jitter = float(gp("publish_delay_jitter"))
        self.blackout_y = [float(v) for v in gp("blackout_y")]
        self.delay_queue = deque()

        self.load_catalog()

        self.bridge = CvBridge()
        self.K = None           # camera matrix (fx, fy, cx, cy)
        self.R_wc = None        # rotation optical -> planning
        self.t_wc = None        # camera origin in planning frame
        self.roi_mask = None    # binary image mask of the belt ROI
        self.frame_count = 0
        self.detection_count = 0
        self.label_window = deque(maxlen=max(1, self.label_filter_window))
        self.last_detection_stamp = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, "/perception/part_pose", 10)
        self.corners_pub = self.create_publisher(PolygonStamped, "/perception/part_corners", 10)
        self.shape_pub = self.create_publisher(String, "/perception/part_shape", 10)
        self.debug_pub = self.create_publisher(Image, "/perception/debug_image", 2)

        if self.publish_delay > 0.0 or self.publish_delay_jitter > 0.0:
            # Wall-clock timer: the hold time must not depend on the
            # sim-clock update granularity.
            self.create_timer(
                0.002, self.flush_delay_queue,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 2)

        self.get_logger().info(
            f"Part perception started (geometry-based classification, "
            f"catalog: {sorted(self.catalog_areas)}, "
            f"part-top plane z = {self.part_top_z} m, ROI {self.belt_roi})"
        )

    # ------------------------------------------------------------------
    # Catalog (same source as the tracker/stream: shapes.yaml)
    # ------------------------------------------------------------------

    def load_catalog(self):
        path = self.shapes_file
        if not path:
            share = get_package_share_directory("robotic_tracking_system")
            path = os.path.join(share, "config", "shapes.yaml")
        with open(path, "r") as f:
            catalog = yaml.safe_load(f)

        # Per shape: physical area, area-weighted centroid offset (the
        # published pose is the part-frame ORIGIN = detected centroid
        # minus this offset, yaw 0), and a reference contour for Hu-moment
        # matching (cv2.matchShapes is scale/rotation invariant, so the
        # metric-space reference matches the pixel-space detection).
        self.part_height = float(catalog.get("height", 0.08))
        self.catalog_areas = {}
        self.catalog_centroids = {}
        self.catalog_contours = {}
        self.circle_names = set()
        for name, spec in catalog["shapes"].items():
            if spec["type"] == "circle":
                r = float(spec["radius"])
                n = self.circle_discretization
                verts = np.array([
                    [r * math.cos(2.0 * math.pi * i / n),
                     r * math.sin(2.0 * math.pi * i / n)]
                    for i in range(n)
                ])
                self.catalog_areas[name] = math.pi * r * r
                self.catalog_centroids[name] = np.zeros(2)
                self.circle_names.add(name)
            elif spec["type"] == "polygon":
                verts = np.array([[float(x), float(y)] for x, y in spec["vertices"]])
                self.catalog_areas[name] = polygon_area(verts)
                self.catalog_centroids[name] = polygon_centroid(verts)
            else:
                continue
            self.catalog_contours[name] = verts.astype(np.float32).reshape(-1, 1, 2)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg):
        if self.K is not None:
            return
        k = np.array(msg.k).reshape(3, 3)
        shift = self.principal_point_shift_px
        self.K = (k[0, 0], k[1, 1], k[0, 2] + shift, k[1, 2] + shift)  # fx, fy, cx, cy
        self.get_logger().info(
            f"Camera intrinsics: fx {k[0, 0]:.1f} fy {k[1, 1]:.1f} "
            f"cx {k[0, 2]:.1f} cy {k[1, 2]:.1f} ({msg.width}x{msg.height})"
        )

    def lookup_camera_pose(self):
        try:
            tr = self.tf_buffer.lookup_transform(
                self.planning_frame, self.camera_optical_frame, rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF {self.planning_frame} -> {self.camera_optical_frame} "
                f"unavailable: {ex}",
                throttle_duration_sec=5.0,
            )
            return False
        t = tr.transform.translation
        q = tr.transform.rotation
        self.t_wc = np.array([t.x, t.y, t.z]) + np.array(self.extrinsic_offset)
        self.R_wc = quat_to_matrix(q.x, q.y, q.z, q.w)
        return True

    def pixel_to_plane(self, u, v):
        """Intersect the pixel ray with the part-top plane (planning frame)."""
        fx, fy, cx, cy = self.K
        d_opt = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d = self.R_wc @ d_opt
        if abs(d[2]) < 1e-9:
            return None
        s = (self.part_top_z - self.t_wc[2]) / d[2]
        return self.t_wc + s * d

    def plane_to_pixel(self, x, y):
        """Project a part-top-plane point into the image."""
        fx, fy, cx, cy = self.K
        p_opt = self.R_wc.T @ (np.array([x, y, self.part_top_z]) - self.t_wc)
        if p_opt[2] <= 1e-9:
            return None
        return (fx * p_opt[0] / p_opt[2] + cx, fy * p_opt[1] / p_opt[2] + cy)

    def build_roi_mask(self, shape):
        x0, x1, y0, y1 = self.belt_roi
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        px = [self.plane_to_pixel(x, y) for x, y in corners]
        if any(p is None for p in px):
            return None
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(px, dtype=np.int32)], 255)
        # A contour touching the ROI edge means the part is only partially
        # visible; publishing its centroid would bias the pose. Contours
        # reaching outside this eroded mask are suppressed.
        self.roi_mask_eroded = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2
        )
        return mask

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def image_callback(self, msg):
        if self.K is None:
            return
        if self.R_wc is None and not self.lookup_camera_pose():
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.roi_mask is None:
            self.roi_mask = self.build_roi_mask(frame.shape)
            if self.roi_mask is None:
                return
            self.get_logger().info("Belt ROI mask built")

        self.frame_count += 1

        # Color-agnostic foreground mask: distance from the belt's
        # per-frame median color inside the ROI (the part covers a small
        # fraction of the ROI, so the median is the belt).
        roi_pixels = frame[self.roi_mask > 0]
        belt_color = np.median(roi_pixels, axis=0)
        distance = np.linalg.norm(frame.astype(np.float32) - belt_color, axis=2)
        fg = ((distance > self.color_distance_threshold).astype(np.uint8) * 255)
        fg = cv2.bitwise_and(fg, self.roi_mask)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_px, self.morph_kernel_px)
        )
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area_px]

        detection = None
        if contours:
            contour = max(contours, key=cv2.contourArea)
            pts = contour.reshape(-1, 2)
            if (self.roi_mask_eroded[pts[:, 1], pts[:, 0]] == 0).any():
                self.get_logger().debug(
                    "Part touches the ROI boundary; detection suppressed"
                )
            else:
                if self.parallax_correction:
                    contour = self.top_face_contour(contour, fg.shape)
                if contour is not None:
                    detection = self.analyze_contour(contour)

        if detection is not None:
            stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if (self.last_detection_stamp is not None
                    and stamp_s - self.last_detection_stamp > self.label_filter_gap):
                self.label_window.clear()  # detection gap = new part
            self.last_detection_stamp = stamp_s
            if abs(detection["center"][1]) <= self.label_trust_band:
                self.label_window.append(detection["shape"])
            raw = detection["shape"]
            if self.label_window:
                counts = Counter(self.label_window)
                smoothed = counts.most_common(1)[0][0]
            else:
                # Entry region: no trusted vote yet. NEVER publish the
                # raw instantaneous guess (the irregular part's oblique
                # entry view classifies as 'square', which would show a
                # wrong outline for a split second): publish the
                # reserved label 'pending' until the vote fills.
                smoothed = "pending" if raw != "unknown" else "unknown"
            detection["shape"] = smoothed
            # Published pose = part-frame ORIGIN (the frame the shape
            # catalog and the tracker use): detected centroid minus the
            # classified shape's catalog centroid offset. Valid for yaw 0
            # (yaw is not estimated); 'unknown' and 'pending' keep the
            # RAW centroid. 'pending' must NOT use the
            # instantaneous guess's offset: that guess flips frame to
            # frame at the oblique entry (square <-> irregular), which
            # would make the pending positions bimodal (8-18 mm apart)
            # and corrupt the estimator's velocity fit under a constant
            # 'pending' label (a +19/-36 mm state excursion at
            # resolution). The raw centroid is one consistent
            # trajectory; the estimator rebases it into the
            # resolved part frame when the label resolves.
            offset_label = "unknown" if smoothed == "pending" else smoothed
            if offset_label != "unknown":
                offset = self.catalog_centroids[offset_label]
                detection["center"][0] -= offset[0]
                detection["center"][1] -= offset[1]
            y_c = detection["center"][1]
            if self.blackout_y[0] < self.blackout_y[1] and \
                    self.blackout_y[0] <= y_c <= self.blackout_y[1]:
                pass  # controlled occlusion: detection dropped
            elif self.publish_delay > 0.0 or self.publish_delay_jitter > 0.0:
                release = time.monotonic() + self.publish_delay + \
                    random.uniform(0.0, self.publish_delay_jitter)
                self.delay_queue.append((release, msg.header.stamp, detection))
            else:
                self.publish_detection(msg.header.stamp, detection)
            self.detection_count += 1

        if self.debug_decimation > 0 and self.frame_count % self.debug_decimation == 0:
            self.publish_debug(frame, msg.header.stamp, detection)

    def top_face_contour(self, contour, shape):
        """Remove the side-wall band from a part silhouette (see the
        parallax_correction parameter); returns the top-face contour."""
        z_base = self.part_top_z - self.part_height
        rho = self.part_height / (self.t_wc[2] - z_base)
        s = 1.0 / (1.0 - rho)
        _, _, cx, cy = self.K
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        warp = np.array([[s, 0.0, (1.0 - s) * cx], [0.0, s, (1.0 - s) * cy]])
        scaled = cv2.warpAffine(mask, warp, (shape[1], shape[0]),
                                flags=cv2.INTER_NEAREST)
        top = cv2.bitwise_and(mask, scaled)
        contours, _ = cv2.findContours(top, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours
                    if cv2.contourArea(c) >= self.min_contour_area_px]
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def analyze_contour(self, contour):
        """Classify a contour and lift it onto the part-top plane."""
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)
        if perimeter_px < 1e-6:
            return None
        circularity = 4.0 * math.pi * area_px / (perimeter_px * perimeter_px)

        moments = cv2.moments(contour)
        if moments["m00"] < 1e-6:
            return None
        cu = moments["m10"] / moments["m00"]
        cv_ = moments["m01"] / moments["m00"]
        center = self.pixel_to_plane(cu, cv_)
        if center is None:
            return None

        # Physical area via the plane projection (shoelace over the
        # projected contour) - scale-independent of the pixel grid.
        pts = contour.reshape(-1, 2)
        plane_pts = np.array([self.pixel_to_plane(u, v)[:2] for u, v in pts])
        area_m2 = polygon_area(plane_pts)

        corners_plane = []
        if circularity < self.circle_circularity_threshold:
            eps = self.approx_epsilon_frac * perimeter_px
            approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
            corners_plane = [self.pixel_to_plane(u, v) for u, v in approx]
            corners_plane = [p for p in corners_plane if p is not None]

        # Classification: Hu-moment contour matching against the catalog
        # (scale/rotation invariant), gated by circularity (circle vs
        # polygon) and a physical-area sanity check. Lowest match score
        # under the threshold wins. Corner count is NOT used: it is
        # unstable (concave notches are smoothed away, rounded corners
        # split).
        shape = "unknown"
        best_score = math.inf
        is_circle = circularity >= self.circle_circularity_threshold
        for name, ref_contour in self.catalog_contours.items():
            if (name in self.circle_names) != is_circle:
                continue
            ref_area = self.catalog_areas[name]
            if abs(area_m2 - ref_area) > self.area_match_tolerance * ref_area:
                continue
            score = cv2.matchShapes(contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if score < best_score and score <= self.match_score_threshold:
                best_score = score
                shape = name

        return {
            "center": np.array(center, dtype=float),
            "corners": corners_plane,
            "shape": shape,
            "circularity": circularity,
            "match_score": best_score,
            "area_m2": area_m2,
            "contour": contour,
        }

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_detection(self, stamp, det):
        # The label goes out BEFORE the pose it belongs to. The estimator
        # pairs each pose with the latest label it has received; with the
        # pose first, the first frame of a resolved shape would be filed
        # under the previous label ('pending') although it is already
        # expressed in the resolved part frame, and the estimator's pending
        # rebase would shift it a second time: one sample 13-18 mm off per
        # part for shapes with a non-zero catalog offset (triangle,
        # irregular), which biases the fitted velocity by 0.2-0.35 mm/s.
        label = String()
        label.data = det["shape"]
        self.shape_pub.publish(label)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.planning_frame
        pose.pose.position.x = float(det["center"][0])
        pose.pose.position.y = float(det["center"][1])
        pose.pose.position.z = float(det["center"][2])
        pose.pose.orientation.w = 1.0  # yaw is not estimated
        self.pose_pub.publish(pose)

        poly = PolygonStamped()
        poly.header = pose.header
        for p in det["corners"]:
            poly.polygon.points.append(
                Point32(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            )
        self.corners_pub.publish(poly)

    def flush_delay_queue(self):
        now = time.monotonic()
        # Release in arrival order (a later frame never overtakes an
        # earlier one, as in a real processing pipeline).
        while self.delay_queue and self.delay_queue[0][0] <= now:
            _, stamp, det = self.delay_queue.popleft()
            self.publish_detection(stamp, det)

    def publish_debug(self, frame, stamp, det):
        # Rendered on a VERTICALLY FLIPPED frame so the view matches the
        # dashboard's workcell canvas orientation (robot at the bottom,
        # belt travel to the right: image right = world +Y, image up =
        # world +X). Every annotation coordinate is flipped accordingly
        # so text stays upright.
        dbg = np.flipud(frame).copy()
        height = dbg.shape[0]

        def fp(px):
            return (int(px[0]), int(height - 1 - px[1]))

        x0, x1, y0, y1 = self.belt_roi
        roi_px = [self.plane_to_pixel(x, y)
                  for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
        if all(p is not None for p in roi_px):
            pts = np.array([fp(p) for p in roi_px], dtype=np.int32)
            cv2.polylines(dbg, [pts], True, (255, 255, 0), 1)
        if det is not None:
            contour = det["contour"].copy()
            contour[:, :, 1] = height - 1 - contour[:, :, 1]
            cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
            for p in det["corners"]:
                px = self.plane_to_pixel(p[0], p[1])
                if px is not None:
                    cv2.circle(dbg, fp(px), 4, (0, 0, 255), -1)
            cpx = self.plane_to_pixel(det["center"][0], det["center"][1])
            if cpx is not None:
                anchor = fp(cpx)
                cv2.drawMarker(dbg, anchor, (255, 0, 255),
                               cv2.MARKER_CROSS, 12, 2)
                # Two short lines in a larger font (a single 0.5-scale
                # line is unreadable at print size), placed below the
                # part so they never cover the contour.
                lines = (
                    det["shape"],
                    f"c={det['circularity']:.2f}  A={det['area_m2'] * 1e4:.0f} cm2",
                )
                # 0.7-scale text is ~230 px wide: centred under the part, kept
                # inside the frame and below the belt region of interest
                # (whose lower edge lies ~105 px under a centred part), so
                # that neither the contour nor the ROI outline is covered.
                x0 = max(8, min(dbg.shape[1] - 240, anchor[0] - 115))
                y0 = min(height - 30, anchor[1] + 132)
                for k, text in enumerate(lines):
                    cv2.putText(dbg, text, (x0, y0 + 24 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                                cv2.LINE_AA)
                    cv2.putText(dbg, text, (x0, y0 + 24 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2,
                                cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_optical_frame
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PartPerceptionNode()
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
