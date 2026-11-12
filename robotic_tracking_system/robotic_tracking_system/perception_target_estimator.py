"""Estimate a continuous part state from camera detections.

Node name:      perception_target_estimator
Purpose:        The target-estimation stage between vision and the
                tracking controller. Raw detections from
                part_perception_node are intermittent
                and become unreliable once the tracking arm reaches over
                the part (occlusion / merged silhouettes). This node
                turns them into a continuous, always-current part state:
                it accepts only classified detections that pass a
                validation gate, fits a constant-velocity trajectory
                x(t), y(t) by least squares over the accepted buffer,
                latches the shape label once confident, and publishes the
                extrapolated state at a fixed rate - coasting on the
                fitted (optionally prior-blended) velocity while the arm
                occludes the part. NO ground-truth topics are consumed:
                everything derives from /perception/*.
Subscriptions:  /perception/part_pose (geometry_msgs/PoseStamped,
                    image-stamped part-frame origin)
                /perception/part_shape (std_msgs/String, paired with the
                    pose by the perception node)
Publications:   /perception/part_state (geometry_msgs/PoseStamped,
                    stamped now, extrapolated part origin)
                /perception/part_state_velocity
                    (geometry_msgs/Vector3Stamped, the velocity used for
                    extrapolation)
                /perception/cycle_start (std_msgs/String = latched shape
                    name, once per new part when the estimate becomes
                    confident) - the perception-mode replacement for the
                    ground-truth /part_stream/cycle_start
Parameters:     publish_rate (50 Hz), min_detections (15),
                min_fit_span (0.3 s), buffer_size (200),
                shapes_file ('' = package share config/shapes.yaml -
                catalog centroid offsets used to rebase the 'pending'
                entry samples when the shape resolves),
                validation_gate (0.06 m - reject detections farther than
                this from the fit prediction once confident),
                new_part_y_max (-0.10 m - a detection this far upstream
                with no active estimate, or jumping back by
                new_part_jump (0.15 m), starts a new part),
                detection_timeout (2.0 s - accepted-detection gap that
                re-arms new-part acceptance),
                coast_time_limit (15 s - hard bound on coasting: with no
                accepted detection for this long the state is suspended
                so the tracker waits and the supervisor re-homes;
                this breaks the occlusion deadlock in which the arm,
                chasing a drifting estimate, keeps blocking the camera),
                velocity_prior [vx, vy] (nominal belt velocity),
                velocity_prior_blend (0.0 = pure measured velocity,
                1.0 = pure prior; the prior models a calibrated belt
                speed as available in a real plant),
                label_consistent_fit / rebase_pending (True; ablation
                switches for comparison runs),
                fit_frame ('part' = fit on the published part-frame origin
                with the label-consistent window and the pending rebase;
                'centroid' = fit on the label-independent raw silhouette
                centroid and apply the catalog offset of the latched label
                only when the state is published; both yield the same
                velocity, see fit_frame below)
Frames:         panda_link0 (all perception outputs already are)
Update rate:    publish_rate timer, sim time
Dependencies:   part_perception_node; run with use_sim_time:=true
Limitations:    constant-velocity part model (matches the conveyor);
                shape latched at confidence time (the perception label
                is vote-stabilized; a wrong early latch is corrected
                through relatch_detections); yaw not estimated
"""

import os
from collections import deque

import yaml

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import String


def stamp_to_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def polygon_centroid(points):
    """Area-weighted centroid of an (N, 2) polygon (pure-python copy of
    part_perception_node.polygon_centroid - importing that module would
    pull in cv2/cv_bridge)."""
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
        return (0.0, 0.0)
    return (cx / (6.0 * a), cy / (6.0 * a))


def load_catalog_centroids(path):
    """shape name -> (ox, oy) catalog centroid offset from shapes.yaml.

    Must equal the offsets part_perception_node subtracts from the
    detected centroid (circle = (0, 0), polygon = area-weighted vertex
    centroid), because rebase_pending_samples() applies exactly this
    offset to raw-centroid 'pending' samples.
    """
    with open(path, "r") as f:
        catalog = yaml.safe_load(f)
    centroids = {}
    for name, spec in catalog["shapes"].items():
        if spec["type"] == "circle":
            centroids[name] = (0.0, 0.0)
        elif spec["type"] == "polygon":
            verts = [(float(x), float(y)) for x, y in spec["vertices"]]
            centroids[name] = polygon_centroid(verts)
    return centroids


def rebase_pending_samples(buffer, resolved_label, offset):
    """Rewrite 'pending'-labeled samples into the resolved part frame.

    While the perception vote is empty the published pose is the RAW
    contour centroid under the constant label 'pending' (applying the
    offset of the flipping instantaneous guess instead would make the
    samples bimodal and corrupt the velocity fit).
    When the label resolves, the origin definition changes to
    'centroid - catalog centroid offset of the resolved shape'. Shifting
    the buffered pending samples by that same offset and relabeling them
    puts the WHOLE buffer in one consistent frame, so the very next fit
    is anchored correctly over the full window. Without the rebase the
    state coasts on a stale pending fit through the arm occlusion right
    after resolution (a +19/-36 mm state excursion over ~4 s).
    """
    ox, oy = offset
    for i, (t, x, y, label) in enumerate(buffer):
        if label == "pending":
            buffer[i] = (t, x - ox, y - oy, resolved_label)


def centroid_frame_offset(label, centroids):
    """Offset that carries a published part-frame origin back to the raw
    silhouette centroid: part_perception_node publishes 'centroid - catalog
    offset of the label', except for 'pending' / 'unknown', which are raw
    centroids already (offset zero)."""
    if label in ("pending", "unknown"):
        return (0.0, 0.0)
    return tuple(centroids.get(label, (0.0, 0.0)))


def label_consistent_tail(buffer):
    """Longest suffix of (t, x, y, label) samples sharing one label.

    The part-frame origin published by part_perception_node is
    'centroid - catalog centroid offset for the CURRENT label', so a
    classification change (e.g. the entry-region square -> irregular
    correction) STEPS the reported origin. Fitting a velocity across
    that step corrupts it (a step of 8-12 mm inside a 1.2 s window
    gives a fitted velocity of (+9.5, +9.5) mm/s instead of (0, +30)
    mm/s) - only samples sharing the current label may enter one fit.
    """
    if not buffer:
        return []
    label = buffer[-1][3]
    for i in range(len(buffer) - 1, -1, -1):
        if buffer[i][3] != label:
            return list(buffer)[i + 1:]
    return list(buffer)


COAST_BLEND_RAMP = 2.0   # s, see blended_coast_velocity


def blended_coast_velocity(fit_v, prior, coast_age, detection_timeout, blend):
    """Velocity used for extrapolation while blind.

    While detections flow (coast_age <= detection_timeout) the fitted
    velocity is trusted as-is. Once coasting longer than that, the best
    model of the part is the calibrated belt (velocity_prior): a fit
    made from a short, parallax-biased entry window (the irregular
    shape's ~1 s window can be 35% slow with a sideways component)
    would otherwise integrate its bias over the whole coast.
    blend = 0 keeps the pure fit, 1 uses the prior fully during long
    coasts. The blend ramps in over
    COAST_BLEND_RAMP seconds past detection_timeout so the published
    position stays continuous (predict() recomputes the trajectory
    from the fit mean, so a velocity step would be a position step).
    """
    if coast_age <= detection_timeout or blend <= 0.0:
        return fit_v
    b = min(1.0, blend) * min(
        1.0, (coast_age - detection_timeout) / COAST_BLEND_RAMP)
    return ((1.0 - b) * fit_v[0] + b * prior[0],
            (1.0 - b) * fit_v[1] + b * prior[1])


class VelocityFit:
    """Least-squares constant-velocity fit over (t, x, y) samples."""

    def __init__(self, samples):
        n = len(samples)
        ts = [s[0] for s in samples]
        self.t_mean = sum(ts) / n
        self.x_mean = sum(s[1] for s in samples) / n
        self.y_mean = sum(s[2] for s in samples) / n
        stt = sum((t - self.t_mean) ** 2 for t in ts)
        if stt < 1e-9:
            self.vx = self.vy = 0.0
        else:
            self.vx = sum((s[0] - self.t_mean) * (s[1] - self.x_mean)
                          for s in samples) / stt
            self.vy = sum((s[0] - self.t_mean) * (s[2] - self.y_mean)
                          for s in samples) / stt

    def predict(self, t, vx=None, vy=None):
        vx = self.vx if vx is None else vx
        vy = self.vy if vy is None else vy
        dt = t - self.t_mean
        return (self.x_mean + vx * dt, self.y_mean + vy * dt)


class PerceptionTargetEstimator(Node):
    def __init__(self):
        super().__init__("perception_target_estimator")

        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("shapes_file", "")
        self.declare_parameter("min_detections", 15)
        self.declare_parameter("min_fit_span", 0.3)
        self.declare_parameter("buffer_size", 200)
        self.declare_parameter("validation_gate", 0.06)
        self.declare_parameter("new_part_y_max", -0.10)
        self.declare_parameter("new_part_jump", 0.15)
        self.declare_parameter("detection_timeout", 2.0)
        self.declare_parameter("velocity_prior", [0.0, 0.03])
        self.declare_parameter("velocity_prior_blend", 0.0)
        # The shape is latched at confidence time, which happens at the
        # belt entry - the most oblique view, where side-wall parallax
        # can distort the first frames' contour (e.g. the irregular part
        # latched as a square at y = -0.215). If the perception label
        # disagrees with the latch for this many
        # consecutive accepted detections, re-latch and re-announce the
        # cycle so the tracker switches to the right perimeter while the
        # part is still early in its pass. 0 = never re-latch.
        self.declare_parameter("relatch_detections", 10)
        # Stop publishing once the estimate coasts to the edge of the
        # validated working zone: the tracker then waits (zero commands)
        # instead of stretching after a part it can no longer usefully
        # work. Two constraints set the default: the plant removes the
        # part at y 0.23 (with a 0.32 limit the arm chases a phantom
        # into a singularity deceleration), and perception can lock a
        # part as late as mid-belt (~y -0.14), which shifts the perimeter
        # path phase so that following to 0.24 still grazes the soft
        # singularity at full extension - 0.20 keeps the reach inside
        # the healthy envelope.
        self.declare_parameter("coast_limit_y", 0.20)
        # Hard time bound on coasting: with no accepted detection for
        # this long, stop publishing the state entirely. A healthy
        # occlusion coast (lock loss ~y -0.1 to coast_limit_y 0.20 at
        # the real belt speed >= 28 mm/s) lasts <= 11 s; a corrupted
        # slow-velocity fit can otherwise coast for 22-45 s with the
        # tracker chasing it and the arm blocking the camera - the
        # occlusion deadlock this bound breaks: once the state stops,
        # the tracker zeroes commands, the supervisor
        # re-homes in the resulting gap, and the next part is
        # reacquired cleanly.
        self.declare_parameter("coast_time_limit", 15.0)
        # Once coasting longer than detection_timeout, blend the
        # extrapolation velocity toward velocity_prior (the calibrated
        # belt). 1.0 = belt velocity during long coasts (default; the
        # entry-window fit bias then does not integrate over the coast);
        # 0.0 = pure fitted velocity.
        self.declare_parameter("coast_prior_blend", 1.0)
        # Ablation switches for comparison runs. Both True is the
        # default behavior; False disables the label-consistent fit
        # window / the pending rebase.
        self.declare_parameter("label_consistent_fit", True)
        self.declare_parameter("rebase_pending", True)
        # Frame of the fit buffer. 'part' (default): the
        # buffer holds the published part-frame origin, so a label change
        # steps the samples and the two measures above keep the window
        # consistent. 'centroid': every sample is carried back to the raw
        # silhouette centroid (offset of its label added), the fit and the
        # gate work in that label-independent frame, and the catalog
        # offset of the LATCHED label is subtracted only when the state is
        # published. In that frame a label change moves nothing, so the
        # label-consistent window and the pending rebase are no-ops (both
        # are skipped). Velocity is identical by construction; the
        # published position differs only in HOW the offset is applied.
        self.declare_parameter("fit_frame", "part")

        gp = lambda name: self.get_parameter(name).value
        self.publish_rate = float(gp("publish_rate"))
        self.min_detections = int(gp("min_detections"))
        self.min_fit_span = float(gp("min_fit_span"))
        self.buffer_size = int(gp("buffer_size"))
        self.validation_gate = float(gp("validation_gate"))
        self.new_part_y_max = float(gp("new_part_y_max"))
        self.new_part_jump = float(gp("new_part_jump"))
        self.detection_timeout = float(gp("detection_timeout"))
        self.velocity_prior = [float(v) for v in gp("velocity_prior")]
        self.velocity_prior_blend = float(gp("velocity_prior_blend"))
        self.coast_limit_y = float(gp("coast_limit_y"))
        self.coast_time_limit = float(gp("coast_time_limit"))
        self.coast_prior_blend = float(gp("coast_prior_blend"))
        self.relatch_detections = int(gp("relatch_detections"))
        self.label_consistent_fit = bool(gp("label_consistent_fit"))
        self.rebase_pending = bool(gp("rebase_pending"))
        self.fit_frame = str(gp("fit_frame"))
        if self.fit_frame not in ("part", "centroid"):
            raise RuntimeError(f"Unknown fit_frame '{self.fit_frame}'")

        # Catalog centroid offsets, needed to rebase 'pending' entry
        # samples when the shape resolves (same source as the perception
        # node / tracker / stream: shapes.yaml).
        shapes_file = str(gp("shapes_file"))
        if not shapes_file:
            from ament_index_python.packages import get_package_share_directory
            shapes_file = os.path.join(
                get_package_share_directory("robotic_tracking_system"),
                "config", "shapes.yaml",
            )
        try:
            self.catalog_centroids = load_catalog_centroids(shapes_file)
        except (OSError, KeyError, ValueError, yaml.YAMLError) as e:
            # Degraded but functional: resolution then proceeds without
            # a rebase (offset (0, 0)).
            self.get_logger().warn(
                f"Could not load shape catalog '{shapes_file}' ({e}); "
                "pending samples will not be rebased at resolution"
            )
            self.catalog_centroids = {}

        # Runtime tuning (operator dashboard): applied immediately.
        self.add_on_set_parameters_callback(self.on_parameters_changed)

        self.buffer = deque(maxlen=self.buffer_size)
        self.fit = None
        self.confident = False
        self.latched_shape = None
        self.latest_label = "unknown"
        self.last_accepted_t = None
        self.rejected_count = 0
        self.label_mismatch_count = 0

        self.state_pub = self.create_publisher(PoseStamped, "/perception/part_state", 10)
        self.vel_pub = self.create_publisher(
            Vector3Stamped, "/perception/part_state_velocity", 10
        )
        self.cycle_pub = self.create_publisher(String, "/perception/cycle_start", 10)

        self.create_subscription(String, "/perception/part_shape", self.shape_callback, 10)
        self.create_subscription(PoseStamped, "/perception/part_pose", self.pose_callback, 10)

        self.create_timer(1.0 / self.publish_rate, self.publish_state)

        self.get_logger().info(
            "Perception target estimator started "
            f"(min_detections {self.min_detections}, gate "
            f"{self.validation_gate} m, velocity_prior_blend "
            f"{self.velocity_prior_blend}, fit_frame {self.fit_frame})"
        )

    def on_parameters_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        tunable = {"velocity_prior_blend", "coast_limit_y",
                   "coast_time_limit", "coast_prior_blend",
                   "validation_gate", "relatch_detections",
                   "velocity_prior"}
        for p in params:
            if p.name not in tunable:
                continue
            if p.name == "relatch_detections":
                self.relatch_detections = int(p.value)
            elif p.name == "velocity_prior":
                self.velocity_prior = [float(v) for v in p.value]
            else:
                setattr(self, p.name, float(p.value))
            self.get_logger().info(f"Parameter {p.name} = {p.value}")
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Detection intake
    # ------------------------------------------------------------------

    def shape_callback(self, msg):
        self.latest_label = msg.data

    def used_velocity(self):
        b = self.velocity_prior_blend
        return (
            (1.0 - b) * self.fit.vx + b * self.velocity_prior[0],
            (1.0 - b) * self.fit.vy + b * self.velocity_prior[1],
        )

    def pose_callback(self, msg):
        t = stamp_to_sec(msg.header)
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Unclassified detections are the arm or a merged arm+part
        # silhouette (arm occlusion) - never trust their pose.
        if self.latest_label == "unknown":
            self.rejected_count += 1
            return

        x_raw, y_raw = x, y
        if self.fit_frame == "centroid":
            # Work in the label-independent raw-centroid frame: undo the
            # catalog offset the perception node applied for this label.
            ox, oy = centroid_frame_offset(self.latest_label, self.catalog_centroids)
            x, y = x + ox, y + oy

        gap = (self.last_accepted_t is None
               or t - self.last_accepted_t > self.detection_timeout)

        if self.confident:
            vx, vy = self.used_velocity()
            px, py = self.fit.predict(t, vx, vy)
            residual = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            jumped_back = y < py - self.new_part_jump
            if jumped_back and y <= self.new_part_y_max:
                if self.fit_frame == "centroid":
                    # The label topic may still carry the previous part's
                    # label when the first pose of the next part arrives;
                    # the perception node publishes an entering part with
                    # its raw centroid ('pending'), so the raw values are
                    # the centroid-frame sample. Converting them with the
                    # stale label would put one sample 13-18 mm off at the
                    # start of the buffer and bias the fit of every part
                    # that follows a non-centred one (2.5-2.9 mm estimator
                    # error on such parts).
                    x, y = x_raw, y_raw
                self.start_new_part(t, x, y, "position jump upstream")
                return
            if residual > self.validation_gate:
                self.rejected_count += 1
                self.get_logger().debug(
                    f"Detection rejected by validation gate "
                    f"({residual * 1000:.0f} mm from prediction)"
                )
                return
        elif gap and y > self.new_part_y_max and not self.buffer:
            # No active estimate and the detection is not at the belt
            # start: likely the arm sweeping over the ROI - ignore.
            self.rejected_count += 1
            return
        elif gap and self.buffer:
            # Not confident, a detection after a gap, and old samples in
            # the buffer: a new part while no state is active. In the part
            # frame the label-consistent suffix already separates the parts
            # (every part enters as 'pending'); in the centroid frame
            # nothing does, so drop the old samples explicitly.
            self.get_logger().info("New part detected (detection gap, no active state)")
            self.buffer.clear()
            self.fit = None
            self.latched_shape = None
            self.label_mismatch_count = 0
            if self.fit_frame == "centroid":
                x, y = x_raw, y_raw       # entering part: raw centroid

        self.accept(t, x, y)

    def start_new_part(self, t, x, y, reason):
        self.get_logger().info(f"New part detected ({reason})")
        self.buffer.clear()
        self.fit = None
        self.confident = False
        self.latched_shape = None
        self.label_mismatch_count = 0
        self.accept(t, x, y)

    def accept(self, t, x, y):
        self.buffer.append((t, x, y, self.latest_label))
        self.last_accepted_t = t

        # 'pending' latch (entry region, before the perception vote has
        # content): upgrade to the first real vote-stabilized label
        # IMMEDIATELY - that label is already trusted, no mismatch count
        # needed. Until then the tracker follows the shape-agnostic
        # 'pending' path.
        if (self.confident and self.latched_shape == "pending"
                and self.latest_label not in ("unknown", "pending")):
            self.get_logger().info(
                f"Shape resolved: pending -> '{self.latest_label}'"
            )
            self.latched_shape = self.latest_label
            self.label_mismatch_count = 0
            # The pending samples are raw centroids; shift them into the
            # resolved part frame so the fit below is immediately valid
            # over the full buffer (see rebase_pending_samples). In the
            # centroid frame every sample already is a raw centroid.
            if self.rebase_pending and self.fit_frame == "part":
                rebase_pending_samples(
                    self.buffer, self.latched_shape,
                    self.catalog_centroids.get(
                        self.latched_shape, (0.0, 0.0)),
                )
            msg = String()
            msg.data = self.latched_shape
            self.cycle_pub.publish(msg)
        # Re-latch a wrong early shape once the (vote-stabilized)
        # perception label has settled on something else.
        elif (self.confident and self.relatch_detections > 0
                and self.latched_shape is not None
                and self.latest_label not in ("unknown", self.latched_shape)):
            self.label_mismatch_count += 1
            if self.label_mismatch_count >= self.relatch_detections:
                self.get_logger().warn(
                    f"Shape re-latched: '{self.latched_shape}' -> "
                    f"'{self.latest_label}' (early oblique-view "
                    "classification corrected)"
                )
                self.latched_shape = self.latest_label
                self.label_mismatch_count = 0
                msg = String()
                msg.data = self.latched_shape
                self.cycle_pub.publish(msg)
        else:
            self.label_mismatch_count = 0

        # Fit ONLY over the label-consistent suffix: a classification
        # change steps the reported origin (different catalog centroid
        # offset), and a fit across the step is corrupted. Until the
        # new-label suffix is long enough, the previous (internally
        # consistent) fit stays in force.
        tail = (label_consistent_tail(self.buffer)
                if self.label_consistent_fit and self.fit_frame == "part"
                else list(self.buffer))
        if len(tail) < self.min_detections:
            return
        if tail[-1][0] - tail[0][0] < self.min_fit_span:
            return
        self.fit = VelocityFit(tail)
        if not self.confident:
            self.confident = True
            self.latched_shape = self.latest_label
            msg = String()
            msg.data = self.latched_shape
            self.cycle_pub.publish(msg)
            vx, vy = self.used_velocity()
            self.get_logger().info(
                f"Confident: part '{self.latched_shape}' at "
                f"({tail[-1][1]:.3f}, {tail[-1][2]:.3f}), "
                f"velocity ({vx:.4f}, {vy:.4f}) m/s from "
                f"{len(tail)} detections"
            )

    # ------------------------------------------------------------------
    # State output
    # ------------------------------------------------------------------

    def publish_state(self):
        if not self.confident or self.fit is None:
            return
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if (self.last_accepted_t is not None
                and t - self.last_accepted_t > self.coast_time_limit):
            # Coasting this long means the fit can no longer be trusted
            # (occlusion deadlock: the tracker chasing a ghost keeps the
            # arm over the camera's view). Suspend the state so the
            # tracker waits and the supervisor re-homes in the gap.
            self.get_logger().warn(
                f"Coast limit reached ({self.coast_time_limit:.0f} s "
                "without an accepted detection) - part state suspended "
                "until fresh detections",
                throttle_duration_sec=10.0,
            )
            return
        vx, vy = blended_coast_velocity(
            self.used_velocity(), self.velocity_prior,
            t - self.last_accepted_t if self.last_accepted_t else 0.0,
            self.detection_timeout, self.coast_prior_blend)
        x, y = self.fit.predict(t, vx, vy)
        if self.fit_frame == "centroid":
            # Fit lives in the raw-centroid frame; publish the part-frame
            # origin of the latched label (pending -> raw centroid, as the
            # perception node does).
            ox, oy = centroid_frame_offset(self.latched_shape, self.catalog_centroids)
            x, y = x - ox, y - oy
        if y > self.coast_limit_y:
            return  # part has left the belt; wait for the next one

        state = PoseStamped()
        state.header.stamp = now.to_msg()
        state.header.frame_id = "panda_link0"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.45  # part-top plane (perception output)
        state.pose.orientation.w = 1.0
        self.state_pub.publish(state)

        vel = Vector3Stamped()
        vel.header = state.header
        vel.vector.x = vx
        vel.vector.y = vy
        self.vel_pub.publish(vel)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionTargetEstimator()
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
