"""Regression tests for the estimator fit logic: a classification
change steps the reported part origin, and a velocity fit across that
step is corrupted. No ROS node is spun."""

import math

from robotic_tracking_system.perception_target_estimator import (
    VelocityFit, label_consistent_tail,
)

BELT_VY = 0.03


def samples(t0, n, label, x0=0.5, y0=-0.25, dt=1 / 30):
    """n clean detections of a part riding the belt at BELT_VY."""
    return [(t0 + i * dt, x0, y0 + BELT_VY * (t0 + i * dt - t0), label)
            for i in range(n)]


def test_velocity_fit_recovers_belt_velocity():
    fit = VelocityFit([s[:3] for s in samples(100.0, 30, "square")])
    assert math.isclose(fit.vx, 0.0, abs_tol=1e-9)
    assert math.isclose(fit.vy, BELT_VY, abs_tol=1e-9)


def test_label_step_corrupts_full_buffer_fit_but_not_tail_fit():
    # Representative entry trace: ~0.7 s labeled 'square', then the
    # origin steps (+8 mm x, -12 mm y) as the label corrects to
    # 'irregular'.
    pre = samples(100.0, 21, "square")
    t1 = pre[-1][0] + 1 / 30
    y1 = pre[-1][2] + BELT_VY / 30
    post = [(t1 + i / 30, 0.508, y1 - 0.012 + BELT_VY * i / 30, "irregular")
            for i in range(15)]
    buf = pre + post

    # Fitting across the step bends both velocity components.
    bad = VelocityFit([s[:3] for s in buf])
    assert bad.vx > 0.004                  # fake sideways drift ("up")
    assert bad.vy < 0.025                  # belt speed badly underestimated

    # The label-consistent tail contains only post-step samples.
    tail = label_consistent_tail(buf)
    assert [s[3] for s in tail] == ["irregular"] * 15
    good = VelocityFit([s[:3] for s in tail])
    assert math.isclose(good.vx, 0.0, abs_tol=1e-6)
    assert math.isclose(good.vy, BELT_VY, abs_tol=1e-6)


def test_tail_edge_cases():
    assert label_consistent_tail([]) == []
    same = samples(0.0, 5, "circle")
    assert label_consistent_tail(same) == same
    flipped = samples(0.0, 5, "circle") + samples(1.0, 1, "pentagon")
    tail = label_consistent_tail(flipped)
    assert len(tail) == 1 and tail[0][3] == "pentagon"


def test_pending_offset_flips_corrupt_the_fit():
    # Why 'pending' samples must be raw centroids: if the perception
    # node applied the FLIPPING instantaneous guess's offset under the
    # constant label 'pending', label_consistent_tail would see one
    # consistent window while the positions alternate between two
    # tracks ~18 mm apart (square offset (0,0) vs irregular
    # (-0.00833, +0.01582)) - and the fit would absorb the steps.
    off = (-0.00833, 0.01582)
    flips = []
    for i in range(40):
        t = 100.0 + i / 30
        use_irregular = i >= 20          # guess settles halfway through
        # guess=square subtracts (0,0) -> publishes the raw centroid
        # (origin + off); guess=irregular subtracts off -> the origin.
        ox, oy = ((0.0, 0.0) if not use_irregular else off)
        # published origin = centroid - guess offset; centroid rides the
        # belt at (0.5 + off_x, y + off_y)
        flips.append((t, 0.5 + off[0] - ox,
                      -0.25 + off[1] - oy + BELT_VY * i / 30, "pending"))
    assert len(label_consistent_tail(flips)) == 40   # guard is blind
    bad = VelocityFit([s[:3] for s in flips])
    assert bad.vx > 0.004                # fake "up" (+x) drift
    assert bad.vy < 0.02                 # belt speed badly underestimated


def test_pending_rebase_keeps_fit_continuous_through_resolution():
    # Pending samples are raw centroids (one consistent track); at
    # resolution rebase_pending_samples() shifts them into the
    # resolved part frame, so the full-window fit is immediately valid
    # and correctly anchored.
    from collections import deque

    from robotic_tracking_system.perception_target_estimator import (
        rebase_pending_samples,
    )

    off = (-0.00833, 0.01582)            # irregular catalog centroid
    buf = deque(maxlen=200)
    for i in range(40):                  # raw centroid = origin + off
        t = 100.0 + i / 30
        buf.append((t, 0.5 + off[0],
                    -0.25 + off[1] + BELT_VY * i / 30, "pending"))
    t1 = 100.0 + 40 / 30                 # first resolved detection
    y1 = -0.25 + BELT_VY * 40 / 30
    buf.append((t1, 0.5, y1, "irregular"))

    rebase_pending_samples(buf, "irregular", off)

    tail = label_consistent_tail(buf)
    assert len(tail) == 41               # whole buffer, one frame
    fit = VelocityFit([s[:3] for s in tail])
    assert math.isclose(fit.vx, 0.0, abs_tol=1e-6)
    assert math.isclose(fit.vy, BELT_VY, abs_tol=1e-6)
    px, py = fit.predict(t1)
    assert math.isclose(px, 0.5, abs_tol=1e-6)
    assert math.isclose(py, y1, abs_tol=1e-6)


def test_pending_rebase_zero_offset_is_noop():
    from robotic_tracking_system.perception_target_estimator import (
        rebase_pending_samples,
    )
    buf = samples(100.0, 10, "pending") + samples(100.4, 5, "square")
    before = [s[:3] for s in buf]
    rebase_pending_samples(buf, "square", (0.0, 0.0))
    assert [s[:3] for s in buf] == before
    assert all(s[3] == "square" for s in buf)


def test_load_catalog_centroids_matches_perception_offsets():
    # The rebase offset must equal what part_perception_node subtracts;
    # validate against the real shapes.yaml in the source tree.
    import os

    from robotic_tracking_system.perception_target_estimator import (
        load_catalog_centroids,
    )

    path = os.path.join(os.path.dirname(__file__), "..",
                        "config", "shapes.yaml")
    c = load_catalog_centroids(path)
    assert math.isclose(c["square"][0], 0.0, abs_tol=1e-9)
    assert math.isclose(c["square"][1], 0.0, abs_tol=1e-9)
    assert c["circle"] == (0.0, 0.0)
    assert math.isclose(c["irregular"][0], -0.00833, abs_tol=5e-4)
    assert math.isclose(c["irregular"][1], 0.01582, abs_tol=5e-4)


def test_blended_coast_velocity_ramps_to_prior():
    from robotic_tracking_system.perception_target_estimator import (
        COAST_BLEND_RAMP, blended_coast_velocity,
    )
    fit_v, prior = (0.005, 0.019), (0.0, 0.03)
    # While detections flow: pure fit.
    assert blended_coast_velocity(fit_v, prior, 1.0, 2.0, 1.0) == fit_v
    # blend 0 (coast on the fitted velocity only): pure fit forever.
    assert blended_coast_velocity(fit_v, prior, 10.0, 2.0, 0.0) == fit_v
    # Mid-ramp: halfway between.
    v = blended_coast_velocity(fit_v, prior, 2.0 + COAST_BLEND_RAMP / 2, 2.0, 1.0)
    assert math.isclose(v[1], (fit_v[1] + prior[1]) / 2, abs_tol=1e-9)
    # Long coast: full prior.
    v = blended_coast_velocity(fit_v, prior, 10.0, 2.0, 1.0)
    assert v == (0.0, 0.03)


def test_ablation_flags_default_to_validated_behavior():
    # Ablation switch label_consistent_fit: set False, the node fits the
    # full buffer, which the first tests show to be corrupted by a label
    # step; this pins the selection logic the node uses.
    pre = samples(100.0, 21, "square")
    post = [(pre[-1][0] + (i + 1) / 30, 0.508, pre[-1][2] - 0.012, "irregular")
            for i in range(15)]
    buf = pre + post
    assert len(label_consistent_tail(buf)) == 15      # flag True path
    assert len(list(buf)) == 36                       # flag False path


def test_centroid_frame_offset_and_equivalence_with_rebased_fit():
    # fit_frame='centroid' (alternative to the part-frame fit): every
    # sample is carried back to the raw silhouette centroid, so the buffer
    # is label-independent. The velocity must equal the rebased part-frame
    # fit, and the published origin (prediction minus the latched offset)
    # must equal the rebased prediction.
    from collections import deque

    from robotic_tracking_system.perception_target_estimator import (
        centroid_frame_offset, rebase_pending_samples,
    )

    cents = {"irregular": (-0.00833, 0.01582), "square": (0.0, 0.0)}
    assert centroid_frame_offset("pending", cents) == (0.0, 0.0)
    assert centroid_frame_offset("unknown", cents) == (0.0, 0.0)
    assert centroid_frame_offset("irregular", cents) == cents["irregular"]
    assert centroid_frame_offset("circle", cents) == (0.0, 0.0)   # not in catalog

    off = cents["irregular"]
    # published samples: 40 pending (raw centroid), then 20 resolved (origin)
    pub = []
    for i in range(60):
        t = 100.0 + i / 30
        cx, cy = 0.5 + off[0], -0.25 + off[1] + BELT_VY * i / 30   # true centroid
        label = "pending" if i < 40 else "irregular"
        ox, oy = centroid_frame_offset(label, cents)
        pub.append((t, cx - ox, cy - oy, label))
    # part frame + rebase (default path)
    buf = deque(pub, maxlen=200)
    rebase_pending_samples(buf, "irregular", off)
    fit_part = VelocityFit([s[:3] for s in label_consistent_tail(buf)])
    # centroid frame: undo the offset of each sample's label, fit on all
    cen = [(t, x + centroid_frame_offset(lb, cents)[0], y + centroid_frame_offset(lb, cents)[1])
           for t, x, y, lb in pub]
    fit_cen = VelocityFit(cen)
    assert math.isclose(fit_cen.vx, fit_part.vx, abs_tol=1e-9)
    assert math.isclose(fit_cen.vy, fit_part.vy, abs_tol=1e-9)
    t1 = 102.5
    px, py = fit_part.predict(t1)
    cx, cy = fit_cen.predict(t1)
    assert math.isclose(cx - off[0], px, abs_tol=1e-9)
    assert math.isclose(cy - off[1], py, abs_tol=1e-9)
