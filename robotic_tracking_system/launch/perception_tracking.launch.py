from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Perception-based perimeter tracking. The full closed loop
    #   camera -> part_perception_node -> perception_target_estimator
    #   -> moving_shape_perimeter_tracker (target_source=perception)
    #   -> MoveIt Servo
    # gazebo_shape_stream_node moves the parts (it is the plant, not the
    # perception system); the tracker receives NO ground-truth topics -
    # its part pose, velocity and cycle boundaries all derive from the
    # camera. tracking_supervisor keeps the arm posture healthy by
    # re-homing; its cycle counting uses the plant's cycle signal,
    # standing in for a real cell's PLC trigger.
    # Requires the Gazebo + MoveIt stack (panda_gazebo_moveit.launch.py)
    # and Servo (panda_gazebo_servo.launch.py, started) running. All
    # nodes on sim time.
    enable_hold_arg = DeclareLaunchArgument(
        "enable_orientation_hold", default_value="true",
        description="Hold the EE orientation captured at the first path start",
    )
    enable_rehome_arg = DeclareLaunchArgument(
        "enable_rehome", default_value="true",
        description="Run tracking_supervisor (posture management)",
    )
    # Idle-mode posture management: spawn_period 24 s leaves a ~9 s
    # empty-belt gap after each part (worked for ~15 s), and the
    # supervisor re-homes in that gap - NO parts are skipped. N=1 is free
    # here (the gap is idle anyway) and keeps the posture always fresh.
    rehome_cycles_arg = DeclareLaunchArgument(
        "rehome_every_n_cycles", default_value="1",
        description="Tracked cycles between gap re-homes (idle mode: 1 is "
                    "free and keeps posture fresh)",
    )
    spawn_period_arg = DeclareLaunchArgument(
        "spawn_period", default_value="24.0",
        description="Seconds between parts; must leave >= ~8 s of empty "
                    "belt after the ~15 s working window for the gap "
                    "re-home",
    )
    prior_blend_arg = DeclareLaunchArgument(
        "velocity_prior_blend", default_value="0.0",
        description="0 = belt velocity purely from perception, "
                    "1 = calibrated nominal belt velocity",
    )

    # The estimator blends long blind coasts toward the belt prior by
    # default (coast_prior_blend 1.0). A purely camera-derived motion
    # estimate needs BOTH blends at 0.
    coast_blend_arg = DeclareLaunchArgument(
        "coast_prior_blend", default_value="1.0",
        description="0 = coast on the fitted velocity only, 1 = ramp long "
                    "coasts to the nominal belt velocity (default)",
    )

    shape_stream_node = Node(
        package="robotic_tracking_system",
        executable="gazebo_shape_stream_node",
        name="gazebo_shape_stream_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "spawn_period": ParameterValue(
                    LaunchConfiguration("spawn_period"), value_type=float
                ),
            },
        ],
    )

    perception_node = Node(
        package="robotic_tracking_system",
        executable="part_perception_node",
        name="part_perception_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    estimator_node = Node(
        package="robotic_tracking_system",
        executable="perception_target_estimator",
        name="perception_target_estimator",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "velocity_prior_blend": ParameterValue(
                    LaunchConfiguration("velocity_prior_blend"),
                    value_type=float,
                ),
                "coast_prior_blend": ParameterValue(
                    LaunchConfiguration("coast_prior_blend"),
                    value_type=float,
                ),
            },
        ],
    )

    tracker_node = Node(
        package="robotic_tracking_system",
        executable="moving_shape_perimeter_tracker",
        name="moving_shape_perimeter_tracker",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "target_source": "perception",
                "sync_with_stream": False,
                "enable_orientation_hold": ParameterValue(
                    LaunchConfiguration("enable_orientation_hold"),
                    value_type=bool,
                ),
            },
        ],
    )

    supervisor_node = Node(
        package="robotic_tracking_system",
        executable="tracking_supervisor",
        name="tracking_supervisor",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_rehome")),
        parameters=[
            {
                "use_sim_time": True,
                "rehome_mode": "idle",
                "rehome_every_n_cycles": ParameterValue(
                    LaunchConfiguration("rehome_every_n_cycles"),
                    value_type=int,
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            enable_hold_arg,
            enable_rehome_arg,
            rehome_cycles_arg,
            spawn_period_arg,
            prior_blend_arg,
            coast_blend_arg,
            shape_stream_node,
            perception_node,
            estimator_node,
            tracker_node,
            supervisor_node,
        ]
    )
