from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Orientation hold in the tracker, switchable per run for A/B
    # comparison (without it the EE orientation drifts steadily).
    enable_hold_arg = DeclareLaunchArgument(
        "enable_orientation_hold",
        default_value="true",
        description="Hold the EE orientation captured at the first path start",
    )
    # Periodic RETURN_HOME between part cycles bounds the null-space
    # posture drift: continuous multi-cycle tracking without re-homing
    # gradually moves the arm toward singular configurations.
    # enable_rehome:=false launches no supervisor (the tracker is enabled
    # by default and then runs every cycle).
    enable_rehome_arg = DeclareLaunchArgument(
        "enable_rehome",
        default_value="true",
        description="Run tracking_supervisor: skip one part every N tracked "
                    "cycles and re-home the arm",
    )
    rehome_cycles_arg = DeclareLaunchArgument(
        "rehome_every_n_cycles",
        default_value="3",
        description="Tracked cycles between re-homes (keep N+1 coprime with "
                    "the shape-sequence length so the skipped shape rotates)",
    )
    # Multi-shape perimeter tracking:
    #   - gazebo_shape_stream_node streams parts of varying shape (from
    #     config/shapes.yaml) along the conveyor in Gazebo and announces
    #     each cycle on /part_stream/cycle_start
    #   - moving_shape_perimeter_tracker follows each part's perimeter
    #     (corners are exact path points), phase-locked to the stream
    # Both need use_sim_time: Servo staleness and the shared cycle timing
    # run on the Gazebo clock.
    shape_stream_node = Node(
        package="robotic_tracking_system",
        executable="gazebo_shape_stream_node",
        name="gazebo_shape_stream_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
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
            shape_stream_node,
            tracker_node,
            supervisor_node,
        ]
    )
