from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Perception stage only: stream shape parts along the conveyor and
    # detect them in the overhead camera image.
    # Requires the Gazebo + MoveIt stack (panda_gazebo_moveit.launch.py,
    # which also publishes the panda_link0 -> camera_link_optical TF).
    # The tracker/Servo are NOT part of this launch: it serves to measure
    # perception accuracy against ground truth without robot motion.
    # Both nodes on sim time, like everything in the Gazebo setup.
    with_stream_arg = DeclareLaunchArgument(
        "with_stream",
        default_value="true",
        description="Also start gazebo_shape_stream_node (set false if the "
                    "tracking launch already runs it)",
    )

    shape_stream_node = Node(
        package="robotic_tracking_system",
        executable="gazebo_shape_stream_node",
        name="gazebo_shape_stream_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("with_stream")),
        parameters=[
            {"use_sim_time": True},
        ],
    )

    perception_node = Node(
        package="robotic_tracking_system",
        executable="part_perception_node",
        name="part_perception_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription(
        [
            with_stream_arg,
            shape_stream_node,
            perception_node,
        ]
    )
