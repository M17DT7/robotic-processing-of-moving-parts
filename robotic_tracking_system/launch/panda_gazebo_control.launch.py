import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_name = "robotic_tracking_system"
    package_share = get_package_share_directory(package_name)

    world_file = os.path.join(package_share, "worlds", "conveyor_world.world")
    xacro_file = os.path.join(package_share, "urdf", "panda_gazebo_control.urdf.xacro")

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", xacro_file]), value_type=str)
    }

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo",
            "--verbose",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            world_file,
        ],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[robot_description],
        output="screen",
    )

    spawn_panda = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "panda",
            # World origin: keeps panda_link0 aligned with the Gazebo world
            # frame (same as panda_gazebo_moveit.launch.py).
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.0",
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "joint_state_broadcaster",
                    "--controller-manager", "/controller_manager",
                ],
                output="screen",
            )
        ],
    )

    panda_arm_controller_spawner = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "panda_arm_controller",
                    "--controller-manager", "/controller_manager",
                ],
                output="screen",
            )
        ],
    )


    panda_hand_controller_spawner = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "panda_hand_controller",
                    "--controller-manager", "/controller_manager",
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            gazebo,
            robot_state_publisher,
            spawn_panda,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
        ]
    )