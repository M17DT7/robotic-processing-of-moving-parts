import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from ament_index_python.packages import get_package_share_directory

from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():
    package_name = "robotic_tracking_system"
    package_share = get_package_share_directory(package_name)

    # World file name inside the package's worlds/ directory. The default
    # is the noise-free scene; conveyor_world_noise02.world and
    # conveyor_world_noise05.world add Gaussian camera noise.
    world_arg = DeclareLaunchArgument(
        "world", default_value="conveyor_world.world",
        description="World file name under share/robotic_tracking_system/worlds",
    )
    world_file = PathJoinSubstitution(
        [package_share, "worlds", LaunchConfiguration("world")]
    )

    xacro_file = os.path.join(
        package_share,
        "urdf",
        "panda_gazebo_control.urdf.xacro",
    )

    # Gazebo-ready Panda robot description.
    # It includes:
    # - mesh fixes
    # - inertias
    # - fixed base/world joint
    # - gazebo_ros2_control plugin
    # - arm + hand ros2_control joints
    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", xacro_file]), value_type=str)
    }

    # The official MoveIt Panda config provides the SRDF, kinematics,
    # joint limits, planning pipelines, etc.
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .to_moveit_configs()
    )

    # MoveIt controller configuration of this package (arm only, same
    # layout as the official MoveIt Panda file).
    moveit_controllers = load_yaml(
        package_name,
        "config/moveit_controllers.yaml",
    )

    # Combine MoveIt parameters.
    # Important:
    # robot_description is overridden with the Gazebo-ready URDF above;
    # the URDF of the MoveIt config targets mock hardware, not Gazebo.
    move_group_params = moveit_config.to_dict()
    move_group_params.update(robot_description)
    move_group_params.update(moveit_controllers)
    move_group_params.update({"use_sim_time": True})

    # gui:=false starts only gzserver (headless experiment runs); the
    # camera sensor still renders.
    gui_arg = DeclareLaunchArgument(
        "gui", default_value="true",
        description="Start the Gazebo client window (false = gzserver only)",
    )
    # /clock publication rate of gazebo_ros_init (Hz). The PLUGIN default
    # is 10 Hz, which makes every sim-time timer in the stack tick on a
    # 0.1 s grid (the tracker then issues its commands in pairs every
    # 0.1 s, although its timer is set to 100 Hz). The default here is
    # 100 Hz, the controller's nominal rate; clock_rate:=10.0 selects the
    # plugin default.
    # Note the long form "--param": gzserver's own "-p" means "play a log
    # file".
    clock_arg = DeclareLaunchArgument(
        "clock_rate", default_value="100.0",
        description="Gazebo /clock publish rate [Hz] (plugin default 10)",
    )
    gazebo_args = [
        "--verbose",
        "-s", "libgazebo_ros_init.so",
        "-s", "libgazebo_ros_factory.so",
        world_file,
        "--ros-args", "--param",
        ["publish_rate:=", LaunchConfiguration("clock_rate")],
    ]
    gazebo = ExecuteProcess(
        cmd=["gazebo"] + gazebo_args,
        output="screen",
        condition=IfCondition(LaunchConfiguration("gui")),
    )
    gazebo_headless = ExecuteProcess(
        cmd=["gzserver"] + gazebo_args,
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("gui")),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            robot_description,
            {"use_sim_time": True},
        ],
        output="screen",
    )

    spawn_panda = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "panda",
            # Spawn at the world origin: panda_link0 then coincides with the
            # Gazebo world frame, so the conveyor/part poses in
            # conveyor_world.world match the tracker frames (and the
            # camera extrinsics) directly.
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
                    "--controller-manager",
                    "/controller_manager",
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
                    "--controller-manager",
                    "/controller_manager",
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
                    "--controller-manager",
                    "/controller_manager",
                ],
                output="screen",
            )
        ],
    )

    # Move the arm from the spawn pose (all joints ~0, which is singular and
    # outside the real Panda joint limits) to the 'ready' pose so MoveIt
    # Servo can operate. Runs once after panda_arm_controller is active.
    home_arm = TimerAction(
        period=7.0,
        actions=[
            Node(
                package="robotic_tracking_system",
                executable="home_arm_node",
                name="home_arm_node",
                parameters=[{"use_sim_time": True}],
                output="screen",
            )
        ],
    )

    # Static TF for the overhead camera's optical frame (REP-103: z
    # forward/down, x right = world +Y so the belt runs horizontally in
    # the image). Must stay consistent with the overhead_camera pose in
    # worlds/conveyor_world.world (0.50, 0.0, 1.20 looking straight
    # down). gazebo_ros_camera stamps images with this frame but does
    # not publish TF itself.
    camera_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_frame_tf",
        arguments=[
            "--x", "0.50", "--y", "0.0", "--z", "1.20",
            "--roll", "3.14159265", "--pitch", "0", "--yaw", "1.57079633",
            "--frame-id", "panda_link0",
            "--child-frame-id", "camera_link_optical",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    move_group_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[move_group_params],
            )
        ],
    )

    return LaunchDescription(
        [
            world_arg,
            gui_arg,
            clock_arg,
            gazebo,
            gazebo_headless,
            robot_state_publisher,
            spawn_panda,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
            home_arm,
            camera_static_tf,
            move_group_node,
        ]
    )