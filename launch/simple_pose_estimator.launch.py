from launch_ros.actions import Node, PushRosNamespace

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_name",
                default_value="front_cam",
            ),
            PushRosNamespace(["/auv4/", LaunchConfiguration("camera_name")]),
            Node(
                package="image_processing",
                executable="image_brighten_node",
                name="image_brighten_node",
                parameters=[
                    {
                        "brightness_factor": 1.0,
                        "input_compressed_image_topic": "color/image/compressed",
                    }
                ],
            ),
            Node(
                package="image_matching",
                executable="simple_matcher_node",
                name="simple_matcher_node",
            ),
            Node(
                package="image_matching",
                executable="simple_pose_estimator_node",
                name="simple_pose_estimator_node",
                parameters=[{"camera_info_topic": "color/camera_info"}],
            ),
            # Node(
            #     package="robot_localization",
            #     executable="ukf_node",
            #     name="ukf_se",
            #     output="screen",
            #     parameters=[
            #         PathJoinSubstitution(
            #             [
            #                 FindPackageShare("image_matching"),
            #                 "cfg",
            #                 "image_matching_ukf.yaml",
            #             ]
            #         )
            #     ],
            #     # remappings=[
            #     #     ("odometry/filtered", LaunchConfiguration("odom_ukf")),
            #     #     ("set_pose", LaunchConfiguration("reset_pose_ukf")),
            #     # ],
            # ),
            Node(
                package="image_transport",
                executable="republish",
                name="image_republisher",
                arguments=["raw", "compressed"],
                output="screen",
                parameters=[{"out.jpeg_quality": 30}],
                remappings=[
                    ("in", "image_matching/image"),
                    ("out/compressed", "image_matching/image/compressed"),
                ],
            ),
        ]
    )
