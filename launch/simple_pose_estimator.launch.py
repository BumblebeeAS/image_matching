from launch_ros.actions import Node, PushRosNamespace

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription(
        [
            PushRosNamespace("/auv4/front_cam"),
            DeclareLaunchArgument(
                "camera_name",
                default_value="front_cam",
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
