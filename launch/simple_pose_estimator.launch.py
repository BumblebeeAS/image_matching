from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription(
        [
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
            Node(
                package="image_matching",
                executable="image_brighten_node",
                name="image_brighten_node",
                parameters=[{"brightness_factor": 1.0}],
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
                    ("in", "/auv4/front_cam/image_matching"),
                    ("out/compressed", "/auv4/front_cam/image_matching/compressed"),
                ],
            ),
        ]
    )
