from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import OpaqueFunction


def launch_setup(context, *args, **kwargs):
    return [
        Node(
            package="image_matching",
            executable="simple_matcher_node",
            name="simple_matcher_node",
        )
    ]


def generate_launch_description():
    ld = LaunchDescription(
        [
            OpaqueFunction(function=launch_setup),
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
    return ld
