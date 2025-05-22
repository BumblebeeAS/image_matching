from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name', default='front_cam')
        
    return LaunchDescription(
        DeclareLaunchArgument(
            'camera_name',
            default_value='front_cam',
        ),

        [
            Node(
                package="image_matching",
                executable="simple_matcher_node",
                name="simple_matcher_node",
                parameters=[{'camera_name': 'camera_name'}]
            ),
            Node(
                package="image_matching",
                executable="simple_pose_estimator_node",
                name="simple_pose_estimator_node",
                parameters=[{'camera_name': 'camera_name'}]
            ),
            Node(
                package="robot_localization",
                executable="ukf_node",
                name="ukf_se",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("image_matching"),
                            "cfg",
                            "image_matching_ukf.yaml",
                        ]
                    )
                ],
                # remappings=[
                #     ("odometry/filtered", LaunchConfiguration("odom_ukf")),
                #     ("set_pose", LaunchConfiguration("reset_pose_ukf")),
                # ],
            ),
            Node(
                package="image_transport",
                executable="republish",
                name="image_republisher",
                arguments=["raw", "compressed"],
                output="screen",
                parameters=[{"out.jpeg_quality": 30}],
                remappings=[
                    ("in", PathJoinSubstitution(['/auv4', camera_name, 'image_matching'])),
                    ("out/compressed", PathJoinSubstitution(['/auv4', camera_name, 'image_matching/compressed'])),
                ],
            ),
        ]
    )
