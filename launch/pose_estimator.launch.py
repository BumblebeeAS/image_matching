from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    front_camera_topic = LaunchConfiguration("front_camera_topic")
    front_camera_info_topic = LaunchConfiguration("front_camera_info_topic")
    bottom_camera_topic = LaunchConfiguration("bottom_camera_topic")
    bottom_camera_info_topic = LaunchConfiguration("bottom_camera_info_topic")
    visualization_topic = LaunchConfiguration("visualization_topic")
    map_ned_frame = LaunchConfiguration("map_ned_frame")
    # base_link_frame = LaunchConfiguration("base_link_frame")
    # odom_ned_frame = LaunchConfiguration("odom_ned_frame")
    matcher = LaunchConfiguration("matcher")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return [
        Node(
            package="image_matching",
            executable="pose_estimator",
            name="pose_estimator",
            parameters=[
                {
                    "front_camera_topic": front_camera_topic.perform(context),
                    "front_camera_info_topic": front_camera_info_topic.perform(
                        context
                    ),
                    "bottom_camera_topic": bottom_camera_topic.perform(
                        context
                    ),
                    "bottom_camera_info_topic": bottom_camera_info_topic.perform(
                        context
                    ),
                    "visualization_topic": visualization_topic.perform(
                        context
                    ),
                    "map_ned_frame": map_ned_frame.perform(context),
                    "matcher": matcher.perform(context),
                    "use_sim_time": use_sim_time,
                    "debug": True,
                }
            ],
        )
    ]


def generate_launch_description():
    ld = LaunchDescription(
        [
            DeclareLaunchArgument(
                "front_camera_topic",
                #   default_value='/auv4/front_cam/image_rect_color/bright/compressed',
                #   default_value='/wamv/sensors/cameras/left_cam_sensor/optical/image_rect_color/compressed',
                default_value="/auv4/front_cam/color/image/compressed",
                description="Front cam topic",
            ),
            DeclareLaunchArgument(
                "front_camera_info_topic",
                default_value="/auv4/front_cam/color/camera_info",
                description="Front cam topic",
            ),
            DeclareLaunchArgument(
                "bottom_camera_topic",
                default_value="/auv4/bot_cam/color/image/compressed",
            ),
            DeclareLaunchArgument(
                "bottom_camera_info_topic",
                default_value="/auv4/bot_cam/color/camera_info",
                description="Bottom cam topic",
            ),
            DeclareLaunchArgument(
                "visualization_topic",
                default_value="/impose_dev_vis/compressed",
            ),
            DeclareLaunchArgument("map_ned_frame", default_value="map_ned"),
            DeclareLaunchArgument(
                "base_link_frame", default_value="auv4/base_link"
            ),
            DeclareLaunchArgument("odom_ned_frame", default_value="map_ned"),
            DeclareLaunchArgument("matcher", default_value="xfeat"), # sift_flann, xfeat
            DeclareLaunchArgument("use_sim_time", default_value="False"),
            OpaqueFunction(function=launch_setup),
        ]
    )
    return ld
