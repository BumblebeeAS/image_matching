import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    front_camera_topic = LaunchConfiguration('front_camera_topic')
    front_camera_info_topic = LaunchConfiguration('front_camera_info_topic')
    bottom_camera_topic = LaunchConfiguration('bottom_camera_topic')
    bottom_camera_info_topic = LaunchConfiguration('bottom_camera_info_topic')
    visualization_topic = LaunchConfiguration('visualization_topic')
    map_ned_frame = LaunchConfiguration('map_ned_frame')
    base_link_frame = LaunchConfiguration('base_link_frame')
    odom_ned_frame = LaunchConfiguration('odom_ned_frame')
    matcher = LaunchConfiguration('matcher')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return [
        Node(package="image_matching",
            executable="pose_estimator",
            name="pose_estimator",
            parameters=[{
            "front_camera_topic": front_camera_topic.perform(context),
            "front_camera_info_topic": front_camera_info_topic.perform(context),
            "bottom_camera_topic": bottom_camera_topic.perform(context),
            "bottom_camera_info_topic": bottom_camera_info_topic.perform(context),
            "visualization_topic": visualization_topic.perform(context),
            "map_ned_frame": map_ned_frame.perform(context),
            "matcher": matcher.perform(context),
            "use_sim_time": use_sim_time,
            "debug": True,
            }])
    ]

def generate_launch_description():
    ld = LaunchDescription([
        DeclareLaunchArgument('front_camera_topic',
                            #   default_value='/auv4/front_cam/image_rect_color/bright/compressed',
                            #   default_value='/wamv/sensors/cameras/left_cam_sensor/optical/image_rect_color/compressed',
                            default_value='/wamv/sensors/cameras/mid_cam_sensor/optical/image_rect_color/compressed',
                              description='Front cam topic'),
        DeclareLaunchArgument('front_camera_info_topic',
                              default_value='/wamv/sensors/cameras/mid_cam_sensor/optical/camera_info',
                              description='Front cam topic'),
        DeclareLaunchArgument('bottom_camera_topic', default_value='/auv4/bot_cam/image_rect_color/bright/compressed'),
        DeclareLaunchArgument('bottom_camera_info_topic',
                              default_value='/auv4/bot_cam/camera_info',
                              description='Bottom cam topic'),
        DeclareLaunchArgument('visualization_topic',
                              default_value='/impose_dev_vis/compressed'),
        DeclareLaunchArgument('map_ned_frame',
                              default_value='map_ned'),
        DeclareLaunchArgument('base_link_frame',
                              default_value='auv4/base_link'),
        DeclareLaunchArgument('odom_ned_frame',
                              default_value='map_ned'),
        DeclareLaunchArgument('matcher',
                              default_value='sift_flann'),
        DeclareLaunchArgument('use_sim_time',
                              default_value='True'),
        OpaqueFunction(function=launch_setup)
    ])
    return ld
