import logging

import cv2
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.msg import PointCorrespondencesStamped
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from transforms3d.quaternions import mat2quat

from pose_estimator.PinholeCamera import PinholeCamera
from utils.ros_np_multiarray import to_numpy_f64
from utils.wait_for_message import wait_for_message


def get_object_pose(
    camera: PinholeCamera,
    object_points: np.ndarray,
    image_points: np.ndarray,
    max_reprojection_error: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the object pose from the camera and point correspondences.

    Args:
        camera (PinholeCamera):
        object_points (np.ndarray): N x 3
        image_points (np.ndarray): N x 2
        max_reprojection_error (float): Maximum reprojection error for RANSAC.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (R, t, inliers)
    """
    # TODO: Account for equidistant distortion
    # TODO: Check if less than 4 points (move check here)
    _, rvec, t, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        useExtrinsicGuess=False,
        reprojectionError=max_reprojection_error,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    R = cv2.Rodrigues(rvec)[0]
    t = t.squeeze()

    return R, t, inliers


def filter_by_homography(object_points: np.ndarray, image_points: np.ndarray) -> tuple:
    """Filter the object points and image points by homography,
    by assuming that the object points are in the same plane.

    Note: The last coordinate of the object points are discarded.
    The object points should first be transformed so that the last
    coordinates are zero.

    Args:
        object_points (np.ndarray): N x 3
        image_points (np.ndarray): N x 2

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (R, t, inliers)
    """
    object_points_2d = object_points[:, :2]
    _, mask = cv2.findHomography(
        object_points_2d,
        image_points,
        cv2.USAC_MAGSAC,
        3.5,
        maxIters=1_000,
        confidence=0.999,
    )
    mask = mask.flatten().astype(bool)
    object_points = object_points[mask]
    image_points = image_points[mask]
    return object_points, image_points


class SimplePoseEstimator(Node):

    def __init__(self):
        super().__init__("pose_estimator")

        self.declare_parameter("camera_info_topic", "/auv4/front_cam/color/camera_info")
        self.declare_parameter("camera_frame_id", "auv4/front_cam_optical")

        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        valid, front_camera_info = wait_for_message(
            CameraInfo, self, camera_info_topic, time_to_wait=10
        )
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            self.camera = PinholeCamera.from_camera_info(
                front_camera_info, rectified=False
            )

        self.camera_frame_id = (
            self.get_parameter("camera_frame_id").get_parameter_value().string_value
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30), node=self)
        self.br = tf2_ros.StaticTransformBroadcaster(self)

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            "/auv4/front_cam/image_matching/point_correspondences",
            self.point_correspondences_callback,
            1,
        )

    def point_correspondences_callback(self, msg: PointCorrespondencesStamped):
        # TODO: Filter using clustering or Kalman
        # TODO: Try catch
        # TODO: Use SVD to handle ill-defined R?
        object_points = to_numpy_f64(msg.object_points)
        image_points = to_numpy_f64(msg.image_points)
        object_points, image_points = filter_by_homography(object_points, image_points)

        # self.get_logger().info(
        #     f"Received {object_points} and {image_points} image points"
        # )
        if len(object_points) < 4:
            return

        R, t, _ = get_object_pose(self.camera, object_points, image_points)
        # self.get_logger().info(f"Pose: {R}, {t}")
        qx, qy, qz, qw = mat2quat(R)

        transform_stamped = TransformStamped()
        transform_stamped.header = msg.header
        transform_stamped.child_frame_id = msg.object_frame_id
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.br.sendTransform(transform_stamped)


def main():
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = SimplePoseEstimator()
    tf2_ros.TransformListener(node.tf_buffer, node, spin_thread=False)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
