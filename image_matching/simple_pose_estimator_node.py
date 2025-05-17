from collections import defaultdict
import logging

from bb_perception_msgs.msg import PointCorrespondencesStamped
import cv2
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from sklearn.cluster import HDBSCAN
import tf2_ros
from transforms3d.quaternions import mat2quat

from pose_estimator.PinholeCamera import PinholeCamera
from utils.ros_np_multiarray import to_numpy_f64


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

    Raises:
        ValueError: If the number of object points is less than 4.
        ValueError: If no inliers are found.
        Exception: If cv2.solvePnPRansac or cv2.solvePnPRefineLM fails.
    """
    # TODO: Account for equidistant distortion
    # TODO: For the planar case, init cv2.solvePnPRefineLM directly with homography
    if len(object_points) < 4:
        raise ValueError(
            f"At least 4 points needed to estimate pose, only {len(object_points)} given"
        )

    # This step gives a rough estimate of the pose for solvePnPRefineLM and
    # allows for quick termination if no inliers are found. This is useful
    # when there are few point correspondences and homography estimation
    # cannot determine if the points are inliers or not.

    # RANSAC accounts for outliers
    # A small max reprojection error is used to get a good pose estimate
    # SQPnP is more robust than EPnP
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        useExtrinsicGuess=False,
        reprojectionError=max_reprojection_error,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if inliers is None:
        raise ValueError("No inliers found")

    # TODO: Split into planar and non-planar cases.
    # Case 1: Object points are non-planar.
    # Use only the inliers from RANSAC as homography estimation does not apply.
    # Case 2: Use all points for the planar case.
    # Homography estimation filters well. RANSAC filtering is too strict, resulting
    # in too few point correspondences and a noisy pose estimate.
    rvec, tvec = cv2.solvePnPRefineVVS(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        rvec,
        tvec,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-6),
    )

    return rvec, tvec


def filter_by_homography(
    object_points: np.ndarray, image_points: np.ndarray
) -> tuple:
    """Filter the object points and image points by homography,
    assuming that the object points are in the same plane.

    Note: The last coordinate of the object points are discarded.
    The object points should first be transformed so that the last
    coordinates are zero.

    Args:
        object_points (np.ndarray): N x 3
        image_points (np.ndarray): N x 2

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (R, t, inliers)

    Raises:
        ValueError: If the number of object points is less than 4.
        ValueError: If the number of object points and image points are not equal.
    """
    if len(object_points) < 4:
        raise ValueError(
            f"At least 4 points needed to estimate homography, only {len(object_points)} given"
        )

    if len(object_points) != len(image_points):
        raise ValueError(
            f"Number of object points and image points must be equal, "
            f"but got {len(object_points)} and {len(image_points)}"
        )

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


def estimate_covariance(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: PinholeCamera,
) -> np.ndarray:
    """Get covariance of pose estimate from reprojection.

    Args:
        object_points (np.ndarray): N x 3
        rvec (np.ndarray): Rotation vector
        tvec (np.ndarray): Translation vector
        camera (PinholeCamera): Camera object

    Returns:
        np.ndarray: 6 x 6 covariance matrix

    Raises:
        np.linalg.LinAlgError: If the inverse of the Jacobian cannot be computed.
    """
    # Jacobian is a 2N x 15 matrix
    # See https://github.com/opencv/opencv/blob/16a3d37dc159dbcaaf8ee74cf63669f0203f9655/modules/calib3d/src/calibration_base.cpp#L1508-L1512
    _, jacobian = cv2.projectPoints(
        object_points, rvec, tvec, camera.camera_matrix(), camera.dist_coeffs()
    )

    # Get jacobian of rotation and translation
    # Interchange rotation and translation covariance
    jacobian = jacobian[:, :6]
    jacobian[:, :3], jacobian[:, 3:] = (
        jacobian[:, 3:].copy(),
        jacobian[:, :3].copy(),
    )

    # Fisher information matrix
    return np.linalg.inv(jacobian.T @ jacobian)


class SimplePoseEstimator(Node):
    def __init__(self):
        super().__init__("pose_estimator")

        self.declare_parameter(
            "camera_info_topic", "/auv4/front_cam/color/camera_info"
        )
        self.declare_parameter("camera_frame_id", "auv4/front_cam_optical")

        camera_info_topic = (
            self.get_parameter("camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        valid, front_camera_info = wait_for_message(
            CameraInfo, self, camera_info_topic
        )
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            self.camera = PinholeCamera.from_camera_info(
                front_camera_info, rectified=False
            )

        self.camera_frame_id = (
            self.get_parameter("camera_frame_id")
            .get_parameter_value()
            .string_value
        )

        self.br = tf2_ros.TransformBroadcaster(self)
        self.cluster_tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            "/auv4/front_cam/image_matching/point_correspondences",
            self.point_correspondences_callback,
            1,
        )

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            "/auv4/front_cam/image_matching/pose",
            1,
        )

        self.declare_parameter(
            "num_detections", 50
        )  # number of detections before clustering
        self.num_detections = (
            self.get_parameter("num_detections")
            .get_parameter_value()
            .integer_value
        )
        self.hdb = HDBSCAN(
            min_cluster_size=25,
            allow_single_cluster=True,
            store_centers="centroid",
        )
        self.q_idx = 0
        self.pos_cluster_q = [
            (None, None, None) for i in range(self.num_detections)
        ]

    def point_correspondences_callback(self, msg: PointCorrespondencesStamped):
        # TODO: Filter using clustering or Kalman
        object_points = to_numpy_f64(msg.object_points)
        image_points = to_numpy_f64(msg.image_points)

        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            self.get_logger().warn(
                f"""Not enough point correspondences:
                {object_points.shape[0]} object points and {image_points.shape[0]} image points"""
            )
            return

        object_points, image_points = filter_by_homography(
            object_points, image_points
        )

        try:
            rvec, tvec = get_object_pose(
                self.camera, object_points, image_points
            )
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.squeeze()
        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        try:
            covariance = estimate_covariance(
                object_points, rvec, tvec, self.camera
            )
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(
                f"Covariance estimation failed, inversion for FIM matrix failed: {e}"
            )
            return

        # self.get_logger().info(
        #     f"Pose estimation std dev: {np.sqrt(covariance.diagonal())}"
        # )

        try:
            qx, qy, qz, qw = mat2quat(R)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(
                f"Error in mat2quat, failed to convert R: {e}"
            )
            return

        pose = PoseWithCovarianceStamped()
        pose.header = msg.header
        pose.header.frame_id = self.camera_frame_id
        # TODO: if the clustering works can remove the below for now leave it
        pose.pose.pose.position = Point(x=t[0], y=t[1], z=t[2])
        pose.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        pose.pose.covariance = covariance.flatten().tolist()

        # add clutering for pose
        pos_xyz = np.array([t[0], t[1], t[2]])
        self.pos_cluster_q[self.q_idx] = (
            pos_xyz,
            [qx, qy, qz, qw],
            pose.pose.covariance,
        )

        if self.q_idx == (self.num_detections - 1):
            # do the clustering and publish the pose
            pose_xyz, q, cov = self.filter_by_clustering()
            if pose_xyz is not None:
                pose.pose.covariance = cov
                pose.pose.pose.orientation = Quaternion(
                    x=q[0], y=q[1], z=q[2], w=q[3]
                )
                self.get_logger().info(
                    f"Publishing posxyz: {pose_xyz} after clustering."
                )
                pose.pose.pose.position = Point(
                    x=pose_xyz[0], y=pose_xyz[1], z=pose_xyz[2]
                )

                transform_stamped = TransformStamped()
                transform_stamped.header = msg.header
                transform_stamped.child_frame_id = (
                    msg.object_frame_id + "/clustered"
                )
                transform_stamped.transform.translation = Vector3(
                    x=pose_xyz[0], y=pose_xyz[1], z=pose_xyz[2]
                )
                transform_stamped.transform.rotation = Quaternion(
                    x=q[0], y=q[1], z=q[2], w=q[3]
                )

                self.cluster_tf_broadcaster.sendTransform(transform_stamped)
            else:
                # we just reset the cluster q in this case
                self.get_logger().warn("No clusters found")
            self.q_idx = 0
        else:
            self.q_idx += 1

        self.pose_publisher.publish(pose)

        transform_stamped = TransformStamped()
        transform_stamped.header = msg.header
        transform_stamped.child_frame_id = msg.object_frame_id
        transform_stamped.transform.translation = Vector3(
            x=t[0], y=t[1], z=t[2]
        )
        transform_stamped.transform.rotation = Quaternion(
            x=qx, y=qy, z=qz, w=qw
        )

        self.br.sendTransform(transform_stamped)

    def filter_by_clustering(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Filter the pose estimates by clustering.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: pose, (qx, qy, qz, qw), covariance
        """
        to_fit = [x[0] for x in self.pos_cluster_q]
        # self.get_logger().info(f"to_fit: {to_fit}")
        self.hdb.fit(to_fit)
        labels = self.hdb.labels_

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(self.pos_cluster_q[i])

        # cluster_sizes = list(map(len, clusters))

        # self.get_logger().info(f"cluster sizes: {cluster_sizes}")
        self.get_logger().info(f"cluster labels: {labels}")

        largest_cluster_size = 0
        largest_cluster_label = -1

        for label, cluster in clusters.items():
            if label == -1:
                continue
            if len(cluster) > largest_cluster_size:
                largest_cluster_size = len(cluster)
                largest_cluster_label = label
        if largest_cluster_label == -1:
            self.get_logger().warn("No clusters found")
            return (None, None, None)

        # Get the largest cluster
        largest_cluster = clusters[largest_cluster_label]
        centroid = np.mean(
            np.array([x[0] for x in largest_cluster]), axis=0
        ).astype(float)
        avg_orientation = np.mean(
            np.array([x[1] for x in largest_cluster]), axis=0
        ).astype(float)
        # Calculate the average covariance for the cluster
        avg_covariance = (
            np.sum(np.array([x[2] for x in largest_cluster]), axis=0)
            / np.square(largest_cluster_size)
        ).astype(float)
        # self.get_logger().info(f"Cluster size: {len(largest_cluster)}, centroid: {centroid}")
        return (centroid, avg_orientation, avg_covariance)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    node = SimplePoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
