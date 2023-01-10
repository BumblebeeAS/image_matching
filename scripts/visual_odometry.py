#!/usr/bin/env python3

import traceback
from typing import List

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, Imu, CameraInfo

from feature_matcher.keypoints_match_producer import get_keypoints_match_producer
from pose_estimator.PinholeCamera import PINHOLE_CAMERAS, PinholeCamera
from utils.logging import BasicLogger


class BasicVisualOdometry:
    """
    A basic visual odometry node that publishes the pose of the vehicle.

    Algorithm:
    1. image matching to get pairwise frame correspondence
    2. Computes essential matrix
    3. Recover vehicle pose from the essential matrix
    """

    def __init__(
        self,
        input_topic,
        visualization_topic,
        imu_topic,
        imu_frame,
        camera_info=None,
        debug=True,
    ):

        self.logger = BasicLogger("BasicVisualOdometry", debug=debug)

        self.bridge = CvBridge()
        self.image_match_producer = get_keypoints_match_producer(
            "alike",
            "bf",
            {"debug": True, "cuda": True},
            {"debug": True, "cuda": True},
        )

        if camera_info is not None:
            self.camera = PinholeCamera.from_camera_info(camera_info)

        else:
            self.logger.warning(
                "No camera info provided, using simulation camera parameters"
            )
            self.camera = PINHOLE_CAMERAS["sim"]

        # Publishers
        self.visualization_pub = rospy.Publisher(
            visualization_topic, CompressedImage, queue_size=1
        )
        self.imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=1)

        # Subscribers
        self.img_sub = rospy.Subscriber(
            input_topic, CompressedImage, self.callback, queue_size=1
        )

        self.imu_frame = imu_frame

        # To avoid floating point loss of precision
        self.time_stamps_sec: List[int] = []
        self.time_stamps_nsec: List[int] = []

    def callback(self, img_msg):
        rospy.logdebug_throttle(1, f"Received image {img_msg.header.seq}")
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            self.logger.error(f"Could not convert image: {e}")
            return

        # Append new image
        start_time = rospy.get_time()
        if self.image_match_producer.add_image(img) == 1:
            self.logger.error("Could not add image to buffer")
            return
        self.time_stamps_sec.append(img_msg.header.stamp.secs)
        self.time_stamps_nsec.append(img_msg.header.stamp.nsecs)

        if len(self.image_match_producer.buffer) < 2:
            # Not enough images to compute pose
            return
        end_time = rospy.get_time()
        self.logger.info(f"Time to add image: {end_time - start_time}")

        # Computing matches
        start_time = rospy.get_time()
        try:
            # Many keypoints are needed for accurate pose estimation
            keypoints1, keypoints2 = self.image_match_producer.compute_matches(5000)
        except Exception as e:
            self.logger.error(f"Could not compute matches: {e}")
            self.logger.error(traceback.format_exc())
            return

        if (
            keypoints1 is None
            or len(keypoints1) < 4
            or keypoints2 is None
            or len(keypoints2) < 4
        ):
            self.logger.error(
                f"Not enough matches to compute pose. Found {0 if keypoints1 is None else len(keypoints1)} matches."
            )
            return
        end_time = rospy.get_time()
        self.logger.info(f"Time to compute matches: {end_time - start_time}")

        # Computing pose
        start_time = rospy.get_time()
        E, mask = cv2.findEssentialMat(
            keypoints1.keypoints,
            keypoints2.keypoints,
            cameraMatrix=self.camera.camera_matrix(),
            method=cv2.USAC_MAGSAC,
            prob=0.99999,  # Taken from QuadTreeAttention's metrics code
            threshold=1.0,  # Taken from QuadTreeAttention's metrics code
        )
        if E is None:
            self.logger.error("Could not compute essential matrix. Got None")
            return
        end_time = rospy.get_time()
        self.logger.info(f"Time to compute essential matrix: {end_time - start_time}")

        # Recover pose from Essential Matrix
        start_time = rospy.get_time()
        _, R, t, mask = cv2.recoverPose(
            E,
            keypoints1.keypoints,
            keypoints2.keypoints,
            cameraMatrix=self.camera.camera_matrix(),
            mask=mask,
        )
        end_time = rospy.get_time()
        self.logger.info(f"Time to recover pose: {end_time - start_time}")

        # Convert to IMU message
        start_time = rospy.get_time()
        R_rad = Rotation.from_matrix(R).as_euler("zyx", degrees=False)

        # OpenCV returns in camera frame i.e. right down front.
        # Hence, z (front) => roll, y (down) => yaw, x => pitch
        roll_rad, yaw_rad, pitch_rad = (
            R_rad[0],
            R_rad[1],
            R_rad[2],
        )

        sec_diff = self.time_stamps_sec[-1] - self.time_stamps_sec[0]
        nsec_diff = self.time_stamps_nsec[-1] - self.time_stamps_nsec[0]

        time_diff = sec_diff + nsec_diff * 1e-9

        imu_msg = Imu()
        imu_msg.header.stamp.secs = self.time_stamps_sec[0] + sec_diff // 2
        imu_msg.header.stamp.nsecs = self.time_stamps_nsec[0] + nsec_diff // 2

        imu_msg.header.frame_id = self.imu_frame
        imu_msg.angular_velocity.x = roll_rad / time_diff
        imu_msg.angular_velocity.y = pitch_rad / time_diff
        imu_msg.angular_velocity.z = yaw_rad / time_diff
        end_time = rospy.get_time()
        self.logger.info(f"Time to produce IMU message: {end_time - start_time}")

        # Publish IMU message
        # print(imu_msg)
        try:
            self.imu_pub.publish(imu_msg)
        except Exception as e:
            self.logger.error(f"Could not publish IMU message: {e}")
            self.logger.error(traceback.format_exc())

        # Debug: Print RPY and translation
        R_deg = Rotation.from_matrix(R).as_euler("zyx", degrees=True)
        roll, yaw, pitch = (
            R_deg[0],
            R_deg[1],
            R_deg[2],
        )
        self.logger.info(f"Rotation (RPY): {roll}, {pitch}, {yaw}")
        self.logger.info(f"Translation: {t}")

        # Pop oldest image
        self.image_match_producer.buffer.pop(0)
        self.time_stamps_sec.pop(0)
        self.time_stamps_nsec.pop(0)

        self.logger.info("")


if __name__ == "__main__":
    rospy.init_node("pose_estimator", anonymous=True)
    camera_topic = rospy.get_param(
        "~camera_topic", "/auv4/front_cam/image_rect_color/compressed"
    )
    camera_info_topic = rospy.get_param(
        "~camera_info_topic", "/auv4/front_cam/camera_info"
    )
    visualization_topic = rospy.get_param(
        "~visualization_topic", "/auv4/visual_odometry/visualisation"
    )
    imu_topic = rospy.get_param("~imu_topic", "/auv4/visual_odometry/imu")
    imu_frame = rospy.get_param("~imu_frame", "auv4/front_cam")

    camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo, 10)
    if camera_info is None:
        print("Time out! Using sim camera info")

    pose_estimator = BasicVisualOdometry(
        camera_topic,
        visualization_topic,
        imu_topic=imu_topic,
        imu_frame=imu_frame,
        camera_info=camera_info,
        debug=True,
    )

    rospy.spin()
