#!/usr/bin/env python3

import json
from operator import attrgetter
import os
from pathlib import Path
import glob

import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
from bb_msgs.msg import DetectedObjects
from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf import transformations
import tf2_geometry_msgs

import threading

from transforms3d.euler import quat2euler

from feature_matcher.keypoints_match_producer import get_keypoints_match_producer
from pose_estimator.PinholeCamera import PinholeCamera
from pose_estimator.pose_estimator import PoseEstimator
mutex = threading.Lock()

class BasicPoseEstimator:
    def __init__(
        self,
        input_topic,
        image_match_producer,
        visualization_topic,
        camera_info,
        template,
        template_path,
        template_dimensions,
        detected_objects_topic=None,
    ):
        self.latest_msg = None
        self.bridge = CvBridge()
        self.template = template
        self.template_img = cv2.imread(template_path)
        self.image_match_producer = image_match_producer

        self.pose_estimator = PoseEstimator(self.image_match_producer)

        self.pose_estimator.register_camera(PinholeCamera.from_camera_info(camera_info))

        self.visualization_pub = rospy.Publisher(
            visualization_topic, CompressedImage, queue_size=1
        )
        self.pose_estimator.visualize_callbacks.append(
            lambda img: self.visualization_pub.publish(
                self.bridge.cv2_to_compressed_imgmsg(img, "jpeg")
            )
        )
        self.pose_estimator.visualize_callbacks.append(
            lambda img: print("visualising")
        )
        self.pose_estimator.register_template(
            template, template_dimensions, cv2.imread(template_path)
        )
        self.transform_stamped = tf2_ros.TransformStamped()
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30))
        self.tf_sub = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()

        # templates = {
        #     "Tommy Gun": ((0.6096, 1.2192), "/home/developer/workspace/src/image_matching/templates/Tommy Gun.jpeg"),
        #     "Bootlegger": ((0.6096, 1.2192), "/home/developer/workspace/src/image_matching/templates/Bootlegger.jpeg")
        # }

        self.PADDING = 10
        self.CROP_IMAGES = detected_objects_topic is not None

        # if self.CROP_IMAGES:
        #     rospy.loginfo("Subscribing to detected objects")
        #     self.detected_objects_sub = message_filters.Subscriber(
        #         detected_objects_topic, DetectedObjects
        #     )
        # self.image_sub = message_filters.Subscriber(input_topic, CompressedImage, queue_size=1, tcp_nodelay=True)
        # ts = message_filters.ApproximateTimeSynchronizer(
        #     [self.image_sub, self.detected_objects_sub]
        #     if self.CROP_IMAGES
        #     else [self.image_sub],
        #     4,
        #     1,
        # )
        # ts.registerCallback(self.cropped_image_callback)
        rospy.Subscriber(input_topic, CompressedImage, self.msg_callback, queue_size=1, tcp_nodelay=True)
        rospy.Timer(rospy.Duration(2), self.cropped_image_callback)

    def msg_callback(self, msg):
        mutex.acquire(blocking=True)
        self.latest_msg = msg
        mutex.release()


    def cropped_image_callback(self, debug=False):
        if self.latest_msg is None:
            return
        mutex.acquire(blocking=True)
        img_msg = self.latest_msg
        detected_objects = None
        rospy.logdebug_throttle(10, f"Received image {img_msg.header.seq}")
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        detected_object = None
        if self.CROP_IMAGES and detected_objects is not None:
            if any([x.name == self.template for x in detected_objects.detected]):
                detected_object = sorted(
                    detected_objects.detected, key=lambda x: x.extra[0], reverse=True
                )[0]

        if detected_object is not None:
            PADDING = 10
            cx, cy, w, h = (
                detected_object.centre_x,
                detected_object.centre_y,
                detected_object.bbox_width,
                detected_object.bbox_height,
            )
            x, y = int(cx - w / 2), int(cy - h / 2)
            lxtyrxby = (
                max(0, x - PADDING),
                max(0, y - PADDING),
                min(img.shape[1], x + w + PADDING),
                min(img.shape[0], y + h + PADDING),
            )
        else:
            lxtyrxby = None

        rot, trans = self.pose_estimator.compute_pose(
            img, template, img_msg.header.frame_id, lxtyrxby=lxtyrxby, debug=True
        )
        if rot is not None and trans is not None and trans[2] > 0:
            self.publish_tf(
                rot, trans, img_msg.header.frame_id, template, img_msg.header.stamp
            )
        mutex.release()

    def publish_tf(self, rot, trans, frame_id, child_frame_id, stamp):
        self.transform_stamped.transform.translation.x = trans[0]
        self.transform_stamped.transform.translation.y = trans[1]
        self.transform_stamped.transform.translation.z = trans[2]
        


        _rot = np.eye(4)
        _rot[:3, :3] = rot
        quaternion = transformations.quaternion_from_matrix(_rot)
        if np.abs(quaternion.dot(quaternion) - 1) < 1e-6:

            pose_stamped = tf2_geometry_msgs.PoseStamped()
            pose_stamped.header.stamp = stamp
            pose_stamped.header.frame_id = frame_id
            pose_stamped.pose.position.x = trans[0]
            pose_stamped.pose.position.y = trans[1]
            pose_stamped.pose.position.z = trans[2]
            pose_stamped.pose.orientation.w = quaternion[3]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.x = quaternion[0]

            try:
                new_pose_stamped = self.tf_buffer.transform(pose_stamped, "world_ned")
                print(new_pose_stamped)
            except Exception as e:
                print(f"Could not transform {e}")
                return
            self.transform_stamped.header = new_pose_stamped.header
            self.transform_stamped.transform.translation.x = new_pose_stamped.pose.position.x
            self.transform_stamped.transform.translation.y = new_pose_stamped.pose.position.y
            self.transform_stamped.transform.translation.z = new_pose_stamped.pose.position.z

            self.transform_stamped.transform.rotation = new_pose_stamped.pose.orientation

            self.transform_stamped.header.stamp = new_pose_stamped.header.stamp
            self.transform_stamped.header.frame_id = new_pose_stamped.header.frame_id
            self.transform_stamped.child_frame_id = child_frame_id

            roll, pitch, yaw = (np.array(quat2euler(attrgetter("w", "x", "y", "z")(self.transform_stamped.transform.rotation)))*180/np.pi) % 90
            print(min(roll, 90-roll), min(pitch, 90-pitch), min(yaw, 90-yaw))
            self.br.sendTransform(self.transform_stamped)


if __name__ == "__main__":
    # image_match_producer = get_keypoints_match_producer("superpoint", "superglue", {"debug": True}, {"debug": True}) # 0.6256848859s
    # image_match_producer = get_keypoints_match_producer(None, "coarse_loftr", {"debug": True}, {"debug": True}) # 0.1275215585s
    # image_match_producer = get_keypoints_match_producer("sift", "flann", {"debug": True}, {"debug": True})  # 0.0294936401s
    # image_match_producer = get_keypoints_match_producer("sift", "bf", {"debug": True}, {"debug": True}) # 0.0318265118s
    # image_match_producer = get_keypoints_match_producer("superpoint", "bf", {"debug": True}, {"debug": True})

    rospy.init_node("pose_estimator", anonymous=True)
    camera_topic = rospy.get_param(
        "~camera_topic", "/auv4/front_cam/image_color/compressed"
    )
    camera_info_topic = rospy.get_param(
        "~camera_info_topic", "/auv4/front_cam/camera_info"
    )
    visualization_topic = rospy.get_param("~visualization_topic", "/visualization")
    template = rospy.get_param("~template", "Bootlegger")


    # Accept either png or jpeg files
    templates_dir = os.path.abspath(Path(RosPack().get_path("image_matching"))/"templates")
    possible_templates = glob.glob(os.path.join(templates_dir, f"{template}.*"))
    if not possible_templates:
        rospy.logwarn_once(f"No template found for {template} in {templates_dir}")

    template_path = rospy.get_param(
        "~template_path",
        possible_templates[0] if possible_templates else "FILE_NOT_FOUND.png",
    )
    print(f"Using template {template_path}")
    rospy.loginfo(f"Using template {template_path}")

    # Retrieve template dimensions from json file
    templates = json.loads(open(os.path.join(templates_dir, "templates.json")).read())
    template_filename = template_path.split("/")[-1]
    if template_filename in templates.keys():
        saved_template_width = templates[template_filename][0]
        saved_template_height = templates[template_filename][1]
    else:
        saved_template_width = None
        saved_template_height = None

    if not rospy.get_param("~use_default_dim", True):
        template_width = rospy.get_param("~template_width", saved_template_width)
        template_height = rospy.get_param("~template_height", saved_template_height)
    else:
        template_width = saved_template_width
        template_height = saved_template_height
    if not template_height or not template_width:
        rospy.logerr("use_default_dim set to true but no template dimensions found")
        exit(1)
    rospy.loginfo(f"Using template dimensions {template_width}x{template_height}")

    detected_objects_topic = rospy.get_param("~detected_objects_topic", None)

    matcher = rospy.get_param("~matcher", "superglue")

    camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)

    if matcher == "coarse_loftr":
        image_match_producer = get_keypoints_match_producer(
            None, "coarse_loftr", {"debug": True}, {"debug": True}
        )
    elif matcher == "sift_flann":
        image_match_producer = get_keypoints_match_producer(
            "sift", "flann", {"debug": True}, {"debug": True}
        )
    elif matcher == "sift_bf":
        image_match_producer = get_keypoints_match_producer(
            "sift", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "superpoint_bf":
        image_match_producer = get_keypoints_match_producer(
            "superpoint", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "superpoint_superglue":
        image_match_producer = get_keypoints_match_producer(
            "superpoint", "superglue", {"debug": True}, {"debug": True}
        )
    elif matcher == "fast_bf":
        image_match_producer = get_keypoints_match_producer(
            "fast", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "orb_bf":
        image_match_producer = get_keypoints_match_producer(
            "orb", "bf", {"debug": True}, {"debug": True}
    )
    elif matcher == "alike_bf":
        image_match_producer = get_keypoints_match_producer(
            "alike", "bf", {"debug": True}, {"debug": True}
        )


    pose_estimator = BasicPoseEstimator(
        camera_topic,
        image_match_producer,
        visualization_topic,
        camera_info,
        template,
        template_path,
        (
            template_width,
            template_height,
        ),  # TODO: get the template dimensions from launch file or service or json file
        detected_objects_topic,
    )

    rospy.spin()
