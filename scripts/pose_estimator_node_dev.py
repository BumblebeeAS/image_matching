#!/usr/bin/env python3
import copy
import glob
import json
import logging
import os
import threading
import traceback
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import rospy
import tf2_ros
from bb_msgs.msg import DetectedObjects
from bb_msgs.srv import (
    IMPoseEstimatorConfig,
    IMPoseEstimatorConfigResponse,
    IMPoseEstimatorGetStatus,
    IMPoseEstimatorGetStatusRequest,
    IMPoseEstimatorGetStatusResponse,
    IMPoseEstimatorGetTemplates,
    IMPoseEstimatorGetTemplatesResponse,
    IMPoseEstimatorRegisterTemplate,
    IMPoseEstimatorRegisterTemplateRequest,
    IMPoseEstimatorRegisterTemplateResponse,
    IMPoseEstimatorToggleTemplate,
    IMPoseEstimatorToggleTemplateResponse,
    IMPoseEstimatorUpdateKeypointMatches,
    IMPoseEstimatorUpdateKeypointMatchesRequest,
    IMPoseEstimatorUpdateKeypointMatchesResponse,
)
from cv_bridge import CvBridge, CvBridgeError
from feature_matcher.keypoints_match_producer import (
    KeypointsMatchProducer,
    get_keypoints_match_producer,
)
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from nav_msgs.msg import Odometry
from pose_estimator.PinholeCamera import PinholeCamera
from pose_estimator.pose_estimator import PoseEstimator
from pose_estimator.pose_weighted_average import get_kmeans_center
from rospkg import RosPack
from sensor_msgs.msg import CameraInfo, CompressedImage
from transforms3d.affines import compose, decompose
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import mat2quat, qinverse, qmult, quat2mat

mutex = threading.Lock()


@dataclass
class Image:
    img: cv2.Mat
    descriptor: Any
    timestamp: float
    pose: PoseStamped


@dataclass
class Template:
    name: str
    matcher: str
    min_matches: int
    reprojection_error_threshold: float
    object_name: str
    offset: Tuple[
        float, float
    ]  # x, y offset of template center from object_name frame in meters


@dataclass
class TemplateObject:
    name: str
    poses: pd.DataFrame
    computed_pose: Optional[PoseWithCovarianceStamped]
    min_buffer_size: int
    max_buffer_size: int
    max_history: float

def filter_forward_facing(pose):
    """
    Given pose np.array([x,y,z,qw,qx,qy,qz])
    return True if valid pose else False
    """
    if pose[2] < -2 or pose[2] > 10:
        rospy.logwarn_throttle(1, "Rubbish z")
        return False
    r, p, y = np.rad2deg(quat2euler(pose[3:], "rzyx"))
    if abs((y % 180) - 90) > 30:
        rospy.logwarn(f"Not vertical {r} {p} {y}")
        return False
    return True


def filter_bottom_facing(pose):
    if pose[2] < 0 or pose[2] > 3:
        rospy.logwarn_throttle(1, "Rubbish z")
        return False
    y, p, r = np.rad2deg(quat2euler(pose[3:], "rzyx"))
    if r >= 45 or r <= -45 or p >= 45 or p <= -45:
        rospy.logwarn_throttle(1, f">>>>>>>>> ignore: r:{r} p: {p}, y: {y}")
        return False
    return True


def transform_buoy_stabilized(quat):
    q1 = euler2quat(np.pi / 2, 0, np.pi / 2, "rzyx")
    qw, qx, qy, qz = qmult(qinverse(q1), quat)
    r, p, y = quat2euler([qw, qx, qy, qz], "rxyz")
    return qmult(q1, euler2quat(r, p, 0, "rxyz"))


class BasicPoseEstimator:
    def publish_img(self, img):
        try:
            self.visualization_pub.publish(
                self.bridge.cv2_to_compressed_imgmsg(img, "jpeg")
            )
        except Exception as e:
            rospy.logerr(e)

    def __init__(
        self,
        image_match_producers: Dict[str, KeypointsMatchProducer],
        visualization_topic,
        detected_objects_topic=None,
        templates_dir="./",
        debug=False,
        map_ned_frame="world_ned"
    ):
        self.latest_msgs: Dict[str, cv2.Mat] = {}
        self.bridge = CvBridge()
        self.templates: Dict[str, Template] = {}
        self.template_objects: Dict[str, TemplateObject] = {}
        self.templates_dir = templates_dir

        self.image_match_producers = image_match_producers
        self.pose_estimator = PoseEstimator(image_match_producers)

        self.visualization_pub = rospy.Publisher(
            visualization_topic, CompressedImage, queue_size=1
        )
        self.topics: Dict[str, str] = {}

        self.debug = debug
        if debug:
            self.pose_estimator.visualize_callbacks.append(self.publish_img)
        self.map_ned_frame = map_ned_frame

        self.active_templates: Set[
            Tuple[str, str]
        ] = set()  # (template_name, camera_frame_id)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30))
        self.tf_sub = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.StaticTransformBroadcaster()
        self.odom_pub = rospy.Publisher("impose_estimates", Odometry, queue_size=1)

        self.update_keypoint_matches_service = rospy.Service(
            "impose_update_keypoint_matches",
            IMPoseEstimatorUpdateKeypointMatches,
            self.update_keypoint_matches,
        )
        self.get_templates_service = rospy.Service(
            "impose_get_templates", IMPoseEstimatorGetTemplates, self.get_templates
        )
        self.register_template_service = rospy.Service(
            "impose_register_template",
            IMPoseEstimatorRegisterTemplate,
            self.register_template_cb,
        )
        self.toggle_template_service = rospy.Service(
            "impose_toggle_template",
            IMPoseEstimatorToggleTemplate,
            self.toggle_template,
        )
        self.get_status_service = rospy.Service(
            "impose_get_status",
            IMPoseEstimatorGetStatus,
            self.get_status,
        )
        self.config_service = rospy.Service(
            "impose_config", IMPoseEstimatorConfig, self.update_config
        )

        self.PADDING = 10
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

        self.subscribers: Dict[str, rospy.Subscriber] = {}
        self.custom_pose_filtering: Dict[str, Callable[[np.ndarray], bool]] = {
            "buoy_seg": filter_forward_facing,
            "gate_seg": filter_forward_facing,
            "torpedo_seg": filter_forward_facing,
            "bin_abydos_1_part-0": filter_bottom_facing,
            "bin_abydos_1_part-1": filter_bottom_facing,
            "bin_abydos_1_part-2": filter_bottom_facing,
            "bin_earth_1_part-0": filter_bottom_facing,
            "bin_earth_1_part-1": filter_bottom_facing,
            "bin_earth_1_part-2": filter_bottom_facing,
            "bin_earth_2_part-0": filter_bottom_facing,
            "bin_earth_2_part-1": filter_bottom_facing,
            "bin_earth_2_part-2": filter_bottom_facing,
        }

        self.custom_stabilized_orientation_transform: Dict[
            str, Callable[[np.ndarray], np.ndarray]
        ] = {
            "buoy1": transform_buoy_stabilized,
            "buoy2": transform_buoy_stabilized,
            "buoy1-2": transform_buoy_stabilized,
            "buoy1-3": transform_buoy_stabilized,
            "buoy1-4": transform_buoy_stabilized,
            "buoy2-2": transform_buoy_stabilized,
            "buoy2-3": transform_buoy_stabilized,
            "buoy2-4": transform_buoy_stabilized,
        }

    def teardown(self):
        for sub in self.subscribers.values():
            sub.unregister()

    def update_config(self, req):
        template_name = req.template_name
        matcher = req.matcher
        if template_name not in self.templates.keys():
            # self.templates[template_name] = self.create_default_template(template_name, template_name, (0, 0))
            rospy.logerr(f"Template {template_name} not registered")
            return IMPoseEstimatorConfigResponse(False)

        if matcher not in self.image_match_producers.keys():
            rospy.logerr(f"Matcher {matcher} not loaded")
            return IMPoseEstimatorConfigResponse(False)

        object_name = self.templates[template_name].object_name
        if object_name not in self.template_objects.keys():
            rospy.logerr(f"Object {object_name} not registered")
            return IMPoseEstimatorConfigResponse(False)

        setattr(
            self.template_objects[object_name], "min_buffer_size", req.min_buffer_size
        )
        setattr(
            self.template_objects[object_name], "max_buffer_size", req.max_buffer_size
        )
        setattr(self.template_objects[object_name], "max_history", req.max_history)

        if req.reset:
            self.template_objects[object_name].poses = pd.DataFrame(
                columns=["stamp", "x", "y", "z", "qw", "qx", "qy", "qz"]
            )
            self.template_objects[object_name].computed_pose = None

        setattr(
            self.templates[template_name],
            "reprojection_error_threshold",
            req.max_reprojection_threshold,
        )
        setattr(self.templates[template_name], "min_matches", req.min_matches)
        setattr(self.templates[template_name], "matcher", matcher)

        rospy.loginfo(
            f"Config updated: {req} {self.templates[template_name]} {self.template_objects[object_name]}"
        )
        return IMPoseEstimatorConfigResponse(success=True)

    def get_templates(self, req):
        active_templates = list(
            set(
                [
                    template_name + ":" + frame_id
                    for (template_name, frame_id) in self.active_templates
                ]
            )
        )
        return IMPoseEstimatorGetTemplatesResponse(
            self.pose_estimator.available_templates,
            active_templates,
            list(self.image_match_producers.keys()),
        )

    def toggle_template(self, req):
        if req.template_name == "" and req.enabled is False:
            print("Disabling all templates")
            self.active_templates.clear()
            return IMPoseEstimatorToggleTemplateResponse(
                False, "All templates disabled"
            )

        if req.template_name not in self.pose_estimator.available_templates:
            return IMPoseEstimatorToggleTemplateResponse(
                False, f"Template {req.template_name} not registered"
            )
        if req.camera_frame_id not in self.pose_estimator.cameras:
            return IMPoseEstimatorToggleTemplateResponse(
                False, f"Camera {req.camera_frame_id} not registered"
            )
        if req.enabled:
            self.active_templates.add((req.template_name, req.camera_frame_id))
        else:
            if (req.template_name, req.camera_frame_id) not in self.active_templates:
                return IMPoseEstimatorToggleTemplateResponse(
                    False, f"Template {req.template_name} not active"
                )
            self.active_templates.remove((req.template_name, req.camera_frame_id))
        print("toggled template", self.subscribers)
        return IMPoseEstimatorToggleTemplateResponse(
            (req.template_name, req.camera_frame_id) in self.active_templates, ""
        )

    @staticmethod
    def create_default_object(name):
        return TemplateObject(
            name,
            pd.DataFrame(columns=["stamp", "x", "y", "z", "qw", "qx", "qy", "qz"]),
            None,
            1,  # min_buffer_size
            20,  # max_buffer_size
            10,  # max_history,
        )

    @staticmethod
    def create_default_template(name, object_name=None, offset=(0, 0)):
        if object_name is None:
            object_name = name
        return Template(
            name,
            "sift_flann",
            4,  # min_matches
            2,  # reprojection_error_threshold
            object_name,
            offset,  # of template center from object center
        )

    def get_status(self, req: IMPoseEstimatorGetStatusRequest):
        res = IMPoseEstimatorGetStatusResponse()
        if req.template_name not in self.templates.keys():
            res.is_valid = False
            return res
        template = self.templates[req.template_name]
        template_object = self.template_objects[template.object_name]
        if template_object.computed_pose is None:
            res.is_valid = False
            return res
        time_since_last = (
            rospy.Time.now().secs - template_object.computed_pose.header.stamp.secs
        )
        if (
            template_object.max_history > 0
            and time_since_last > template_object.max_history
        ):
            res.is_valid = False
            res.num_poses = len(template_object.poses)
            return res
        res.pose = template_object.computed_pose
        res.is_valid = True
        res.num_poses = len(template_object.poses)
        return res

    def register_template(self, img, name, dimensions, object_name, offset=(0, 0)):
        self.pose_estimator.register_template(name, dimensions, img)
        if object_name not in self.template_objects.keys():
            self.template_objects[object_name] = self.create_default_object(object_name)
        self.templates[name] = BasicPoseEstimator.create_default_template(
            name, object_name, offset
        )

    @staticmethod
    def equalize_green_blue(img):
        b, g, r = cv2.split(img)

        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)

        img_eq = cv2.merge((b_eq, g_eq, r))

        return img_eq

    def register_template_cb(self, req: IMPoseEstimatorRegisterTemplateRequest):
        compressed_image_topic_name = req.image_topic_name
        detected_objects_topic_name = req.detected_objects_topic_name
        object_name = req.object_name
        detected_object = None
        try:
            if detected_objects_topic_name != "" and object_name != "":
                for i in range(3):
                    try:
                        detected_objects = rospy.wait_for_message(
                            detected_objects_topic_name, DetectedObjects, timeout=2
                        )
                    except rospy.ROSException:
                        continue
                    if any([x.name == object_name for x in detected_objects.detected]):
                        detected_object = sorted(
                            detected_objects.detected,
                            key=lambda x: x.extra[0],
                            reverse=True,
                        )[0]
                        break
            img: CompressedImage = rospy.wait_for_message(
                compressed_image_topic_name, CompressedImage, timeout=2
            )
            cv2_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(img)
            if detected_object is not None:
                PADDING = 10
                cx, cy, w, h = (
                    detected_object.centre_x,
                    detected_object.centre_y,
                    detected_object.width,
                    detected_object.height,
                )
                x, y = int(cx - w / 2), int(cy - h / 2)
                cv2_img = cv2_img[
                    y - PADDING : y + h + PADDING, x - PADDING : x + w + PADDING, :
                ]
            cv2.imwrite(
                os.path.join(
                    self.templates_dir,
                    f"{req.template_name}.{rospy.Time.now().secs}.jpg",
                ),
                cv2_img,
            )
            if req.template_name in self.templates:
                rospy.loginfo("Replacing existing template %s", req.template_name)
            self.register_template(
                cv2_img, req.template_name, (req.width, req.height), object_name, (0, 0)
            )
            return IMPoseEstimatorRegisterTemplateResponse(True, "")
        except Exception as e:
            return IMPoseEstimatorRegisterTemplateResponse(False, str(e))

    def check_subscribers(self):
        for subscriber in self.subscribers.values():
            if subscriber.get_num_connections() == 0:
                return False

    def register_camera(self, camera_topic: str, camera: PinholeCamera):
        if camera_topic not in self.subscribers:
            self.subscribers[camera_topic] = rospy.Subscriber(
                camera_topic,
                CompressedImage,
                self.msg_callback(camera.frame_id),
                queue_size=1,
            )
            self.pose_estimator.register_camera(camera)
            self.topics[camera.frame_id] = camera_topic

    def msg_callback(self, camera_frame_id):
        def callback(msg):
            # print(f"cb: {self.active_templates}")
            if len(self.active_templates) == 0:
                return
            if mutex.acquire(blocking=False):
                try:
                    # print("Saving to: ", camera_frame_id)
                    if camera_frame_id not in self.latest_msgs or msg.header.stamp > self.latest_msgs[camera_frame_id].header.stamp:
                        self.latest_msgs[camera_frame_id] = msg
                        # print("Saving recent timestamp: ", msg.header.stamp)
                    # else: 
                        # print("Received old timestamp! ", msg.header.stamp)
                except Exception:
                    print("ee")
                    print(traceback.format_exc())
                finally:
                    # print("Mutex released")
                    mutex.release()
            else:
                rospy.logwarn_throttle(1.0, "Dropping message for %s", camera_frame_id)

        return callback

    def cropped_image_callback(self, debug=True):
        # rospy.loginfo_throttle(5, self.active_templates)
        with mutex:
            images = {}
            camera_stamp_poses: Dict[Tuple[float, np.ndarray]] = {}
            for camera_frame_id, msg in self.latest_msgs.items():
                try:
                    img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError as e:
                    rospy.logerr(e)
                    continue
                # CLAHE to L in LAB space
                # lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                # lab_img[:, :, 0] = self.clahe.apply(lab_img[:, :, 0])
                # img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

                # Contrast Normalization
                # img = cv2.normalize(
                #     img,
                #     None,
                #     alpha=0,
                #     beta=1.0,
                #     norm_type=cv2.NORM_MINMAX,
                #     dtype=cv2.CV_32F,
                # )
                # img = (255 * img).astype(np.uint8)

                images[camera_frame_id] = img
                try:
                    camera_tf = self.tf_buffer.lookup_transform(
                        self.map_ned_frame,
                        camera_frame_id,
                        msg.header.stamp,
                        rospy.Duration(3.0),
                    )
                except Exception as e:
                    rospy.logerr(e)
                    continue
                camera_stamp_poses[camera_frame_id] = (
                    msg.header.stamp,
                    compose(
                        attrgetter("x", "y", "z")(camera_tf.transform.translation),
                        quat2mat(
                            attrgetter("w", "x", "y", "z")(camera_tf.transform.rotation)
                        ),
                        np.ones(3),
                    ),
                )
            self.latest_msgs = {}
            # rospy.loginfo("End cb")
        active_templates = copy.deepcopy(self.active_templates)

        for active_template in active_templates:
            rospy.sleep(0.01)
            template_name, camera_frame_id = active_template
            if camera_frame_id not in self.pose_estimator.available_cameras:
                rospy.logerr(f"Camera {camera_frame_id} not registered")
                continue
            if camera_frame_id not in images.keys() or images[camera_frame_id] is None:
                rospy.logerr_throttle_identical(1.0, f"Camera {camera_frame_id} image not received, trying to restart")
                try:
                    print("Registering Camera...")
                    self.register_camera(self.topics[camera_frame_id], self.pose_estimator.cameras[camera_frame_id])
                    print("Camera registered!")
                except Exception as e:
                    print(traceback.format_exc())
                continue
            if template_name not in self.templates:
                rospy.logerr_throttle_identical(
                    1.0, f"Template {template_name} not registered"
                )
                continue
            if (
                camera_frame_id not in camera_stamp_poses
                or len(camera_stamp_poses[camera_frame_id]) == 0
            ):
                rospy.logerr(f"No camera poses found for {camera_frame_id}")
                continue
            _s = camera_stamp_poses[camera_frame_id][0].secs
            _ns = camera_stamp_poses[camera_frame_id][0].nsecs
            if _s == 0 and _ns == 0:
                rospy.logerr(f"Camera {camera_frame_id} has no timestamp, skipping")
                continue
            rospy.logdebug_throttle(
                10,
                f"Processing {template_name}<->{camera_frame_id}: {_s}.{_ns}",
            )
            template = self.templates[template_name]
            print("Computing pose...")
            rot, trans = self.pose_estimator.compute_pose(
                images[camera_frame_id],
                template_name,
                camera_frame_id,
                matcher=template.matcher,
                num_keypoints=300,
                lxtyrxby=None,
                debug=True,
                is_planar=False,  # Use homography to do rejection
                max_reprojection_error=template.reprojection_error_threshold,
                min_matches=template.min_matches,
            )
            print("Pose computed")
            print("Rot: ", rot)
            print("Trans: ", trans)
            if rot is not None and trans is not None and trans[2] > 0:
                yaw, pitch, roll = mat2euler(rot, axes="szyx")
                yaw = np.rad2deg(yaw)
                pitch = np.rad2deg(pitch)
                roll = np.rad2deg(roll)

                rospy.loginfo_throttle(
                    1,
                    f"YPR: {yaw:.2f}, {pitch:.2f}, {roll:.2f} {template_name} {trans}",
                )

                if np.all(np.abs(trans) < 1e-10) or np.all(np.abs([yaw, pitch, roll]) < 1e-10):
                    continue

                self.update_pose(
                    rot,
                    trans,
                    camera_frame_id,
                    template,
                    *camera_stamp_poses[camera_frame_id],
                    debug,
                )

    def update_keypoint_matches(self, req: IMPoseEstimatorUpdateKeypointMatchesRequest):
        """
        Update the keypoint matches for a template -> image without performing feature matching

        Ensure image is rectified version if available.
        """
        template_name = req.template_name
        camera_frame_id = req.header.frame_id
        if req.template_name not in self.pose_estimator.available_templates:
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False, f"Template {req.template_name} not registered"
            )
        if camera_frame_id not in self.pose_estimator.cameras:
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False, f"Camera {camera_frame_id} not registered"
            )
        if template_name not in self.templates:
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False, f"Template {req.template_name} not registered"
            )
        try:
            camera_tf = self.tf_buffer.lookup_transform(
                self.map_ned_frame, camera_frame_id, req.header.stamp, rospy.Duration(3.0)
            )
            camera_stamp_pose = (
                req.header.stamp,
                compose(
                    attrgetter("x", "y", "z")(camera_tf.transform.translation),
                    quat2mat(
                        attrgetter("w", "x", "y", "z")(camera_tf.transform.rotation)
                    ),
                    np.ones(3),
                ),
            )
        except Exception as e:
            rospy.logerr(e)
            return IMPoseEstimatorUpdateKeypointMatchesResponse(False, str(e))

        kp1 = np.array([x.coord for x in req.keypoints.ref_keypoints])
        kp2 = np.array([x.coord for x in req.keypoints.cur_keypoints])
        if len(kp1) != len(kp2):
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False,
                f"Invalid keypoints: got different numbers of correspondences: \
{len(kp1)}, {len(kp2)}",
            )
        if len(kp1) < max(4, self.templates[template_name].min_matches):
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False,
                f"Invalid keypoints: Need at least {max(4, self.templates[template_name].min_matches)} pairs of keypoints",
            )
        rot, trans, _ = self.pose_estimator.compute_pose_from_keypoints(
            template_name,
            camera_frame_id,
            kp1,
            kp2,
            is_planar=False,
            max_reprojection_error=self.templates[
                template_name
            ].reprojection_error_threshold,
            debug=self.debug,
        )

        euler = mat2euler(rot, "szyx")
        y = np.rad2deg(euler[0])
        p = np.rad2deg(euler[1])
        r = np.rad2deg(euler[2])

        print(f"Estimated Rot: {y}, {p}, {r}")
        print(f"Estimated trans: {trans}")

        if np.all(np.abs(trans) < 1e-10) or np.all(np.abs([y, p, r]) < 1e-10):
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False, "Failed to compute pose! All values are near zero!"
            )

        if rot is not None and trans is not None and trans[2] > 0:
            self.update_pose(
                rot,
                trans,
                camera_frame_id,
                self.templates[template_name],
                *camera_stamp_pose,
                debug,
            )
        else:
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False, "Failed to compute pose!"
            )
        return IMPoseEstimatorUpdateKeypointMatchesResponse(True, "")

    def update_raw_pose(self, rot, trans, frame_id, template: Template, stamp):
        """
        Updates the pose estimate of the template in the camera frame with no filtering

        Args:
            rot: 3x3 rotation matrix
            trans: 3x1 translation vector
            frame_id: camera frame id
            template: Template object
            stamp: timestamp of the image
        """
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = frame_id
        transform.child_frame_id = f"{template.name}_raw_optical"

        transform.transform.translation = Vector3(trans[0], trans[1], trans[2])
        q = mat2quat(rot)
        transform.transform.rotation = Quaternion(q[1], q[2], q[3], q[0])

        self.br.sendTransform(transform)

    def update_pose(
        self, rot, trans, frame_id, template: Template, stamp, camera_pose, debug=False
    ):
        """
        Updates the pose estimate of the template in the world frame
        params:
            rot: 3x3 rotation matrix
            trans: 3x1 translation vector
            frame_id: camera frame id
            template: Template object
            camera_pose: 4x4 homogeneous transformation matrix from
            world to camera frame
        """
        template_object = self.template_objects[template.object_name]
        object_trans = np.array(
            [trans[0] - template.offset[0], trans[1] - template.offset[1], trans[2]]
        )
        tfm_camera_to_frame = np.eye(4)
        tfm_camera_to_frame[:3, :3] = rot
        tfm_camera_to_frame[:3, -1] = object_trans

        tfm_world_to_frame = camera_pose @ tfm_camera_to_frame

        try:
            T, R, _, _ = decompose(tfm_world_to_frame)
        except Exception as e:
            rospy.logwarn_throttle(
                1,
                f"Failed to decompose {e}",
            )
            return
        object_quat = mat2quat(R)

        x, y, z = T

        pose = np.array([x, y, z, *object_quat])
        if any(np.isnan(pose)) or any(np.isinf(pose)) or any(np.abs([x, y, z]) > 1000):
            rospy.logwarn_throttle(
                1,
                f"Invalid pose estimate for {template.object_name} in {frame_id}: \
{x:.2f}, {y:.2f}, {z:.2f}, {object_quat}",
            )
            return

        if (
            template.object_name in self.custom_pose_filtering
            and not self.custom_pose_filtering[template.object_name](pose)
        ):
            rospy.logwarn_throttle(
                1,
                f"Invalid pose estimate for {template.object_name} based on custom filter",
            )
            return

        qw, qx, qy, qz = object_quat
        template_object.poses.loc[len(template_object.poses)] = [stamp.secs, *pose]
        if len(template_object.poses) > template_object.min_buffer_size:
            old_rows = template_object.poses.iloc[
                : len(template_object.poses) - template_object.min_buffer_size
            ]
            keep_rows = template_object.poses.iloc[
                len(template_object.poses) - template_object.min_buffer_size :
            ]
            if template_object.max_history > 0:
                template_object.poses = pd.concat(
                    [
                        old_rows.loc[
                            (
                                old_rows.stamp
                                < stamp.secs - template_object.max_history
                            ).index
                        ],
                        keep_rows,
                    ]
                )
            else:
                template_object.poses = pd.concat(
                    [
                        old_rows,
                        keep_rows,
                    ]
                )
            template_object.poses = template_object.poses.iloc[
                -template_object.max_buffer_size :
            ]
        template_object.poses.reset_index(drop=True, inplace=True)
        if self.debug:
            global debug_file
            debug_file.write(
                f"{template_object.name}, {stamp}, {x}, {y}, \
{z}, {qw}, {qx}, {qy}, {qz}\n"
            )

        if len(template_object.poses) < template_object.min_buffer_size:
            return
        poses = template_object.poses.to_numpy()[:, 1:]
        fused_pose = get_kmeans_center(poses)

        fused_pose_ang = quat2euler(fused_pose[3:])
        _fused_pose = np.hstack([fused_pose[:3], fused_pose_ang])
        _poses = np.hstack(
            [poses[:, :3], np.array([quat2euler(q) for q in poses[:, 3:]])]
        )
        _err = _poses - _fused_pose
        variance = np.maximum(np.var(_err, 0), 0.00001)

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = stamp
        transform_stamped.header.frame_id = self.map_ned_frame
        transform_stamped.child_frame_id = template.object_name + "_optical"

        transform_stamped.transform.translation = Vector3(*fused_pose[:3])
        qw, qx, qy, qz = fused_pose[3:]
        transform_stamped.transform.rotation = Quaternion(qx, qy, qz, qw)

        self.br.sendTransform(transform_stamped)
        transform_zeroed = transform_stamped

        rangle = np.pi / 2
        r, p, y = quat2euler(fused_pose[3:], axes="rzyx")
        new_r, new_p, new_y = (
            r,
            np.round(p / rangle) * rangle,
            np.round(y / rangle) * rangle,
        )
        qw, qx, qy, qz = euler2quat(new_r, new_p, new_y, axes="rzyx")

        if template.object_name in self.custom_stabilized_orientation_transform:
            qw, qx, qy, qz = self.custom_stabilized_orientation_transform[
                template.object_name
            ](np.array([qw, qx, qy, qz])).tolist()

        transform_zeroed.transform.rotation = Quaternion(qx, qy, qz, qw)
        transform_zeroed.child_frame_id = template.object_name + "_stabilized"
        self.br.sendTransform(transform_zeroed)

        fused_pose_covariance_stamped = PoseWithCovarianceStamped()
        fused_pose_covariance_stamped.header.stamp = stamp
        fused_pose_covariance_stamped.header.frame_id = self.map_ned_frame
        fused_pose_covariance_stamped.pose.pose.position = Point(*fused_pose[:3])
        fused_pose_covariance_stamped.pose.pose.orientation = (
            transform_zeroed.transform.rotation
        )
        fused_pose_covariance_stamped.pose.covariance = (
            np.diag(variance).flatten().tolist()
        )

        template_object.computed_pose = fused_pose_covariance_stamped

        rospy.loginfo(
            f"Published transform {template.object_name}_stabilized:\
                {transform_stamped.transform.translation}",
        )

        odometry = Odometry()
        odometry.header = fused_pose_covariance_stamped.header
        odometry.pose.pose = fused_pose_covariance_stamped.pose
        odometry.child_frame_id = template.object_name + "_stabilized"
        odometry.pose = fused_pose_covariance_stamped.pose
        self.odom_pub.publish(odometry)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rospy.init_node("pose_estimator_dev", anonymous=False, log_level=rospy.INFO)
    debug = rospy.get_param("~debug", True)
    if debug:
        debug_file = open("debug_poses.csv", "w")
        rospy.loginfo(f"Writing debug poses to {os.path.abspath(debug_file.name)}")
        rospy.on_shutdown(lambda: debug_file.close())
    front_camera_topic = rospy.get_param(
        "~front_camera_topic", "/auv4/front_cam/image_rect_color/compressed"
    )
    front_camera_info_topic = rospy.get_param(
        "~front_camera_info_topic", "/auv4/front_cam/camera_info"
    )
    bottom_camera_topic = rospy.get_param(
        "~bottom_camera_topic", "/auv4/bot_cam/image_rect_color/compressed"
    )
    bottom_camera_info_topic = rospy.get_param(
        "~bottom_camera_info_topic", "/auv4/bot_cam/camera_info"
    )
    visualization_topic = rospy.get_param(
        "~visualization_topic", "/pose_estimator_vis/compressed"
    )

    detected_objects_topic = rospy.get_param("~detected_objects_topic", None)

    matcher = rospy.get_param("~matcher", "superpoint_lightglue")
    map_ned_frame = rospy.get_param("~map_ned_frame", "world_ned")

    # Register templates, template dimensions from json file
    templates_dir = os.path.abspath(
        Path(RosPack().get_path("image_matching")) / "templates"
    )

    def get_matcher(matcher):
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
        elif matcher == "superpoint_lightglue":
            image_match_producer = get_keypoints_match_producer(
                "superpoint", "lightglue", {"debug": True}, {"debug": True}
            )
        elif matcher == "fast_bf":
            image_match_producer = get_keypoints_match_producer(
                "fast", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher == "orb_bf":
            image_match_producer = get_keypoints_match_producer(
                "orb", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher == "orb_flann":
            image_match_producer = get_keypoints_match_producer(
                "orb", "flann", {"debug": True}, {"debug": True}
            )
        elif matcher == "alike_bf":
            image_match_producer = get_keypoints_match_producer(
                "alike", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher == "dkm":
            image_match_producer = get_keypoints_match_producer(
                None, "dkm", {"debug": True}, {"debug": True}
            )
        elif matcher == "keyaffhard_flann":
            image_match_producer = get_keypoints_match_producer(
                "keyaffhard", "flann", {"debug": True}, {"debug": True}
            )
        elif matcher == "disk_lightglue": 
            image_match_producer = get_keypoints_match_producer(
                "disk", "lightglue", {"debug": True}, {"debug": True, "weights": "disk"}
            )
        elif matcher == "dalf_flann": 
            image_match_producer = get_keypoints_match_producer(
                "dalf", "flann", {"debug": True}, {"debug": True}
            )
        elif matcher == "dalf_bf": 
            image_match_producer = get_keypoints_match_producer(
                "dalf", "bf", {"debug": True}, {"debug": True}
            )
        else:
            raise NotImplementedError(f"Matcher {matcher} unimplemented!")
        return image_match_producer

    matchers = {}
    matchers["sift_flann"] = get_matcher("sift_flann")
    # matchers["dalf_bf"] = get_matcher("dalf_bf")
    # matchers["keyaffhard_flann"] = get_matcher("keyaffhard_flann")
    matchers["superpoint_lightglue"] = get_matcher("superpoint_lightglue") # specify in launch file
    if matcher not in matchers:
        matchers[matcher] = get_matcher(matcher)
    pose_estimator = BasicPoseEstimator(
        matchers,
        visualization_topic,
        detected_objects_topic,
        templates_dir,
        debug,
        map_ned_frame,
    )

    # NOTE: template.json values are real world dimensions corresponding to
    # width and height of image: [width, height] in meters.
    templates = json.loads(open(os.path.join(templates_dir, "templates.json")).read())
    for template in templates.keys():
        if template.startswith("_"):
            continue
        template = os.path.splitext(template)[0]
        template_path = os.path.join(templates_dir, template)
        possible_templates = glob.glob(os.path.join(templates_dir, f"{template}.*"))
        if not possible_templates:
            rospy.logwarn_once(f"No template found for {template} in {templates_dir}")
            continue
        template_path = possible_templates[-1]  # take newest template
        rospy.loginfo(f"Registering template {template_path}")

        template_filename = template_path.split("/")[-1]

        template_img = cv2.imread(template_path)
        regions = {}
        regions[template] = [0, 0, 1, 1]
        if isinstance(templates[template_filename], list):
            template_width = templates[template_filename][0]
            template_height = templates[template_filename][1]
        else:
            template_width = templates[template_filename]["dimensions"][0]
            template_height = templates[template_filename]["dimensions"][1]
            for region_name, region in templates[template_filename]["regions"].items():
                template_name = f"{template}_{region_name}"
                if template_name in templates.keys():
                    rospy.logwarn_once(f"{template_name} already registered!")
                    continue
                regions[template_name] = region

        for region_name, region in regions.items():
            if not isinstance(region, list) or len(region) != 4:
                continue
            x1, y1, x2, y2 = region
            x1, x2 = int(x1 * template_img.shape[1]), int(x2 * template_img.shape[1])
            y1, y2 = int(y1 * template_img.shape[0]), int(y2 * template_img.shape[0])
            template_img_width, template_img_height = x2 - x1, y2 - y1
            region_img = template_img[y1:y2, x1:x2]
            print(template_img.shape, region_img.shape)
            region_px_offset = (
                (x1 + x2) / 2 - template_img.shape[1] / 2,
                (y1 + y2) / 2 - template_img.shape[0] / 2,
            )
            region_offset = (
                region_px_offset[0] / template_img.shape[1] * template_width,
                region_px_offset[1] / template_img.shape[0] * template_height,
            )
            region_width, region_height = (
                (template_img_width / template_img.shape[1]) * template_width,
                (template_img_height / template_img.shape[0]) * template_height,
            )

            if region_img.shape[0] > 480 or region_img.shape[1] > 480:
                rospy.logerr(f"Region {region_name} is too large! Resizing the image")
                if template_img_height > template_img_width:
                    template_img_width, template_img_height = 480, int(
                        480 * template_img_width / template_img_height
                    )
                else:
                    template_img_width, template_img_height = (
                        int(480 * region_img.shape[0] / region_img.shape[1]),
                        480,
                    )
                region_img = cv2.resize(
                    region_img, (template_img_height, template_img_width)
                )
            rospy.loginfo(
                f"Using template dimensions {region_width}x{region_height} \
    for template of size {region_img.shape[:2]} with offset {region_offset}"
            )
            pose_estimator.register_template(
                region_img,
                region_name,
                (region_width, region_height),
                template,
                region_offset,
            )
            rospy.sleep(0.05)

    print("Registered templates")
    try:
        if front_camera_topic is not None and front_camera_info_topic is not None:
            front_camera_info = rospy.wait_for_message(
                front_camera_info_topic, CameraInfo, timeout=5
            )
            pose_estimator.register_camera(
                front_camera_topic,
                PinholeCamera.from_camera_info(
                    front_camera_info, "rect" in front_camera_topic
                ),
            )
    except:
        rospy.logwarn("Front camera not found! Using default")
        pose_estimator.register_camera(
            front_camera_topic,
            PinholeCamera(
                "auv4/front_cam_optical",
                1024,
                768,
                452.3013610839844,
                482.3131408691406,
                526.00118954543,
                396.61607947004813,
            ),
        )
    try:
        if bottom_camera_topic is not None and bottom_camera_info_topic is not None:
            bottom_camera_info = rospy.wait_for_message(
                bottom_camera_info_topic, CameraInfo, timeout=5
            )
            pose_estimator.register_camera(
                bottom_camera_topic,
                PinholeCamera.from_camera_info(
                    bottom_camera_info, "rect" in bottom_camera_topic
                ),
            )
    except:
        rospy.logwarn("Bottom camera not found! Using default")
        pose_estimator.register_camera(
            bottom_camera_topic,
            PinholeCamera(
                "auv4/bot_cam_optical",
                1024,
                768,
                436.40875244140625,
                467.6256103515625,
                510.88065980075044,
                376.3738157469634,
            ),
        )

    rospy.Timer(
        rospy.Duration(0.05), pose_estimator.cropped_image_callback, reset=False
    )

    rospy.on_shutdown(pose_estimator.teardown)
    rospy.spin()
