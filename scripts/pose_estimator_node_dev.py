#!/usr/bin/env python3

from dataclasses import dataclass
import json
import logging
from operator import attrgetter
import os
from pathlib import Path
import glob
import copy

import cv2
import pandas as pd
from typing import Any, Dict, Optional, Set, Tuple
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import (
    Point, PoseStamped, Vector3, Quaternion, TransformStamped
)
from nav_msgs.msg import Odometry
from bb_msgs.msg import DetectedObjects
from bb_msgs.srv import (
    IMPoseEstimatorToggleTemplate,
    IMPoseEstimatorToggleTemplateResponse,
    IMPoseEstimatorGetTemplates,
    IMPoseEstimatorGetTemplatesResponse,
    IMPoseEstimatorRegisterTemplate,
    IMPoseEstimatorRegisterTemplateRequest,
    IMPoseEstimatorRegisterTemplateResponse,
    IMPoseEstimatorConfig,
    IMPoseEstimatorConfigResponse,
    IMPoseEstimatorUpdateKeypointMatches,
    IMPoseEstimatorUpdateKeypointMatchesRequest,
    IMPoseEstimatorUpdateKeypointMatchesResponse,
    IMPoseEstimatorGetStatus,
    IMPoseEstimatorGetStatusRequest,
    IMPoseEstimatorGetStatusResponse,
)

import threading
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat, mat2euler
from transforms3d.affines import compose, decompose

from feature_matcher.keypoints_match_producer\
    import KeypointsMatchProducer, get_keypoints_match_producer
from pose_estimator.pose_weighted_average\
    import get_kmeans_center
from pose_estimator.PinholeCamera import PinholeCamera
from pose_estimator.pose_estimator import PoseEstimator

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
    poses: pd.DataFrame
    matcher: str
    computed_pose: Optional[PoseStamped]
    min_buffer_size: int
    max_buffer_size: int
    min_matches: int
    max_history: float
    reprojection_error_threshold: float

class BasicPoseEstimator:
    def __init__(
        self,
        image_match_producers: Dict[str, KeypointsMatchProducer],
        visualization_topic,
        detected_objects_topic=None,
        templates_dir="./",
        debug=False,
    ):

        self.latest_msgs: Dict[str, cv2.Mat] = {}
        self.bridge = CvBridge()
        self.templates: Dict[str, Template] = {}
        self.templates_dir = templates_dir

        self.image_match_producers = image_match_producers
        self.pose_estimator = PoseEstimator(image_match_producers)

        self.visualization_pub = rospy.Publisher(
            visualization_topic, CompressedImage, queue_size=1
        )

        self.debug = debug
        if debug:
            self.pose_estimator.visualize_callbacks.append(
                lambda img: self.visualization_pub.publish(
                    self.bridge.cv2_to_compressed_imgmsg(img, "jpeg")
                )
            )

        self.active_templates: Set[
            Tuple[str, str]
        ] = set()  # (template_name, camera_frame_id)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(15))
        self.tf_sub = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.odom_pub = rospy.Publisher(
            "impose_estimates",
            Odometry,
            queue_size=1
        )

        self.update_keypoint_matches_service = rospy.Service(
            "impose_update_keypoint_matches",
            IMPoseEstimatorUpdateKeypointMatches,
            self.update_keypoint_matches
        )
        self.get_templates_service = rospy.Service(
            "impose_get_templates",
            IMPoseEstimatorGetTemplates,
            self.get_templates
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

        self.subscribers: Dict[str, rospy.Subscriber] = {}
        rospy.Timer(rospy.Duration(0.3), self.cropped_image_callback)

    def update_config(self, req):
        template_name = req.template_name
        matcher = req.matcher
        if template_name not in self.templates.keys():
            self.templates[template_name] = self.create_default_template(template_name)
            rospy.logerr(f"Template {template_name} not registered")
            return IMPoseEstimatorConfigResponse(False)

        if matcher not in self.image_match_producers.keys():
            rospy.logerr(f"Matcher {matcher} not loaded")
            return IMPoseEstimatorConfigResponse(False)

        setattr(self.templates[template_name], "reprojection_error_threshold",
                req.max_reprojection_threshold)
        setattr(self.templates[template_name], "max_history", req.max_history)
        setattr(self.templates[template_name], "min_buffer_size", req.min_buffer_size)
        setattr(self.templates[template_name], "max_buffer_size", req.max_buffer_size)
        setattr(self.templates[template_name], "min_matches", req.min_matches)
        setattr(self.templates[template_name], "matcher", matcher)

        if req.reset:
            self.templates[template_name].poses = pd.DataFrame(
                columns=["stamp", "x", "y", "z", "qw", "qx", "qy", "qz"])
            self.templates[template_name].computed_pose = None
        rospy.loginfo(f"Config updated: {req}")
        return IMPoseEstimatorConfigResponse(success=True)

    def get_templates(self, req):
        active_templates = list(set([template_name for template_name,
                      _ in self.active_templates]))
        active_templates = [template_name + ": " + self.templates[template_name].matcher for template_name in active_templates]
        return IMPoseEstimatorGetTemplatesResponse(
            self.pose_estimator.available_templates,
            active_templates,
            list(self.image_match_producers.keys()),
        )

    def toggle_template(self, req):
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
            if not (req.template_name, req.camera_frame_id) in \
                    self.active_templates:
                return IMPoseEstimatorToggleTemplateResponse(
                    False, f"Template {req.template_name} not active"
                )
            self.active_templates.remove((req.template_name,
                                          req.camera_frame_id))
        print("toggled template")
        return IMPoseEstimatorToggleTemplateResponse(
            (req.template_name, req.camera_frame_id) in self.active_templates,
            ""
        )

    @staticmethod
    def create_default_template(name):
        return Template(
            name,
            pd.DataFrame(
                columns=["stamp", "x", "y", "z", "qw", "qx", "qy", "qz"]),
            "sift_flann",
            None,
            1,  # min_buffer_size
            20,  # max_buffer_size
            4, # min_matches
            10,  # max_history,
            2,  # reprojection_error_threshold
        )

    def get_status(self, req: IMPoseEstimatorGetStatusRequest):
        res = IMPoseEstimatorGetStatusResponse()
        if req.template_name not in self.templates.keys():
            res.is_valid = False
            return res
        template = self.templates[req.template_name]
        if (template.computed_pose is None):
            res.is_valid = False
            return res
        time_since_last = rospy.Time.now().secs-template.computed_pose.header.stamp.secs
        if time_since_last > template.max_history:
            res.is_valid = False
            res.num_poses = len(template.poses)
            return res
        res.pose = template.computed_pose
        res.is_valid = True
        res.num_poses = len(template.poses)
        return res

    def register_template(self, img, name, dimensions):
        self.pose_estimator.register_template(name, dimensions, img)
        self.templates[name] = BasicPoseEstimator.create_default_template(name)

    def register_template_cb(self,
                             req: IMPoseEstimatorRegisterTemplateRequest):
        compressed_image_topic_name = req.image_topic_name
        detected_objects_topic_name = req.detected_objects_topic_name
        object_name = req.object_name
        detected_object = None
        try:
            if detected_objects_topic_name != "" and object_name != "":
                for i in range(3):
                    try:
                        detected_objects = rospy.wait_for_message(
                            detected_objects_topic_name, DetectedObjects,
                            timeout=2
                        )
                    except rospy.ROSException:
                        continue
                    if any([x.name == object_name
                            for x in detected_objects.detected]):
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
                    y - PADDING:y + h + PADDING,
                    x - PADDING:x + w + PADDING,
                    :
                ]
            cv2.imwrite(
                os.path.join(
                    self.templates_dir,
                    f"{req.template_name}.{rospy.Time.now().secs}.jpg",
                ),
                cv2_img,
            )
            if req.template_name in self.templates:
                rospy.loginfo("Replacing existing template %s",
                              req.template_name)
            self.register_template(
                cv2_img,
                req.template_name,
                (req.width, req.height),
            )
            return IMPoseEstimatorRegisterTemplateResponse(True, "")
        except Exception as e:
            return IMPoseEstimatorRegisterTemplateResponse(False, str(e))

    def register_camera(self, camera_topic: str, camera: PinholeCamera):
        self.subscribers[camera_topic] = rospy.Subscriber(
            camera_topic,
            CompressedImage,
            self.msg_callback(camera.frame_id),
            queue_size=1,
        )
        self.pose_estimator.register_camera(camera)

    def msg_callback(self, camera_frame_id):
        def callback(msg):
            rospy.loginfo_throttle(10, f"Received image from {camera_frame_id}")
            mutex.acquire(blocking=True)
            self.latest_msgs[camera_frame_id] = msg
            mutex.release()

        return callback

    def cropped_image_callback(self, debug=True):
        rospy.loginfo_throttle(5, self.active_templates)
        mutex.acquire(blocking=True)
        images = {}
        camera_stamp_poses: Dict[Tuple[float, np.ndarray]] = {}
        for camera_frame_id, msg in self.latest_msgs.items():
            try:
                img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
                continue
            images[camera_frame_id] = img
            try:
                camera_tf = self.tf_buffer.lookup_transform(
                    "world_ned", camera_frame_id, msg.header.stamp,
                    rospy.Duration(2)
                )
            except Exception as e:
                rospy.logerr(e)
                continue
            camera_stamp_poses[camera_frame_id] = (
                msg.header.stamp,
                compose(
                    attrgetter("x", "y", "z")(camera_tf.transform.translation),
                    quat2mat(attrgetter("w", "x", "y", "z")(
                        camera_tf.transform.rotation
                    )),
                    np.ones(3),
                ),
            )

        active_templates = copy.deepcopy(self.active_templates)
        mutex.release()

        for active_template in active_templates:
            template_name, camera_frame_id = active_template
            if camera_frame_id not in self.pose_estimator.available_cameras:
                rospy.logerr(f"Camera {camera_frame_id} not registered")
                continue
            if camera_frame_id not in images.keys() or images[camera_frame_id] is None:
                rospy.logerr(f"Camera {camera_frame_id} image not received")
                continue
            if template_name not in self.templates:
                rospy.logerr(f"Template {template_name} not registered")
                continue
            if camera_frame_id not in camera_stamp_poses or len(camera_stamp_poses[camera_frame_id]) == 0:
                rospy.logerr(f"No camera poses found for {camera_frame_id}")
                continue
            _s = camera_stamp_poses[camera_frame_id][0].secs
            _ns = camera_stamp_poses[camera_frame_id][0].nsecs
            rospy.logdebug_throttle(
                10,
                f"Processing {template_name}<->{camera_frame_id}: {_s}.{_ns}",
            )
            template = self.templates[template_name]

            rot, trans = self.pose_estimator.compute_pose(
                images[camera_frame_id],
                template_name,
                camera_frame_id,
                matcher = template.matcher,
                num_keypoints=300,
                lxtyrxby=None,
                debug=True,
                is_planar=False,  # Use homography to do rejection
                max_reprojection_error=template.reprojection_error_threshold,
                min_matches=template.min_matches
            )
            print(rot, trans, template_name)
            if rot is not None and trans is not None and trans[2] > 0:
                yaw, pitch, roll = mat2euler(rot, axes="szyx")
                rospy.loginfo_throttle(
                    1,
                    f"YPR: {np.rad2deg(yaw):.2f}, {np.rad2deg(pitch):.2f}, {np.rad2deg(roll):.2f}",
                )

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
                "world_ned", camera_frame_id, req.header.stamp,
                rospy.Duration(2)
            )
            camera_stamp_pose = (
                req.header.stamp,
                compose(
                    attrgetter("x", "y", "z")(camera_tf.transform.translation),
                    quat2mat(attrgetter("w", "x", "y", "z")(
                        camera_tf.transform.rotation
                    )),
                    np.ones(3),
                ),
            )
        except Exception as e:
            rospy.logerr(e)
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False,
                str(e)
            )

        kp1 = np.array([x.coord for x in req.keypoints.ref_keypoints])
        kp2 = np.array([x.coord for x in req.keypoints.cur_keypoints])
        if len(kp1) != len(kp2):
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False,
                f"Invalid keypoints: got different numbers of correspondences: \
{len(kp1)}, {len(kp2)}"
            )
        if len(kp1) < max(4, self.templates[template_name].min_matches):
            return IMPoseEstimatorUpdateKeypointMatchesResponse(
                False,
                f"Invalid keypoints: Need at least {max(4, self.templates[template_name].min_matches)} pairs of keypoints"
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
            debug=self.debug
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
                False, 
                "Failed to compute pose!"
            )
        return IMPoseEstimatorUpdateKeypointMatchesResponse(
            True,
            ""
        )

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

        tfm_camera_to_frame = np.eye(4)
        tfm_camera_to_frame[:3, :3] = rot
        tfm_camera_to_frame[:3, -1] = trans

        tfm_world_to_frame = camera_pose @ tfm_camera_to_frame

        T, R, _, _ = decompose(tfm_world_to_frame)
        object_quat = mat2quat(R)

        x, y, z = T

        if any(np.isnan([x, y, z, *object_quat])) or any(
            np.isinf([x, y, z, *object_quat])
        ) or any(np.abs([x, y, z]) > 10000):
            rospy.logwarn_throttle(
                1,
                f"Invalid pose estimate for {template.name} in {frame_id}: \
{x:.2f}, {y:.2f}, {z:.2f}, {object_quat}",
            )
            return

        qw, qx, qy, qz = object_quat
        template.poses.loc[len(template.poses)] = [
            stamp.secs, x, y, z, qw, qx, qy, qz
        ]
        if len(template.poses) > template.min_buffer_size:
            old_rows = template.poses.iloc[
                :len(template.poses) - template.min_buffer_size
            ]
            keep_rows = template.poses.iloc[
                len(template.poses) - template.min_buffer_size:
            ]
            template.poses = pd.concat(
                [
                    old_rows.loc[
                        (old_rows.stamp < stamp.secs - template.max_history).index
                    ],
                    keep_rows,
                ]
            )
            template.poses = template.poses.iloc[-template.max_buffer_size:]
        template.poses.reset_index(drop=True, inplace=True)
        if self.debug:
            global debug_file
            debug_file.write(
                f"{template.name}, {stamp}, {x}, {y}, \
{z}, {qw}, {qx}, {qy}, {qz}\n"
            )

        if len(template.poses) < template.min_buffer_size:
            return
        poses = template.poses.to_numpy()[:, 1:]
        fused_pose = get_kmeans_center(poses)

        fused_pose_ang = quat2euler(fused_pose[3:])
        _fused_pose = np.hstack([
            fused_pose[:3],
            fused_pose_ang
        ])
        _poses = np.hstack([
            poses[:, :3],
            np.array([quat2euler(q) for q in poses[:, 3:]])])
        _err = _poses - _fused_pose
        variance = np.maximum(np.var(_err, 0), 0.00001)

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = rospy.Time.now()
        transform_stamped.header.frame_id = "world_ned"
        transform_stamped.child_frame_id = template.name + "_optical"

        transform_stamped.transform.translation = Vector3(*fused_pose[:3])
        qw, qx, qy, qz = fused_pose[3:]
        transform_stamped.transform.rotation = Quaternion(qx, qy, qz, qw)

        self.br.sendTransform(transform_stamped)
        transform_zeroed = transform_stamped

        rangle = np.pi/2
        r, p, y = quat2euler(fused_pose[3:], axes='rzyx')
        new_r, new_p, new_y = r, np.round(p / rangle) * rangle, np.round(y / rangle) * rangle
        qw, qx, qy, qz = euler2quat(new_r, new_p, new_y, axes='rzyx')
        transform_zeroed.transform.rotation = Quaternion(qx, qy, qz, qw)
        transform_zeroed.child_frame_id = template.name + "_stabilized"
        self.br.sendTransform(transform_zeroed)


        fused_pose_stamped = PoseStamped()
        fused_pose_stamped.header.stamp = rospy.Time.now()
        fused_pose_stamped.header.frame_id = "world_ned"
        fused_pose_stamped.pose.position = Point(*fused_pose[:3])
        fused_pose_stamped.pose.orientation = transform_zeroed.transform.rotation

        template.computed_pose = fused_pose_stamped

        rospy.loginfo_throttle(
            5,
            f"Published transform {template.name}_stabilized:\
                {transform_stamped.transform.translation}",
        )

        odometry = Odometry()
        odometry.header = fused_pose_stamped.header
        odometry.pose.pose = fused_pose_stamped.pose
        odometry.child_frame_id = template.name + "_stabilized"
        odometry.pose.covariance = np.diag(variance).flatten().tolist()
        self.odom_pub.publish(odometry)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rospy.init_node("pose_estimator_dev", anonymous=True)
    debug = rospy.get_param("~debug", False)
    print(debug)
    if debug:
        debug_file = open(f"debug_poses.csv", "w")
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

    matcher = rospy.get_param("~matcher", "superpoint_superglue")

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
        else:
            raise NotImplementedError(f"Matcher {matcher} unimplemented!")
        return image_match_producer
    matchers = {}
    matchers["sift_flann"] = get_matcher("sift_flann")
    matchers["superpoint_superglue"] = get_matcher("superpoint_superglue")
    if matcher not in matchers:
        matchers[matcher] = get_matcher(matcher)
    pose_estimator = BasicPoseEstimator(
        matchers,
        visualization_topic,
        detected_objects_topic,
        templates_dir,
        debug,
    )

    # NOTE: template.json values are real world dimensions corresponding to
    # width and height of image: [width, height] in meters.
    templates = json.loads(open(
        os.path.join(templates_dir, "templates.json")).read())
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
        template_width = templates[template_filename][0]
        template_height = templates[template_filename][1]
        template_img = cv2.imread(template_path)
        if template_img.shape[0] > 1000 or template_img.shape[1] > 1000:
            rospy.logerr(
                f"Template {template_path} is too large! Resize the image")
            continue
        rospy.loginfo(
            f"Using template dimensions {template_width}x{template_height} \
for template of size {template_img.shape[:2]}"
        )

        pose_estimator.register_template(
            template_img,
            template,
            (
                template_width,
                template_height,
            )
        )
    print("Registered templates")
    try:
        if front_camera_topic is not None and front_camera_info_topic is not None:
            front_camera_info = rospy.wait_for_message(front_camera_info_topic,
                                                    CameraInfo, timeout=5)
            pose_estimator.register_camera(
                front_camera_topic,
                PinholeCamera.from_camera_info(front_camera_info,
                                            "rect" in front_camera_topic),
            )
    except:
        rospy.logwarn("Front camera not found! Using default")
        pose_estimator.register_camera(
            front_camera_topic,
            PinholeCamera(
                "auv4/front_cam_optical",
                1024,
                768,
                452.3013610839844, 482.3131408691406, 526.00118954543, 396.61607947004813
            ))
    try:
        if bottom_camera_topic is not None and\
                bottom_camera_info_topic is not None:
            bottom_camera_info = rospy.wait_for_message(
                bottom_camera_info_topic, CameraInfo
                , timeout=5
            )
            pose_estimator.register_camera(
                bottom_camera_topic,
                PinholeCamera.from_camera_info(bottom_camera_info,
                                            "rect" in bottom_camera_topic),
            )
    except:
        rospy.logwarn("Bottom camera not found! Using default")
        pose_estimator.register_camera(
            bottom_camera_topic,
            PinholeCamera(
                "auv4/bot_cam_optical",
                1024,
                768,
                436.40875244140625, 467.6256103515625, 510.88065980075044, 376.3738157469634
            ))
    rospy.spin()
