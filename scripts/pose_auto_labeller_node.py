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
from typing import Any, Dict, Set, Tuple, List
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

import threading
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.affines import compose, decompose
from transforms3d.quaternions import quat2mat
from shapely.geometry import Polygon

from pose_estimator.PinholeCamera import PinholeCamera

@dataclass
class Image:
    img: cv2.Mat
    descriptor: Any
    timestamp: float
    pose: PoseStamped


@dataclass
class Template:
    name: str
    width: int
    height: int

mutex = threading.Lock()

class BasicPoseLabeller:
    PADDING=0
    def __init__(
        self,
        autolabel: bool,
        annotations_dir: str,
    ):
        self.autolabel = autolabel
        self.annotations_dir = annotations_dir
        print(autolabel, self.annotations_dir)
        if self.autolabel:
            self.df = pd.DataFrame(columns=["stamp", "uuid", "dataset_creation_date", "camera_id", "tags", "extrinsics", "intrinsics", "width", "height", "detection_valid", "segmentation_valid"])
            os.makedirs(self.annotations_dir, exist_ok=True)
            os.makedirs(self.annotations_dir + "/images", exist_ok=True)
            os.makedirs(self.annotations_dir + "/detect", exist_ok=True)
            os.makedirs(self.annotations_dir + "/segment", exist_ok=True)
        self.latest_msgs: Dict[str, cv2.Mat] = {}
        self.start_time = rospy.Time.now()
        self.bridge = CvBridge()
        self.templates: Dict[str, Template] = {}
        self.cameras: Dict[str, PinholeCamera] = {}
        self.subscribers = {}
        self.object_poses = {}
        self.records = []

        self.front_visualization_pub = rospy.Publisher(
            front_visualization_topic, CompressedImage, queue_size=1
        )
        self.bot_visualization_pub = rospy.Publisher(
            bot_visualization_topic, CompressedImage, queue_size=1
        )

        self.active_templates: Set[
            Tuple[str, str]
        ] = set()  # (template_name, camera_frame_id)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(15))
        self.tf_sub = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.debug_img_pub = rospy.Publisher(
            "/debug_pose_estimate/compressed", CompressedImage, queue_size=1)

        self.object_pose_sub = rospy.Subscriber(
            "/impose_estimates", Odometry, self.object_pose_callback, queue_size=1
        )
        
        rospy.Timer(rospy.Duration(1), self.cropped_image_callback)

    def register_template(self, name, dimensions):
        self.templates[name] = Template(name, dimensions[0], dimensions[1])

    def register_camera(self, camera_topic: str, camera: PinholeCamera):
        self.subscribers[camera_topic] = rospy.Subscriber(
            camera_topic,
            CompressedImage,
            self.msg_callback(camera.frame_id),
            queue_size=1,
        )
        self.cameras[camera.frame_id] = camera

    def msg_callback(self, camera_frame_id):
        def callback(msg):
            mutex.acquire(blocking=True)
            self.latest_msgs[camera_frame_id] = msg
            mutex.release()

        return callback
    
    def object_pose_callback(self, msg):
        self.object_poses[msg.child_frame_id] = msg

    def pose_to_pq(self, msg):
        """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([msg.orientation.x, msg.orientation.y,
                    msg.orientation.z, msg.orientation.w])
        return p, q
    
    def save_image_labels_yolov8(self, image, detections: List[Tuple[str, np.ndarray, np.ndarray]]):
        """Save the image and labels for yolov8

        Args:
            image (np.ndarray): image
            detections (List[Tuple[str, np.ndarray]]): list of (label, bbox)
        """
        if not self.autolabel:
            return
        if len(detections) == 0:
            return

        time_str = f"{rospy.get_time():.6f}"
        image_name = f"{time_str}.jpg"
        label_name = f"{time_str}.txt"
        img_h, img_w = image.shape[:2]

        with open(os.path.join(self.annotations_dir, "detect", label_name), "a") as f:
            for label, bbox, _ in detections:
                f.write(f"{label} {bbox[0]/img_w} {bbox[1]/img_h} {bbox[2]/img_w} {bbox[3]/img_h}\n")
        with open(os.path.join(self.annotations_dir, "segment", label_name), "a") as f:
            for label, _, polygon in detections:
                polygon[:,0]/=img_w
                polygon[:,1]/=img_h
                f.write(f"{label} " + " ".join(map(str, list(polygon.flatten()))) +"\n")
        cv2.imwrite(os.path.join(self.annotations_dir, "images", image_name), image)

    def write_df(self):
        df = pd.DataFrame.from_records(self.records)
        df.to_csv(f"{self.annotations_dir}/data.csv")
    def cropped_image_callback(self, debug=True):
        images = {}
        camera_stamp_poses: Dict[Tuple[float, np.ndarray]] = {}
        mutex.acquire(blocking=True)
        latest_msgs = self.latest_msgs.copy()
        mutex.release()
        if len(self.object_poses) == 0 or len(latest_msgs) == 0:
            return
        
        for camera_frame_id, msg in latest_msgs.items():
            try:
                img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
                continue
            images[camera_frame_id] = img
            try:
                camera_tf = self.tf_buffer.lookup_transform(
                    camera_frame_id, "world_ned", msg.header.stamp,
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
        
        if len(camera_stamp_poses) == 0:
            return
        for camera_frame in latest_msgs:
            img = images[camera_frame]
            vis = img.copy()
            if not camera_frame in camera_stamp_poses:
                continue
            camera_pose = camera_stamp_poses[camera_frame]
            image_height, image_width = images[camera_frame].shape[:2]
            image_polygon = Polygon([[0, 0], [image_width, 0],[image_width, image_height], [0, image_height]])
            detections = []
            stamp = rospy.Time.now()
            stamp_str = str(stamp.secs) + str(stamp.nsecs)[0]
            for world_object in self.object_poses.values():
                template_name = world_object.child_frame_id.split("_stabilized")[0]
                if not template_name in self.templates:
                    print(f"{template_name} not found")
                    continue
                width, height = attrgetter("width", "height")(self.templates[template_name])
                point_coords = np.array([[-width / 2 - self.PADDING, -height / 2 - self.PADDING, 0, 1],
                                      [width / 2 + self.PADDING, -height / 2 - self.PADDING, 0, 1],
                                      [width / 2 + self.PADDING, height / 2 + self.PADDING, 0, 1],
                                      [-width / 2 - self.PADDING, height / 2 + self.PADDING, 0, 1]])
                
                p, q = self.pose_to_pq(world_object.pose.pose)
                
                norm = np.linalg.norm(q)
                if np.abs(norm - 1.0) > 1e-3:
                    raise ValueError(
                        "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                            str(q), np.linalg.norm(q)))
                elif np.abs(norm - 1.0) > 1e-6:
                    q = q / norm
                g = np.eye(4)
                g[:3,:3] = quat2mat(q[[3, 0, 1, 2]])
                g[:3, 3] = p
                world_points = point_coords @ g.T
                world_points = world_points[:,:3] / world_points[:,3:]

                image_points = cv2.projectPoints(world_points, camera_pose[1][:3, :3], camera_pose[1][:3, 3], self.cameras[camera_frame].camera_matrix(), self.cameras[camera_frame].dist_coeffs())

                

                try:
                    poly = image_polygon.intersection(Polygon(image_points[0][:,0]))
                    if poly.is_empty:
                        continue
                except Exception as e:                    
                    print(e)
                    continue
                coords = poly.boundary.coords.xy
                min_x = int(np.min(coords[0]))
                max_x = int(np.max(coords[0]))
                min_y = int(np.min(coords[1]))
                max_y = int(np.max(coords[1]))
                cx = int((min_x + max_x) / 2)
                cy = int((min_y + max_y) / 2)
                w = max_x - min_x
                h = max_y - min_y
                coords = np.array(poly.boundary.coords.xy).T

                cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                for point in coords:
                    cv2.circle(vis, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                
                cv2.circle(vis, (int(image_points[0][0][0][0]), int(image_points[0][0][0][1])), 5, (255, 0, 0), -1)
                
                detections.append((template_name, (cx, cy, w, h), coords))
            tx, ty, tz = camera_pose[1][:3, 3].flatten()
            qw, qx, qy, qz = mat2quat(camera_pose[1][:3, :3])
            intrinsics = self.cameras[camera_frame].camera_matrix()

            self.records.append(
                {
                    "uuid": None,
                    "tags": "",
                    "stamp": stamp_str,
                    "camera_id": 288 if camera_frame == "auv4/front_cam_optical" else 289,
                    "width": image_width,
                    "height": image_height,
                    "dataset_creation_date": self.start_time,
                    "extrinsics": ";".join(map(str,[tx,ty,tz,qw,qx,qy,qz])),
                    "intrinsics": ";".join(map(str,[intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]])),
                    "detection_valid": False,
                    "segmentation_valid": False
                }
            )
            self.save_image_labels_yolov8(img, detections)    

            {"auv4/front_cam_optical": self.front_visualization_pub,
             "auv4/bot_cam_optical": self.bot_visualization_pub}[camera_frame].publish(self.bridge.cv2_to_compressed_imgmsg(vis))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rospy.init_node("pose_estimator_dev", anonymous=True)

    output_folder = rospy.get_param("~output_folder", "")
    debug = rospy.get_param("~debug", False)
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
    front_visualization_topic = rospy.get_param(
        "~front_visualization_topic", "/debug_front_pose_estimate/compressed"
    )
    bot_visualization_topic = rospy.get_param(
        "~bot_visualization_topic", "/debug_bot_pose_estimate/compressed"
    )
    templates_dir = os.path.abspath(
        Path(RosPack().get_path("image_matching")) / "templates"
    )
    # NOTE: template.json values are real world dimensions corresponding to
    # width and height of image: [width, height] in meters.
    templates = json.loads(open(
        os.path.join(templates_dir, "templates.json")).read())
    autolabel = rospy.get_param("~autolabel", False)
    annotations_dir = rospy.get_param("~output_folder", "")
    if autolabel and annotations_dir == "":
        raise("Missing annotations dir")

    pose_labeller = BasicPoseLabeller(autolabel=autolabel, annotations_dir=annotations_dir)
    rospy.on_shutdown(pose_labeller.write_df)
    for template in templates.keys():
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
        rospy.loginfo(
            f"Using template dimensions {template_width}x{template_height} \
for template of size {template_img.shape[:2]}"
        )

        pose_labeller.register_template(
            template,
            (
                template_width,
                template_height,
            )
        )

    try:
        if front_camera_topic is not None and front_camera_info_topic is not None:
            front_camera_info = rospy.wait_for_message(front_camera_info_topic,
                                                    CameraInfo, timeout=1)
            pose_labeller.register_camera(
                front_camera_topic,
                PinholeCamera.from_camera_info(front_camera_info,
                                            "rect" in front_camera_topic),
            )
    except:
        rospy.logwarn("Front camera not found! Using default")
        pose_labeller.register_camera(
            bottom_camera_topic,
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
                , timeout=1
            )
            pose_labeller.register_camera(
                bottom_camera_topic,
                PinholeCamera.from_camera_info(bottom_camera_info,
                                            "rect" in bottom_camera_topic),
            )
    except:
        rospy.logwarn("Bottom camera not found! Using default")
        pose_labeller.register_camera(
            bottom_camera_topic,
            PinholeCamera(
                "auv4/bot_cam_optical",
                1024,
                768,
                436.40875244140625, 467.6256103515625, 510.88065980075044, 376.3738157469634
            ))
    rospy.spin()
