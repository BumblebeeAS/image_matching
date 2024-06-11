#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import threading

from ament_index_python import get_package_share_directory
from bb_msgs.msg import DetectedObject
from bb_msgs.msg import DetectedObjects
from bb_msgs.srv import MLDetector
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from src.feature_matcher.keypoints_match_producer import (
    get_keypoints_match_producer,
)


class BasicFeatureMatcher(Node):
    def __init__(
        self,
        name,
        input_topic,
        visualization_topic,
        template,
        template_path,
        matcher_name: str,
        detected_objects_topic=None,
        save_detections_folder=None,
    ):
        # Create ROS Node
        super().__init__(name, allow_undeclared_parameters=True)

        # Init params
        input_topic = self.get_parameter_or("~camera_topic", input_topic)
        visualization_topic = self.get_parameter_or(
            "~visualization_topic", visualization_topic
        )
        template = self.get_parameter_or("~template", template)

        template_path = os.path.abspath(
            Path(get_package_share_directory("image_matching")) / "templates"
        )

        possible_templates = glob.glob(
            os.path.join(template_path, f"{template}.*")
        )
        if not possible_templates:
            self.get_logger().warn(
                f"No template found for {template} in {template_path}"
            )

        template_path = self.get_parameter_or(
            "~template_path",
            os.path.abspath(
                Path(get_package_share_directory("image_matching"))
                / "templates"
                / f"{template}.png"
            ),
        )

        save_detections_folder = self.get_parameter_or(
            "~output_dir", save_detections_folder
        )
        detected_objects_topic = self.get_parameter_or(
            "~detected_objects_topic", detected_objects_topic
        )
        matcher_name = self.get_parameter_or("~matcher", matcher_name)

        self.bridge = CvBridge()
        self.template = template
        self.template_img = cv2.imread(template_path)

        # Setup matcher
        if matcher_name == "coarse_loftr":
            self.image_match_producer = get_keypoints_match_producer(
                None, "coarse_loftr", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "loftr":
            # TODO: Doesn't work
            self.image_match_producer = get_keypoints_match_producer(
                None, "loftr", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "sift_flann":
            self.image_match_producer = get_keypoints_match_producer(
                "sift", "flann", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "sift_bf":
            self.image_match_producer = get_keypoints_match_producer(
                "sift", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "superpoint_bf":
            self.image_match_producer = get_keypoints_match_producer(
                "superpoint", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "superpoint_superglue":
            self.image_match_producer = get_keypoints_match_producer(
                "superpoint", "superglue", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "fast_bf":
            self.image_match_producer = get_keypoints_match_producer(
                "fast", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "orb_bf":
            self.image_match_producer = get_keypoints_match_producer(
                "orb", "bf", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "orb_flann":
            self.image_match_producer = get_keypoints_match_producer(
                "orb", "flann", {"debug": True}, {"debug": True}
            )
        elif matcher_name == "alike_bf":
            self.image_match_producer = get_keypoints_match_producer(
                "alike", "bf", {"debug": True}, {"debug": True}
            )
        else:
            raise NotImplementedError(f"Matcher: {matcher_name} is unknown!")

        self.visualization_pub = self.create_publisher(
            CompressedImage, visualization_topic, 1
        )

        self.detected_object_pub = self.create_publisher(
            DetectedObjects, detected_objects_topic, 1
        )

        # self.visualization_pub = rospy.Publisher(
        #     visualization_topic, CompressedImage, queue_size=1
        # )
        # self.detected_object_pub = rospy.Publisher(
        #     detected_objects_topic, DetectedObjects, queue_size=10
        # )
        self.image_match_producer.visualize_callbacks.append(
            lambda img: self.visualization_pub.publish(
                self.bridge.cv2_to_compressed_imgmsg(img, "jpeg")
            )
        )

        self.image_match_producer.register_template(
            template, cv2.imread(template_path)
        )

        self.PADDING = 10
        self.save_detections_folder = save_detections_folder

        self.image_sub = message_filters.Subscriber(
            self, CompressedImage, input_topic
        )
        # self.image_sub = message_filters.Subscriber(input_topic, CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub], 10, 1
        )
        ts.registerCallback(self.cropped_image_callback)

        # Detector service for Vision
        self.vision_service = self.create_service(
            MLDetector,
            "/auv4/vision/KP/detector",
            self.detector_srv_cb,
        )

        # Data
        self.lock = threading.Lock()
        self.img = None
        self.lxtyrxby = None
        self.detected_object = None

        # Thread to actually process images
        self.t = threading.Thread(target=self.process_image, daemon=True)
        self.t.start()

    def detector_srv_cb(self, req):
        # err = "" if req.dryRun else Detector.load_detectors(req.detectors)
        resp = MLDetector.Response()
        resp.success = True
        resp.runningDetectors = [self.template]
        return resp

    def cropped_image_callback(
        self, img_msg, detected_objects=None, debug=False
    ):
        self.get_logger().debug(
            f"Received image {img_msg.header.seq}", throttle_duration_sec=10
        )
        # rospy.logdebug_throttle(10, f"Received image {img_msg.header.seq}")
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        self.lock.acquire()
        self.img = img
        self.lock.release()

    def process_image(self):
        while True:
            lxtyrxby = None
            detected_object = None

            self.lock.acquire()
            img = None if self.img is None else self.img.copy()
            self.lock.release()
            if img is None:
                continue

            kp1, kp2 = self.image_match_producer.process_image(
                img,
                self.template,
                lxtyrxby=lxtyrxby,
                debug=True,
                num_keypoints=500,
            )

            # Process KP2
            if kp2 is None or kp2.shape[0] == 0:
                continue

            kp2 = np.round(kp2).astype(int)
            min_x = np.min(kp2[:, 0])
            max_x = np.max(kp2[:, 0])

            min_y = np.min(kp2[:, 1])
            max_y = np.max(kp2[:, 1])

            contours = [
                (min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, max_y),
            ]

            cx = int((max_x + min_x) / 2)
            cy = int((max_y + min_y) / 2.0)

            w = max_x - min_x
            h = max_y - min_y

            # No one publishing on the topic => our turn to publish
            if detected_object is None:
                objs = DetectedObjects()
                objs.nodeName = "keypoint_based_detector"
                obj = DetectedObject()
                obj.source = 288
                obj.name = self.template

                obj.centre_x = cx
                obj.centre_y = cy

                obj.bbox_height = h
                obj.bbox_width = w

                obj.contour = [
                    int(coord) for point in contours for coord in point
                ]

                obj.image_height = 768
                obj.image_width = 1024

                objs.detected = [obj]
                self.detected_object_pub.publish(objs)


def main():
    rclpy.init()

    camera_topic = "/auv4/front_cam/image_rect_color/compressed"
    visualization_topic = "/visualization/compressed"
    template = "Bootlegger"
    template_path = os.path.abspath(
        Path(get_package_share_directory("image_matching"))
        / "templates"
        / f"{template}.png"
    )
    matcher = "coarse_loftr"
    detected_objects_topic = "/auv4/vision/external/detected"
    save_detections_folder = "detections"
    detector = BasicFeatureMatcher(
        "kp_based_detector",
        camera_topic,
        visualization_topic,
        template,
        template_path,
        matcher,
        detected_objects_topic,
        save_detections_folder,
    )
    rclpy.spin(detector)

    # rospy.init_node("kp_based_detector", anonymous=True, log_level=rospy.DEBUG)
    # camera_topic = rospy.get_param(
    #     "~camera_topic", "/auv4/front_cam/image_rect_color/compressed"
    # )
    # visualization_topic = rospy.get_param(
    #     "~visualization_topic", "/visualization/compressed"
    # )
    # template = rospy.get_param("~template", "Bootlegger")

    # # Accept either png or jpeg files
    # templates_dir = os.path.abspath(
    #     Path(RosPack().get_path("image_matching")) / "templates"
    # )
    # possible_templates = glob.glob(os.path.join(templates_dir, f"{template}.*"))
    # if not possible_templates:
    #     rospy.logwarn_once(f"No template found for {template} in {templates_dir}")

    # template_path = rospy.get_param(
    #     "~template_path",
    #     os.path.abspath(
    #         Path(RosPack().get_path("image_matching"))
    #         / "templates"
    #         / f"{template}.png"
    #     ),
    # )
    # save_detections_folder = rospy.get_param("~output_dir", "detections")
    # detected_objects_topic = rospy.get_param(
    #     "~detected_objects_topic", "/auv4/vision/external/detected"
    # )
    # matcher = rospy.get_param("~matcher", "coarse_loftr")
    # detector = BasicFeatureMatcher(
    #     camera_topic,
    #     visualization_topic,
    #     template,
    #     template_path,
    #     matcher,
    #     detected_objects_topic,
    #     save_detections_folder,
    # )
    # rospy.spin()


if __name__ == "__main__":
    main()
