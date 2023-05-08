#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import numpy as np
import message_filters
import rospy
from bb_msgs.msg import DetectedObjects, DetectedObject
from bb_msgs.srv import MLDetector, MLDetectorResponse

from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage

from feature_matcher.keypoints_match_producer import get_keypoints_match_producer


class BasicFeatureMatcher:
    def __init__(
        self,
        input_topic,
        visualization_topic,
        template,
        template_path,
        detected_objects_topic=None,
        save_detections_folder=None,
    ):
        self.bridge = CvBridge()
        self.template = template
        self.template_img = cv2.imread(template_path)
        # self.image_match_producer = TwoStageMatchProducer(self.template_img, SuperPointKeypointProducer(), SuperglueKeypointMatcher())
        # self.image_match_producer = get_keypoints_match_producer(
        #     None, "loftr_ts", {"debug": True}, {"debug": True, "cuda": True}
        # )
        self.image_match_producer = get_keypoints_match_producer(
            "superpoint", "superglue", {"debug": True}, {"debug": True, "cuda": True}
        )

        self.visualization_pub = rospy.Publisher(
            visualization_topic, CompressedImage, queue_size=1
        )
        self.detected_object_pub = rospy.Publisher(
            detected_objects_topic, DetectedObjects, queue_size=10
        )
        self.image_match_producer.visualize_callbacks.append(
            lambda img: self.visualization_pub.publish(
                self.bridge.cv2_to_compressed_imgmsg(img, "jpeg")
            )
        )

        self.image_match_producer.register_template(template, cv2.imread(template_path))

        self.PADDING = 10
        self.CROP_IMAGES = False  # detected_objects_topic is not None
        self.save_detections_folder = save_detections_folder

        if self.CROP_IMAGES:
            rospy.loginfo("Subscribing to detected objects")
            self.detected_objects_sub = message_filters.Subscriber(
                detected_objects_topic, DetectedObjects
            )
        self.image_sub = message_filters.Subscriber(input_topic, CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.detected_objects_sub]
            if self.CROP_IMAGES
            else [self.image_sub],
            10,
            1,
        )
        ts.registerCallback(self.cropped_image_callback)

        # Detector service for Vision
        self.vision_service = rospy.Service(
            "/auv4/vision/KP/detector",
            MLDetector,
            self.detector_srv_cb,
        )

    def detector_srv_cb(self, req):
        # err = "" if req.dryRun else Detector.load_detectors(req.detectors)
        resp = MLDetectorResponse()
        resp.success = True
        resp.runningDetectors = [self.template]
        return resp

    def cropped_image_callback(self, img_msg, detected_objects=None, debug=False):
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

        kp1, kp2 = self.image_match_producer.process_image(
            img, self.template, lxtyrxby=lxtyrxby, debug=True, num_keypoints=150
        )

        # Process KP2
        if kp2 is None or kp2.shape[0] == 0:
            return

        kp2 = np.round(kp2).astype(int)
        min_x = np.min(kp2[:, 0])
        max_x = np.max(kp2[:, 0])

        min_y = np.min(kp2[:, 1])
        max_y = np.max(kp2[:, 1])

        contours = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, max_y)]

        cx = int((max_x + min_x) / 2)
        cy = int((max_y + min_y) / 2.0)

        w = max_x - min_x
        h = max_y - min_y

        if detected_object is None:
            # No one publishing on the topic => our turn to publish
            objs = DetectedObjects()
            objs.nodeName = "keypoint_based_detector"
            obj = DetectedObject()
            obj.source = 288
            obj.name = self.template

            obj.centre_x = cx
            obj.centre_y = cy

            obj.bbox_height = h
            obj.bbox_width = w

            obj.contour = [int(coord) for point in contours for coord in point]

            obj.image_height = 768
            obj.image_width = 1024

            objs.detected = [obj]
            self.detected_object_pub.publish(objs)


if __name__ == "__main__":
    rospy.init_node("basic_feature_matcher", anonymous=True, log_level=rospy.DEBUG)
    camera_topic = rospy.get_param(
        "~camera_topic", "/auv4/front_cam/image_rect_color/compressed"
    )
    visualization_topic = rospy.get_param(
        "~visualization_topic", "/visualization/compressed"
    )
    template = rospy.get_param("~template", "Bootlegger")
    template_path = rospy.get_param(
        "~template_path",
        os.path.abspath(
            Path(RosPack().get_path("image_matching"))
            / "templates"
            / f"{template}.jpeg"
        ),
    )
    save_detections_folder = rospy.get_param("~output_dir", "detections")
    detected_objects_topic = rospy.get_param(
        "~detected_objects_topic", "/auv4/vision/external/detected"
    )
    detector = BasicFeatureMatcher(
        camera_topic,
        visualization_topic,
        template,
        template_path,
        detected_objects_topic,
        save_detections_folder,
    )
    rospy.spin()
