#!/usr/bin/env python3
import os
from pathlib import Path

from rospkg import RosPack
import cv2
from feature_matcher.tools import time_func
import rospy
from sensor_msgs.msg import Image, CompressedImage
from feature_matcher.two_stage_match_producer import TwoStageMatchProducer
from feature_matcher.coarse_loftr_matcher import Coarse_LoFTRMatchProducer
from feature_matcher.keypoint_producer import OrbKeypointProducer, SiftKeypointProducer, SuperPointKeypointProducer, FastKeypointProducer
from feature_matcher.keypoint_matcher.bf import BFKeypointMatcher
from feature_matcher.keypoint_matcher.superglue import SuperglueKeypointMatcher

from cv_bridge import CvBridge, CvBridgeError
# from feature_matcher.loftr_matcher import LoFTRMatchProducer # requires CUDA
import rospy


class BasicFeatureMatcher:
    def __init__(self, input_topic, visualization_topic, template_path):
        self.bridge = CvBridge()

        self.template_img = cv2.imread(template_path)
        # self.image_match_producer = TwoStageMatchProducer(self.template_img, SuperPointKeypointProducer(), SuperglueKeypointMatcher())
        self.image_match_producer = Coarse_LoFTRMatchProducer(self.template_img)


        self.visualization_pub = rospy.Publisher(visualization_topic, Image, queue_size=1)
        self.image_match_producer.visualize_callbacks.append(lambda img: self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8")))

        self.PADDING = 10
        self.CROP_IMAGES = True

        self.image_sub = rospy.Subscriber(
            input_topic, CompressedImage, self.process_image_msg, queue_size=1)
        
        rospy.spin()

    @time_func
    def process_image_msg(self, msg, bboxes = None):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        if bboxes is not None:
            x, y, w, h = [int(_) for _ in bboxes[i]]

            if self.CROP_IMAGES:
                img = img[y-self.PADDING:y+h+self.PADDING,
                        x-self.PADDING:x+w+self.PADDING, :]

        kp1, kp2 = self.image_match_producer.process_image(img)
        rospy.loginfo_throttle(10, f"Found {len(kp1)} keypoints in image")


if __name__ == "__main__":
    rospy.init_node("basic_feature_matcher", anonymous=True)
    BasicFeatureMatcher("/auv4/front_cam/image_color/compressed",
                        "/visualization",
                        os.path.abspath(Path(RosPack().get_path("feature_matcher"))/"templates"/"Bootlegger.jpeg"))