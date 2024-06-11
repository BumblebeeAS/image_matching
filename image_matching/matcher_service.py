#!/usr/bin/env python3
import json
import os
from typing import Callable, Dict, List, Tuple, Union

from bb_msgs.msg import DetectedObjects
from bb_msgs.msg import Keypoint
from bb_msgs.msg import KeypointsDict
from bb_msgs.srv import ClearBuffer
from bb_msgs.srv import ClearBufferRequest
from bb_msgs.srv import ClearBufferResponse
from bb_msgs.srv import MatchImages
from bb_msgs.srv import MatchImagesRequest
from bb_msgs.srv import MatchImagesResponse
from bb_msgs.srv import MatchToTemplate
from bb_msgs.srv import MatchToTemplateRequest
from bb_msgs.srv import MatchToTemplateResponse
from bb_msgs.srv import RegisterImage
from bb_msgs.srv import RegisterImageRequest
from bb_msgs.srv import RegisterImageResponse
import cv2
from cv_bridge import CvBridge
from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer
from feature_matcher.keypoints_match_producer import (
    get_keypoints_match_producer,
)
from feature_matcher.tools import create_save_image
import numpy as np
from rospkg import RosPack
import rospy
from sensor_msgs.msg import CompressedImage


class MatcherNode:
    """
    ROSWrapper to offer matching services
    """

    def __init__(
        self,
        is_debug=False,
        detector_name="sift",
        matcher_name="flann",
        node_name="matcher",
        detector_config: Union[Dict, None] = {},
        matcher_config: Union[Dict, None] = {},
    ):
        rospy.init_node(node_name)
        self.services: List[rospy.Service] = []
        self.bridge = CvBridge()

        self.detector_config = detector_config
        self.matcher_config = matcher_config

        self.image_match_producer: KeypointsMatchProducer = (
            get_keypoints_match_producer(
                detector_name,
                matcher_name,
                self.detector_config,
                self.matcher_config,
            )
        )  # 0.1275215585s

        self.path = os.path.abspath(RosPack().get_path("image_matching"))

        self.debug = is_debug
        if is_debug:
            self.debug_path = os.path.join(self.path, "debug")
            if not os.path.isdir(self.debug_path):
                os.mkdir(self.debug_path)
            self.image_match_producer.visualize_callbacks.append(
                create_save_image(
                    os.path.join(self.debug_path, "debug_matches_py.jpg")
                )
            )

        template_path = os.path.join(self.path, "templates")
        templates = json.loads(
            open(os.path.join(template_path, "templates.json")).read()
        )
        for filename, _ in templates.items():
            name = ".".join(filename.split(".")[:-1])
            self.image_match_producer.register_template(
                name, cv2.imread(os.path.join(template_path, filename))
            )

        print("Warming up models")
        self.warmup()

        self.offer_services()
        print("Services offered!")

    def offer_services(self):
        try:
            # Add the services here
            self.services.append(
                rospy.Service(
                    "registerImage", RegisterImage, self.register_img
                )
            )
        except rospy.service.ServiceException:
            pass
        try:
            self.services.append(
                rospy.Service("clearBuffer", ClearBuffer, self.clear_buffer)
            )
        except rospy.service.ServiceException:
            pass
        try:
            self.services.append(
                rospy.Service("matchImages", MatchImages, self.match_images)
            )
        except rospy.service.ServiceException:
            pass
        try:
            self.services.append(
                rospy.Service(
                    "matchToTemplate", MatchToTemplate, self.match_to_template
                )
            )
        except rospy.service.ServiceException:
            pass

    def _to_correct_format(
        self,
        results: Tuple[Keypoints, Keypoints],
        response_creator: Callable[
            [], Union[MatchImagesResponse, MatchToTemplateResponse]
        ],
    ) -> Union[MatchImagesResponse, MatchToTemplateResponse]:
        response = response_creator()
        response.keypoints_dict = KeypointsDict()

        response.keypoints_dict.ref_keypoints = []
        response.keypoints_dict.cur_keypoints = []

        for k in results[0].keypoints:
            kp = Keypoint()
            kp.coord = k
            response.keypoints_dict.ref_keypoints.append(kp)

        for k in results[1].keypoints:
            kp = Keypoint()
            kp.coord = k
            response.keypoints_dict.cur_keypoints.append(kp)
        response.keypoints_dict.match_score = results[
            "match_score"
        ]  # List of floats
        return response

    def register_img(self, req: RegisterImageRequest) -> RegisterImageResponse:
        topic_name = req.topic_name
        detected_objects_topic_name = req.detected_objects_topic_name
        object_name = req.object_name
        detected_object = None
        if detected_objects_topic_name != "" and object_name != "":
            for i in range(3):
                try:
                    detected_objects = rospy.wait_for_message(
                        detected_objects_topic_name, DetectedObjects, timeout=2
                    )
                except rospy.ROSException:
                    continue
                if any(
                    [x.name == object_name for x in detected_objects.detected]
                ):
                    detected_object = sorted(
                        detected_objects.detected,
                        key=lambda x: x.extra[0],
                        reverse=True,
                    )[0]
                    break
        img: CompressedImage = rospy.wait_for_message(
            topic_name, CompressedImage, timeout=2
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
                self.debug_path, f"current_{len(self.buffer) % 2}.jpg"
            ),
            cv2_img,
        )
        return self.image_match_producer.add_image(cv2_img)

    def clear_buffer(
        self, _: ClearBufferRequest = None
    ) -> ClearBufferResponse:
        return self.image_match_producer.clear_buffer()

    def match_images(self, request: MatchImagesRequest) -> MatchImagesResponse:
        num_keypoints = request.numKeypoints
        if num_keypoints <= 0:
            resp = MatchImagesResponse()
            resp.result = 1
            return resp

        results = self.image_match_producer.process_image(
            num_keypoints=num_keypoints, template=None
        )

        return self._to_correct_format(results, lambda: MatchImagesResponse())  # type: ignore

    def match_to_template(
        self, request: MatchToTemplateRequest
    ) -> MatchToTemplateResponse:
        num_keypoints = request.numKeypoints
        template_name = request.template_name

        print(num_keypoints)
        print(template_name)

        if num_keypoints <= 0 or not self.image_match_producer.has_template(
            template_name
        ):
            resp = MatchToTemplateResponse()
            resp.result = 1
            return resp

        results = self.image_match_producer.process_image(
            num_keypoints=num_keypoints, template=template_name
        )

        return self._to_correct_format(
            results, lambda: MatchToTemplateResponse()
        )  # type: ignore

    def warmup(self):
        debug_state = self.debug
        self.debug = False

        self.image_match_producer.add_image(
            (np.random.rand(720, 640, 3) * 255).astype(np.uint8)
        )
        for _ in range(10):
            self.image_match_producer.add_image(
                (np.random.rand(720, 640, 3) * 255).astype(np.uint8)
            )
            self.image_match_producer.compute_matches()
        self.image_match_producer.clear_buffer()
        self.debug = debug_state


if __name__ == "__main__":
    DEBUG_MODE = True
    matcher_node = MatcherNode(
        DEBUG_MODE,
        detector_config={
            "cuda": True,
        },
        matcher_config={"cuda": True},
    )
    rospy.spin()
