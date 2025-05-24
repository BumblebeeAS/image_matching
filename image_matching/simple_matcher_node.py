import json
from pathlib import Path
from typing import Dict

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from bb_msgs.srv import IMPoseEstimatorToggleTemplate
from bb_perception_msgs.msg import PointCorrespondencesStamped
from cv2.typing import MatLike
from cv_bridge import CvBridge
from imutils import resize
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from feature_matcher.tools import (
    get_image_match_empty_canvas,
    get_template_specs,
    warp_corners_and_draw_matches,
)
from feature_matcher.xfeat import XFeatMatcher
from utils.ros_np_multiarray import to_multiarray_f64

custom_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
)  # Reliable due to image transport, will change to best effort soon


class SimpleMatcherNode(Node):
    def __init__(self):
        super().__init__("simple_pose_estimator_node")

        default_templates_dir = (
            Path(get_package_share_directory("image_matching"))
            / "templates"
            / "robosub25"
        )
        self.declare_parameter("templates_dir", default_templates_dir.as_posix())
        self.declare_parameter(
            "toggle_template_topic", "image_matching/toggle_template"
        )
        self.declare_parameter("input_compressed_image_topic", "color/image/compressed")

        self.matcher = XFeatMatcher()

        templates_dir = Path(
            self.get_parameter("templates_dir").get_parameter_value().string_value
        )
        template_json: Dict[str, Dict] = json.loads(
            open(templates_dir / "templates.json", "r").read()
        )
        self.template_specs = get_template_specs(templates_dir, template_json)
        for template_name, template in self.template_specs.items():
            self.get_logger().info(
                f"Template {template_name}: {template.dimensions}, {template.offset}, {template.image.shape}"
            )
        self.matcher.set_all_templates(self.template_specs)

        self.cv_bridge = CvBridge()
        input_compressed_image_topic = (
            self.get_parameter("input_compressed_image_topic")
            .get_parameter_value()
            .string_value
        )
        self.img_subscriber = self.create_subscription(
            CompressedImage, input_compressed_image_topic, self.image_callback, 1
        )
        self.points_publisher = self.create_publisher(
            PointCorrespondencesStamped, "image_matching/point_correspondences", 10
        )
        self.img_publisher = self.create_publisher(Image, "image_matching/image", 10)

        # Toggle Template Service
        self.template_name: None | str = None
        toggle_template_topic = (
            self.get_parameter("toggle_template_topic")
            .get_parameter_value()
            .string_value
        )
        self.toggle_template_service = self.create_service(
            IMPoseEstimatorToggleTemplate,
            toggle_template_topic,
            self.toggle_template_callback,
        )

    def image_callback(self, msg: CompressedImage):
        if self.template_name is None:
            return

        img: MatLike = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        template_mkps, image_mkps = self.matcher.get_matches(self.template_name, img)

        # Send 3D object - 2D image correspondences
        template_spec = self.template_specs[self.template_name]
        image_dims = np.array(template_spec.image.shape[:2][::-1])
        object_dims = np.array(template_spec.dimensions)
        object_mkps = template_mkps / image_dims * object_dims
        object_mkps = np.hstack([object_mkps, np.zeros((object_mkps.shape[0], 1))])

        points_msg = PointCorrespondencesStamped()
        points_msg.header = msg.header
        points_msg.object_frame_id = Path(self.template_name).stem + "_optical"
        points_msg.object_points = to_multiarray_f64(object_mkps)
        points_msg.image_points = to_multiarray_f64(image_mkps)
        self.points_publisher.publish(points_msg)

        # TODO: Move annotation to separate node
        template = self.matcher.templates_with_keypoints[self.template_name]
        _template = resize(template.image, width=200)
        scale_template = template.image.shape[1] / _template.shape[1]
        _img = resize(img, width=600)
        scale_img = img.shape[1] / _img.shape[1]

        if len(template_mkps) < 4 or len(image_mkps) < 4:
            self.get_logger().warn(
                f"Insufficient matches found. Found {len(template_mkps)}, 4 or more required."
            )
            canvas = get_image_match_empty_canvas(_template, _img)
            self.img_publisher.publish(
                self.cv_bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            )
            return

        canvas: MatLike = warp_corners_and_draw_matches(
            template_mkps // scale_template, image_mkps // scale_img, _template, _img
        )

        img_msg = self.cv_bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        self.img_publisher.publish(img_msg)

    def toggle_template_callback(
        self,
        request: IMPoseEstimatorToggleTemplate.Request,
        response: IMPoseEstimatorToggleTemplate.Response,
    ):
        if not request.enabled:
            self.get_logger().info("Disabling template matching.")
            self.template_name = None
            response.new_state = False
            response.error_message = "All templates disabled."
            return response

        if request.template_name not in self.matcher.templates_with_keypoints:
            self.get_logger().warn(f"Template {request.template_name} not found.")
            response.new_state = False
            response.error_message = f"Template {request.template_name} not found."
            return response

        if request.enabled:
            self.get_logger().info(
                f"Enabling template matching for {request.template_name}."
            )
            self.template_name = request.template_name
            response.new_state = True
            response.error_message = ""
            return response


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMatcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
