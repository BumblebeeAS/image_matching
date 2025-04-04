import json
from pathlib import Path
from typing import Dict

import rclpy
from ament_index_python import get_package_share_directory
from bb_msgs.srv import IMPoseEstimatorToggleTemplate
from cv2.typing import MatLike
from cv_bridge import CvBridge
from imutils import resize
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

import feature_matcher
from feature_matcher.tools import warp_corners_and_draw_matches
from feature_matcher.xfeat import XFeatMatcher

custom_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
)  # Reliable due to image transport, will change to best effort soon


templates_dir = Path(get_package_share_directory("image_matching")) / "templates"
template_json: Dict[str, Dict] = json.loads(
    open(Path(templates_dir) / "templates.json", "r").read()
)


class SimpleMatcherNode(Node):
    def __init__(self):
        super().__init__("simple_pose_estimator_node")
        self.get_logger().info(f"{feature_matcher.__path__}")

        self.matcher = XFeatMatcher()

        self.cv_bridge = CvBridge()
        self.img_subscriber = self.create_subscription(
            CompressedImage,
            "/auv4/front_cam/color/image/compressed",
            self.image_callback,
            1,
        )
        self.img_publisher = self.create_publisher(
            Image, "/auv4/front_cam/image_matching", 10
        )

        # Toggle Template Service
        self.declare_parameter(
            "toggle_template_topic", "/auv4/image_matching/toggle_template"
        )
        self.active_template = None
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
        if self.active_template is None:
            self.get_logger().info("No active template. Skipping image processing.")
            return
        self.active_template: MatLike

        img: MatLike = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        template_mkps, image_mkps = self.matcher.get_matches(self.template_name, img)

        _template = resize(self.active_template["image"], width=200)
        scale_template = self.active_template["image"].shape[1] / _template.shape[1]
        _img = resize(img, width=600)
        scale_img = img.shape[1] / _img.shape[1]

        canvas: MatLike = warp_corners_and_draw_matches(
            template_mkps // scale_template, image_mkps // scale_img, _template, _img
        )
        self.get_logger().info(f"{canvas.shape}")

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


def main():
    rclpy.init()
    node = SimpleMatcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
