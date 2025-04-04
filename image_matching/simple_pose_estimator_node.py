from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from imutils import resize
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

import feature_matcher
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat
from feature_matcher.tools import warp_corners_and_draw_matches

template_dir = Path(get_package_share_directory("image_matching")) / "templates"
weights = Path(feature_matcher.__path__[0]) / Path(
    "models/accelerated_features/weights/xfeat.pt"
)


custom_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
)  # Reliable due to image transport, will change to best effort soon


class SimplePoseEstimatorNode(Node):
    def __init__(self):
        super().__init__("simple_pose_estimator_node")
        self.get_logger().info(f"{feature_matcher.__path__}")

        self.model = XFeat(str(weights))

        self.template = cv2.imread(str(template_dir / "2023_torpedo1.png"))
        template_output = self.model.detectAndCompute(self.template)[0]
        self.template_kps = template_output["keypoints"]
        self.template_descs = template_output["descriptors"]

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

    def image_callback(self, msg: CompressedImage):
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)

        outputs = self.model.detectAndCompute(img)[0]
        keypoints = outputs["keypoints"]
        descriptors = outputs["descriptors"]

        idxs0, idxs1 = self.model.match(self.template_descs, descriptors)

        _template = resize(self.template, width=200)
        scale_template = self.template.shape[1] / _template.shape[1]
        _img = resize(img, width=600)
        scale_img = img.shape[1] / _img.shape[1]

        canvas = warp_corners_and_draw_matches(
            self.template_kps[idxs0].cpu().numpy() // scale_template,
            keypoints[idxs1].cpu().numpy() // scale_img,
            _template,
            _img,
        )

        self.get_logger().info(f"{canvas.shape}")

        img_msg = self.cv_bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        self.img_publisher.publish(img_msg)


def main():
    rclpy.init()
    node = SimplePoseEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
