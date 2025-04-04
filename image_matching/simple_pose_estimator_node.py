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
from feature_matcher.tools import plot_matches

template_dir = Path(get_package_share_directory("image_matching")) / "templates"
weights = Path(feature_matcher.__path__[0]) / Path(
    "models/accelerated_features/weights/xfeat.pt"
)


custom_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
)  # Reliable due to image transport, will change to best effort soon


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(
            img2_with_corners, start_point, end_point, (0, 255, 0), 4
        )  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(
        img1,
        keypoints1,
        img2_with_corners,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=2,
    )

    return img_matches


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

        # canvas = plot_matches(
        #     _template,
        #     img,
        #     self.template_kps[idxs0].cpu().numpy() // scale_template,
        #     keypoints[idxs1].cpu().numpy() // scale_img,
        # )

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
