import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ImageBrightenNode(Node):
    def __init__(self):
        super().__init__("image_brighten_node")

        self.bridge = CvBridge()
        self.brightness_factor = (
            self.declare_parameter("brightness_factor", 1.2)
            .get_parameter_value()
            .double_value
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            "/auv4/front_cam/color/image/compressed",
            self.image_callback,
            10,
        )

        self.publisher = self.create_publisher(
            CompressedImage, "/auv4/front_cam/color/image/brighten/compressed", 10
        )

        self.get_logger().info(
            f"ImageBrightenNode started with brightness factor {self.brightness_factor}"
        )

    def image_callback(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Brighten
        brightened = cv2.convertScaleAbs(cv_image, alpha=self.brightness_factor, beta=0)

        # Sharpen
        kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
        sharpened_image = cv2.filter2D(brightened, -1, kernel)

        output_msg = self.bridge.cv2_to_compressed_imgmsg(sharpened_image)
        output_msg.header = msg.header
        self.publisher.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageBrightenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
