import json
from itertools import chain
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from bb_perception_msgs.msg import PointCorrespondencesStamped
from bb_perception_msgs.srv import IMPoseEstimatorToggleTemplate
from cv2.typing import MatLike
from cv_bridge import CvBridge
from feature_matcher.tools import get_template_specs, get_warped_corners
from feature_matcher.xfeat import XFeatMatcher
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation
from imutils import resize
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from image_processing.utils.image_annotations import get_image_annotations
from image_processing.utils.ros_np_multiarray import to_multiarray_f64


class SimpleMatcherNode(Node):
    def __init__(self):
        super().__init__("simple_pose_estimator_node")

        default_templates_dir = (
            Path(get_package_share_directory("image_matching"))
            / "templates"
            / "robosub25"
        )

        # Parameters
        templates_dir = (
            self.declare_parameter("templates_dir", default_templates_dir.as_posix())
            .get_parameter_value()
            .string_value
        )
        toggle_template_topic = (
            self.declare_parameter(
                "toggle_template_topic", "image_matching/toggle_template"
            )
            .get_parameter_value()
            .string_value
        )
        input_image_topic = (
            self.declare_parameter("input_image_topic", "image")
            .get_parameter_value()
            .string_value
        )
        output_annotations_topic = (
            self.declare_parameter(
                "output_annotations_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )

        # Initialize matcher
        self.matcher = XFeatMatcher()

        templates_dir_path = Path(templates_dir)
        template_json: Dict[str, Dict] = json.loads(
            open(templates_dir_path / "templates.json", "r").read()
        )
        self.template_specs = get_template_specs(templates_dir_path, template_json)
        for template_name, template in self.template_specs.items():
            self.get_logger().info(
                f"Template {template_name}: {template.dimensions}, {template.offset}, {template.image.shape}"
            )
        self.matcher.set_all_templates(self.template_specs)

        # Subscribers and Publishers
        self.cv_bridge = CvBridge()
        self.img_subscriber = self.create_subscription(
            Image, input_image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.points_publisher = self.create_publisher(
            PointCorrespondencesStamped,
            "image_matching/point_correspondences",
            qos_profile_sensor_data,
        )
        self.annotations_pub = self.create_publisher(
            ImageAnnotations,
            output_annotations_topic,
            qos_profile=qos_profile_sensor_data,
        )

        # Toggle Template Service
        self.template_name: None | str = None
        self.toggle_template_service = self.create_service(
            IMPoseEstimatorToggleTemplate,
            toggle_template_topic,
            self.toggle_template_callback,
        )

    def publish_image_annotations(
        self,
        header: Header,
        template_mkps: np.ndarray,
        image_mkps: np.ndarray,
        template,
    ) -> None:
        try:
            _, warped_corners = get_warped_corners(
                template_mkps, image_mkps, template.image
            )
        except cv2.error:
            self.get_logger().info(
                "Failed to get warped corners for image points, skipping annotations..."
            )
            return

        polygon_image_annotations: ImageAnnotations = get_image_annotations(
            header, [[warped_corners.squeeze()]], ["#00FF00"]
        )
        points_image_annotations: ImageAnnotations = get_image_annotations(
            header, [[image_mkps]], ["#00FF00"], PointsAnnotation.POINTS
        )
        points_annotations = list(
            chain(polygon_image_annotations.points, points_image_annotations.points)
        )
        output_msg = ImageAnnotations(points=points_annotations)
        self.annotations_pub.publish(output_msg)

    def image_callback(self, msg: Image) -> None:
        if self.template_name is None:
            return

        img: MatLike = self.cv_bridge.imgmsg_to_cv2(msg)
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

        # Publish image containing template matches
        # TODO: Move annotation to separate node
        template = self.matcher.templates_with_keypoints[self.template_name]
        self.publish_image_annotations(msg.header, template_mkps, image_mkps, template)

    def toggle_template_callback(
        self,
        request: IMPoseEstimatorToggleTemplate.Request,
        response: IMPoseEstimatorToggleTemplate.Response,
    ):
        if not request.enable:
            self.get_logger().info("Disabling template matching.")
            self.template_name = None
            response.new_state = False
            response.error_message = "All templates disabled."
            return response

        elif request.template_name not in self.matcher.templates_with_keypoints:
            self.get_logger().warn(f"Template {request.template_name} not found.")
            response.new_state = False
            response.error_message = f"Template {request.template_name} not found."
            return response

        else:
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
