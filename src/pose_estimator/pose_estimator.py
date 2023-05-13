import logging

import cv2
import numpy as np

from feature_matcher.keypoints_match_producer import (
    KeypointsMatchProducer,
    get_keypoints_match_producer,
)
from feature_matcher.tools import (
    create_show_image,
    create_save_image,
    plot_matches,
)
from pose_estimator.PinholeCamera import PinholeCamera


class PoseEstimator:
    def __init__(self, keypoints_match_producer: KeypointsMatchProducer, debug=False):
        self.keypoints_match_producer = keypoints_match_producer
        self.visualize_callbacks = []
        self.cameras = {}
        self.templates = {}
        self.min_inliers = 6
        self.debug = debug

    def register_camera(self, camera: PinholeCamera):
        self.cameras[camera.frame_id] = camera

    def register_template(self, name, world_dimensions, template_img):
        self.keypoints_match_producer.register_template(name, template_img)
        self.templates[name] = (
            np.array(world_dimensions),
            np.array(template_img.shape[1::-1]),
        )  # width, height

    @staticmethod
    def draw_object_points(img, object_points, R, t, camera):
        imgpts, _ = cv2.projectPoints(
            object_points, R, t, camera.camera_matrix(), camera.dist_coeffs()
        )
        return cv2.polylines(img, [np.int32(imgpts)], True, (0, 255, 0), 3)

    def visualize(self, image):
        for callback in self.visualize_callbacks:
            callback(image)

    def compute_pose(
        self,
        img,
        template,
        camera_frame,
        *,
        num_keypoints=20,
        lxtyrxby=None,
        debug=False,
        logger=None,
    ):
        if template is None:
            raise Exception("Template has to be specified.")
        if camera_frame not in self.cameras:
            raise Exception(f"Camera {camera_frame} not registered.")
        else:
            camera = self.cameras[camera_frame]

        keypoints1, keypoints2 = self.keypoints_match_producer.process_image(
            img, template, debug, num_keypoints=100, lxtyrxby=lxtyrxby, logger=logger
        )

        if not keypoints1 is None:
            print(f"keypoints1: {len(keypoints1)}, keypoints2: {len(keypoints2)}")
        else:
            print("no keypoints found")
        if keypoints1 is None or len(keypoints1) < 4:
            logging.warning(
                f"Not enough matches to compute pose. Found {0 if keypoints1 is None else len(keypoints1)} matches."
            )
            return None, None

        source_dimensions, source_image_size = self.templates[template]

        object_coord = (
            (keypoints1.keypoints - source_image_size / 2)
            * source_dimensions
            / source_image_size
        )
        object_coord = np.hstack((object_coord, np.zeros((len(object_coord), 1))))
        kp2 = keypoints2.keypoints

        _, R, t, mask = cv2.solvePnPRansac(
            object_coord,
            kp2,
            camera.camera_matrix(),
            camera.dist_coeffs(),
            iterationsCount=100,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # TODO: Better but bimodal at some positions
        R, t = cv2.solvePnPRefineVVS(
            object_coord, kp2, camera.camera_matrix(), camera.dist_coeffs(), R, t
        )
        print(f"t_z: {t.squeeze()[2]}, mask: {mask}")
        t_z = t.squeeze()[2]
        if t_z < 0 or t_z > 100 or mask is None or len(mask) < self.min_inliers:
            logging.info("Not enough inliers.")
            return None, None

        print("Passed inliers check.")

        if debug:
            _x = source_dimensions[0] / 2
            _y = source_dimensions[1] / 2
            object_rect = np.array(
                [[-_x, -_y, 0], [_x, -_y, 0], [_x, _y, 0], [-_x, _y, 0]]
            )
            # object_rect = np.array([[0, 0, 0], [_x, 0, 0], [_x, _y, 0], [0, _y, 0]])
            img = self.draw_object_points(img, object_rect, R, t, camera)
            img = plot_matches(
                self.keypoints_match_producer.get_template(template).img,
                img,
                keypoints1.keypoints,
                kp2,
                keypoints1.scores,
            )
            # create_save_image("/home/nvidia/catkin_ws/src/image_matching/debug.png")(img)
            # exit(1)
            self.visualize(img)
        return cv2.Rodrigues(R)[0], t.squeeze()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os

    # folder_path = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/_auv4_front_cam_image_rect_color"
    # bboxes_file = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/tommygun_gt.csv"
    folder_path = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/Images"
    bboxes_file = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/bootlegger1.csv"

    # image_match_producers = {
    #     # 0.0103288540s
    #     "sift": get_keypoints_match_producer("sift", "bf", {"debug": True}, {"debug": True}),
    #     # 0.0042234153s
    #     "orb": get_keypoints_match_producer("orb", "bf", {"debug": True}, {"debug": True}),
    #     # 0.0026472004s
    #     "fast": get_keypoints_match_producer("fast", "bf", {"debug": True}, {"debug": True}),
    #     # 0.2101211615s
    #     "superpoint": get_keypoints_match_producer("superpoint", "superglue", {"debug": True}, {"debug": True}),
    #     # 0.0883604178s
    #     "coarse_loftr": get_keypoints_match_producer(None, "coarse_loftr", {"debug": True}, {"debug": True}),
    #     # "loftr": get_keypoints_match_producer(None, "loftr", {"debug": True}, {"debug": True}),
    # }

    # image_match_producer = get_keypoints_match_producer("superpoint", "superglue", {"debug": True}, {"debug": True}) # 0.6256848859s
    # image_match_producer = get_keypoints_match_producer(None, "coarse_loftr", {"debug": True}, {"debug": True}) # 0.1275215585s
    # image_match_producer = get_keypoints_match_producer("sift", "flann", {"debug": True}, {"debug": True}) # 0.0294936401s
    image_match_producer = get_keypoints_match_producer(
        "sift", "bf", {"debug": True}, {"debug": True}
    )  # 0.0318265118s
    # image_match_producer = get_keypoints_match_producer("superpoint", "bf", {"debug": True}, {"debug": True})

    pose_estimator_1 = PoseEstimator(image_match_producer)
    camera = PinholeCamera(
        "auv4/front_cam",
        768,
        492,
        407.0646129842357,
        407.0646129842357,
        384.5,
        246.5,
        0,
        0,
        0,
        0,
        0,
    )
    pose_estimator_1.register_camera(camera)

    pose_estimator_1.visualize_callbacks.append(
        create_show_image(pose_estimator_1.__class__.__name__)
    )
    # pose_estimator_1.visualize_callbacks.append(create_save_image())

    templates = {
        "Tommy Gun": (
            (0.6096, 1.2192),
            "/home/developer/workspace/src/image_matching/templates/Tommy Gun.jpeg",
        ),
        "Bootlegger": (
            (0.6096, 1.2192),
            "/home/developer/workspace/src/image_matching/templates/Bootlegger.jpeg",
        ),
    }
    for key, value in templates.items():
        pose_estimator_1.register_template(key, value[0], cv2.imread(value[1]))

    bboxes = np.loadtxt(bboxes_file, delimiter=",")
    PADDING = 10
    CROP_IMAGES = True
    for i, file in enumerate(os.listdir(folder_path)):
        try:
            img = cv2.imread(f"{folder_path}/{file}")
            l, t, w, h = [int(_) for _ in bboxes[i]]
            l = max(0, l - PADDING)
            t = max(0, t - PADDING)
            r = min(img.shape[1], l + w + PADDING)
            b = min(img.shape[0], t + h + PADDING)

            # 0.48s
            rot, trans = pose_estimator_1.compute_pose(
                img,
                "Bootlegger",
                "auv4/front_cam",
                lxtyrxby=(l, t, r, b) if CROP_IMAGES else None,
                debug=True,
            )
            # print(", ".join(map(str,np.rad2deg(rot.squeeze()))), ", ".join(map(str, trans.squeeze())))

        except Exception as e:
            logging.error(e)
            continue
