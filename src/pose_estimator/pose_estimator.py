import logging

import cv2
import numpy as np

from feature_matcher.keypoints_match_producer import (
    KeypointsMatchProducer,
    get_keypoints_match_producer,
)
from feature_matcher.tools import create_show_image, plot_matches
from pose_estimator.PinholeCamera import PinholeCamera
from transforms3d.euler import mat2euler


def homography_based_filter(
    kp1,
    kp2,
    camera_matrix,
    dist_coeffs,
    min_inliers=6,
):
    if (kp1.shape[0] < min_inliers) or (kp2.shape[0] < min_inliers):
        print("NOT ENOUGH POINTS -> Object not in view? ")
        return None, None, None, None

    kp1_undistort = cv2.undistortPoints(kp1.reshape(-1, 1, 2), np.eye(3), dist_coeffs)
    kp2_undistort = cv2.undistortPoints(kp2.reshape(-1, 1, 2), np.eye(3), dist_coeffs)

    H, mask = cv2.findHomography(kp1_undistort, kp2_undistort, cv2.RANSAC)
    if H is None or sum(mask) < min_inliers:
        print("NO INLIERS")
        return None, None, None, None

    _, Rs, ts, ns = cv2.decomposeHomographyMat(H, camera_matrix)
    for (R, tvec, n) in zip(Rs, ts, ns):
        if n[2] > 0:  # Wrong direction -> skip
            continue
        if abs(n[2] + 1) > 0.1:  # Wrong normal -> skip
            continue

        # yaw, pitch, roll = mat2euler(R, axes="szyx")
        # print("Yaw (H): ", np.rad2deg(yaw))
        # print("Pitch (H): ", np.rad2deg(pitch))
        # print("Roll (H): ", np.rad2deg(roll))

        if mask is not None:
            # bef = len(kp1)
            kp1 = kp1[mask.ravel() == 1]
            kp2 = kp2[mask.ravel() == 1]

            # rospy.loginfo("Inliers (H): ", len(kp1), " / ", bef)
            return kp1, kp2, R, tvec
    return None, None, None, None


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
        if "_seg" in name:
            print("Will not perform matching for segmentation templates")
        else:
            self.keypoints_match_producer.register_template(name, template_img)
        self.templates[name] = (
            np.array(world_dimensions),
            np.array(template_img.shape[1::-1]),
        )  # width, height

    @property
    def available_cameras(self):
        return self.cameras.keys()

    @property
    def available_templates(self):
        return self.templates.keys()

    @staticmethod
    def draw_object_points(img, object_points, R, t, camera):
        imgpts, _ = cv2.projectPoints(
            object_points, R, t, camera.camera_matrix(), camera.dist_coeffs()
        )
        return cv2.polylines(img, [np.int32(imgpts)], True, (0, 255, 0), 3)

    def visualize(self, image):
        for callback in self.visualize_callbacks:
            callback(image)

    def compute_pose_from_keypoints(
        self,
        template,
        camera_frame,
        keypoints1,  # np.ndarray (x, y) * N
        keypoints2,  # np.ndarray (x, y) * N
        is_planar=False,  # If true, we assume object is planar => homography is used first
        max_reprojection_error=2.0,  # Maximum reprojection error for pose to be accepted
    ):
        if camera_frame not in self.cameras:
            raise Exception(f"Camera {camera_frame} not registered.")
        else:
            camera = self.cameras[camera_frame]

        R_H = None
        t_H = None
        if is_planar:
            # Homography-based filtering
            kp1, kp2, R_H, t_H = homography_based_filter(
                keypoints1,
                keypoints2,
                camera.camera_matrix(),
                camera.dist_coeffs(),
                min_inliers=self.min_inliers,
            )
            if kp1 is None or kp2 is None:
                return None, None, None

        source_dimensions, source_image_size = self.templates[template]
        object_coord = (
            (keypoints1 - source_image_size / 2) * source_dimensions / source_image_size
        )
        object_coord = np.hstack((object_coord, np.zeros((len(object_coord), 1))))

        try:
            _, rvec, t, inliers = cv2.solvePnPRansac(
                object_coord.astype(np.float64),
                keypoints2.astype(np.float32),
                camera.camera_matrix(),
                camera.dist_coeffs(),
                useExtrinsicGuess=False,
                reprojectionError=max_reprojection_error,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except Exception as e:
            print(e)
            return None, None, None
        R = cv2.Rodrigues(rvec)[0]
        t = t.squeeze()

        if R_H is not None:
            r_diff = np.arccos((np.trace(R @ R_H.T) - 1) / 2)
            if r_diff > 0.2:  # Radian -> 11 degrees
                print(f"Homography vs PnP: {r_diff}, skipping")
                return None, None, None

        return R, t, inliers

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
        is_planar=True,  # If true, we assume object is planar => homography is used first
        max_reprojection_error=2.0,  # Maximum reprojection error for pose to be accepted
    ):
        if template is None:
            raise Exception("Template has to be specified.")
        if camera_frame not in self.cameras:
            raise Exception(f"Camera {camera_frame} not registered.")
        else:
            camera = self.cameras[camera_frame]

        keypoints1, keypoints2 = self.keypoints_match_producer.process_image(
            img,
            template,
            debug,
            num_keypoints=num_keypoints,
            lxtyrxby=lxtyrxby,
            logger=logger,
        )

        if keypoints1 is None or keypoints2 is None: 
            return None, None
        
        print(len(keypoints1.keypoints))
        print(len(keypoints2.keypoints))

        R, t, inliers = self.compute_pose_from_keypoints(
            template,
            camera_frame,
            keypoints1.keypoints,
            keypoints2.keypoints,
            is_planar=is_planar,
            max_reprojection_error=max_reprojection_error,
        )

        if debug and R is not None and t is not None:
            source_dimensions, _ = self.templates[template]
            _x = source_dimensions[0] / 2
            _y = source_dimensions[1] / 2
            object_rect = np.array(
                [[-_x, -_y, 0], [_x, -_y, 0], [_x, _y, 0], [-_x, _y, 0]]
            )
            # Draw object bbox
            img = self.draw_object_points(img, object_rect, R, t, camera)

            # Draw axes
            img = cv2.drawFrameAxes(
                img,
                camera.camera_matrix(),
                camera.dist_coeffs(),
                R,
                t,
                length=0.1,
            )

            mask = np.zeros(keypoints1.keypoints.shape[0], dtype=np.uint8)
            if inliers is not None:
                mask[inliers.squeeze()] = 1
            else:
                mask = np.ones(keypoints1.keypoints.shape[0], dtype=np.uint8)

            img = plot_matches(
                self.keypoints_match_producer.get_template(template).img,
                img,
                keypoints1.keypoints,
                keypoints2.keypoints,
                scores=mask,
            )
            # create_save_image("/home/nvidia/catkin_ws/src/image_matching/debug.png")(img)
            # exit(1)
            self.visualize(img)
        return R, t


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
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
