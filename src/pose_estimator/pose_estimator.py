from enum import Enum
import logging
import cv2
from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer
from feature_matcher.tools import create_show_image, time_func

from pose_estimator.PinholeCamera import PinholeCamera
import numpy as np


class PoseEstimator:
    class Mode(Enum):
        AFFINE = 1
        HOMOGRAPHY = 2

    def __init__(self, camera: PinholeCamera, keypoints_match_producer: KeypointsMatchProducer, source_dimensions: tuple):
        self.keypoints_match_producer = keypoints_match_producer
        self.camera = camera
        self.source_image_size = np.array(self.keypoints_match_producer.image.shape[:2])
        self.source_dimensions = np.array(source_dimensions)

    @time_func
    def compute_pose(self, img, ltwh=None, mode=Mode.AFFINE):
        if ltwh is not None:
            l, t, w, h = ltwh
            img = img[t:t+h, l:l+w]
        keypoints1, keypoints2 = self.keypoints_match_producer.process_image(img)

        if len(keypoints1) < 4:
            logging.warning(
                f'Not enough matches to compute pose. Found {len(keypoints1)} matches.')
            return None
        
        object_coord = (keypoints1.keypoints - self.source_image_size / 2) * self.source_dimensions / self.source_image_size
        object_coord = np.hstack((object_coord, np.zeros((len(object_coord), 1))))
        kp2 = keypoints2.keypoints + np.array([l, t])

        _, R, t, mask = cv2.solvePnPRansac(
            object_coord,
            kp2,
            self.camera.camera_matrix(),
            self.camera.dist_coeffs(),
            iterationsCount=100,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return R, t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    from feature_matcher.two_stage_match_producer import TwoStageMatchProducer
    from feature_matcher.coarse_loftr_matcher import Coarse_LoFTRMatchProducer
    from feature_matcher.keypoint_producer import OrbKeypointProducer, SiftKeypointProducer, SuperPointKeypointProducer, FastKeypointProducer
    from feature_matcher.keypoint_matcher.bf import BFKeypointMatcher
    from feature_matcher.keypoint_matcher.superglue import SuperglueKeypointMatcher
    # from feature_matcher.loftr_matcher import LoFTRMatchProducer # requires CUDA

    # folder_path = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/_auv4_front_cam_image_rect_color"
    # bboxes_file = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/tommygun_gt.csv"
    # template_img = cv2.imread("/home/developer/workspace/src/image_matching/templates/Tommy Gun.jpeg")
    folder_path = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/Images"
    bboxes_file = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/bootlegger1.csv"
    template_img = cv2.imread(
        "/home/developer/workspace/src/image_matching/templates/Bootlegger.jpeg")

    # 0.0402967947s
    # match_image = TwoStageMatchProducer(template_img, SiftKeypointProducer(), BFKeypointMatcher())

    # 0.0242464252s
    # match_image = TwoStageMatchProducer(template_img, OrbKeypointProducer(), BFKeypointMatcher())

    # requires opencv contrib
    # 0.0210344344s
    # match_image = TwoStageMatchProducer(template_img, FastKeypointProducer(), BFKeypointMatcher())

    # 0.4429747616s
    # 3.3s on full image
    match_image_1 = TwoStageMatchProducer(
        template_img, SuperPointKeypointProducer(), SuperglueKeypointMatcher())

    # 0.16616s
    # 1s on full image
    # match_image = TwoStageMatchProducer(template_img, SuperPointKeypointProducer(), BFKeypointMatcher())

    # 0.2477135228s
    match_image_2 = Coarse_LoFTRMatchProducer(template_img)

    # match_image = LoFTRMatchProducer(template_img)

    #   hfov="1.5125"
    #   width="768"
    #   height="492"
    # fl = 768 / (2.0 *tan(1.5125/2.0)) = 407.064612984

    camera = PinholeCamera(768, 492, 407.0646129842357, 407.0646129842357, 384.5, 246.5, 0, 0, 0, 0, 0)

    pose_estimator_1 = PoseEstimator(camera, match_image_1, (1.2192, 0.6096))
    pose_estimator_2 = PoseEstimator(camera, match_image_2, (1.2192, 0.6096))


    # match_image_1.visualize_callbacks.append(create_show_image(match_image_1.__class__.__name__))
    # match_image_2.visualize_callbacks.append(create_show_image(match_image_2.__class__.__name__))

    bboxes = np.loadtxt(bboxes_file, delimiter=",")
    PADDING = 10
    CROP_IMAGES = True
    for i, file in enumerate(os.listdir(folder_path)):
        try:
            img = cv2.imread(
                f"{folder_path}/{file}")
            l, t, w, h = [int(_) for _ in bboxes[i]]
            l = max(0, l-PADDING)
            t = max(0, t-PADDING)
            w = min(img.shape[1], w+PADDING)
            h = min(img.shape[0], h+PADDING)


            #0.48s
            rot, trans = pose_estimator_1.compute_pose(img, (l, t, w, h) if CROP_IMAGES else None)
            print(", ".join(map(str,np.rad2deg(rot.squeeze()))), ", ".join(map(str, trans.squeeze())))

            # 0.3828051131s
            rot, trans = pose_estimator_2.compute_pose(img, (l, t, w, h) if CROP_IMAGES else None)
            print(", ".join(map(str,np.rad2deg(rot.squeeze()))), ", ".join(map(str, trans.squeeze())))

        except Exception as e:
            logging.error(e)
            continue
