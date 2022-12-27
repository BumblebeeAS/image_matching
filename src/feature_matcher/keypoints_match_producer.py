from abc import ABC, abstractmethod
import copy
from typing import Tuple
import cv2
import numpy as np
import logging
from feature_matcher.keypoints import Keypoints

from feature_matcher.tools import create_show_image, plot_matches, time_func

class KeypointsMatchProducer(ABC):
    def __init__(self, image):
        self.image = image
        self.visualize_callbacks = []

    def visualize(self, image):
        for callback in self.visualize_callbacks:
            callback(image)

    def get_template(self):
        return self.image

    @abstractmethod
    def compute_matches(self, other) -> Tuple[Keypoints, Keypoints]:
        """
        Returns:
            M Keypoints in the first image
            M Keypoints in the second image that matches to keypoints in first image.
        """
        pass

    @staticmethod
    def draw_keypoints(image1: np.ndarray, image2: np.ndarray, keypoints1: np.ndarray, keypoints2: np.ndarray,
                       color: tuple = (0, 255, 0)):
        assert len(keypoints1) == len(keypoints2)
        # concatenate the images along their widths.
        w = image1.shape[1] + image2.shape[1]
        h = max(image1.shape[0], image2.shape[0])
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:image1.shape[0], :image1.shape[1], :] = image1
        image[:image2.shape[0], image1.shape[1]:, :] = image2
        _keypoints2 = copy.deepcopy(keypoints2)
        _keypoints2.keypoints[:, 0] += float(image1.shape[1])

        # draw the keypoint matches.
        for i, keypoint1 in enumerate(keypoints1.keypoints):
            keypoint2 = _keypoints2.keypoints[i]
            image = cv2.line(img=image, pt1=tuple(keypoint1[:2].astype(int)),
                             pt2=tuple(keypoint2[:2].astype(int)), thickness=1, color=color)
        return image

    @time_func
    def process_image(self, img):
        kp1, kp2 = self.compute_matches(img)

        # img = self.draw_keypoints(self.get_template(), img, kp1, kp2)
        img = plot_matches(self.get_template(), img, kp1.keypoints, kp2.keypoints, kp1.scores)
        self.visualize(img)
        return kp1, kp2


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
    # template_img = cv2.imread("/home/developer/workspace/src/feature_matcher/templates/Tommy Gun.jpeg")
    folder_path = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/Images"
    bboxes_file = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/bootlegger1.csv"
    template_img = cv2.imread("/home/developer/workspace/src/feature_matcher/templates/Bootlegger.jpeg")

    # 0.0402967947s
    # image_match_producer = TwoStageMatchProducer(template_img, SiftKeypointProducer(), BFKeypointMatcher())

    # 0.0242464252s
    # image_match_producer = TwoStageMatchProducer(template_img, OrbKeypointProducer(), BFKeypointMatcher())

    # requires opencv contrib
    # 0.0210344344s
    # image_match_producer = TwoStageMatchProducer(template_img, FastKeypointProducer(), BFKeypointMatcher())

    # 0.4429747616s
    # 3.3s on full image
    # image_match_producer = TwoStageMatchProducer(template_img, SuperPointKeypointProducer(), SuperglueKeypointMatcher())
    
    #0.16616s 
    #1s on full image
    # image_match_producer = TwoStageMatchProducer(template_img, SuperPointKeypointProducer(), BFKeypointMatcher())

    #0.2477135228s
    image_match_producer = Coarse_LoFTRMatchProducer(template_img)

    # image_match_producer = LoFTRMatchProducer(template_img)

    image_match_producer.visualize_callbacks.append(create_show_image(image_match_producer.__class__.__name__))

    bboxes = np.loadtxt(bboxes_file, delimiter=",")
    PADDING = 10
    CROP_IMAGES = True
    for i, file in enumerate(os.listdir(folder_path)):
        try:
            x, y, w, h = [int(_) for _ in bboxes[i]]
            img = cv2.imread(
                f"{folder_path}/{file}")
                
            if CROP_IMAGES:
                img = img[y-PADDING:y+h+PADDING, x-PADDING:x+w+PADDING, :]
            kp1, kp2 = image_match_producer.process_image(img)

        except Exception as e:
            logging.error(e)
            continue
