import cv2
import numpy as np

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer


class SiftKeypointProducer(KeypointProducer):
    def __init__(self, config={"num_keypoints": 500}):
        self.num_keypoints = config.get("num_keypoints", 500)
        self.sift = cv2.SIFT_create(
            self.num_keypoints,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6,
        )

    def __call__(self, image: np.ndarray) -> Keypoints:
        """
        Finds keypoints within image using SIFT.

        Args:
            image: numpy array of image
            num_keypoints: the number of retrieved keypoints
        Returns:
            N keypoints.
        """
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        keypoints = np.array([list(kp.pt) for kp in keypoints])
        return Keypoints(
            image.shape[:2], keypoints, descriptors, np.ones(len(keypoints))
        )
