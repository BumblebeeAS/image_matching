import cv2
import numpy as np

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer


class SiftKeypointProducer(KeypointProducer):
    def __init__(self, config={"num_keypoints": 500, "rootsift": True}):
        self.num_keypoints = config.get("num_keypoints", 500)
        self.use_root = config.get("rootsift", True)
        self.eps = 1e-7

        self.sift = cv2.SIFT_create(
            self.num_keypoints,
            nOctaveLayers=6,
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

        if self.use_root and len(keypoints) > 0:
            descriptors /= (descriptors.sum(axis=1, keepdims=True) + self.eps)
            descriptors = np.sqrt(descriptors)

        return Keypoints(
            image.shape[:2], keypoints, descriptors, np.ones(len(keypoints))
        )
