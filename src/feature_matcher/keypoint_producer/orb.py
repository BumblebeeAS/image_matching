import cv2
import numpy as np
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer

class OrbKeypointProducer(KeypointProducer):
    def __init__(self, config={"num_keypoints": 500}):
        self.num_keypoints = config.get("num_keypoints", 500)
        self.orb = cv2.ORB_create(self.num_keypoints, nlevels=8, edgeThreshold=15,
            patchSize=31, fastThreshold=20, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE, firstLevel=0)

    def __call__(self, image: np.ndarray) -> Keypoints:
        """
        Finds keypoints within image using SIFT.

        Args:
            image: numpy array of image
            num_keypoints: the number of retrieved keypoints
        Returns:
            N keypoints.
        """
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        keypoints = np.array([list(kp.pt) for kp in keypoints])
        return Keypoints(image.shape[:2], keypoints, descriptors, np.ones(len(keypoints)))