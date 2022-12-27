import cv2
import numpy as np
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer

class FastKeypointProducer(KeypointProducer):
    def __init__(self, num_keypoints: int = 500):
        self.num_keypoints = num_keypoints
        self.fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        # self.fast = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()


    def __call__(self, image: np.ndarray) -> Keypoints:
        """
        Finds keypoints within image using SIFT.

        Args:
            image: numpy array of image
            num_keypoints: the number of retrieved keypoints
        Returns:
            N keypoints.
        """
        keypoints = self.fast.detect(image, None)
        keypoints, descriptors = self.brief.compute(image, keypoints)
        keypoints = np.array([list(kp.pt) for kp in keypoints])
        return Keypoints(image.shape[:2], keypoints, descriptors, np.ones(len(keypoints)))