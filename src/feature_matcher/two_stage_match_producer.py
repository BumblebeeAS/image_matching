from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer


class KeypointProducer(ABC):
    """
    Produces keypoints to be used in TwoStageMatcher.
    """
    @abstractmethod
    def __call__(self, image) -> Keypoints:
        """
        Returns:
            M x 3 new keypoints in the first image
            M descriptors of the keypoints
            M scores of the keypoints
        """
        pass


class KeypointMatcher(ABC):
    """
    Matches keypoints from two images. Used in TwoStageMatcher.
    """
    @abstractmethod
    def __call__(self, keypoints1: Keypoints, keypoints2: Keypoints) -> Tuple[Keypoints, Keypoints]:
        """
        Returns:
            M Keypoints in the first image
            M Keypoints in the second image that matches to keypoints in first image.
        """
        pass

class TwoStageMatchProducer(KeypointsMatchProducer):
    NUM_MATCHES = 20

    def __init__(self, template_img, producer: KeypointProducer, matcher: KeypointMatcher):
        super(TwoStageMatchProducer, self).__init__(template_img)
        self.producer = producer
        self.matcher = matcher
        self.template_kp = self.producer(template_img)

    def compute_matches(self, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes matches between template image and input image.
        Args:
            image: numpy array of image
        """
        
        kp = self.producer(other)
        return self.matcher(self.template_kp, kp)

