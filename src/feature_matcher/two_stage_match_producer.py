from abc import ABC, abstractmethod
from typing import Tuple

from typing_extensions import override

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
    def __call__(
        self, keypoints1: Keypoints, keypoints2: Keypoints, num_keypoints: int
    ) -> Tuple[Keypoints, Keypoints]:
        """
        Args:
            keypoints1: Keypoints in the first image
            keypoints2: Keypoints in the second image
            num_keypoints: Number of keypoints to match
        Returns:
            num_keypoints Keypoints in the first image
            num_keypoints Keypoints in the second image that matches to keypoints in first image.
        """
        pass


class TwoStageMatchProducer(KeypointsMatchProducer):
    def __init__(self, producer: KeypointProducer, matcher: KeypointMatcher):
        super(TwoStageMatchProducer, self).__init__()
        self.producer = producer
        self.matcher = matcher

    @override
    def preprocess_img(self, image):
        """Preprocess the cropped image."""
        return self.producer.__call__(image)

    def compute_matches(
        self, num_keypoints: int = 20, template: str = None
    ) -> Tuple[Keypoints, Keypoints]:
        img0, img1 = self.get_images(template)
        assert img0.results is not None, "Template image not registered"
        assert img1.results is not None, "Camera image not registered"
        if img0 is None or img1 is None:
            raise Exception("Images not registered")

        template_kp = img0.results
        if img0.lxtyrxby is not None:
            template_kp = Keypoints(
                template_kp.image_size,
                template_kp.keypoints - img0.lxtyrxby[:2],
                template_kp.descriptors,
                template_kp.scores,
            )
        kp: Keypoints = img1.results
        if img1.lxtyrxby is not None:
            kp = Keypoints(
                kp.image_size,
                kp.keypoints + img1.lxtyrxby[:2],
                kp.descriptors,
                kp.scores,
            )

        return self.matcher(template_kp, kp, num_keypoints)
