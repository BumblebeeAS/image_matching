from typing import Any
import cv2
import numpy as np
import torch
import kornia
import kornia.feature as kornia_feature

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer


class KeyAffHardKeypointProducer(KeypointProducer):
    def __init__(self, config: dict = {"num_keypoints": 5000}):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keynet_affnet_hardnet = kornia_feature.KeyNetAffNetHardNet(
            num_features=config.get("num_keypoints", 5000),
            upright=False,
            device=self.device,
        )

    def __call__(self, image: Any) -> Keypoints:
        """
        Finds keypoints within image using SIFT.

        Args:
            image: numpy array of image
            num_keypoints: the number of retrieved keypoints
        Returns:
            N keypoints.
        """
        with torch.no_grad():
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = kornia.image_to_tensor(rgb, False).float() / 255.0
            image_tensor = kornia.color.rgb_to_grayscale(image_tensor).to(self.device)
            lafs, _, descriptors = self.keynet_affnet_hardnet(image_tensor)
        keypoints = kornia_feature.get_laf_center(lafs)
        keypoints = keypoints.cpu().numpy()
        descriptors = descriptors.cpu().numpy()

        return Keypoints(
            image.shape[:2],
            keypoints.squeeze(0),
            descriptors.squeeze(0),
            np.ones(keypoints.shape[1]),
        )
