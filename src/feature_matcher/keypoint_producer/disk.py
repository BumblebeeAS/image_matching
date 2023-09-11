import threading

import cv2
import kornia
import numpy as np
import torch
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer
from lightglue import DISK


class DISKKeypointProducer(KeypointProducer):
    def __init__(self, config=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.disk = DISK(weights="epipolar").eval().to(self.device)
        self.lock = threading.Lock()

    def __call__(self, image: np.ndarray) -> Keypoints:
        """
        Finds keypoints within image using DISK.

        Args:
            image (_type_): numpy array of image
            num_keypoints: the number of retrieved keypoints

        Returns:
            Keypoints: _description_
        """

        with torch.no_grad():
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = kornia.image_to_tensor(rgb, False).float() / 255.0
            image_tensor = image_tensor.to(self.device)

            img_dict = {"image": image_tensor}
            with self.lock:
                res_dict = self.disk.forward(img_dict)

            keypoints = np.squeeze(res_dict["keypoints"].cpu().numpy(), axis=0)
            descriptors = np.squeeze(res_dict["descriptors"].cpu().numpy(), axis=0)
            scores = np.squeeze(res_dict["keypoint_scores"].cpu().numpy(), axis=0)

        return Keypoints(
            image.shape[:2],
            keypoints,
            descriptors,
            scores,
        )
