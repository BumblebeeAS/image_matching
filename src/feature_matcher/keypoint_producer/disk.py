import torch
import numpy as np

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer

from lightglue import DISK



class DISKKeypointProducer(KeypointProducer): 
    def __init__(self, config=None) -> None:
        self.disk = DISK().eval().to("cuda" if torch.cuda.is_available() else "cpu")
 

    def __call__(self, image:np.ndarray) -> Keypoints:
        """
        Finds keypoits within image using DISK. 

        Args:
            image (_type_): numpy array of image
            num_keypoints: the number of retrieved keypoints

        Returns:
            Keypoints: _description_
        """
        img_dict = {"image": image}
        res_dict = self.disk.forward(img_dict)

        keypoints= res_dict["keypoints"]
        descriptors = res_dict["descriptors"]
        scores = res_dict["keypoint_scores"]

        return Keypoints(
            image.shape[:2], keypoints, descriptors, scores, 
        )