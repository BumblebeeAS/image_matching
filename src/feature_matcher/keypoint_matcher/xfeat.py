import os
import sys
import logging
import threading
from typing import Dict
from pathlib import Path
from ament_index_python import get_package_share_directory

XFEAT_DIR = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[0] / "models/accelerated_features/modules"
)

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat
from feature_matcher.two_stage_match_producer import KeypointMatcher

class XFeatKeypointMatcher(KeypointMatcher):
    default_config = {
        "weights": os.path.join(get_package_share_directory("image_matching"),
                                "models", "accelerated_features",
                                "weights", "xfeat.pt"),
        "top_k": 4096,
        "cuda": True
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = self.default_config
        self.config = {
            **self.config,
            **config
        }

        logging.info("XFeat matcher config")
        logging.info("self.config")

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        self.model = XFeat(self.config["weights"], self.config["top_k"])

        self.lock = threading.lock()
    
    # not sure if preprocessing is necessary, ideally keypoints also generated from xfeat
    def __call__(self, keypoints1: Keypoints, keypoints2: Keypoints):
        # matching occurs on descriptors
        idxs0, idxs1 = self.model.match(keypoints1.descriptors, keypoints2.descriptors)
        
        # does not give confidence of match?
        # select relevant keypoints
        keypoints1 = keypoints1[idxs0]
        keypoints2 = keypoints2[idxs1]

        return keypoints1, keypoints2