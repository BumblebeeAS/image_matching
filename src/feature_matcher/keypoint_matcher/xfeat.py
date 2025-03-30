import logging
import os
import threading
from pathlib import Path

import torch

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat
from feature_matcher.two_stage_match_producer import KeypointMatcher

XFEAT_DIR = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[1] / "models/accelerated_features"
)  # noqa E402


class XFeatKeypointMatcher(KeypointMatcher):
    default_config = {
        "weights": os.path.join(XFEAT_DIR, "weights", "xfeat.pt"),
        "top_k": 4096,
        "cuda": True,
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = self.default_config
        self.config = {**self.config, **config}

        logging.info("XFeat matcher config")
        logging.info("self.config")

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        self.model = XFeat(self.config["weights"], self.config["top_k"])

        self.lock = threading.Lock()

    # not sure if preprocessing is necessary, ideally keypoints also generated from xfeat
    def __call__(self, keypoints1: Keypoints, keypoints2: Keypoints, num_keypoints=20):
        # matching occurs on descriptors
        idxs0, idxs1 = self.model.match(
            torch.tensor(keypoints1.descriptors),
            torch.tensor(keypoints2.descriptors),
            min_cossim=0.82,
        )

        # PyTorch tensors are not officially supported as indexers in NumPy
        # e.g., a[tensor([1])] reduces dimensions of a by 1
        # but a[tensor([1,2])] maintains the number of dimensions
        idxs0 = idxs0.numpy()
        idxs1 = idxs1.numpy()

        # does not give confidence of match?
        # select relevant keypoints
        keypoints1 = keypoints1[idxs0]
        keypoints2 = keypoints2[idxs1]

        return keypoints1[:num_keypoints], keypoints2[:num_keypoints]
