import os
import sys
import threading
from pathlib import Path

import numpy as np

SuperGlue_dir = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[1] / "models/DALF_CVPR_2023"
)  # noqa E402
sys.path.insert(0, SuperGlue_dir)  # noqa E402
import torch
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer
from modules.models.DALF import DALF_extractor as DALF


class DALFKeypointProducer(KeypointProducer):
    def __init__(self, config=None) -> None:
        self.lock = threading.Lock()
        self.dalf = DALF(dev=torch.device("cuda" if torch.cuda.is_available else "cpu"))

    def __call__(self, image) -> Keypoints:
        with self.lock:
            keypoints, descriptors = self.dalf.detectAndCompute(image)
        keypoints = np.array([kp.pt for kp in keypoints])
        return Keypoints(
            image.shape[:2], keypoints, descriptors, np.ones(len(keypoints))
        )
