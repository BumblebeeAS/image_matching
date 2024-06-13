import logging
import os
from pathlib import Path
import sys
import threading

from ament_index_python import get_package_share_directory
import cv2
import numpy as np
import torch

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.tools import image2tensor
from feature_matcher.two_stage_match_producer import KeypointProducer

SuperGlue_dir = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[1]
    / "models/SuperGluePretrainedNetwork"
)  # noqa E402
sys.path.insert(0, SuperGlue_dir)  # noqa E402

from models.superpoint import SuperPoint  # noqa E402


class SuperPointKeypointProducer(KeypointProducer):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": os.path.join(
            get_package_share_directory("image_matching"),
            "models",
            "SuperGluePretrainedNetwork",
            "models",
            "weights",
            "superpoint_v1.pth",
        ),
        "cuda": True,
    }

    def __init__(self, config={}):
        self.debug = config.get("debug", False)
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO
        )

        self.config = {**SuperPointKeypointProducer.default_config, **config}
        logging.info("SuperPoint detector config: ")
        logging.info(self.config)

        self.device = (
            "cuda"
            if torch.cuda.is_available() and self.config["cuda"]
            else "cpu"
        )
        path_ = self.config["path"]
        parent_dir = os.path.dirname(path_)
        ref_file = os.path.basename(path_).split(".")[0]
        ts_file = os.path.join(parent_dir, ref_file + ".zip")

        logging.info("Creating SuperPoint detector...")
        if False:  # os.path.isfile(ts_file):
            self.superpoint = torch.jit.load(ts_file).eval().to(self.device)
        else:
            self.superpoint = SuperPoint(self.config).eval().to(self.device)
        self.lock = threading.Lock()

    def preprocess(self, image) -> torch.Tensor:
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logging.error(e)
            pass  # Squeezed gray image array

        logging.debug("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        return image_tensor

    def __call__(self, image: np.ndarray) -> Keypoints:
        """
        Finds keypoints within image using SIFT.

        Args:
            image: numpy array of image
            num_keypoints: the number of retrieved keypoints
        Returns:
            N keypoints.
        """
        preprocessed = self.preprocess(image)
        with torch.no_grad():
            with self.lock:
                pred = self.superpoint(preprocessed)
            keypoints = pred["keypoints"][0].cpu().numpy()
            descriptors = pred["descriptors"][0].cpu().numpy().transpose()
            scores = pred["scores"][0].cpu().numpy()

        return Keypoints(image.shape[:2], keypoints, descriptors, scores)
