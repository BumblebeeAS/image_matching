import os
import sys
import threading
from pathlib import Path
from ament_index_python import get_package_share_directory

XFEAT_DIR = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[0] / "models/accelerated_features"
)

sys.path.insert(0, XFEAT_DIR)

import logging

import cv2
import numpy as np
import torch
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer
from feature_matcher.tools import image2tensor

class XFeatKeypointProducer(KeypointProducer):
    default_config = {
        "weights": os.path.join(get_package_share_directory("image_matching"),
                                "models", "accelerated_features",
                                "weights", "xfeat.pt"),
        "top_k": 4096,
        "cuda": True
    }

    def __init__(self, config={}) -> None:
        self.debug = config.get("debug", False)
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)

        self.config = {
            **XFeatKeypointProducer.default_config,
            **config,
        }

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        self.config = {
            **self.config,
            "device": self.device,
        }

        logging.info("XFeat dectector config: ")
        logging.info(self.config)

        logging.info("Creating XFeat detector...")

        self.config.pop("cuda")
        self.config.pop("device")

        self.model = XFeat(**self.config)
        self.lock = threading.Lock()

    # Up to us to change this, XFeat can accept RGB or grayscale
    def preprocess(self, image) -> np.ndarray:
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            logging.error(e)
            pass

        logging.debug("Detecting keypoints with XFeat")
        return image2tensor(image, self.device)
    
    def __call__(self, image: np.ndarray) -> Keypoints:
        """
        Find keypoints within image using XFeat

        Args:
            image: numpy array of image

        Returns:
            N keypoints

        Consider adding option to use detectAndComputeDense
        """
        preprocessed = self.preprocess(image)
        print(preprocessed.shape)
        with self.lock:
            # XFeat handles additional preprocessing, such as conversion into tensor
            pred = self.model.detectAndCompute(preprocessed)[0]
            # pred contains keypoints, scores and descriptors
        return Keypoints(
            image.shape[:2], pred["keypoints"], pred["descriptors"], pred["scores"]
        )