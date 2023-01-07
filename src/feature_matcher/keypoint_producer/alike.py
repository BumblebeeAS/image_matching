import os
import sys
from pathlib import Path

ALIKE_DIR = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[1] / "models/ALIKE"
)  # noqa E402
sys.path.insert(0, ALIKE_DIR)  # noqa E402

import logging

import cv2
import numpy as np
import torch

from alike import ALike, configs

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointProducer


class AlikeKeypointProducer(KeypointProducer):
    default_config = {
        "model": "alike-n",
        "top_k": -1,
        "scores_th": 0.1,
        "n_limit": 5000,
        "cuda": True,
    }

    def __init__(self, config={}) -> None:
        self.debug = config.get("debug", False)
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)

        self.config = {
            **AlikeKeypointProducer.default_config,
            **config,
        }

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        self.config = {
            **self.config,
            "device": self.device,
        }

        logging.info("ALIKE detector config: ")
        logging.info(self.config)

        logging.info("Creating ALIKE detector...")

        # Remove unnecessary config
        model = self.config.pop("model", "alike-n")
        self.config.pop("cuda")
        self.config.pop("debug")

        if model not in configs.keys():
            raise ValueError(
                f"Model {model} not found. Available models: {configs.keys()}"
            )

        self.model = ALike(**configs[model], **self.config)
        self.sub_pixel = True

    def preprocess(self, image) -> np.ndarray:
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __call__(self, image) -> Keypoints:
        """
        Finds keypoints within image using ALIKE.

        Args:
            image: numpy array of image
            num_keypoints: the number of retrieved keypoints

        Returns:
            N keypoints.
        """
        pred = self.model(self.preprocess(image), sub_pixel=self.sub_pixel)
        return Keypoints(
            image.shape[:2], pred["keypoints"], pred["descriptors"], pred["scores"]
        )
