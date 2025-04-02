import logging
import os
import threading
from pathlib import Path

from cv2.typing import MatLike

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat
from feature_matcher.two_stage_match_producer import KeypointProducer

XFEAT_DIR = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[1] / "models/accelerated_features"
)  # noqa E402


class XFeatKeypointProducer(KeypointProducer):
    default_config = {
        "weights": os.path.join(XFEAT_DIR, "weights", "xfeat.pt"),
        "top_k": 4096,
    }

    def __init__(self, config: dict = {}) -> None:
        self.debug = config.get("debug", False)
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)

        self.config = {
            **XFeatKeypointProducer.default_config,
            **config,
        }

        logging.info("XFeat dectector config: ")
        logging.info(self.config)
        logging.info("Creating XFeat detector...")

        if "debug" in self.config:
            self.config.pop("debug")

        self.model = XFeat(**self.config)
        self.device = self.model.dev
        self.lock = threading.Lock()

    def __call__(self, image: MatLike) -> Keypoints:
        """
        Find keypoints within image using XFeat

        Args:
            image: numpy array of image

        Returns:
            N keypoints

        Consider adding option to use detectAndComputeDense
        """
        with self.lock:
            # XFeat handles additional preprocessing, such as conversion into tensor
            pred = self.model.detectAndCompute(image)[0]

        return Keypoints(
            image.shape[:2],
            pred["keypoints"].cpu().numpy(),
            pred["descriptors"].cpu().numpy(),
            pred["scores"].cpu().numpy(),
        )
