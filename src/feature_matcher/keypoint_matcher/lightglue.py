import logging
import threading
from typing import Dict, Tuple

import torch

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.models.LightGlue.lightglue import LightGlue
from feature_matcher.two_stage_match_producer import KeypointMatcher


class LightglueKeypointMatcher(KeypointMatcher):
    default_config = {
        "weights": "superpoint",  # One of "superpoint", "disk"
        "cuda": True,
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = self.default_config
        self.config = {**self.config, **config}

        logging.info("LightGlue matcher config: ")
        logging.info(self.config)

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        assert self.config["weights"] in ["disk", "superpoint"]
        self.superglue = (
            LightGlue(pretrained=self.config.get("weights", "superpoint"))
            .eval()
            .to(self.device)
        )
        self.lock = threading.Lock()

    def preprocess(self, keypoints1: Keypoints, keypoints2: Keypoints) -> Keypoints:
        data0, data1 = {}, {}
        data0["image_size"] = (
            torch.tensor(
                (keypoints1.image_size[1], keypoints1.image_size[0]),
                device=self.device,
            )
            .float()
            .unsqueeze(0)
        )
        data1["image_size"] = (
            torch.tensor(
                (keypoints2.image_size[1], keypoints2.image_size[0]),
                device=self.device,
            )
            .float()
            .unsqueeze(0)
        )

        data0["scores"] = (
            torch.from_numpy(keypoints1.scores).float().to(self.device).unsqueeze(0)
        )
        data0["keypoints"] = (
            torch.from_numpy(keypoints1.keypoints).float().to(self.device).unsqueeze(0)
        )
        data0["descriptors"] = (
            torch.from_numpy(keypoints1.descriptors)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )

        data1["scores"] = (
            torch.from_numpy(keypoints2.scores).float().to(self.device).unsqueeze(0)
        )
        data1["keypoints"] = (
            torch.from_numpy(keypoints2.keypoints).float().to(self.device).unsqueeze(0)
        )
        data1["descriptors"] = (
            torch.from_numpy(keypoints2.descriptors)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )

        data = {"image0": data0, "image1": data1}

        return data

    def forward(self, data) -> Dict:
        with torch.no_grad():
            with self.lock:
                pred: Dict = self.superglue(data)
        return pred

    def __call__(
        self,
        keypoints1: Keypoints,
        keypoints2: Keypoints,
        num_keypoints: int = 20,
    ) -> Tuple[Keypoints, Keypoints]:
        preprocessed = self.preprocess(keypoints1, keypoints2)

        with torch.no_grad():
            preds = self.superglue(preprocessed)
            matches = preds["matches0"][0].cpu().numpy()
            confidence = preds["matching_scores0"][0].cpu().numpy()

        # Sort them in the order of their confidence.
        match_conf = []
        for i, (m, c) in enumerate(zip(matches, confidence)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: -x[2])

        valid = [[x[0], x[1]] for x in match_conf if x[1] > -1]
        v0 = [x[0] for x in valid]
        v1 = [x[1] for x in valid]
        keypoints1 = keypoints1[v0][:num_keypoints]
        keypoints2 = keypoints2[v1][:num_keypoints]

        keypoints1 = Keypoints(
            keypoints1.image_size,
            keypoints1.keypoints,
            keypoints1.descriptors,
            keypoints1.scores * confidence[v0][:num_keypoints],
        )
        keypoints2 = Keypoints(
            keypoints2.image_size,
            keypoints2.keypoints,
            keypoints2.descriptors,
            keypoints2.scores * confidence[v0][:num_keypoints],
        )

        return keypoints1, keypoints2
