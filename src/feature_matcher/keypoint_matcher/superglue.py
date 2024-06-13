import logging
import os
from pathlib import Path
from typing import Dict

import torch

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.models.SuperGluePretrainedNetwork.models.superglue import (
    SuperGlue,
)
from feature_matcher.two_stage_match_producer import KeypointMatcher

SuperGlue_dir = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[0]
    / "models/SuperGluePretrainedNetwork"
)


class SuperglueKeypointMatcher(KeypointMatcher):
    default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "cuda": True,
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = self.default_config
        self.config = {**self.config, **config}

        logging.info("SuperGlue matcher config: ")
        logging.info(self.config)

        self.device = (
            "cuda"
            if torch.cuda.is_available() and self.config["cuda"]
            else "cpu"
        )

        assert self.config["weights"] in ["indoor", "outdoor"]
        path_ = os.path.join(
            SuperGlue_dir,
            "models",
            "weights",
            f'superpoint_{self.config["weights"]}.pth',
        )
        self.config["path"] = path_
        ts_file = os.path.join(
            SuperGlue_dir,
            "models",
            "weights",
            f'superpoint_{self.config["weights"]}.zip',
        )

        logging.info("Creating SuperGlue matcher...")
        if False:  # os.path.isfile(ts_file):
            self.superglue = torch.jit.load(ts_file).eval().to(self.device)
        else:
            self.superglue = SuperGlue(self.config).eval().to(self.device)

    def preprocess(
        self, keypoints1: Keypoints, keypoints2: Keypoints
    ) -> Keypoints:
        data = {}
        data["image_size0"] = torch.tensor(
            keypoints1.image_size, device=self.device
        ).float()
        data["image_size1"] = torch.tensor(
            keypoints2.image_size, device=self.device
        ).float()

        data["scores0"] = (
            torch.from_numpy(keypoints1.scores)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        data["keypoints0"] = (
            torch.from_numpy(keypoints1.keypoints)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        data["descriptors0"] = (
            torch.from_numpy(keypoints1.descriptors)
            .float()
            .to(self.device)
            .unsqueeze(0)
            .transpose(1, 2)
        )

        data["scores1"] = (
            torch.from_numpy(keypoints2.scores)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        data["keypoints1"] = (
            torch.from_numpy(keypoints2.keypoints)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        data["descriptors1"] = (
            torch.from_numpy(keypoints2.descriptors)
            .float()
            .to(self.device)
            .unsqueeze(0)
            .transpose(1, 2)
        )

        return data

    def forward(self, data) -> Dict:
        with torch.no_grad():
            pred: Dict = self.superglue(data)
        return pred

    def __call__(
        self,
        keypoints1: Keypoints,
        keypoints2: Keypoints,
        num_keypoints: int = 20,
    ) -> Dict:
        preprocessed = self.preprocess(keypoints1, keypoints2)
        preds = self.forward(preprocessed)

        matches = preds["matches0"][0].cpu().numpy()
        confidence = preds["matching_scores0"][0].cpu().detach().numpy()

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
