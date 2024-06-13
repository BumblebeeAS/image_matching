import os
from pathlib import Path
import sys
from typing import Optional

import cv2
import torch

from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer

DKM_dir = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[0] / "models/DKM"
)  # noqa E402
sys.path.append(DKM_dir)  # noqa E402

from dkm import DKMv3_outdoor  # noqa E402
from dkm.utils import numpy_to_pil  # noqa E402


class DKMv3MatchProducer(KeypointsMatchProducer):
    def __init__(self, config={"cuda": True}):
        super(DKMv3MatchProducer, self).__init__()

        self.config = config
        self.config["cuda"] = True
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and config.get("cuda", False)
            else "cpu"
        )

        print("Loading DKM...")
        self.model = DKMv3_outdoor(device=self.device)
        print("Loaded!")
        print(self.device)

        self.model.w_resized = 640  # width of image used
        self.model.h_resized = 480  # height of image used
        self.model.upsample_preds = False
        self.img_size = (640, 480)

    def preprocess_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return numpy_to_pil(img), original_size

    def compute_matches(
        self, num_keypoints: int = 20, template: Optional[str] = None
    ):
        img0, img1 = self.get_images(template)

        if img0 is None or img1 is None:
            raise Exception("Images not registered")

        template_img, template_size = img0.results
        query_img, query_size = img1.results

        warp, certainty = self.model.match(
            template_img, query_img, device=self.device
        )

        # samples from the warp using the certainty
        matches, certainty = self.model.sample(
            warp, certainty, num=5 * num_keypoints
        )

        # convenience function to convert normalized matches to pixel coordinates
        kpts_A, kpts_B = self.model.to_pixel_coordinates(
            matches,
            template_size[0],  # HA
            template_size[1],  # WA
            query_size[0],  # HB
            query_size[1],  # WB
        )

        mkpts0 = kpts_A.cpu().numpy()
        mkpts1 = kpts_B.cpu().numpy()
        mconf = certainty.cpu().numpy()
        keypoints1 = Keypoints(img0.img.shape[:2], mkpts0, None, mconf)
        keypoints2 = Keypoints(img1.img.shape[:2], mkpts1, None, mconf)
        print("DONE")

        return keypoints1, keypoints2
