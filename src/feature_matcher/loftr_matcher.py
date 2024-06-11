import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from typing_extensions import override

from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer

LoFTR_dir = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[0]
    / "models/QuadTreeAttention/FeatureMatching/src/"
)  # noqa E402
sys.path.append(LoFTR_dir)  # noqa E402
QuadTreeAttn_DIR = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[0]
    / "models/QuadTreeAttention/QuadTreeAttention/"
)  # noqa E402
sys.path.append(QuadTreeAttn_DIR)  # noqa E402

from loftr import LoFTR  # noqa E402
from loftr import default_cfg  # noqa E402

# Requires CUDA.
# Download weights from github: `https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt`
# cd to `QuadTreeAttention` and run `python setup.py install`


class LoFTRMatchProducer(KeypointsMatchProducer):
    NUM_MATCHES = 20

    def __init__(self, config={}):
        super(LoFTRMatchProducer, self).__init__()

        self.config = default_cfg
        self.config = {**self.config, **config}
        self.matcher = LoFTR(config=self.config)

        if not torch.cuda.is_available:
            raise Exception("LoFTR requires CUDA")

        self.device = torch.device("cuda")

        # https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt
        weight_path = os.path.join(LoFTR_dir, "weights", "outdoor.ckpt")
        self.matcher.load_state_dict(torch.load(weight_path)["state_dict"])
        self.matcher = self.matcher.eval().to(device=self.device)

        self.img_size = (640, 480)

    @override
    def preprocess_img(self, img):
        """Convert cropped image into form required by matcher."""
        img_tensor, scale, lxty = self.make_query_image(img)
        img_tensor = (
            torch.from_numpy(img_tensor)[None][None].to(device=self.device)
            / 255.0
        )
        return (img_tensor, scale, lxty)

    def compute_matches(
        self, num_keypoints: int = 20, template: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        img0, img1 = self.get_images(template)

        if img0 is None or img1 is None:
            raise Exception("Images not registered")

        template_tensor, template_scale, template_lxty = img0.results
        other_tensor, scale, lxty = img1.results

        with torch.no_grad():
            self.last_data = {
                "image0": template_tensor,
                "image1": other_tensor,
            }
            self.matcher(self.last_data)

        mkpts0 = self.last_data["mkpts0_f"].cpu().numpy()
        mkpts1 = self.last_data["mkpts1_f"].cpu().numpy()
        mconf = self.last_data["mconf"].cpu().numpy()

        # filter only the most confident features
        indices = np.argsort(mconf)[::-1]
        indices = indices[:num_keypoints]
        mkpts0 = mkpts0[indices, :]
        mkpts1 = mkpts1[indices, :]
        mconf = mconf[indices]

        # get keypoints corresponding to original image
        mkpts0[:, :2] = (
            mkpts0[:, :2] - np.array(template_lxty)
        ) / template_scale
        mkpts1[:, :2] = (mkpts1[:, :2] - np.array(lxty)) / scale
        if img0.lxtyrxby is not None:
            mkpts0[:, :2] = mkpts0[:, :2] + img0.lxtyrxby[:2]
        if img1.lxtyrxby is not None:
            mkpts1[:, :2] = mkpts1[:, :2] + img1.lxtyrxby[:2]
        keypoints1 = Keypoints(img0.img.shape[:2], mkpts0, None, mconf)
        keypoints2 = Keypoints(img1.img.shape[:2], mkpts1, None, mconf)
        return keypoints1, keypoints2

    def make_query_image(self, frame):
        query_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        query_img, scale, lxty = LoFTRMatchProducer.ratio_preserving_resize(
            query_img, self.img_size
        )
        return query_img, scale, lxty

    @staticmethod
    def ratio_preserving_resize(image, img_size):
        # ratio preserving resize
        img_h, img_w = image.shape
        scale_h = img_size[1] / img_h
        scale_w = img_size[0] / img_w
        scale_max = min(scale_h, scale_w)
        new_size = (int(img_w * scale_max), int(img_h * scale_max))
        result = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        # center crop
        x = img_size[0] // 2 - new_size[0] // 2
        y = img_size[1] // 2 - new_size[1] // 2
        result[y : y + image.shape[0], x : x + image.shape[1]] = image
        return result, scale_max, (x, y)
