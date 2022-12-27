import os
import sys
from pathlib import Path
LoFTR_dir = os.path.abspath(Path(os.path.realpath(
    __file__)).parents[0] / "models/QuadTreeAttention/FeatureMatching/src/")  # noqa E402
sys.path.append(LoFTR_dir)  # noqa E402

from loftr import LoFTR, default_cfg
from typing import Tuple
import cv2
import numpy as np
from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer
import torch

# Requires CUDA.
# Download weights from github: `https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt`
# cd to `QuadTreeAttention` and run `python setup.py install`


class LoFTRMatchProducer(KeypointsMatchProducer):
    NUM_MATCHES = 20

    def __init__(self, template_img, config=None):
        super(LoFTRMatchProducer, self).__init__(template_img)
        if config is None:
            config = {}

        self.config = default_cfg
        self.config = {**self.config, **config}
        self.matcher = LoFTR(config=self.config)

        self.device = (
            "cuda" if torch.cuda.is_available(
            ) and self.config["cuda"] else "cpu"
        )

        # https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt
        weight_path = os.path.join(LoFTR_dir, 'weights', 'outdoor.ckpt')
        matcher.load_state_dict(torch.load(weight_path)['state_dict'])
        matcher = matcher.eval().to(device=self.device)

        self.img_size = (640, 480)

        self.template_img, self.template_scale, self.template_lxty = self.make_query_image(
            template_img)
        self.template_img = torch.from_numpy(self.template_img)[
            None][None].to(device=self.device) / 255.0
        self.last_data = {"image0": self.template_img}

    def compute_matches(self, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes matches between template image and input image.
        Args:
            image: numpy array of image
        """
        other_shape = other.shape[:2]
        other, scale, lxty = self.make_query_image(other)
        other = torch.from_numpy(other)[None][None].to(
            device=self.device) / 255.0
        with torch.no_grad():
            self.last_data = {**self.last_data, 'image1': other}
            self.matcher(self.last_data)
            conf_matrix = conf_matrix.cpu().numpy()

            total_n_matches = len(self.last_data['mkpts0_f'])
            mkpts0 = self.last_data['mkpts0_f'].cpu().numpy()
            mkpts1 = self.last_data['mkpts1_f'].cpu().numpy()
            mconf = self.last_data['mconf'].cpu().numpy()

            # Normalize confidence.
            # if len(mconf) > 0:
            #     conf_vis_min = 0.
            #     conf_min = mconf.min()
            #     conf_max = mconf.max()
            #     mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)

            # filter only the most confident features
            n_top = 20
            indices = np.argsort(mconf)[::-1]
            indices = indices[:n_top]
            mkpts0 = mkpts0[indices, :]
            mkpts1 = mkpts1[indices, :]

            # get keypoints corresponding to original image
            mkpts0[:, :2] = (
                mkpts0[:, :2] - np.array(self.template_lxty)) / self.template_scale
            mkpts1[:, :2] = (mkpts1[:, :2] - np.array(lxty)) / scale
            keypoints1 = Keypoints(
                self.get_template().shape[:2], mkpts0, None, mconf)
            keypoints2 = Keypoints(other_shape, mkpts1, None, mconf)
            return keypoints1, keypoints2

    def make_query_image(self, frame):
        query_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        query_img, scale, lxty = LoFTRMatchProducer.ratio_preserving_resize(
            query_img, self.img_size)
        return query_img, scale, lxty

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
        result[y:y + image.shape[0], x:x + image.shape[1]] = image
        return result, scale_max, (x, y)
