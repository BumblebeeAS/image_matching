from copy import copy
import logging
import time
from typing_extensions import override
from feature_matcher.models.Coarse_LoFTR_TRT.utils import get_coarse_match, make_student_config
from feature_matcher.models.Coarse_LoFTR_TRT.loftr.utils.cvpr_ds_config import default_cfg
from feature_matcher.models.Coarse_LoFTR_TRT.loftr import LoFTR
from typing import Optional, Tuple
import cv2
import numpy as np
from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer
import torch
import os
from pathlib import Path

from feature_matcher.tools import time_func
LOFTR_dir = os.path.abspath(Path(os.path.realpath(
    __file__)).parents[0] / "models/Coarse_LoFTR_TRT")


class Coarse_LoFTRMatchProducer(KeypointsMatchProducer):
    NUM_MATCHES = 20

    def __init__(self, config={}):
        super(Coarse_LoFTRMatchProducer, self).__init__(config)
        model_cfg = make_student_config(default_cfg)
        self.loftr_coarse_resolution = model_cfg['resolution'][0]
        self.img_size = (model_cfg['input_width'], model_cfg['input_height'])
        self.matcher = LoFTR(config=model_cfg)
        self.device = torch.device("cpu")
        checkpoint = torch.load(os.path.join(
            LOFTR_dir, 'weights', 'LoFTR_teacher.pt'), map_location=self.device)
        if checkpoint is not None:
            state_dict = checkpoint['model_state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            self.matcher = self.matcher.eval().to(device=self.device)
            logging.info('Successfully loaded pre-trained weights.')
        else:
            logging.error('Failed to load weights')
            raise Exception('Failed to load weights')

    @override
    def preprocess_img(self, img):
        """Convert cropped image into form required by matcher."""
        img_tensor, scale, lxty = self.make_query_image(img)
        img_tensor = torch.from_numpy(img_tensor)[None][None].to(device=self.device) / 255.0
        return (img_tensor, scale, lxty)

    def compute_matches(self, num_keypoints: int = 20, template: Optional[str] = None) -> Tuple[Keypoints, Keypoints]:
        img0, img1 = self.get_images(template)
        if img0 is None or img1 is None:
            raise Exception("Images not registered")

        template_tensor, template_scale, template_lxty = img0.results
        other_tensor, scale, lxty = img1.results

        with torch.no_grad():
            conf_matrix, _ = self.matcher(template_tensor, other_tensor)
            conf_matrix = conf_matrix.cpu().numpy()

            mkpts0, mkpts1, mconf = get_coarse_match(
                conf_matrix, self.img_size[1], self.img_size[0], self.loftr_coarse_resolution)

            # Normalize confidence.
            if len(mconf) > 0:
                conf_min = mconf.min()
                conf_max = mconf.max()
                mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)

        # filter only the most confident features
        indices = np.argsort(mconf)[::-1]
        indices = indices[:num_keypoints]
        mkpts0 = mkpts0[indices, :]
        mkpts1 = mkpts1[indices, :]

        # get keypoints corresponding to non-cropped image
        mkpts0[:, :2] = (mkpts0[:, :2] - np.array(template_lxty)) / template_scale
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
        query_img, scale, lxty = Coarse_LoFTRMatchProducer.ratio_preserving_resize(
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
