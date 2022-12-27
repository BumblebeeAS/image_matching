from feature_matcher.models.Coarse_LoFTR_TRT.utils import get_coarse_match, make_student_config
from feature_matcher.models.Coarse_LoFTR_TRT.loftr.utils.cvpr_ds_config import default_cfg
from feature_matcher.models.Coarse_LoFTR_TRT.loftr import LoFTR
from typing import Tuple
import cv2
import numpy as np
from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer
import torch
import os
from pathlib import Path
LOFTR_dir = os.path.abspath(Path(os.path.realpath(
    __file__)).parents[0] / "models/Coarse_LoFTR_TRT")


class Coarse_LoFTRMatchProducer(KeypointsMatchProducer):
    NUM_MATCHES = 20

    def __init__(self, template_img):
        super(Coarse_LoFTRMatchProducer, self).__init__(template_img)
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
            print('Successfully loaded pre-trained weights.')
        else:
            print('Failed to load weights')

        self.template_img, self.template_scale, self.template_lxty = self.make_query_image(
            template_img)
        self.template_img = torch.from_numpy(self.template_img)[
            None][None].to(device=self.device) / 255.0

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
            conf_matrix, _ = self.matcher(self.template_img, other)
            conf_matrix = conf_matrix.cpu().numpy()

            mkpts0, mkpts1, mconf = get_coarse_match(
                conf_matrix, self.img_size[1], self.img_size[0], self.loftr_coarse_resolution)

            # Normalize confidence.
            if len(mconf) > 0:
                conf_min = mconf.min()
                conf_max = mconf.max()
                mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)

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
