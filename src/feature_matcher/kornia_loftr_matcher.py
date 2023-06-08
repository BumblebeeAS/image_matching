import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

from typing_extensions import override

from feature_matcher.keypoints import Keypoints
from feature_matcher.keypoints_match_producer import KeypointsMatchProducer


class Kornia_LoFTRMatchProducer(KeypointsMatchProducer):
    NUM_MATCHES = 20

    def __init__(self, config={"cuda": True}):
        super(Kornia_LoFTRMatchProducer, self).__init__(config)
        self.W = 1024
        self.H = 768
        self.matcher = KF.LoFTR(pretrained='outdoor').eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("cuda", False) else "cpu"
        )
        self.matcher = self.matcher.to(self.device)
        logging.info("Successfully loaded pre-trained weights.")

    @override
    def preprocess_img(self, img):
        """Convert cropped image into form required by matcher."""
        timg1 = K.image_to_tensor(img, None).float().to(self.device) / 255.
        h1, w1 = timg1.shape[2:]
        timg1 = K.geometry.transform.resize(timg1, (self.H, self.W))
        return timg1, h1, w1

    def compute_matches(
        self, num_keypoints: int = 20, template: Optional[str] = None
    ) -> Tuple[Keypoints, Keypoints]:
        img0, img1 = self.get_images(template)
        if img0 is None or img1 is None:
            raise Exception("Images not registered")

        template_tensor, h_template, w_template = img0.results
        other_tensor, h_other, w_other = img1.results

        batch = {'image0': template_tensor, 'image1': other_tensor}
        with torch.no_grad():
            out = self.matcher(batch)
        
        template_pts = out['keypoints0'].detach().cpu().numpy() 
        image_pts = out['keypoints1'].detach().cpu().numpy()
        template_pts[:,0] *= float (w_template) / float(self.W)
        template_pts[:,1] *= float (h_template) / float(self.H)
        image_pts[:,0] *= float (w_other) / float(self.W)
        image_pts[:,1] *= float (h_other) / float(self.H)

        print("Confidence")
        mconf = out['confidence'].detach().cpu().numpy().reshape(-1)

        # Normalize confidence.
        if len(mconf) > 0:
            conf_min = mconf.min()
            conf_max = mconf.max()
            mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)

        # filter only the most confident features
        indices = np.argsort(mconf)[::-1]
        indices = indices[:num_keypoints]
        mkpts0 = template_pts[indices, :]
        mkpts1 = image_pts[indices, :]
        mconf = mconf[indices]

        # get keypoints corresponding to non-cropped image
        keypoints1 = Keypoints(img0.img.shape[:2], mkpts0, None, mconf)
        keypoints2 = Keypoints(img1.img.shape[:2], mkpts1, None, mconf)
        print("T: ", keypoints1)
        return keypoints1, keypoints2

    # def make_query_image(self, frame):
    #     query_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     query_img, scale, lxty = Coarse_LoFTRMatchProducer.ratio_preserving_resize(
    #         query_img, self.img_size
    #     )
    #     return query_img, scale, lxty

    # def ratio_preserving_resize(image, img_size):
    #     # ratio preserving resize
    #     img_h, img_w = image.shape
    #     scale_h = img_size[1] / img_h
    #     scale_w = img_size[0] / img_w
    #     scale_max = min(scale_h, scale_w)
    #     new_size = (int(img_w * scale_max), int(img_h * scale_max))
    #     result = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    #     image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    #     # center crop
    #     x = img_size[0] // 2 - new_size[0] // 2
    #     y = img_size[1] // 2 - new_size[1] // 2
    #     result[y : y + image.shape[0], x : x + image.shape[1]] = image
    #     return result, scale_max, (x, y)
