import numpy as np
import torch
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.keypoint_producer.xfeat import XFeatKeypointProducer
from feature_matcher.keypoint_matcher.xfeat import XFeatKeypointMatcher
import os
from pathlib import Path
from ament_index_python import get_package_share_directory
import cv2
from feature_matcher.tools import plot_matches, create_show_image
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat

template_dir = os.path.abspath(Path(get_package_share_directory("image_matching")) / "templates")
weights = os.path.join(get_package_share_directory("image_matching"),
                                "models", "accelerated_features",
                                "weights", "xfeat.pt")

def main():
    producer = XFeatKeypointProducer()
    matcher = XFeatKeypointMatcher()
    bobby = cv2.imread(f"{template_dir}/Bootlegger.jpeg")
    gun = cv2.imread(f"{template_dir}/Tommy Gun.jpeg")
    model = XFeat(weights)
    kp1 = producer(bobby)
    kp2 = producer(gun)
    kp1, kp2 = matcher(kp1, kp2)
    out = plot_matches(bobby, gun, kp1.keypoints, kp2.keypoints)
    create_show_image()(out)
    # kp1, kp2 = model.match_xfeat(bobby, gun)
    # out = plot_matches(bobby, gun, kp1, kp2)
    # create_show_image()(out)
if __name__ == "__main__":
    main()
