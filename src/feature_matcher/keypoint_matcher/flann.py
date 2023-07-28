import threading
from typing import Tuple

import cv2
import numpy as np
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointMatcher


class FlannKeypointMatcher(KeypointMatcher):
    def __init__(self, config={"cross_check": True, "num_matches": 500}):
        self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        # self.matcher = cv2.BFMatcher(crossCheck=config.get("cross_check", False), normType=cv2.NORM_L1)
        num_matches = config.get("num_matches", 500)
        if not isinstance(num_matches, int):
            raise ValueError("num_matches must be an integer")
        self.num_matches = num_matches
        self.lock = threading.Lock()

    def __call__(
        self, keypoints1: Keypoints, keypoints2: Keypoints, num_keypoints: int = 20
    ) -> Tuple[Keypoints, Keypoints]:
        """
        Finds matches between keypoints1 and keypoints2 using Brute Force Matcher.

        Args:
            keypoints1: Keypoints
            keypoints2: Keypoints
        Returns:
            M Keypoints in the first image
            M Keypoints in the second image that matches to keypoints in first image.
        """
        with self.lock:
            matches = self.matcher.knnMatch(
                np.ascontiguousarray(keypoints1.descriptors),
                np.ascontiguousarray(keypoints2.descriptors),
                2,
            )
        selected_matches = []
        matches1, matches2 = [], []
        # -- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.75
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        for match in sorted(good_matches, key=lambda x: x.distance):
            if len(selected_matches) >= self.num_matches:
                break
            i1, i2 = match.queryIdx, match.trainIdx
            keypoint1, keypoint2 = i1, i2

            # check if the keypoint matches
            selected_match = "-".join([str(keypoint1), str(keypoint2)])
            if selected_match in selected_matches:
                continue

            # add matched keypoint
            matches1.append(keypoint1)
            matches2.append(keypoint2)
            selected_matches.append(selected_match)
        keypoints1, keypoints2 = keypoints1[matches1], keypoints2[matches2]
        return keypoints1, keypoints2

    def _filter_matches(
        self, keypoints1: Keypoints, keypoints2: Keypoints, matches: list
    ) -> Tuple[Keypoints, Keypoints]:
        """
        Filters matches based on distance and ratio test.

        Args:
            keypoints1: Keypoints
            keypoints2: Keypoints
            matches: list of cv2.Matches
        Returns:
            M Keypoints in the first image
            M Keypoints in the second image that matches to keypoints in first image.
        """
        # filter matches based on distance
        matches = [match for match in matches if match.distance < 0.7]

        # filter matches based on ratio test
        matches = [
            match for match in matches if match.distance < 0.8 * matches[1].distance
        ]

        # get keypoints from matches
        keypoints1 = [keypoints1.keypoints[match.queryIdx] for match in matches]
        keypoints2 = [keypoints2.keypoints[match.trainIdx] for match in matches]
        return Keypoints(keypoints1), Keypoints(keypoints2)
