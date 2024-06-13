import threading
from typing import Tuple

import cv2

from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.two_stage_match_producer import KeypointMatcher


def get_norm_type(norm_str: str) -> int:
    if norm_str == "l2":
        return cv2.NORM_L2
    elif norm_str == "l1":
        return cv2.NORM_L1
    elif norm_str == "hamming":
        return cv2.NORM_HAMMING
    elif norm_str == "hamming2":
        return cv2.NORM_HAMMING2
    else:
        raise ValueError(f"Invalid norm type: {norm_str}")


class BFKeypointMatcher(KeypointMatcher):
    def __init__(self, config={"cross_check": True, "norm_type": "l2"}):
        self.matcher = cv2.BFMatcher(
            crossCheck=config.get("cross_check", True),
            normType=get_norm_type(config.get("norm_type", "l2")),
        )
        self.lock = threading.Lock()

    def __call__(
        self,
        keypoints1: Keypoints,
        keypoints2: Keypoints,
        num_keypoints: int = 500,
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
            matches = self.matcher.match(
                keypoints1.descriptors, keypoints2.descriptors
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
            if len(selected_matches) >= num_keypoints:
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
            match
            for match in matches
            if match.distance < 0.8 * matches[1].distance
        ]

        # get keypoints from matches
        keypoints1 = [
            keypoints1.keypoints[match.queryIdx] for match in matches
        ]
        keypoints2 = [
            keypoints2.keypoints[match.trainIdx] for match in matches
        ]
        return Keypoints(keypoints1), Keypoints(keypoints2)
