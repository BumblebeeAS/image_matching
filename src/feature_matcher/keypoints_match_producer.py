import copy
import logging
from abc import ABC, abstractmethod
from threading import Lock, Thread
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np

from feature_matcher.keypoints import Keypoints
from feature_matcher.tools import create_show_image, plot_matches, white_balance
from utils.logging import Logger

T = TypeVar("T")


class ImageWrapper(Generic[T]):
    def __init__(
        self, id: Union[str, int], img: np.ndarray, lxtyrxby: Tuple[int] = None
    ) -> None:
        self.id = id
        self.img = img
        self.lxtyrxby = lxtyrxby
        self.results: T = None
        self.thread = None

    def register_thread(self, thread: Thread):
        self.thread = thread

    def add_results(self, results: T):
        self.results = results


class Buffer(Generic[T]):
    def __init__(self) -> None:
        self.images: List[ImageWrapper[T]] = []
        self.templates: Dict[str, ImageWrapper[T]] = {}
        self.lock = Lock()

    def add_template(self, img: ImageWrapper[T]) -> None:
        self.lock.acquire()
        if isinstance(img.id, str):
            self.templates[img.id] = img
        self.lock.release()

    def remove_template(self, name: str) -> None:
        self.lock.acquire()
        if name in self.templates:
            self.templates.pop(name)
        self.lock.release()

    def append(self, x: ImageWrapper[T]) -> None:
        self.lock.acquire()
        if isinstance(x.id, int):
            self.images.append(x)
        self.lock.release()

    def pop(self, idx: int) -> ImageWrapper[T]:
        self.lock.acquire()
        image = self.images.pop(idx)
        self.lock.release()
        return image

    def clear(self) -> None:
        self.lock.acquire()
        self.images.clear()
        self.lock.release()

    def __len__(self) -> int:
        self.lock.acquire()
        length = len(self.images)
        self.lock.release()
        return length


class KeypointsMatchProducer(ABC):
    def __init__(self, config={}):
        self.debug = config.get("debug", False)
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)

        self.visualize_callbacks = []
        self.last_data = [None, None]
        self.buffer = Buffer()
        self.lock = Lock()
        self.idx = 0

    def _get_buffer(self):
        return self.buffer

    def visualize(self, image):
        for callback in self.visualize_callbacks:
            callback(image)

    @abstractmethod
    def preprocess_img(self, img):
        """Convert image into form required by matcher."""
        pass

    def _preprocess(self, idx):
        self.buffer.lock.acquire()
        if isinstance(idx, str):
            relevant = (
                [self.buffer.templates[idx]] if idx in self.buffer.templates else []
            )
        else:
            relevant = [x for x in self.buffer.images if x.id == idx]
        if len(relevant) != 1:
            self.buffer.lock.release()
            raise ValueError("Impossible!")
        else:
            img = copy.deepcopy(relevant[0].img)
            self.buffer.lock.release()

        bounds = relevant[0].lxtyrxby
        if bounds is not None:
            img = img[bounds[1] : bounds[3], bounds[0] : bounds[2]]
        preprocessed = self.preprocess_img(img)

        self.buffer.lock.acquire()
        if isinstance(idx, str) and idx in self.buffer.templates:
            self.buffer.templates[idx].add_results(preprocessed)
        else:
            for x in self.buffer.images:
                if x.id == idx:
                    x.add_results(preprocessed)
        self.buffer.lock.release()

    def get_images(self, name=None) -> Tuple[ImageWrapper, ImageWrapper]:
        """if name is not None, get the template image and the first image in the buffer.
        else, get both images in the buffer.
        """
        self.buffer.lock.acquire()
        image0, image1 = None, None
        if name is None:
            if len(self.buffer.images) > 0:
                image0 = self.buffer.images[0]
            if len(self.buffer.images) > 1:
                image1 = self.buffer.images[1]
        else:
            if name in self.buffer.templates:
                image0 = self.buffer.templates[name]
            if len(self.buffer.images) > 0:
                image1 = self.buffer.images[0]
        self.buffer.lock.release()
        if image0 is not None and image1 is not None:
            image0.thread.join()
            image1.thread.join()
        else:
            raise ValueError("No matching possible!")
        return image0, image1

    def has_template(self, name):
        return name in self.buffer.templates

    def get_template(self, name):
        if self.has_template(name):
            return self.buffer.templates[name]
        return None

    def register_template(self, name, template_img):
        """
        Register a template image / keypoint (for 2 stage matcher) to be used for matching.
        """
        self.buffer.remove_template(name)
        content = ImageWrapper(name, template_img)

        self.buffer.add_template(content)
        # Get keypoints
        t = Thread(target=self._preprocess, args=(name,))
        t.start()
        content.register_thread(t)
        return 0

    def add_image(self, img, lxtyrxby=None):
        if len(self.buffer) >= 2:
            return 1
        self.idx += 1
        content = ImageWrapper(self.idx, img, lxtyrxby)
        self.buffer.append(content)

        # Get keypoints
        t = Thread(target=self._preprocess, args=(self.idx,))
        t.start()
        content.register_thread(t)
        return 0

    def clear_buffer(self) -> int:
        if len(self.buffer) < 0:
            return 1
        self.buffer.clear()
        return 0

    @abstractmethod
    def compute_matches(
        self, num_keypoints=20, template=None
    ) -> Tuple[Keypoints, Keypoints]:
        """
        if template is None, match the first 2 images in buffer
        Else, match the last image in buffer with the template
        Args:
            num_keypoints: number of keypoints to use for matching
            template: template image to match with
        Returns:
            M Keypoints in the first image
            M Keypoints in the second image that matches to keypoints in first image.
        """

    @staticmethod
    def draw_keypoints(
        image1: np.ndarray,
        image2: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        color: tuple = (0, 255, 0),
    ):
        assert len(keypoints1) == len(keypoints2)
        # concatenate the images along their widths.
        w = image1.shape[1] + image2.shape[1]
        h = max(image1.shape[0], image2.shape[0])
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[: image1.shape[0], : image1.shape[1], :] = image1
        image[: image2.shape[0], image1.shape[1] :, :] = image2
        _keypoints2 = copy.deepcopy(keypoints2)
        _keypoints2.keypoints[:, 0] += float(image1.shape[1])

        # draw the keypoint matches.
        for i, keypoint1 in enumerate(keypoints1.keypoints):
            keypoint2 = _keypoints2.keypoints[i]
            image = cv2.line(
                img=image,
                pt1=tuple(keypoint1[:2].astype(int)),
                pt2=tuple(keypoint2[:2].astype(int)),
                thickness=1,
                color=color,
            )
        return image

    def process_image(
        self,
        img=None,
        template: Optional[str] = None,
        debug=False,
        num_keypoints=20,
        lxtyrxby=None,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            img: image to process
            template: template to use for matching. If None, use the previous image.
        Returns:
            M Keypoints in the first image
            M Keypoints in the second image that matches to keypoints in first image.
        Raises:
            Exception if no template is registered.
        """
        if img is not None:
            img = white_balance(img)
            self.add_image(img, lxtyrxby)
        try:
            kp1, kp2 = self.compute_matches(num_keypoints, template=template)
        except Exception as e:
            if logger:
                logger.error(e)
            return None, None
        if debug:
            img1, img2 = self.get_images(template)
            img = plot_matches(
                img1.img, img2.img, kp1.keypoints, kp2.keypoints, kp1.scores
            )
            self.visualize(img)
        self.buffer.lock.acquire()
        self.buffer.images.pop(0)
        self.buffer.lock.release()
        return kp1, kp2


def get_keypoints_match_producer(
    extractor=None,
    matcher="superglue",
    extractor_config={},
    matcher_config={"debug": False},
):
    valid_combinations = [
        # SuperPoint detectors
        ("superpoint", "superglue"),
        ("superpoint", "lightglue"),
        ("superpoint", "bf"),
        ("superpoint", "flann"),
        # DISK detectors
        ("disk", "lightglue"),
        # DALF detectors
        ("dalf", "bf"),
        ("dalf", "flann"),
        # ALIKE detectors
        ("alike", "bf"),
        ("alike", "flann"),
        # ORB detectors
        ("orb", "bf"),
        ("orb", "flann"),
        # SIFT detectors
        ("sift", "bf"),
        ("sift", "flann"),
        # FAST detectors
        ("fast", "bf"),
        ("fast", "flann"),
        # KeyAffHard detectors
        ("keyaffhard", "bf"),
        ("keyaffhard", "flann"),
        # Detector-free matchers
        (None, "loftr"),
        (None, "coarse_loftr"),
        (None, "loftr_ts"),
        (None, "dkm"),
    ]
    if not (extractor, matcher) in valid_combinations:
        raise ValueError(
            f"Invalid combination of extractor and matcher: {extractor}, {matcher}"
        )

    # Feature extractors:

    def get_superpoint(config):
        from feature_matcher.keypoint_producer import SuperPointKeypointProducer

        return SuperPointKeypointProducer(config)
    
    def get_disk(config): 
        from feature_matcher.keypoint_producer import DISKKeypointProducer

        return DISKKeypointProducer(config)

    def get_orb(config):
        from feature_matcher.keypoint_producer import OrbKeypointProducer

        return OrbKeypointProducer(config)

    def get_sift(config):
        from feature_matcher.keypoint_producer import SiftKeypointProducer

        return SiftKeypointProducer(config)

    def get_fast(config):
        from feature_matcher.keypoint_producer import FastKeypointProducer

        return FastKeypointProducer(config)

    def get_alike(config):
        from feature_matcher.keypoint_producer import AlikeKeypointProducer

        return AlikeKeypointProducer(config)
    
    def get_dalf(config):
        from feature_matcher.keypoint_producer import DALFKeypointProducer

        return DALFKeypointProducer(config)

    # Feature matchers:

    def get_bf(config):
        from feature_matcher.keypoint_matcher.bf import BFKeypointMatcher

        return BFKeypointMatcher(config)

    def get_flann(config):
        from feature_matcher.keypoint_matcher.flann import FlannKeypointMatcher

        return FlannKeypointMatcher(config)

    def get_superglue(config):
        from feature_matcher.keypoint_matcher.superglue import SuperglueKeypointMatcher

        return SuperglueKeypointMatcher(config)

    def get_lightglue(config):
        from feature_matcher.keypoint_matcher.lightglue import LightglueKeypointMatcher

        return LightglueKeypointMatcher(config)

    # extractor + matchers:

    def get_loftr(config):
        from feature_matcher.loftr_matcher import LoFTRMatchProducer

        return LoFTRMatchProducer(config)

    def get_coarse_loftr(config):
        from feature_matcher.kornia_loftr_matcher import Kornia_LoFTRMatchProducer

        # return Coarse_LoFTRMatchProducer(config)
        return Kornia_LoFTRMatchProducer(config)

    def get_loftr_ts(config):
        from feature_matcher.loftr_torchscript_matcher import (
            LoFTRTorchscriptMatchProducer,
        )

        return LoFTRTorchscriptMatchProducer(config)

    def get_dkm(config):
        from feature_matcher.dkm_matcher import DKMv3MatchProducer

        return DKMv3MatchProducer(config)

    def get_keyaffhard(config):
        from feature_matcher.keypoint_producer import KeyAffHardKeypointProducer

        return KeyAffHardKeypointProducer(config)

    extractors = {
        "superpoint": get_superpoint,
        "disk": get_disk, 
        "orb": get_orb,
        "sift": get_sift,
        "fast": get_fast,
        "alike": get_alike,
        "keyaffhard": get_keyaffhard,
        "dalf": get_dalf
    }
    matchers = {
        "superglue": get_superglue,
        "bf": get_bf,
        "flann": get_flann,
        "lightglue": get_lightglue,
    }
    extractor_matcher = {
        "loftr": get_loftr,
        "coarse_loftr": get_coarse_loftr,
        "loftr_ts": get_loftr_ts,
        "dkm": get_dkm,
    }

    if extractor is not None and extractor in extractors:
        extractor = extractors[extractor](extractor_config)
    if matcher in matchers:
        if matcher == "lightglue" and extractors == "disk":
            matcher = get_lightglue({"weights": "disk"})
        else: 
            matcher = matchers[matcher](matcher_config)

    if extractor is not None:
        from feature_matcher.two_stage_match_producer import TwoStageMatchProducer

        return TwoStageMatchProducer(extractor, matcher)
    else:
        matcher = extractor_matcher[matcher](matcher_config)

    return matcher


if __name__ == "__main__":
    import os
    from pathlib import Path

    # folder_path = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/_auv4_front_cam_image_rect_color"
    # bboxes_file = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/tommygun_gt.csv"
    folder_path = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/Images"
    bboxes_file = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/bootlegger1.csv"

    image_match_producers = {
        # 0.0103288540s
        # "sift": get_keypoints_match_producer("sift", "flann", {"debug": True}, {"debug": True}),
        # # 0.0042234153s
        # "orb": get_keypoints_match_producer("orb", "bf", {"debug": True}, {"debug": True}),
        # # 0.0026472004s
        # "fast": get_keypoints_match_producer("fast", "bf", {"debug": True}, {"debug": True}),
        # # 0.2101211615s
        # "superpoint": get_keypoints_match_producer("superpoint", "superglue", {"debug": True}, {"debug": True}),
        # "superpoint": get_keypoints_match_producer("superpoint", "bf", {"debug": True}, {"debug": True}),
        # "superpoint": get_keypoints_match_producer("superpoint", "flann", {"debug": True}, {"debug": True}),
        # # 0.0883604178s
        # "coarse_loftr": get_keypoints_match_producer(None, "coarse_loftr", {"debug": True}, {"debug": True}),
        # "loftr": get_keypoints_match_producer(None, "loftr", {"debug": True}, {"debug": True}),
        "alike": get_keypoints_match_producer(
            "alike", "bf", {"debug": True}, {"debug": True}
        ),
    }
    for key, image_match_producer in image_match_producers.items():
        image_match_producer.visualize_callbacks.append(
            create_show_image(image_match_producer.__class__.__name__)
        )

        templates_dir = os.path.abspath(
            Path(os.path.realpath(__file__)).parents[2] / "templates"
        )
        templates = {
            "Tommy Gun": os.path.join(templates_dir, "Tommy Gun.jpeg"),
            "Bootlegger": os.path.join(templates_dir, "Bootlegger.jpeg"),
        }
        for key, value in templates.items():
            image_match_producer.register_template(key, cv2.imread(value))

        bboxes = np.loadtxt(bboxes_file, delimiter=",")
        PADDING = 10
        CROP_IMAGES = True
        for i, file in enumerate(os.listdir(folder_path)):
            try:
                x, y, w, h = [int(_) for _ in bboxes[i]]
                img = cv2.imread(f"{folder_path}/{file}")
                lxtyrxby = (
                    max(0, x - PADDING),
                    max(0, y - PADDING),
                    min(img.shape[1], x + w + PADDING),
                    min(img.shape[0], y + h + PADDING),
                )

                if not CROP_IMAGES:
                    lxtyrxby = None

                # matching against bootlegger template
                kp1, kp2 = image_match_producer.process_image(
                    img, "Bootlegger", lxtyrxby=lxtyrxby, debug=True
                )

                # For matching between consecutive images
                # kp1, kp2 = image_match_producer.process_image(img, None, debug=True)

            except Exception as e:
                logging.error(e)
                continue
