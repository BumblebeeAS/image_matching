import unittest
import os
import sys
import numpy as np

from rospkg import RosPack
sys.path.append(os.path.abspath(RosPack().get_path("image_matching")))  # noqa E402
from scripts.matcher_service import MatcherNode  # noqa E402
TEMPLATE_DIR = os.path.abspath(RosPack().get_path("image_matching") + "/templates/")


class TestMatcherNode(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.matcher_node: MatcherNode = MatcherNode(True,
                                                     detector_config={
                                                         "cuda": False,
                                                     },
                                                     matcher_config={"cuda": False},)

    def test_pipeline(self):
        import cv2
        img1 = cv2.imread(TEMPLATE_DIR + "/temp/G-Man_real.jpeg")

        image_center = tuple(np.array(img1.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 0.2, 1.0)
        img2 = cv2.warpAffine(img1, rot_mat, img1.shape[1::-1], flags=cv2.INTER_LINEAR)

        resp1 = self.matcher_node.image_match_producer.add_image(img1)
        resp2 = self.matcher_node.image_match_producer.add_image(img2)
        self.assertEqual(resp1, 0)
        self.assertEqual(resp2, 0)
        self.assertEqual(len(self.matcher_node.image_match_producer.buffer.images), 2)

        matches = self.matcher_node.image_match_producer.compute_matches(500, None)
        
        from feature_matcher.tools import plot_matches

        img = plot_matches(
            img1,
            img2,
            matches[0][0:200].keypoints,
            matches[1][0:200].keypoints,
            matches[0][0:200].scores,
            layout="lr",
        )

        cv2.imshow("MATCHES", img)
        cv2.waitKey(0)
        self.matcher_node.clear_buffer()

    def test_pipeline2(self):
        import cv2
        # img1 = cv2.imread(TEMPLATE_DIR + "G-Man.jpeg")
        # resp1 = self.matcher_node.image_match_producer.add_img(img1)
        # self.assertEqual(resp1.result, 0)

        img2 = cv2.imread(TEMPLATE_DIR + "/temp/G-Man_real.jpeg")
        resp2 = self.matcher_node.image_match_producer.add_image(img2)
        self.assertEqual(resp2, 0)

        self.assertEqual(len(self.matcher_node.image_match_producer.buffer.images), 1)

        matches = self.matcher_node.image_match_producer.compute_matches(500, "G-Man")
        
        from feature_matcher.tools import plot_matches

        img = plot_matches(
            self.matcher_node.image_match_producer.get_template("G-Man").img,
            img2,
            matches[0][0:200].keypoints,
            matches[1][0:200].keypoints,
            matches[0][0:200].scores,
            layout="lr",
        )

        cv2.imshow("MATCHES", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
