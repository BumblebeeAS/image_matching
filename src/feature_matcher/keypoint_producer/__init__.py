from feature_matcher.keypoint_producer.fast import FastKeypointProducer
from feature_matcher.keypoint_producer.orb import OrbKeypointProducer

try:
    from feature_matcher.keypoint_producer.sift import SiftKeypointProducer
except ImportError as e:
    print(e)
try:
    from feature_matcher.keypoint_producer.superpoint import SuperPointKeypointProducer
except ImportError as e:
    print(e)
try:
    from feature_matcher.keypoint_producer.alike import AlikeKeypointProducer
except ImportError as e:
    print(e)
try:
    from feature_matcher.keypoint_producer.KeyAffHard import KeyAffHardKeypointProducer
except ImportError as e:
    print(e)

try:
    from feature_matcher.keypoint_producer.disk import DISKKeypointProducer
except ImportError as e:
    print(e)

try:
    from feature_matcher.keypoint_producer.dalf import DALFKeypointProducer
except ImportError as e:
    print(e)
