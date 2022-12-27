from typing import NamedTuple, Tuple
import numpy as np

class Keypoints(NamedTuple):
    image_size: Tuple[int, int] # width, height
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray

    def __getitem__(self, key):
        return Keypoints(self.image_size,
                         self.keypoints[key],
                         None if self.descriptors is None else self.descriptors[key],
                         None if self.descriptors is None else self.scores[key])
    def __len__(self):
        return len(self.keypoints)
