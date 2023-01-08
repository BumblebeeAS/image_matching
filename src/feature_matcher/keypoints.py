from typing import Optional, Tuple
import numpy as np

class Keypoints(np.ndarray):

    def __new__(cls, image_size: Tuple[int, int], keypoints: np.ndarray, descriptors: Optional[np.ndarray] = None, scores: np.ndarray = None):
        '''
        params:
            image_size: (width, height)
            keypoints: np.ndarray (x, y) * N
            descriptors: np.ndarray (descriptor_size) * N
            scores: np.ndarray (score) * N 
        Represented as a single numpy array with the following columns:
        keypoints (x, y): 0-1, scores: 2, descriptors: rest
        '''
        to_be_stacked = [keypoints, scores.squeeze()[:, None]]
        if descriptors is not None:
            to_be_stacked.append(descriptors)

        obj = np.hstack(to_be_stacked).astype(np.float32).view(cls)
        obj.image_size = image_size
        obj.has_descriptors = descriptors is not None
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.image_size = getattr(obj, 'image_size', None)
        self.has_descriptors = getattr(obj, "has_descriptors", None)
    @property
    def descriptors(self):
        if not self.has_descriptors:
            return None
        return self[:, 3:]
    @property
    def keypoints(self):
        return self[:, :2]
    @property
    def scores(self):
        return self[:, 2]
