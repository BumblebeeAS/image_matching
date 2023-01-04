from typing import Tuple
import numpy as np

class Keypoints(np.ndarray):

    def __new__(cls, image_size: Tuple[int, int], keypoints: np.ndarray, descriptors: np.ndarray = None, scores: np.ndarray = None):
        '''
        params:
            image_size: (width, height)
            keypoints: np.ndarray (x, y) * N
            descriptors: np.ndarray (descriptor_size) * N
            scores: np.ndarray (score) * N 
        Represented as a single numpy array with the following columns:
        keypoints (x, y): 0-1, scores: 2, descriptors: rest
        '''
        obj = np.hstack([keypoints, scores.squeeze()[:, None], descriptors]).astype(np.float32).view(cls)
        obj.image_size = image_size
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.image_size = getattr(obj, 'image_size', None)
    @property
    def descriptors(self):
        return self[:, 3:]
    @property
    def keypoints(self):
        return self[:, :2]
    @property
    def scores(self):
        return self[:, 2]
