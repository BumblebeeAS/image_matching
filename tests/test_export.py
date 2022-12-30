import os
import sys

import os
import sys

from rospkg import RosPack  # noqa E402
sys.path.append(os.path.abspath(RosPack().get_path("image_matching") + "/src/feature_matcher"))  # noqa E402

from feature_matcher.models.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from feature_matcher.models.SuperGluePretrainedNetwork.models.superglue import SuperGlue

import torch
import torch.onnx

file_path = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(
    file_path, "..", "models", "SuperGluePretrainedNetwork", "models", "weights"
)

superpoint_config = {
    "descriptor_dim": 256,
    "nms_radius": 4,
    "keypoint_threshold": 0.005,
    "max_keypoints": -1,
    "remove_borders": 4,
    "path": os.path.join(
        file_path,
        "..",
        "models/SuperGluePretrainedNetworks/models/weights/superpoint_v1.pth",
    ),
    "cuda": True,
}
ts_superpoint = SuperPoint(superpoint_config).eval().cuda()

superglue_config = {
    "descriptor_dim": 256,
    "weights": "outdoor",
    "keypoint_encoder": [32, 64, 128, 256],
    "GNN_layers": ["self", "cross"] * 9,
    "sinkhorn_iterations": 100,
    "match_threshold": 0.2,
    "cuda": True,
}
ts_superglue = SuperGlue(superglue_config).eval().cuda()
print("Loaded")

torch.jit.save(ts_superpoint, os.path.join(weight_path, "superpoint_v1.zip"))
torch.jit.save(
    ts_superglue,
    os.path.join(weight_path, f"superglue_{superglue_config['weights']}.zip"),
)
