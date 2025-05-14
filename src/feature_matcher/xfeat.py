from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from cv2.typing import MatLike
from torch import Tensor

import feature_matcher
from feature_matcher.models.accelerated_features.modules.xfeat import XFeat
from feature_matcher.tools import TemplateSpec

from numpy.typing import ArrayLike

weights = Path(feature_matcher.__path__[0]) / Path(
    "models/accelerated_features/weights/xfeat.pt"
)


@dataclass
class TemplateWithKeypoints:
    """
    Class representing a template with keypoints and descriptors.

    Keypoints and descriptors are obtained from the template image using the XFeat model.
    They are stored to reduce repeated computation during matching.
    They are stored as tensors to eliminate overhead from copying between CPU and GPU.

    Attributes:
        image (torch.Tensor): The image of the template.
        keypoints (list): The keypoints of the template.
        descriptors (torch.Tensor): The descriptors of the template.
        dimensions (tuple): The dimensions of the template.
        offset (tuple): The offset of the template.
    """

    image: MatLike
    dimensions: tuple
    offset: tuple
    keypoints: Tensor
    descriptors: Tensor


class XFeatMatcher:
    """
    Wrapper for XFeat that stores templates with keypoints and descriptors.

    Keypoints and descriptors are obtained from the template image using the XFeat model.
    They are stored to reduce repeated computation during matching.
    They are stored as tensors to eliminate overhead from copying between CPU and GPU.
    """

    def __init__(self):
        """
        Initialize the XFeatMatcher.
        """
        self.model = XFeat(weights=str(weights))
        self.templates_with_keypoints: Dict[str, TemplateWithKeypoints] = {}

    def get_template_with_keypoints(
        self,
        template_image: MatLike,
        template_dims: Tuple,
        template_offset: Tuple,
    ):
        # Get the keypoints of the template image
        template_output = self.model.detectAndCompute(template_image)[0]
        template_kps = template_output["keypoints"]
        template_descs = template_output["descriptors"]

        template = TemplateWithKeypoints(
            image=template_image,
            dimensions=template_dims,
            offset=template_offset,
            keypoints=template_kps,
            descriptors=template_descs,
        )
        return template

    def set_all_templates(self, template_specs: Dict[str, TemplateSpec]):
        for template_name, template_spec in template_specs.items():
            template_image = template_spec.image
            template_dims = template_spec.dimensions
            template_offset = template_spec.offset

            self.templates_with_keypoints[template_name] = (
                self.get_template_with_keypoints(
                    template_image, template_dims, template_offset
                )
            )

    def get_matches(
        self, template_name: str, image: MatLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Get matches between the template and the image.
        This method uses the XFeat model to detect and compute keypoints and descriptors
        for both the template and the image. It then matches the keypoints and descriptors
        using Lighterglue.

        Args:
            template_name (str): Template name
            image (MatLike): Image to match against the template

        Returns:
            Tuple[ArrayLike, ArrayLike]: Matched keypoints from the template and the image.
        """
        template = self.templates_with_keypoints[template_name]
        template_data = {
            "keypoints": template.keypoints,
            "descriptors": template.descriptors,
            "image_size": (template.image.shape[1], template.image.shape[0]),
        }

        image_data = self.model.detectAndCompute(image)[0]
        image_data.update({"image_size": (image.shape[1], image.shape[0])})

        mkpts_0, mkpts_1, _ = self.model.match_lighterglue(template_data, image_data)

        return mkpts_0, mkpts_1

    def get_matches_cossim(
        self, template_name: str, image: MatLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Get matches between the template and the image.
        This method uses the XFeat model to detect and compute keypoints and descriptors
        for both the template and the image. It then matches the keypoints and descriptors
        using cosine similarity.

        Args:
            template_name (str): Template name
            image (MatLike): Image to match against the template

        Returns:
            Tuple[ArrayLike, ArrayLike]: Matched keypoints from the template and the image.
        """
        template = self.templates_with_keypoints[template_name]
        template_kps = template.keypoints
        template_descs = template.descriptors

        outputs = self.model.detectAndCompute(image)[0]
        image_kps = outputs["keypoints"]
        image_descs = outputs["descriptors"]

        idxs0, idxs1 = self.model.match(template_descs, image_descs)
        template_mkps = template_kps[idxs0]
        image_mkps = image_kps[idxs1]

        return template_mkps.cpu().numpy(), image_mkps.cpu().numpy()
