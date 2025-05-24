import logging
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from cv2.typing import MatLike


@dataclass
class TemplateSpec:
    image: MatLike
    dimensions: tuple  # (width, height)
    offset: tuple


def get_region_template_specs(
    template_image: MatLike,
    template_dims: Tuple,
    region: List,
):
    x1, y1, x2, y2 = region
    x1, x2 = (
        int(x1 * template_image.shape[1]),
        int(x2 * template_image.shape[1]),
    )
    y1, y2 = (
        int(y1 * template_image.shape[0]),
        int(y2 * template_image.shape[0]),
    )

    template_image_width = x2 - x1
    template_image_height = y2 - y1
    region_image = template_image[y1:y2, x1:x2]

    region_px_offset = (
        (x1 + x2) / 2 - template_image.shape[1] / 2,
        (y1 + y2) / 2 - template_image.shape[0] / 2,
    )

    template_width, template_height = template_dims

    # Offset in metres
    region_offset = (
        region_px_offset[0] / template_image.shape[1] * template_width,
        region_px_offset[1] / template_image.shape[0] * template_height,
    )

    # Region dimensions in metres
    region_width, region_height = (
        template_image_width / template_image.shape[1] * template_width,
        template_image_height / template_image.shape[0] * template_height,
    )
    return region_image, (region_width, region_height), region_offset


def get_template_specs(
    templates_dir: Path, template_files: Dict[str, Dict]
) -> Dict[str, TemplateSpec]:
    """
    Get the template specifications from the template files.

    Args:
        templates_dir (Path): Path to the directory containing the template files.
        template_files (Dict[str, Dict]): Dictionary containing the template files and their specifications.

    Returns:
        Dict[str, TemplateSpec]: A dictionary with the template names as keys and their specifications as values.
    """
    # Create a dictionary to store the template specifications
    template_specs = {}

    for template_name in template_files.keys():
        # Templates starting with "_" are ignored
        if template_name.startswith("_"):
            continue

        template_file_path = templates_dir / template_name
        template_image = cv2.imread(str(template_file_path))

        template_data = template_files[template_name]
        if isinstance(template_data, list):
            # Template comes with a list of dimensions
            template_dims = tuple(template_data)
        else:
            # Template comes with a dictionary of dimensions and regions of interest
            template_dims = tuple(template_data["dimensions"])

            for region_name, region in template_data["regions"].items():
                region_name = f"{template_name}_{region_name}"
                region_image, region_dims, region_offset = get_region_template_specs(
                    template_image, template_dims, region
                )
                template_specs[region_name] = TemplateSpec(
                    image=region_image,
                    dimensions=region_dims,
                    offset=region_offset,
                )

        template_specs[template_name] = TemplateSpec(
            image=template_image,
            dimensions=template_dims,
            offset=(0, 0),
        )

    return template_specs


def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_show_image(window_name="image"):
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)

    def show_image(img):
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    return show_image


def create_save_image(filename="matched_image.png"):
    def save_image(img):
        cv2.imwrite(filename, img)

    return save_image


def time_func(name=None):
    def time_func_decorator(func):
        nonlocal name
        # This function shows the execution time of
        # the function object passed
        if name is None:
            name = func.__name__
        total_time = 0
        count = 0

        def wrap_func(*args, **kwargs):
            nonlocal total_time, count, name, func
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            total_time += t2 - t1
            count += 1
            if kwargs.get("debug", False):
                if count % 10 == 0:
                    logging.info(
                        f"Function {name!r} executed in {(total_time / count):.10f}s"
                    )
            return result

        return wrap_func

    return time_func_decorator


# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert 0 <= smin <= 1 and 0 <= smax <= 1

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image


# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None and len(scores) > 0:
        smin, smax = scores.min(), scores.max()
        assert 0 <= smin <= 1 and 0 <= smax <= 1

        unique_values = np.unique(scores)
        if len(unique_values) == 2:
            color = np.zeros((kpts0.shape[0], 3), dtype=int)
            color[scores > 0.5, 1] = 255
            color[scores <= 0.5, 2] = 255
        else:
            color = cm.gist_rainbow(scores * 0.4)
            color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(
                out,
                (x0, y0),
                (x1 + W0, y1),
                color=c,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(
                out,
                (x0, y0),
                (x1, y1 + H0),
                color=c,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out


def get_image_match_empty_canvas(template: MatLike, img: MatLike) -> MatLike:
    """
    Concatenates template and input image horizontally, with template
    on the left.

    Args:
        template (MatLike): Template image.
        img (MatLike): Input image.

    Returns:
        MatLike: Template and input image concatenated.
    """
    combined = cv2.drawMatches(
        template, [], img, [], [], None, matchColor=(0, 255, 0), flags=2
    )
    return combined


# based on: https://github.com/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb
def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    try:
        warped_corners = cv2.perspectiveTransform(corners_img1, H)
    except cv2.error as e:
        # TODO: Handle cv2.error: OpenCV(4.11.0) ... error: (-215:Assertion failed) scn + 1 == m.cols in function 'perspectiveTransform'
        print(f"Error in perspectiveTransform: {e}")
        return get_image_match_empty_canvas(img1, img2)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(
            img2_with_corners, start_point, end_point, (0, 255, 0), 4
        )  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(
        img1,
        keypoints1,
        img2_with_corners,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=2,
    )

    return img_matches
