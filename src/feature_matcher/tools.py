#!/usr/bin/env python3

from time import time
import cv2
import numpy as np
import torch
import logging

import matplotlib.cm as cm


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
        cv2.waitKey(1)

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

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(
                out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA
            )
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(
                out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA
            )
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out
