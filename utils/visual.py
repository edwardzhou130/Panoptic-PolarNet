#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def flow_to_img(flow, normalize=True):
    """Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.
    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
    Returns:
        img: viewable representation of the dense optical flow in RGB format
    Ref:
        https://github.com/philferriere/tfoptflow/blob/33e8a701e34c8ce061f17297d40619afbd459ade/tfoptflow/optflow.py
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img