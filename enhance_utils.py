# enhance_utils.py
#
# Shared utilities for:
#  - converting frames to uint8
#  - enhancing ultrasound frames for better bone visibility
#  - basic "blank / flat" and "has content" checks
#
# These functions mirror the logic you already use in filtering1.py,
# just centralized so run_unet_filter_intervals.py, run_sam_on_kept.py,
# OR_auto_calculator.py, and window.py can all import the same behavior.

import numpy as np
import cv2 as cv

## --- Enhancement defaults (same spirit as filtering1.py) ---
CLAHE_CLIP       = 2.0
CLAHE_TILE_GRIDS = (8, 8)
GAUSS_SIGMA_PX   = 1
MEDIAN_KSIZE     = 3

# --- Blank frame thresholds (tune per dataset) ---
BLANK_STD_THR   = 0.8   # stddev in uint8
BLANK_RANGE_THR = 3     # p99-p1 range in uint8
BLANK_GRAD_THR  = 1.5   # mean Sobel |grad| (uint8 scale)

# --- Central ROI params for "has content?" ---
MID_ROI_TOP = 0.35
MID_ROI_BOT = 0.65
MID_KEEP_VERT_ABS_THR = 4.0   # uint8 grad units
MID_KEEP_VAR_ABS_THR  = 12.0  # intensity variance in ROI

def to_uint8(image_):
    """
    Convert any numeric image array to uint8 [0..255], robust to constant images.
    """
    image_ = np.asanyarray(image_)
    if image_.dtype == np.uint8:
        return image_

    imin = float(np.min(image_))
    imax = float(np.max(image_))
    if imax <= imin:
        return np.zeros_like(image_, dtype=np.uint8)

    scaled = (image_ - imin) / (imax - imin)
    return (scaled * 255.0).astype(np.uint8)


def enhance_frame_only(frame,
                       clip_limit=CLAHE_CLIP,
                       tile_grid_size=CLAHE_TILE_GRIDS,
                       sigma=GAUSS_SIGMA_PX):
    """
    Median -> normalize -> CLAHE -> slight Gaussian blur, all in uint8.
    This is the same style of enhancement you were using in filtering1.py.
    """
    f = frame
    if f.dtype != np.uint8:
        f = to_uint8(f)

    # median to knock small speckle
    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        f = cv.medianBlur(f, MEDIAN_KSIZE)

    # stretch contrast to [0,255]
    f = cv.normalize(f, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # CLAHE (local contrast)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    f = clahe.apply(f)

    # small Gaussian blur to smooth noise
    if sigma and sigma > 0:
        k = max(1, int(2 * round(3 * sigma) + 1))  # odd kernel
        f = cv.GaussianBlur(f, (k, k), sigmaX=sigma, sigmaY=sigma)

    return f


def is_blank_or_flat(frame):
    """
    Decide if a frame is basically empty / flat ultrasound (no useful structure).

    Uses:
      - central crop (10% border trimmed)
      - intensity std
      - intensity range (p1..p99)
      - Sobel gradient magnitude
    """
    f8 = to_uint8(frame)

    h, w = f8.shape[:2]
    y0, y1 = int(0.10 * h), int(0.90 * h)
    x0, x1 = int(0.10 * w), int(0.90 * w)
    roi = f8[y0:y1, x0:x1]

    std  = float(roi.std())
    rng  = float(np.percentile(roi, 99) - np.percentile(roi, 1))

    gx   = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
    gy   = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(np.hypot(gx, gy)))

    # conservative: require all three to be tiny
    return (std < BLANK_STD_THR) and (rng < BLANK_RANGE_THR) and (grad < BLANK_GRAD_THR)


def has_central_content(frame):
    """
    Quick test: does the central band of the image have enough structure?

    Uses:
      - vertical edge energy (Sobel x)
      - intensity variance
    """
    f8 = to_uint8(frame)
    h, w = f8.shape[:2]
    y0, y1 = int(MID_ROI_TOP * h), int(MID_ROI_BOT * h)
    roi = f8[y0:y1, :]

    gx = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
    vert_edges = float(np.mean(np.abs(gx)))
    var_int    = float(roi.var())

    return (vert_edges >= MID_KEEP_VERT_ABS_THR) or (var_int >= MID_KEEP_VAR_ABS_THR)