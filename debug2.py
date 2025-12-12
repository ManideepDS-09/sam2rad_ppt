# save_debug_frame.py

import cv2 as cv
import numpy as np
import pydicom
from skimage.morphology import skeletonize

# ---------------- HARD-CODED INPUT ----------------
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/H007.dcm"
FRAME_NO   = 216
OUT_PATH   = "debug.png"
# --------------------------------------------------


# ---------- EXACTLY FROM YOUR FILTERING SCRIPT ----------
CLAHE_CLIP       = 2.0
CLAHE_TILE_GRIDS = (8, 8)
GAUSS_SIGMA_PX   = 1
MEDIAN_KSIZE     = 3

def to_uint8(image_):
    image_ = np.asanyarray(image_)
    if image_.dtype == np.uint8:
        return image_
    imin, imax = np.min(image_), np.max(image_)
    if imax <= imin:
        return np.zeros_like(image_, dtype=np.uint8)
    scaled = (image_ - imin) / (imax - imin)
    return (scaled * 255).astype(np.uint8)

def enhance_frame_only(frame, clip_limit=CLAHE_CLIP,
                       tile_grid_size=CLAHE_TILE_GRIDS,
                       sigma=GAUSS_SIGMA_PX):

    if frame.dtype != np.uint8:
        frame = to_uint8(frame)

    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        frame = cv.medianBlur(frame, MEDIAN_KSIZE)

    frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    clahe = cv.createCLAHE(clipLimit=clip_limit,
                           tileGridSize=tile_grid_size)
    frame = clahe.apply(frame)

    if sigma and sigma > 0:
        k = max(1, int(2 * round(3 * sigma) + 1))
        frame = cv.GaussianBlur(frame, (k, k),
                                sigmaX=sigma, sigmaY=sigma)
    return frame
# -----------------------------------------------------------


# ================= AXIS TICKS ONLY (NO TEXT) =================
def draw_axes(img, tick_step=50):
    vis = img.copy()
    h, w = vis.shape[:2]

    # X-axis ticks (bottom)
    for x in range(0, w, tick_step):
        cv.line(vis, (x, h-10), (x, h), (0, 255, 0), 2)

    # Y-axis ticks (left)
    for y in range(0, h, tick_step):
        cv.line(vis, (0, y), (10, y), (0, 255, 0), 2)

    # Outer border
    cv.rectangle(vis, (0, 0), (w-1, h-1), (0, 255, 0), 1)

    return vis


def draw_roi_bounds(img, top_end, bottom_sta):
    vis = img.copy()
    h, w = vis.shape[:2]

    # ROI cut lines
    cv.line(vis, (0, top_end),     (w, top_end),     (255, 0, 0), 2)
    cv.line(vis, (0, bottom_sta), (w, bottom_sta),  (255, 0, 0), 2)

    return vis
# ============================================================


def main():
    ds = pydicom.dcmread(DICOM_PATH)
    arr = ds.pixel_array

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] > 1:
        arr = arr[..., 0]

    n_frames = arr.shape[0]

    if FRAME_NO < 1 or FRAME_NO > n_frames:
        print(f"[ERROR] Frame must be in range 1–{n_frames}")
        return

    raw = arr[FRAME_NO - 1]
    raw_u8 = to_uint8(raw)

    # === ENHANCED IMAGE ===
    enhanced = enhance_frame_only(raw)

    # === BINARIZATION ===
    _, binary = cv.threshold(
        enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    binary_bool = binary > 0

    # === SKELETONIZATION ===
    skeleton_std = skeletonize(binary_bool)
    skeleton_lee = skeletonize(binary_bool, method='lee')

    skel_std_u8 = (skeleton_std * 255).astype(np.uint8)
    skel_lee_u8 = (skeleton_lee * 255).astype(np.uint8)

    # === RGB CONVERSION ===
    raw_rgb      = cv.cvtColor(raw_u8, cv.COLOR_GRAY2RGB)
    enh_rgb      = cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)
    skel_std_rgb = cv.cvtColor(skel_std_u8, cv.COLOR_GRAY2RGB)
    skel_lee_rgb = cv.cvtColor(skel_lee_u8, cv.COLOR_GRAY2RGB)

    # ================= ROI DEFINITION =================
    h, w = raw_rgb.shape[:2]
    top_end    = int(0.30 * h)
    bottom_sta = int(0.90 * h)

    # Draw axes + ROI bounds on FULL image (for reference)
    raw_full  = draw_axes(draw_roi_bounds(raw_rgb, top_end, bottom_sta))
    enh_full  = draw_axes(draw_roi_bounds(enh_rgb, top_end, bottom_sta))
    skel_full = draw_axes(draw_roi_bounds(skel_std_rgb, top_end, bottom_sta))
    lee_full  = draw_axes(draw_roi_bounds(skel_lee_rgb, top_end, bottom_sta))

    # ================= ROI CROP (✅ FIXED HERE) =================
    def apply_roi(img):
        # KEEP ONLY the middle region (blue band)
        return img[top_end:bottom_sta, :]

    raw_roi  = draw_axes(apply_roi(raw_rgb))
    enh_roi  = draw_axes(apply_roi(enh_rgb))
    skel_roi = draw_axes(apply_roi(skel_std_rgb))
    lee_roi  = draw_axes(apply_roi(skel_lee_rgb))

    # === STACK FINAL DEBUG IMAGE ===
    debug_img = np.vstack([
        np.hstack([raw_full, enh_full, skel_full, lee_full]),
        np.hstack([raw_roi,  enh_roi,  skel_roi,  lee_roi ])
    ])

    cv.imwrite(OUT_PATH, debug_img)

    print("[OK] debug.png saved with AXES + ROI")
    print(" Frame     :", FRAME_NO, "/", n_frames)
    print(" Full Size :", raw_rgb.shape)
    print(" ROI Height:", raw_roi.shape[0])
    print(" Final Img :", debug_img.shape)
    print(" Output    :", OUT_PATH)


if __name__ == "__main__":
    main()