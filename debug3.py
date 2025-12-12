import os
import math
import numpy as np
import cv2 as cv
import pydicom

# ---------------- HARD CODED DEBUG VALUES ----------------
DICOM_PATH  = "/home/ds/Desktop/Hand_dicom/K04.dcm"   # change if needed
FRAME_INDEX = 114   # 1-based index
OUT_PATH    = "debug_hough.png"

# ---------------- LOAD DICOM ----------------
ds = pydicom.dcmread(DICOM_PATH)
arr = ds.pixel_array

if arr.ndim == 2:
    arr = arr[None, ...]
if arr.ndim == 4:
    arr = arr[..., 0]

n_frames = arr.shape[0]
if FRAME_INDEX < 1 or FRAME_INDEX > n_frames:
    raise ValueError(f"Invalid frame index {FRAME_INDEX}. Valid range is 1–{n_frames}")

frame = arr[FRAME_INDEX - 1]

# ---------------- CONVERT TO UINT8 ----------------
def to_uint8(image_):
    image_ = np.asanyarray(image_)
    if image_.dtype == np.uint8:
        return image_
    imin, imax = np.min(image_), np.max(image_)
    if imax <= imin:
        return np.zeros_like(image_, dtype=np.uint8)
    scaled = (image_ - imin) / (imax - imin)
    return (scaled * 255).astype(np.uint8)

src = to_uint8(frame)

# ---------------- CANNY ----------------
dst = cv.Canny(src, 50, 200, None, 3)

# Convert to BGR for drawing
cdst  = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = cdst.copy()

# ---------------- STANDARD HOUGH (YOUR EXACT CODE) ----------------
lines = cv.HoughLines(dst, 1, np.pi / 180, 150)

std_count = 0
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)
        std_count += 1

# ---------------- PROBABILISTIC HOUGH (YOUR EXACT CODE) ----------------
linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

prob_count = 0
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2, cv.LINE_AA)
        prob_count += 1

# ---------------- SAVE IMAGE ----------------
combined = np.hstack([cdst, cdstP])
cv.imwrite(OUT_PATH, combined)

print("\n=== HOUGH DEBUG (RAW, UNFILTERED) ===")
print(f"DICOM        : {os.path.basename(DICOM_PATH)}")
print(f"Frame index  : {FRAME_INDEX}")
print(f"Image shape  : {src.shape}")
print(f"Std lines    : {std_count}  (RED)")
print(f"Prob lines   : {prob_count} (BLUE)")
print(f"Saved image  : {OUT_PATH}")