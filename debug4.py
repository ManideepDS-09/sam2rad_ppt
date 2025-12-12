import numpy as np
import cv2 as cv
import pydicom

# ---------------- SETTINGS ----------------
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K11.dcm"
START_F = 1
END_F   = 85

ABS_VAR_THR = 1.1e-05
BOTTOM_FRAC_IGNORE = 0.15

# thresholds tuned from your K04 curves
BAND_THR_PX      = 55      # thick band = bar1-type
MEAN_VAR_THR_B2  = 0.016   # bar2 mean_var
SOBEL_THR_B2     = 17.0    # bar2 sobel
MIN_SEG_LEN      = 3       # min frames to keep a bar segment
EXPAND_MARGIN    = 8       # expand each bar segment by this many frames on each side

# ---------------- LOAD DICOM ----------------
ds = pydicom.dcmread(DICOM_PATH)
arr = ds.pixel_array

if arr.ndim == 2:
    arr = arr[None, ...]
if arr.ndim == 4:
    arr = arr[..., 0]

H, W = arr.shape[1], arr.shape[2]
BOTTOM_Y = int((1.0 - BOTTOM_FRAC_IGNORE) * H)

frames   = list(range(START_F, END_F + 1))
mean_var_arr = []
band_px_arr  = []
sobel_arr    = []

# ---- 1) compute features for all frames ----
for f in frames:
    frame = arr[f - 1].astype(np.float32)

    # normalize
    fmin, fmax = frame.min(), frame.max()
    frame_n = (frame - fmin) / (fmax - fmin + 1e-6)

    # row variance
    row_var = np.var(frame_n, axis=1)
    row_var_smooth = cv.GaussianBlur(row_var.reshape(-1,1), (1,21), 0).ravel()
    mean_v = float(row_var_smooth.mean())

    # low-var rows (absolute)
    low_var_rows = row_var_smooth < ABS_VAR_THR
    bands = []
    i = 0
    while i < H:
        if not low_var_rows[i]:
            i += 1
            continue
        j = i
        while j + 1 < H and low_var_rows[j + 1]:
            j += 1
        bands.append((i, j))
        i = j + 1

    max_valid_band = 0
    for y1, y2 in bands:
        h_band = y2 - y1 + 1
        center_y = (y1 + y2) // 2
        if center_y < BOTTOM_Y:   # ignore floor
            max_valid_band = max(max_valid_band, h_band)

    # Sobel vertical gradient = horizontal edges
    f8 = (frame_n * 255).astype(np.uint8)
    gy = cv.Sobel(f8, cv.CV_32F, 0, 1, ksize=3)
    sobel_dy = float(np.mean(np.abs(gy)))

    mean_var_arr.append(mean_v)
    band_px_arr.append(max_valid_band)
    sobel_arr.append(sobel_dy)

# ---- 2) apply Rule A + Rule B to get BAR candidates ----
n = len(frames)
bar_cand = np.zeros(n, dtype=bool)

for i in range(n):
    mv = mean_var_arr[i]
    bp = band_px_arr[i]
    sd = sobel_arr[i]

    ruleA = (bp >= BAND_THR_PX)  # bar1 type
    ruleB = (mv >= MEAN_VAR_THR_B2 and sd >= SOBEL_THR_B2)  # strong bar2
    ruleC = (mv >= 0.014 and sd >= 14.0)  # transitional bar2 (new)
    ruleD = (mv < 0.012 and sd < 13)

    bar = (ruleA or ruleB or ruleC)
    
    if ruleD:
        bar = False
    bar_cand[i] = bar

# ---- 3) temporal cleanup: drop very short segments ----
def runs_from_mask(mask):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs

runs = runs_from_mask(bar_cand)
kept_runs = []
for a, b in runs:
    length = b - a + 1
    if length >= MIN_SEG_LEN:
        kept_runs.append((a, b))

# ---- 4) expand segments by margin ----
final_mask = np.zeros(n, dtype=bool)
for a, b in kept_runs:
    aa = max(0, a - EXPAND_MARGIN)
    bb = min(n - 1, b + EXPAND_MARGIN)
    final_mask[aa:bb+1] = True

# ---- 5) convert final_mask to intervals ----
final_runs = runs_from_mask(final_mask)

print("\nFrame | Label | mean_var | band_px | sobel_dy")
print("------------------------------------------------")
for i, f in enumerate(frames):
    lab = "BAR" if final_mask[i] else "BONE"
    print(f"{f:03d}   | {lab:4s} | {mean_var_arr[i]:.6f} | {band_px_arr[i]:7d} | {sobel_arr[i]:7.2f}")

print("\n Predicted Intervals (after smoothing):")
for (a, b) in final_runs:
    s = frames[a]
    e = frames[b]
    print(f"BAR : {s} → {e}")