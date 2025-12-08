# filtering1.py  (with two fused-window tables printed)

import os
import math
import numpy as np
import cv2 as cv
import pydicom

# ---------------- Config toggles (tune freely) ----------------
ENHANCE_FOR_DETECTION      = True   # use enhanced frames during Hough detection
REFINE_WINDOWS_WITH_SOBEL  = False   # trim/split fused windows with Sobel+brightness rule

# Enhancement params
CLAHE_CLIP       = 2.0
CLAHE_TILE_GRIDS = (8, 8)
GAUSS_SIGMA_PX   = 1   # ~1-1.5 px is good after CLAHE
MEDIAN_KSIZE     = 3

# Sobel refinement params
TOP_FRAC            = 0.18
BOTTOM_FRAC         = 0.15
PERCENTILE_THRESH   = 95
ROW_COUNT_THRESH    = 3
GRAD_PERC_THRESH    = 85
MIN_BAD_RUN_SPLIT   = 2
DROP_IF_BAD_FRAC_GE = 0.85
MIN_LEN_KEEP        = 5

# New safety rails for Sobel refinement
ENHANCE_FOR_SOBEL = True           # start by NOT enhancing for Sobel test
NEVER_DROP_IF_BONE_OVERLAP_GE = 0.5 # if ≥50% of window overlaps the bone-run, do not drop
NEVER_DROP_IF_D_AT_LEAST = 9         # if bone–bar center discrepancy is large, bias to keep
ALWAYS_DROP_IF_BLANK_FRAC_GE  = 0.9  # if ≥80% of window is blank (zero), drop it
EXCLUDE_BLANK_FROM_BONE = True     # exclude blank frames from bone-runs

# --- Blank frame thresholds (tune per dataset) ---
BLANK_STD_THR   = 0.8   # stddev in uint8
BLANK_RANGE_THR = 3   # p99-p1 range in uint8
BLANK_GRAD_THR  = 1.5   # mean Sobel |grad| (uint8 scale)

# --- Gap smoothing in 'bad' mask (helps tiny hiccups) ---
IGNORE_BAD_GAPS_LE = 2   # treat bad runs ≤2 frames as good

# Stitch params (unchanged)
STITCH_MAX_GAP = 6

# ---------------- helper function to detect Blank frame -------------------
def is_blank_or_flat(frame_u8):
    f = to_uint8(frame_u8)

    # crop away borders (vignettes/black edges skew stats)
    h, w = f.shape[:2]
    y0, y1 = int(0.10 * h), int(0.90 * h)
    x0, x1 = int(0.10 * w), int(0.90 * w)
    roi = f[y0:y1, x0:x1]

    # stats in the central ROI
    std  = float(roi.std())
    rng  = float(np.percentile(roi, 99) - np.percentile(roi, 1))
    gx   = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
    gy   = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(np.hypot(gx, gy)))

    # be CONSERVATIVE: require ALL THREE to be tiny
    return (std < BLANK_STD_THR) and (rng < BLANK_RANGE_THR) and (grad < BLANK_GRAD_THR)

# central ROI evidence of structure (bones/texture)
MID_ROI_TOP = 0.35
MID_ROI_BOT = 0.65
MID_KEEP_VERT_ABS_THR = 4.0   # uint8 grad units; start conservative
MID_KEEP_VAR_ABS_THR  = 12.0  # intensity variance in ROI

def has_central_content(f8):
    h, w = f8.shape[:2]
    y0, y1 = int(MID_ROI_TOP*h), int(MID_ROI_BOT*h)
    roi = f8[y0:y1, :]
    # vertical edges are informative for bones/texture
    gx = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
    vert_edges = float(np.mean(np.abs(gx)))   # vertical-edge strength
    var_int    = float(roi.var())
    return (vert_edges >= MID_KEEP_VERT_ABS_THR) or (var_int >= MID_KEEP_VAR_ABS_THR)

# ---------------- Utilities -------------------
def to_uint8(image_):
    image_ = np.asanyarray(image_)
    if image_.dtype == np.uint8:
        return image_
    imin, imax = np.min(image_), np.max(image_)
    if imax <= imin:
        return np.zeros_like(image_, dtype=np.uint8)
    scaled = (image_ - imin) / (imax - imin)
    return (scaled * 255).astype(np.uint8)

def angle_deg_from_segment(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

def is_angle_bar(angle_deg, tol_deg=8.0):
    return min(angle_deg, 180 - angle_deg) <= tol_deg

def consecutive_intervals(indices, merge_gap=0):
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for x in indices[1:]:
        if x <= prev + 1 + merge_gap:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs

def format_intervals(runs):
    return ", ".join([str(a) if a == b else f"{a}-{b}" for a, b in runs]) if runs else "None"

# ---------------- Enhancement + Sobel tests -------------------
def enhance_frame_only(frame, clip_limit=CLAHE_CLIP, tile_grid_size=CLAHE_TILE_GRIDS, sigma=GAUSS_SIGMA_PX):
    """Median -> normalize -> CLAHE -> slight Gaussian blur, all uint8."""
    if frame.dtype != np.uint8:
        frame = to_uint8(frame)
    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        frame = cv.medianBlur(frame, MEDIAN_KSIZE)
    frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    frame = clahe.apply(frame)
    if sigma and sigma > 0:
        k = max(1, int(2 * round(3 * sigma) + 1))  # odd kernel ~ 6*sigma
        frame = cv.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)
    return frame

def is_horizontal_intensity_bad(enhanced_frame,
                                top_frac=TOP_FRAC, bottom_frac=BOTTOM_FRAC,
                                percentile_thresh=PERCENTILE_THRESH,
                                row_count_thresh=ROW_COUNT_THRESH,
                                grad_perc_thresh=GRAD_PERC_THRESH):
    """Bright rows near top/bottom + weak vertical gradients within those rows."""
    f = enhanced_frame
    h, w = f.shape[:2]
    top_rows    = max(1, int(top_frac * h))
    bottom_rows = max(1, int(bottom_frac * h))

    row_means = np.mean(f, axis=1)
    row_means = cv.blur(row_means.reshape(-1, 1), (7, 1)).ravel()

    intensity_threshold = np.percentile(row_means, percentile_thresh)
    top_bright_rows    = int(np.sum(row_means[:top_rows] > intensity_threshold))
    bottom_bright_rows = int(np.sum(row_means[-bottom_rows:] > intensity_threshold))

    sobel_y = cv.Sobel(f, cv.CV_32F, dx=0, dy=1, ksize=3)
    vert_grad_profile = np.mean(np.abs(sobel_y), axis=1)
    vert_grad_profile = cv.blur(vert_grad_profile.reshape(-1, 1), (7, 1)).ravel()

    bright_mask = row_means > intensity_threshold
    mean_bright_grad = float(vert_grad_profile[bright_mask].mean()) if bright_mask.any() else 0.0
    grad_thresh = np.percentile(vert_grad_profile, grad_perc_thresh)

    too_many_bright = (top_bright_rows > row_count_thresh) or (bottom_bright_rows > row_count_thresh)
    is_bad = bool(too_many_bright and (mean_bright_grad < grad_thresh))
    return is_bad

# ---------------- Hough detection -----------
def detect_horizontal_bar_in_frame(img_u8,
                                   canny_low=50, canny_high=150, canny_aperture=3,
                                   hough_std_rho=1, hough_std_theta=np.pi/180, hough_std_threshold=100,
                                   houghp_rho=1, houghp_theta=np.pi/180, houghp_threshold=20,
                                   min_line_length=50, min_line_len_ratio=0.75, max_line_gap=5,
                                   angle_tol_deg=2):
    """
    Detects near-horizontal bars using both Standard and Probabilistic Hough.
    Returns in order expected by the caller:
      has_std_bar, has_prob_bar, n_std, n_prob, lines_prob
    """
    img_u8 = cv.GaussianBlur(img_u8, (3, 3), 0)
    h, w = img_u8.shape[:2]
    top_limit = int(0.25 * h)
    bottom_limit = int(0.75 * h)
    min_line_horizontal = max(1, int(w * min_line_len_ratio))

    edges = cv.Canny(img_u8, canny_low, canny_high, apertureSize=canny_aperture)

    # Standard Hough
    has_std_bar = False
    lines_std = cv.HoughLines(edges, hough_std_rho, hough_std_theta, hough_std_threshold)
    n_std = 0 if lines_std is None else len(lines_std)
    if lines_std is not None:
        for (rho, theta) in lines_std[:, 0, :]:
            angle_deg = abs(np.degrees(theta)) % 180
            if is_angle_bar(angle_deg, tol_deg=angle_tol_deg):
                has_std_bar = True
                break

    # Probabilistic Hough
    has_prob_bar = False
    lines_prob = cv.HoughLinesP(edges, houghp_rho, houghp_theta, houghp_threshold,
                                minLineLength=min_line_horizontal,
                                maxLineGap=max_line_gap)
    n_prob = 0 if lines_prob is None else len(lines_prob)
    if lines_prob is not None:
        valid_lines = 0
        for (x1, y1, x2, y2) in lines_prob[:, 0, :]:
            ang = angle_deg_from_segment(x1, y1, x2, y2)
            if not is_angle_bar(ang, tol_deg=angle_tol_deg):
                continue
            if not (top_limit <= y1 <= bottom_limit and top_limit <= y2 <= bottom_limit):
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            if length >= min_line_horizontal:
                valid_lines += 1
            has_prob_bar = valid_lines >= 2

    return has_std_bar, n_std, has_prob_bar, n_prob, lines_prob

# ---------------- window helpers & fusion tools ----------------
def interval_len(a, b): return max(0, b - a + 1)

def interval_overlap(a_start, a_end, b_start, b_end):
    lo = max(a_start, b_start); hi = min(a_end, b_end)
    return max(0, hi - lo + 1)

def adaptive_radius_from_len(iv_len, rmin=4, rmax=10):
    return int(max(rmin, min(rmax, iv_len // 4)))

def make_window(center, radius, n_frames):
    start = max(1, center - radius); end = min(n_frames, center + radius)
    return start, end

def stitch_intervals(runs, max_gap=STITCH_MAX_GAP):
    if not runs:
        return []
    merged = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        if s <= cur_e + max_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def compute_bone_centered_windows(bone_runs, n_frames, radius=None, rmin=4, rmax=10):
    wins = []
    for s_bone, e_bone in bone_runs:
        c_bone = (s_bone + e_bone) // 2
        r = adaptive_radius_from_len(interval_len(s_bone, e_bone), rmin=rmin, rmax=rmax) if radius is None else int(radius)
        start, end = make_window(c_bone, r, n_frames)
        wins.append({
            "s_bone": s_bone, "e_bone": e_bone, "c_bone": c_bone,
            "radius": r, "start": start, "end": end, "length": interval_len(start, end)
        })
    return wins

def compute_bar_gap_windows(bar_runs, n_frames, radius=5):
    windows = []
    if len(bar_runs) < 2:
        return windows
    for (a_s, a_e), (b_s, b_e) in zip(bar_runs[:-1], bar_runs[1:]):
        a_end, b_start = a_e, b_s
        c = (a_end + b_start) // 2
        start, end = make_window(c, radius, n_frames)
        windows.append({
            "a_end": a_end, "b_start": b_start, "c_bar": c,
            "radius": radius, "start": start, "end": end, "length": interval_len(start, end)
        })
    return windows

def fuse_bone_and_bar_windows(bone_wins, bar_wins, n_frames, bias_thresh=5, r_cap=None):
    fused_rows, fused_spans = [], []
    used_bar = set()

    def nearest_bar(center):
        if not bar_wins:
            return None, None
        best_i, best_d = None, None
        for i, bw in enumerate(bar_wins):
            d = abs(center - bw["c_bar"])
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

    for bw in bone_wins:
        c_bone, r_bone = bw["c_bone"], bw["radius"]
        s_bone, e_bone = bw["s_bone"], bw["e_bone"]
        bi, d = nearest_bar(c_bone)
        if bi is None:
            fs, fe = bw["start"], bw["end"]
            fused_rows.append({
                "source": "bone-only",
                "s_bone": s_bone, "e_bone": e_bone, "c_bone": c_bone,
                "c_bar": None, "d": None, "c_fused": c_bone,
                "start": fs, "end": fe, "length": interval_len(fs, fe),
                "bar_start": None, "bar_end": None
            })
            fused_spans.append((fs, fe))
            continue

        used_bar.add(bi)
        cw = bar_wins[bi]
        c_bar, r_bar = cw["c_bar"], cw["radius"]
        bs, be = cw["start"], cw["end"]

        c_fused = c_bone if (d is not None and d > bias_thresh) else (c_bone + c_bar) // 2
        r_fused = max(r_bone, r_bar)
        if r_cap is not None:
            r_fused = min(r_fused, r_cap)

        fs, fe = make_window(c_fused, r_fused, n_frames)
        fused_rows.append({
            "source": "fused",
            "s_bone": s_bone, "e_bone": e_bone, "c_bone": c_bone,
            "c_bar": c_bar, "d": d, "c_fused": c_fused,
            "start": fs, "end": fe, "length": interval_len(fs, fe),
            "bar_start": bs, "bar_end": be
        })
        fused_spans.append((fs, fe))

    for i, cw in enumerate(bar_wins):
        if i in used_bar:
            continue
        fused_rows.append({
            "source": "bar-only",
            "s_bone": None, "e_bone": None, "c_bone": None,
            "c_bar": cw["c_bar"], "d": None, "c_fused": cw["c_bar"],
            "start": cw["start"], "end": cw["end"], "length": cw["length"],
            "bar_start": cw["start"], "bar_end": cw["end"]
        })
        fused_spans.append((cw["start"], cw["end"]))

    return fused_rows, fused_spans

# ---------------- Pretty printing helpers -----------------
def _rng(a, b): return f"{a}-{b}" if a is not None and b is not None else ""
def _color(s, code): return f"\033[{code}m{s}\033[0m"

def print_fused_windows_table(title, fused_rows, use_color=True):
    """Print a single fused-windows table (used twice: pre- and post-refine)."""
    counts = {
        "fused":     sum(1 for r in fused_rows if r["source"] == "fused"),
        "bone-only": sum(1 for r in fused_rows if r["source"] == "bone-only"),
        "bar-only":  sum(1 for r in fused_rows if r["source"] == "bar-only"),
    }
    print(f"\n{title}")
    print(f"counts → fused: {counts['fused']}, bone-only: {counts['bone-only']}, bar-only: {counts['bar-only']}")
    print(" src        bone_rng  c_bone  c_bar    d  c_fused        window    len  ov_bone  ov_bar")
    print("-" * 96)
    for r in fused_rows:
        if r["source"] == "bar-only":  # skip bar-only rows for readability
            continue
        src_disp = {"fused": "fused", "bone-only": "bone-only"}[r["source"]]
        if use_color:
            src_disp = _color(src_disp, "32") if r["source"] == "fused" else _color(src_disp, "35")

        ov_bone = interval_overlap(r["start"], r["end"], r["s_bone"], r["e_bone"]) if r["s_bone"] is not None else 0
        ov_bar  = interval_overlap(r["start"], r["end"], r["bar_start"], r["bar_end"]) if r["bar_start"] is not None else 0

        ddisp = "" if r["d"] is None else str(r["d"])
        if use_color and r["d"] is not None and r["d"] >= 5:
            ddisp = _color(ddisp, "33")

        print(f"{src_disp:>9}  {_rng(r['s_bone'], r['e_bone']):>9}  "
              f"{'' if r['c_bone'] is None else r['c_bone']:>6}  "
              f"{'' if r['c_bar']  is None else r['c_bar']:>5}  "
              f"{ddisp:>3}  {r['c_fused']:>7}  "
              f"{_rng(r['start'], r['end']):>11}  {r['length']:>5}  "
              f"{ov_bone:>7}  {ov_bar:>7}")

# ---------------- Refinement of fused windows -----------------
def _runs_from_mask(mask_true):
    runs = []
    i, n = 0, len(mask_true)
    while i < n:
        if not mask_true[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask_true[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs

def refine_windows_with_sobel(arr, fused_rows):
    """Trim/split windows using Sobel+brightness rule on enhanced frames."""
    n = arr.shape[0]
    refined_rows = []
    refined_spans = []

    for r in fused_rows:
        if r["source"] == "bar-only":
            # keep bar-only as-is
            refined_rows.append(r.copy())
            refined_spans.append((r["start"], r["end"]))
            continue

        s, e = int(r["start"]), int(r["end"])
        if e < s:
            s, e = e, s
        s = max(1, min(s, n))
        e = max(1, min(e, n))
        L = e - s + 1
        if L <= 0:
            continue

        # build bad-mask for frames in [s..e]
        bad = np.zeros(L, dtype=bool)
        blank_mask = np.zeros(L, dtype=bool)

        for idx, k in enumerate(range(s, e + 1)):
            frame = arr[k - 1]
            f8 = enhance_frame_only(frame) if ENHANCE_FOR_SOBEL else to_uint8(frame)
            bright_border_bad = is_horizontal_intensity_bad(f8)
            central_ok        = has_central_content(f8)
            #bad_intensity = is_horizontal_intensity_bad(f8)
            blank_here    = is_blank_or_flat(frame)
            bad[idx]        = ( (bright_border_bad and not central_ok) or blank_here )
            blank_mask[idx] = blank_here

        # ignore short bad runs for splitting
        if IGNORE_BAD_GAPS_LE > 0:
            bad_runs = _runs_from_mask(bad)
            for a, b in bad_runs:
                if (b - a + 1) <= IGNORE_BAD_GAPS_LE:
                    bad[a:b + 1] = False

        bad_frac   = float(bad.mean())
        blank_frac = float(blank_mask.mean())
        blank_mask = np.zeros(L, dtype=bool)
        # quick recheck (cheap): mark blanks so we can compute blank fraction separately
        for idx, k in enumerate(range(s, e + 1)):
            if is_blank_or_flat(arr[k - 1]):
                blank_mask[idx] = True

        # --- VETO: strong bone evidence? don't drop, just split/trim if needed
        ov_bone_len = interval_overlap(s, e, r["s_bone"], r["e_bone"]) if r["s_bone"] is not None else 0
        bone_overlap_frac = (ov_bone_len / L) if L > 0 else 0.0
        d_val = r.get("d", None)
        veto_drop = (bone_overlap_frac >= NEVER_DROP_IF_BONE_OVERLAP_GE) or \
                    (d_val is not None and d_val >= NEVER_DROP_IF_D_AT_LEAST)
        
        # Hard-drop if the window is mostly blank (even if veto would apply)
        if blank_frac >= ALWAYS_DROP_IF_BLANK_FRAC_GE:
            print(f"[refine] DROP {s}-{e}  L={L}  blank={blank_mask.sum()} ({blank_frac:.1%})")
            continue
        # Normal drop (only if not vetoed)
        if (bad_frac >= DROP_IF_BAD_FRAC_GE) and (not veto_drop):
            print(f"[refine] DROP {s}-{e}  L={L}  bad={bad.sum()} ({bad_frac:.1%})")
            continue

        # find contiguous "good" segments
        good = ~bad
        good_runs = _runs_from_mask(good)
        kept = []
        for a, b in good_runs:
            len_ab = b - a + 1
            if len_ab >= MIN_LEN_KEEP:
                ss = s + a
                ee = s + b
                kept.append((ss, ee))

        if not kept:
            # fallback: keep original if nothing survived
            refined_rows.append(r.copy())
            refined_spans.append((s, e))
            continue

         # After computing bad, just before removing short bad runs:
        print(f"[refine] window {s}-{e} L={L}  bad={bad.sum()} ({bad.mean():.1%})")

        # After building `kept` (and before emitting rows):
        if bad.mean() >= DROP_IF_BAD_FRAC_GE:
            print(f"[refine] DROPPED {s}-{e}  bad_frac={bad.mean():.1%}")
        elif kept and (len(kept) != 1 or kept[0] != (s, e)):
            print(f"[refine] SPLIT {s}-{e} -> {kept}")
        else:
            print(f"[refine] UNCHANGED {s}-{e}")

        # emit refined segments (may split one window into multiple)
        for (ss, ee) in kept:
            row = r.copy()
            row["start"], row["end"] = ss, ee
            row["length"] = interval_len(ss, ee)
            row["refined"] = True
            refined_rows.append(row)
            refined_spans.append((ss, ee))

    # keep sorted
    refined_rows.sort(key=lambda x: x["start"])
    refined_spans.sort(key=lambda x: x[0])
    return refined_rows, refined_spans

# ---------------- Main scan function -------------------------
def scan_dicom_for_horizontal_bars(
        dicom_path,
        canny_low=50, canny_high=150,
        hough_std_threshold=150,
        houghp_threshold=60,
        min_line_len_ratio=0.75,
        max_line_gap=5,
        angle_tol_deg=2,
        verbose_every=200):

    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] > 1:
        arr = arr[..., 0]

    n_frames = arr.shape[0]
    subject = os.path.splitext(os.path.basename(dicom_path))[0]
    print(f"Scanning {subject}: {n_frames} frames")

    std_bar, prob_bar = [], []
    std_counts, prob_counts = [], []

    for i in range(n_frames):
        img = to_uint8(arr[i])
        if ENHANCE_FOR_DETECTION:
            img = enhance_frame_only(img)

        has_std, has_prob, n_std, n_prob, _lines_prob = detect_horizontal_bar_in_frame(
            img,
            canny_low=canny_low, canny_high=canny_high,
            hough_std_threshold=hough_std_threshold,
            houghp_threshold=houghp_threshold,
            min_line_len_ratio=min_line_len_ratio,
            max_line_gap=max_line_gap,
            angle_tol_deg=angle_tol_deg
        )

        if has_std:
            std_bar.append(i + 1)
        if has_prob:
            prob_bar.append(i + 1)
        std_counts.append(n_std)
        prob_counts.append(n_prob)

    # intervals
    std_runs = consecutive_intervals(std_bar)
    prob_runs = consecutive_intervals(prob_bar)
    all_bar_frames = sorted(set(std_bar) | set(prob_bar))
    bar_runs = consecutive_intervals(all_bar_frames)

    # Optionally precompute blanks and exclude them from bone runs
    if EXCLUDE_BLANK_FROM_BONE:
        blank_frames = []
        for i in range(n_frames):
            if is_blank_or_flat(arr[i]):
                blank_frames.append(i + 1)
        blank_set = set(blank_frames)

    else:
        blank_set = set()

    # Precompute blank frames once
    blank_frames = []
    for i in range(n_frames):
        f = to_uint8(arr[i])
        if is_blank_or_flat(f):
            blank_frames.append(i + 1)

    # bone = frames NOT in any bar and NOT blank
    bar_set   = set(all_bar_frames)
    no_bar = [k for k in range(1, n_frames + 1) if k not in bar_set and k not in blank_set]
    bone_runs = consecutive_intervals(no_bar, merge_gap=4)

    # windows
    bone_windows = compute_bone_centered_windows(bone_runs, n_frames=n_frames, radius=None, rmin=4, rmax=10)
    bar_radius = 5
    bar_windows = compute_bar_gap_windows(bar_runs, n_frames=n_frames, radius=bar_radius)

    fused_rows_raw, fused_spans_raw = fuse_bone_and_bar_windows(
        bone_windows, bar_windows, n_frames,
        bias_thresh=5, r_cap=None
    )

    # --------- Fused table BEFORE Sobel refinement ----------
    #print_fused_windows_table("[4] Fused windows (pre-refine: before Sobel)", fused_rows_raw, use_color=True)

    # refine with sobel+brightness
    if REFINE_WINDOWS_WITH_SOBEL:
        fused_rows, fused_spans = refine_windows_with_sobel(arr, fused_rows_raw)
    else:
        fused_rows, fused_spans = fused_rows_raw, fused_spans_raw

    # --------- Fused table AFTER Sobel refinement ----------
    print_fused_windows_table("[4R] Fused windows (post-refine: after Sobel)", fused_rows, use_color=True)

    # stitch for display (optional for diagnostics)
    fused_spans.sort(key=lambda x: x[0])
    stitched_fused = stitch_intervals(fused_spans, max_gap=STITCH_MAX_GAP)

    return {
        "subject": subject,
        "std_bar_frames": std_bar,
        "prob_bar_frames": prob_bar,
        "std_bar_intervals": std_runs,
        "prob_bar_intervals": prob_runs,
        "no_bar_intervals": bone_runs,
        "std_counts": std_counts,
        "prob_counts": prob_counts,
        "bar_intervals": bar_runs,
        "bone_centered_windows": bone_windows,
        "bar_centered_windows": bar_windows,
        "fused_windows_raw": fused_rows_raw,     # pre-refine (for diagnostics)
        "fused_windows": fused_rows,             # refined windows (used downstream)
        "stitched_fused_spans": stitched_fused
    }

if __name__ == "__main__":
    # Example run (keep your original params)
    result = scan_dicom_for_horizontal_bars(
        "/home/ds/Desktop/Hand_dicom/K11.dcm",
        canny_low=60, canny_high=180,
        hough_std_threshold=150,
        houghp_threshold=200,
        min_line_len_ratio=0.90,
        angle_tol_deg=4,
    )