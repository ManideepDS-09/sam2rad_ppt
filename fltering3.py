#!/usr/bin/env python
import os
import numpy as np
import pydicom
import cv2 as cv

from skimage.morphology import skeletonize

# HARD-CODED DICOM PATH
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K02.dcm"

from filtering1 import (
    enhance_frame_only,
    to_uint8,
    is_blank_or_flat,
)

# ----------------- Parameters -----------------
BAR_SPAN_RATIO    = 0.50
BAR_MAX_ANGLE_DEG = 15.0
BAR_MIN_LINEARITY = 15.0

MIN_CONSEC_BAR   = 3
MIN_CONSEC_BONE  = 5
MIN_CONSEC_BLANK = 10

# --- Adaptive Temporal Voting ---
BAR_ENTER_TH = 4     # how many consecutive BAR votes to enter BAR
BONE_ENTER_TH = 2    # how many consecutive BONE votes to enter BONE

BAR_EXIT_TH  = 2     # how many BONE votes needed to exit BAR
BONE_EXIT_TH = 3     # how many BAR votes needed to exit BONE

def adaptive_temporal_filter(raw_labels):
    final_labels = []

    current_state = raw_labels[0]
    streak = 1

    bar_streak = 0
    bone_streak = 0

    for i, lbl in enumerate(raw_labels):
        if lbl == "BAR":
            bar_streak += 1
            bone_streak = 0
        else:
            bone_streak += 1
            bar_streak = 0

        # ---------------- ENTER CONDITIONS ----------------
        if current_state == "BONE":
            if bar_streak >= BAR_ENTER_TH:
                current_state = "BAR"
                bar_streak = 0
                bone_streak = 0

        elif current_state == "BAR":
            if bone_streak >= BONE_ENTER_TH:
                current_state = "BONE"
                bar_streak = 0
                bone_streak = 0

        # ---------------- EXIT CONDITIONS (HYSTERESIS) ----------------
        if current_state == "BAR":
            if bone_streak >= BAR_EXIT_TH:
                current_state = "BONE"
                bar_streak = 0
                bone_streak = 0

        elif current_state == "BONE":
            if bar_streak >= BONE_EXIT_TH:
                current_state = "BAR"
                bar_streak = 0
                bone_streak = 0

        final_labels.append(current_state)

    return final_labels

# ----------------- Classification --------------------
def classify_frame_with_skeleton(frame):
    f8 = to_uint8(frame)
    f8 = enhance_frame_only(f8)

    blank = is_blank_or_flat(f8)

    h, w = f8.shape[:2]

    # ---------------- ROI DEFINITION ----------------
    top_end    = int(0.15 * h)
    bottom_sta = int(0.50 * h)

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[:top_end, :] = True                 
    roi_mask[bottom_sta:, :] = True              
    # ------------------------------------------------

    # Gradient strength (critical for blank detection)
    gx = cv.Sobel(f8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(f8, cv.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.hypot(gx, gy)))

    var_int = float(f8.var())

    _, bw = cv.threshold(f8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = bw > 0

    # ---------------- APPLY ROI BEFORE SKELETON ----------------
    mask = mask & roi_mask
    # -----------------------------------------------------------

    skel = skeletonize(mask)
    coords = np.column_stack(np.nonzero(skel))
    skel_pixels = int(coords.shape[0])

    if coords.shape[0] < 10:
        return "BONE", {
            "span": 0.0,
            "angle": 0.0,
            "lin": 0.0,
            "grad": grad_mean,
            "var": var_int,
            "skel_px": skel_pixels,
            "blank": blank,
        }

    ys = coords[:, 0]
    xs = coords[:, 1]

    x_span = xs.max() - xs.min() + 1
    span_ratio = x_span / float(w)

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    v = eigvecs[:, 0]
    angle = abs(np.degrees(np.arctan2(v[1], v[0]))) % 180.0
    angle = min(angle, 180.0 - angle)

    lin = float(eigvals[0]) / float(eigvals[1] + 1e-6)

    label = "BONE"

    if not blank:
        if (
            span_ratio >= BAR_SPAN_RATIO
            and angle <= BAR_MAX_ANGLE_DEG
            and lin >= BAR_MIN_LINEARITY
        ):
            label = "BAR"
    else:
        label = "BLANK"

    info = {
        "span": span_ratio,
        "angle": angle,
        "lin": lin,
        "grad": grad_mean,
        "var": var_int,
        "skel_px": skel_pixels,
        "blank": blank,
    }

    return label, info


# ----------------- State Machine --------------------

def collect_intervals_with_state_machine(frame_stream, logs):
    pockets = {i: [] for i in range(8)}
    state = 0

    consec_bar = 0
    consec_bone = 0
    consec_blank = 0

    for i, (lbl, info) in enumerate(frame_stream, start=1):

        pockets[state].append(i)

        if lbl == "BAR":
            consec_bar += 1
            consec_bone = 0
            consec_blank = 0
        elif lbl == "BONE":
            consec_bone += 1
            consec_bar = 0
            consec_blank = 0
        else:
            consec_blank += 1
            consec_bar = 0
            consec_bone = 0

        prev_state = state

        if state == 0 and consec_bone >= MIN_CONSEC_BONE:
            state = 1
        elif state == 1 and consec_bar >= MIN_CONSEC_BAR:
            state = 2
        elif state == 2 and consec_bone >= MIN_CONSEC_BONE:
            state = 3
        elif state == 3 and consec_bar >= MIN_CONSEC_BAR:
            state = 4
        elif state == 4 and consec_bone >= MIN_CONSEC_BONE:
            state = 5
        elif state == 5 and consec_bar >= MIN_CONSEC_BAR:
            state = 6
        elif state == 6 and consec_blank >= MIN_CONSEC_BLANK:
            state = 7

        logs.append((i, lbl, info, prev_state, state))

    intervals = {}
    for k, v in pockets.items():
        intervals[k] = (v[0], v[-1]) if v else None

    return intervals


# ----------------- MAIN ----------------------
def main():
    if not os.path.isfile(DICOM_PATH):
        raise FileNotFoundError(f"DICOM not found: {DICOM_PATH}")

    ds = pydicom.dcmread(DICOM_PATH)
    arr = ds.pixel_array

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] > 1:
        arr = arr[..., 0]

    n_frames = min(350, arr.shape[0])
    print(f"[INFO] Scanning: {os.path.basename(DICOM_PATH)} ({n_frames} frames)\n")

    raw_labels = []
    infos = []
    logs = []

    for i in range(n_frames):
        lbl, info = classify_frame_with_skeleton(arr[i])
        raw_labels.append(lbl)
        infos.append(info)

    # --------- APPLY ADAPTIVE MULTI-VOTE FILTER ----------
    labels = adaptive_temporal_filter(raw_labels)

    intervals = collect_intervals_with_state_machine(list(zip(labels, infos)), logs)
    BEST_BONE_FRAMES = list(range(80,89)) + list(range(120,130)) + list(range(170,180))

    print("\n================ FULL 350-FRAME DIAGNOSTIC ================\n")
    print(f"{'Frame':>4} | {'Label':>5} | {'Span':>6} | {'Angle':>6} | {'Lin':>8} | {'Grad':>7} | {'Var':>9} | {'Skel':>6} | {'Blank':>5} | Mark")
    print("-"*115)

    for i in range(n_frames):
        lbl, info = classify_frame_with_skeleton(arr[i])

        mark = ""
        if (i+1) in BEST_BONE_FRAMES:
            mark = "★BEST"
        elif lbl == "BAR":
            mark = "bar"
        elif lbl == "BLANK":
            mark = "blank"

        print(
            f"{i+1:4d} | {lbl:>5} | "
            f"{info.get('span',0):6.3f} | "
            f"{info.get('angle',0):6.2f} | "
            f"{info.get('lin',0):8.2f} | "
            f"{info.get('grad',0):7.3f} | "
            f"{info.get('var',0):9.1f} | "
            f"{info.get('skel_px',0):6d} | "
            f"{str(info.get('blank',False)):>5} | {mark}"
        )

    names = [
        "Bar 1", "Bone (PD4)", "Bar 2", "Bone (PD3)",
        "Bar 3", "Bone (PD2)", "Bar 4", "Blank"
    ]

    print("\n================ FINAL INTERVAL TABLE ================\n")
    print(f"{'Slot':<15} | Interval")
    print("-" * 40)

    for name, pocket_id in zip(names, range(8)):
        iv = intervals[pocket_id]
        print(f"{name:<15} | {iv if iv else 'None'}")

    print("\n======================================================\n")


if __name__ == "__main__":
    main()