# energy_intervals.py
import numpy as np
import cv2 as cv
import pydicom
import matplotlib.pyplot as plt
import os

def analyze_bone_interval(arr, bone_interval,
                          roi_y1, roi_y2,
                          sustain_k=5,
                          onset_frac=0.65):
    """
    Analyze one bone interval and find onset-based bone frame
    """

    s, e = bone_interval
    scores = []

    #print(f"\nBone interval {s}–{e}")
    #print("frame | mean_energy")

    if e - s + 1 <= 2:
        #print(f"--> Trivial interval, using center frame {s}")
        #print(f"--> Window: [{s} , {s}]")
        return s

    for f in range(s-1, e):   # 0-based
        frame = arr[f].astype(np.float32)

        fmin, fmax = frame.min(), frame.max()
        frame_n = (frame - fmin) / (fmax - fmin + 1e-6)

        roi = frame_n[roi_y1:roi_y2, :]
        energy = float(np.mean(roi))

        scores.append(energy)
        #print(f"{f+1:5d} | {energy:8.5f}")

    scores = np.array(scores)
    ENERGY_FLOOR = 0.05

    if scores.max() < ENERGY_FLOOR:
        #print(f"--> Dropped interval (low energy): max={scores.max():.4f}")
        return None
    N = len(scores)

    # --- temporal derivative ---
    dE = np.zeros_like(scores)
    if N > 2:
        dE[1:-1] = np.abs(scores[2:] - scores[:-2]) / 2
        dE[0] = dE[1]
        dE[-1] = dE[-2]
    else:
        dE[:] = 0

    # --- thresholds (SUBJECT RELATIVE, NOT ABSOLUTE) ---
    HIGH_ENERGY_THR = 0.85 * np.percentile(scores, 95)
    FLAT_THR = np.percentile(dE, 40)   # low variation

    mask = (scores >= HIGH_ENERGY_THR) & (dE <= FLAT_THR)

    idxs = np.where(mask)[0]

    if len(idxs) > 0:
        segments = np.split(
            idxs,
            np.where(np.diff(idxs) != 1)[0] + 1
        )

        # choose longest flat high-energy band
        # reject bands hugging interval edges
        VALID_MARGIN = 0.15 * (e - s + 1)

        valid_segs = []
        for seg in segments:
            seg_start = s + seg[0]
            seg_end   = s + seg[-1]
            if (seg_start - s) > VALID_MARGIN and (e - seg_end) > VALID_MARGIN:
                valid_segs.append(seg)

        if len(valid_segs) > 0:
            # Prefer band closest to next bar (late bone region)
            best_seg = max(
                valid_segs,
                key=lambda seg: (seg[-1])   # highest frame index
            )
        else:
            best_seg = max(segments, key=len)


        seg_start = s + best_seg[0]
        seg_end   = s + best_seg[-1]

        center = (seg_start + seg_end) // 2

        #print(f"--> Flat high-energy band: [{seg_start} , {seg_end}]")
        #print(f"--> Output window: [{center-10} , {center+10}]")
    else:
        center = s + np.argmax(scores)
        #print(f"--> Fallback window: [{center-10} , {center+10}]")

    return center

def detect_bar1_regions_debug(dicom_path,
                              start_f=1,
                              end_f=None,
                              roi_top_frac=0.30,
                              roi_bottom_frac=0.70,
                              bright_ratio=0.55,
                              width_ratio=0.40,
                              smooth_len=4,
                              show_frames=[x for x in range(1,3)]):
    """
    Standalone Bar-1 detector.
    NO SAVING. Notebook-friendly visualization.
    """

    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array

    # Normalize to [N,H,W]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4:
        arr = arr[..., 0]

    N = arr.shape[0]
    #end_f = min(end_f, N)
    if end_f is None:
        end_f = N
    else:
        end_f = min(end_f, N)

    frames = range(start_f - 1, end_f)

    bar_mask = np.zeros(len(frames), dtype=bool)

    H, W = arr.shape[1], arr.shape[2]
    roi_y1 = int(roi_top_frac * H)
    roi_y2 = int(roi_bottom_frac * H)

    #print("\nFrame | max_run | thresh | BAR")
    #print("--------------------------------")

    #vis_data = {}

    for idx, f in enumerate(frames):
        frame = arr[f].astype(np.float32)

        # Normalize intensity
        fmin, fmax = frame.min(), frame.max()
        frame_n = (frame - fmin) / (fmax - fmin + 1e-6)

        roi = frame_n[roi_y1:roi_y2, :]

        # ----- NEW: detect empty ROI -----
        roi_energy = np.mean(roi)

        if roi_energy < 0.02:   # you can tune to 0.015–0.03
            is_bar = True
            max_run = 0
            vis_mask = np.zeros_like(roi, dtype=np.uint8)
            bar_mask[idx] = True
            #print(f"{f+1:03d}   |  EMPTY ROI |   -   | True")
            
            #if (f+1) in show_frames:
            #    vis_data[f+1] = (frame_n, vis_mask, is_bar)
            continue

        # Dynamic brightness threshold
        thr = bright_ratio * np.max(roi)

        # Compute max contiguous width of bright region
        max_run = 0
        vis_mask = np.zeros_like(roi, dtype=np.uint8)

        for r in range(roi.shape[0]):
            row = roi[r] > thr
            run_len = 0

            for j in range(W):
                if row[j]:
                    run_len += 1
                    vis_mask[r, j] = 255
                    max_run = max(max_run, run_len)
                else:
                    run_len = 0

        #is_bar = max_run >= width_ratio * W
        #is_bar = (max_run >= width_ratio * W) and (thr > 0.50)
        roi_mean   = float(np.mean(roi))
        roi_std    = float(np.std(roi))
        roi_max    = float(np.max(roi))
        roi_p95    = float(np.percentile(roi, 95))
        roi_energy = roi_mean

        bright_ratio_used = thr / roi_max if roi_max > 0 else 0
        width_ratio_used  = max_run / W

        is_bar = (max_run >= width_ratio * W) and (thr >= 0.52) and (max_run >= 300)
        bar_mask[idx] = is_bar

        #print(f"{f+1:03d} | run={max_run:4d} {width_ratio_used:.2f} | thr={thr:.3f} | p95={roi_p95:.3f} | energy={roi_energy:.3f} | mean={roi_mean:.3f} | std={roi_std:.3f} | BAR={is_bar}")

        #if (f+1) in show_frames:
        #    vis_data[f+1] = (frame_n, vis_mask, is_bar)

    # ------------------------------------------
    #  TEMPORAL REPAIR PIPELINE (C)
    # ------------------------------------------

    raw = bar_mask.copy()
    N = len(raw)

    # T1 — Neighbor fill (1 0 1 → 1 1 1)
    filled = raw.copy()
    for i in range(1, N-1):
        if raw[i] == 0 and raw[i-1] == 1 and raw[i+1] == 1:
            filled[i] = 1

    # T2 — Temporal dilation (expand ±3)
    dilated = filled.copy()
    dilation_radius = 3
    for i in range(N):
        if filled[i]:
            left  = max(0, i - dilation_radius)
            right = min(N-1, i + dilation_radius)
            dilated[left:right+1] = True

    # T3 — Remove micro-bursts (<5 frames)
    clean = dilated.copy()

    # detect continuous segments
    segments = []
    i = 0
    while i < N:
        if clean[i] == 0:
            i += 1
            continue
        j = i
        while j+1 < N and clean[j+1] == 1:
            j += 1
        length = j - i + 1
        if length < 5:           # remove tiny noisy bars
            clean[i:j+1] = 0
        else:
            segments.append((i, j))
        i = j + 1

    # this is the final bar mask
    bar_mask_final = clean


    # ---- Convert to intervals (FINAL MASK) ----
    intervals = []
    i = 0
    while i < len(bar_mask_final):
        if not bar_mask_final[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(bar_mask_final) and bar_mask_final[j+1]:
            j += 1
        intervals.append((frames[i] + 1, frames[j] + 1))
        i = j + 1
    
    # -------- COMPUTE BONE INTERVALS INSIDE FUNCTION --------
    bone_intervals = []
    for i in range(len(intervals) - 1):
        bar_end   = intervals[i][1]
        next_start = intervals[i+1][0]

        bone_s = bar_end + 1
        bone_e = next_start - 1

        if bone_s <= bone_e:
            bone_intervals.append((bone_s, bone_e))
    
    print("\nBone interval | bone frames interval")

    final_bone_frame_intervals = []

    for (bone_s, bone_e) in bone_intervals:
        center = analyze_bone_interval(
            arr,
            (bone_s, bone_e),
            roi_y1,
            roi_y2,
            sustain_k=5,
            onset_frac=0.65
        )

        if center is None:
            continue

        win_s = max(bone_s, center - 10)
        win_e = min(bone_e, center + 10)

        print(f"{bone_s}-{bone_e} | {win_s}-{win_e}")
        final_bone_frame_intervals.append((win_s, win_e))


    # -------- SHOW VISUAL DEBUG --------
    #for fnum, (frame_n, vis_mask, is_bar) in vis_data.items():
    #    plt.figure(figsize=(6, 6))
    #    plt.title(f"Frame {fnum}  |  {'BAR' if is_bar else 'BONE'}")

        #vis = cv.cvtColor((frame_n * 255).astype(np.uint8), cv.COLOR_GRAY2RGB)

        # ROI boundaries
        #vis[roi_y1, :, :] = [255, 255, 0]   # top ROI boundary
        #vis[roi_y2, :, :] = [0, 255, 255]   # bottom ROI boundary

        # Overlay bright mask
        #mask_rgb = np.zeros_like(vis)
        #mask_rgb[roi_y1:roi_y2][vis_mask == 255] = [255, 0, 0]

        #overlay = cv.addWeighted(vis, 0.6, mask_rgb, 0.8, 0)
        #plt.imshow(overlay)
        #plt.axis("off")
        #plt.show()

    return bar_mask_final, intervals, final_bone_frame_intervals

#def get_bone_intervals(dicom_path):
#    mask, bar_intervals, bone_intervals = detect_bar1_regions_debug(dicom_path)
#    return bone_intervals

dicom_path = "/home/ds/Desktop/Hand_dicom/K07.dcm"
subject_name =  os.path.splitext(os.path.basename(dicom_path))[0]

mask, ivals, bone_intervals = detect_bar1_regions_debug(dicom_path)

print("****************************")
print(f"subject name -- {subject_name}")
print("****************************")
#print("\nDetected BAR-1 intervals:")

# Numbered BAR intervals
#for idx, (s, e) in enumerate(ivals, start=1):
#    print(f"BAR {idx} : {s} to {e}")

# -------- COMPUTE BONE INTERVALS --------
#print("\nDetected Bone intervals:")

#bone_id = 1
#for i in range(len(ivals) - 1):
#    bar_end   = ivals[i][1]
#    next_start = ivals[i+1][0]

#    bone_s = bar_end + 1
#    bone_e = next_start - 1

#    if bone_s <= bone_e:
#        print(f"Bone {bone_id} : {bone_s} - {bone_e}")
#        bone_id += 1