# energy_unet_intervals.py

import cv2
import torch
import numpy as np
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed

from filtering1 import to_uint8                      
from unet_model import UNet
from energy_intervals import detect_bar1_regions_debug   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_H, IMG_W = 480, 640
PAD = 0
THR = 0.6
MAX_WORKERS = 8

# Default paths (edit if needed)
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K11.dcm"
UNET_WTS   = "/home/ds/Desktop/SAM_HAND_FINAL/trained_weights/unet_bones_best.pth"

# -------- UNET ----------
def build_unet(weight_path):
    model = UNet(in_channels=3, out_channels=1, base_c=32, bilinear=True)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

GLOBAL_UNET = build_unet(UNET_WTS)  
GLOBAL_UNET.eval()

def preprocess(frame):
    f8 = to_uint8(frame)
    f8 = cv2.resize(f8, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    f8 = cv2.cvtColor(f8, cv2.COLOR_GRAY2RGB)
    x = (f8.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def infer_score(model, frame):
    x = preprocess(frame)
    probs = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    flat = probs.ravel()
    k = max(1, int(0.05 * flat.size))            # top-5% pixel mean
    return float(np.mean(np.sort(flat)[-k:]))


def merge_consecutive(frames):
    if not frames:
        return []
    frames = sorted(frames)
    out, s, p = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == p + 1:
            p = f
        else:
            out.append((s, p))
            s = p = f
    out.append((s, p))
    return out


def load_dicom_frames(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] > 1:
        arr = arr[..., 0]
    return arr


# -------- PUBLIC API you can import ----------
def get_kept_frames(
    dicom_path=DICOM_PATH,
    unet_wts=UNET_WTS,
    thr=THR,
    pad=PAD,
    max_workers=MAX_WORKERS,
    min_thr=0.4,      # hard floor to block weak bone patterns
    top_k=5,          # keep strongest K frames per interval
):
    """
    Improved UNet selection, now using bone intervals from energy_intervals.py:

    1. Use detect_bar1_regions_debug(dicom_path) to get bone_intervals (1-based).
    2. For each bone interval (s,e), expand by PAD on both sides.
    3. Score every frame in the expanded interval with UNet.
    4. Drop frames with raw score < min_thr.
    5. Keep only top_k frames per interval by score.
    """
    model = build_unet(unet_wts)
    frames = load_dicom_frames(dicom_path)
    n_frames = frames.shape[0]

    # --------------------------------------------------------
    # Get bone intervals from energy_intervals.py
    # detect_bar1_regions_debug returns:
    #   bar_mask_final, bar_intervals, bone_intervals
    # bone_intervals are 1-based frame indices (s,e)
    # --------------------------------------------------------
    print(f"[INFO] Getting bone intervals from energy_intervals.py for: {dicom_path}")
    _, bar_intervals, final_bone_frame_intervals, bone_intervals = detect_bar1_regions_debug(dicom_path)

    intervals = bone_intervals

    kept_per_interval = []

    for (s, e) in intervals:
        # expand, but clamp to valid frame range
        s2, e2 = max(1, s - pad), min(n_frames, e + pad)
        idxs = list(range(s2, e2 + 1))
        
        interval_center = 0.5 * (s + e)
        half_len = max(1.0, 0.5 * (e - s))

        scored = []  # (score, frame_idx)
        # score all frames in this expanded interval
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(infer_score, model, frames[i - 1]): i for i in idxs}
            for fut in as_completed(futs):
                i = futs[fut]
                score = fut.result()
                # distance from bone center (normalized 0..1)
                dist = abs(i - interval_center) / half_len
                # center-biased score (small, smooth penalty)
                weighted_score = score * (1.0 - 0.35 * dist)
                scored.append((weighted_score, score, i))

        # reject low-confidence frames
        scored = [(wsc, sc, fr) for (wsc, sc, fr) in scored if sc >= min_thr]

        if not scored:
            # entire interval is too weak → drop
            kept_per_interval.append({
                "interval": (s, e),
                "expanded": (s2, e2),
                "kept_frames": []
            })
            continue

        # sort by score desc
        scored.sort(key=lambda x: x[0], reverse=True)
        # take top_k only
        top = scored[:top_k]
        kept = sorted([fr for (_, _, fr) in top])

        kept_per_interval.append({
            "interval": (s, e),   # original bone interval
            "expanded": (s2, e2),
            "kept_frames": kept
        })

    return kept_per_interval


# -------- CLI print mode ----------
def main():
    print(f"[INFO] Loading UNet weights from: {UNET_WTS}")
    per_iv = get_kept_frames(DICOM_PATH, UNET_WTS, THR, PAD, MAX_WORKERS)

    def fmt_runs(frames):
        runs = merge_consecutive(frames)
        return ", ".join([f"{a}-{b}" if a != b else f"{a}" for a, b in runs]) if runs else ""

    print(f"[INFO] Extracting intervals from energy_intervals.py ...")
    print(f"[INFO] Found {len(per_iv)} bone intervals from energy_intervals.py")

    for item in per_iv:
        (s, e) = item["interval"]
        (s2, e2) = item["expanded"]
        kept = item["kept_frames"]
        if not kept:
            print(f"{s}-{e}  →  expanded {s2}-{e2}  →  dropped")
        else:
            kept_str = fmt_runs(kept)
            print(f"{s}-{e}  →  expanded {s2}-{e2}  →  kept {kept_str} ({len(kept)} frames)")


def unet_score_frame(frame):
    """Standalone scorer for window2.py"""
    f8 = to_uint8(frame)
    f8 = cv2.resize(f8, (640, 480))
    f8 = cv2.cvtColor(f8, cv2.COLOR_GRAY2RGB)
    x = (f8.astype(np.float32) / 255.0).transpose(2,0,1)
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = GLOBAL_UNET(x)
        probs = torch.sigmoid(out).squeeze().detach().cpu().numpy()

    flat = probs.ravel()
    k = max(1, int(0.05 * flat.size))
    return float(np.mean(np.sort(flat)[-k:]))

if __name__ == "__main__":
    main()