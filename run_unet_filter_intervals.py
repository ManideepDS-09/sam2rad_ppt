#run_unet_filter_intervals.py
import cv2
import torch
import numpy as np
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed

from filtering1 import scan_dicom_for_horizontal_bars, to_uint8
from unet_model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_H, IMG_W = 480, 640
PAD = 15
THR = 0.4
MAX_WORKERS = 8

# Default paths (edit if needed)
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K04.dcm"
UNET_WTS = "/home/ds/Desktop/SAM_HAND_FINAL/trained_weights/unet_bones_best.pth"


# -------- UNET ----------
def build_unet(weight_path):
    model = UNet(in_channels=3, out_channels=1, base_c=32, bilinear=True)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


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
    top_k=7,           # keep strongest K frames per interval
):
    """
    Improved UNet selection:
    - score every frame in the expanded interval
    - reject frames with raw score < min_thr
    - from remaining, keep only top_k
    """
    model = build_unet(unet_wts)
    frames = load_dicom_frames(dicom_path)
    n_frames = frames.shape[0]

    # Get fused intervals from filtering1.py
    result = scan_dicom_for_horizontal_bars(dicom_path)
    fused_rows = result["fused_windows"]
    intervals = [(r["start"], r["end"]) for r in fused_rows if r["source"] == "fused"]

    kept_per_interval = []

    for (s, e) in intervals:
        s2, e2 = max(1, s - pad), min(n_frames, e + pad)
        idxs = list(range(s2, e2 + 1))

        scored = []  # (score, frame_idx)

        # score all frames
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(infer_score, model, frames[i - 1]): i for i in idxs}
            for fut in as_completed(futs):
                i = futs[fut]
                score = fut.result()
                scored.append((score, i))

        # reject low-confidence frames
        scored = [(sc, fr) for (sc, fr) in scored if sc >= min_thr]

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
        kept = sorted([fr for (_, fr) in top])

        kept_per_interval.append({
            "interval": (s, e),
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

    print(f"[INFO] Extracting intervals from filtering1.py ...")
    print(f"[INFO] Found {len(per_iv)} intervals from filtering1.py")

    for item in per_iv:
        (s, e) = item["interval"]
        (s2, e2) = item["expanded"]
        kept = item["kept_frames"]
        if not kept:
            print(f"{s}-{e}  →  expanded {s2}-{e2}  →  dropped")
        else:
            kept_str = fmt_runs(kept)
            print(f"{s}-{e}  →  expanded {s2}-{e2}  →  kept {kept_str} ({len(kept)} frames)")


if __name__ == "__main__":
    main()