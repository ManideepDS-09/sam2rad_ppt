import cv2
import torch
import numpy as np
import pydicom
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from filtering1 import scan_dicom_for_horizontal_bars, to_uint8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_H, IMG_W = 480, 640
PAD = 4
THR = 0.5
MAX_WORKERS = 8

# Default paths (edit if needed)
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/H036.dcm"
UNET_WTS  = "/home/ds/Desktop/sam_hand/sam_hand/runs/unet_bones/last_epoch_multiclass.pth"
CFG_PATH  = "/home/ds/Desktop/sam_hand/sam_hand/config_unet.yaml"


# -------- LOAD CLASS NAMES --------
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

CLASS_LIST = cfg["coco"]["classes"][:]   # 19 names
BONE_CLASSES = CLASS_LIST[:-1]           # ignore last "or"


# -------- UNET ----------
def build_unet(weight_path):
    # model that matches checkpoint
    from unet_model import UNet32   # we will define below

    model = UNet32(
        n_channels=3,
        n_classes=19
    )

    ckpt = torch.load(weight_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)

    model.load_state_dict(state, strict=True)
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
    """
    Compute:
      - multiclass softmax
      - ignore last "or" channel
      - max class probability per pixel
      - top-5% mean
      - also return best bone class name
    """
    x = preprocess(frame)
    logits = model(x)                        # (1,19,H,W)
    probs  = torch.softmax(logits, dim=1)[0] # (19,H,W)

    # ignore 'or'
    probs_bone = probs[:-1]                  # (18,H,W)

    # max prob per pixel
    max_per_pixel = probs_bone.max(dim=0).values.cpu().numpy()

    # best class per pixel
    class_map = probs_bone.argmax(dim=0).cpu().numpy()

    # find dominant bone class index
    vals, counts = np.unique(class_map, return_counts=True)
    dominant_idx = vals[np.argmax(counts)]
    dominant_name = BONE_CLASSES[dominant_idx]

    # top 5%
    flat = max_per_pixel.ravel()
    k = max(1, int(0.05 * flat.size))
    score = float(np.mean(np.sort(flat)[-k:]))

    return score, dominant_name


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


def get_kept_frames(
    dicom_path=DICOM_PATH,
    unet_wts=UNET_WTS,
    thr=THR,
    pad=PAD,
    max_workers=MAX_WORKERS,
    min_thr=0.60,
    top_k=4,
):
    model = build_unet(unet_wts)
    frames = load_dicom_frames(dicom_path)
    n_frames = frames.shape[0]

    result = scan_dicom_for_horizontal_bars(dicom_path)
    fused_rows = result["fused_windows"]
    intervals = [(r["start"], r["end"]) for r in fused_rows if r["source"] == "fused"]

    kept_per_interval = []

    for (s, e) in intervals:
        s2, e2 = max(1, s - pad), min(n_frames, e + pad)
        idxs = list(range(s2, e2 + 1))

        scored = []  # (score, name, frame_idx)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(infer_score, model, frames[i - 1]): i for i in idxs}
            for fut in as_completed(futs):
                i = futs[fut]
                score, name = fut.result()
                scored.append((score, name, i))

        scored = [(sc, nm, fr) for (sc, nm, fr) in scored if sc >= min_thr]

        if not scored:
            kept_per_interval.append({
                "interval": (s, e),
                "expanded": (s2, e2),
                "kept_frames": [],
                "best_class": None
            })
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        kept = sorted([fr for (_, _, fr) in top])

        # dominant bone class among kept
        names = [nm for (_, nm, _) in top]
        vals, cnts = np.unique(names, return_counts=True)
        best_name = vals[np.argmax(cnts)]

        kept_per_interval.append({
            "interval": (s, e),
            "expanded": (s2, e2),
            "kept_frames": kept,
            "best_class": best_name
        })

    return kept_per_interval


def main():
    print(f"[INFO] Loading UNet weights from: {UNET_WTS}")
    print(f"[INFO] Bone classes ({len(BONE_CLASSES)}): {BONE_CLASSES}")

    per_iv = get_kept_frames(DICOM_PATH, UNET_WTS, THR, PAD, MAX_WORKERS)

    def fmt_runs(frames):
        runs = merge_consecutive(frames)
        return ", ".join([f"{a}-{b}" if a != b else f"{a}" for a, b in runs]) if runs else ""

    print(f"[INFO] Found {len(per_iv)} intervals")

    for item in per_iv:
        (s, e) = item["interval"]
        (s2, e2) = item["expanded"]
        kept = item["kept_frames"]
        bestc = item["best_class"]

        if not kept:
            print(f"{s}-{e}  → expanded {s2}-{e2}  → dropped (no strong bone)")
        else:
            kept_str = fmt_runs(kept)
            print(f"{s}-{e}  → expanded {s2}-{e2}  → kept {kept_str} ({len(kept)} frames)  | bone: {bestc}")


if __name__ == "__main__":
    main()
