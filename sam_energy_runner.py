import torch
import torch.nn.functional as F
import numpy as np
import pydicom
import cv2

from sam2rad_model import SAM2RadModel
from energy_intervals import detect_bar1_regions_debug
from enhance_utils import enhance_frame_only

# -------------------------------------------------
# CLASS MAP
# -------------------------------------------------
CLASS_NAMES = [
    "PD4","PD3","PD2","PD5",
    "PM5","PM4","PM3","PM2",
    "PP5","MC5","PP4","MC4",
    "PP3","MC3","PP2","MC2",
    "PD1","PP1","or"
]

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CKPT  = "/home/ds/Desktop/sam_hand/sam_hand/sam_vit_b_01ec64.pth"
RAD_CKPT  = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad_train_sam_b_2/checkpoints/best_model.pth"

# <<< HARD-CODE YOUR SUBJECT HERE >>>
DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K02.dcm"
# DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K05.dcm"
# change as needed


# ============================================================
# LOAD TRAINED MODEL
# ============================================================
def load_model():
    print("\n[INFO] Loading SAM2Rad checkpoint...")

    model = SAM2RadModel(
        sam_checkpoint=SAM_CKPT,
        lora_rank=4
    )

    state = torch.load(RAD_CKPT, map_location="cpu")
    if "model" in state:
        state = state["model"]

    clean = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)

    print(f"[INFO] Loaded checkpoint.")
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.to(DEVICE).eval()
    return model


# ============================================================
# SOFT IoU + Dice (NO GT)
# ============================================================
def compute_iou_dice(mask_logits, thr=0.5):
    """
    mask_logits: (N, H, W)
    Returns:
      iou:  (N,)
      dice: (N,)
    """
    prob = mask_logits.sigmoid()              # (N,H,W)
    binmask = (prob > thr).float()           # threshold = 0.5

    # flatten
    prob_f = prob.view(prob.shape[0], -1)
    bin_f  = binmask.view(binmask.shape[0], -1)

    inter = (prob_f * bin_f).sum(dim=1)
    sum_prob = prob_f.sum(dim=1)
    sum_bin  = bin_f.sum(dim=1)

    union = sum_prob + sum_bin - inter + 1e-6

    iou  = inter / union
    dice = (2.0 * inter) / (sum_prob + sum_bin + 1e-6)

    return iou, dice


# ============================================================
# PROCESS SINGLE FRAME
# ============================================================
def run_frame(model, frame):
    """
    frame: 2D numpy (H,W) float32
    Returns:
      best_score, best_bone_name, best_iou, best_dice
    """
    # --- enhancement FIRST ---
    frame_enh = enhance_frame_only(frame)          # uint8
    img = cv2.resize(frame_enh, (1024, 1024))      # uint8
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    # H,W,3

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,3,1024,1024)

    # always run with max bones = 5 (how it was trained)
    out = model(img, num_bones=5)

    # ----- class logits -----
    cls = out["pred_class_logits"]   # (1,5,19) OR (1,5,5,19)
    if cls.dim() == 4:
        # collapse weird extra dim if present
        cls = cls.mean(dim=2)
    cls = cls[0]                      # (5,19)

    # ----- masks -----
    mask_logits = out["mask_logits"][0]  # (5,Hm,Wm)

    # compute IoU and Dice per bone
    iou, dice = compute_iou_dice(mask_logits, thr=0.5)  # (5,), (5,)
    score_per_bone = 0.5 * (iou + dice)                 # (5,)

    # pick best bone for this frame
    best_idx = torch.argmax(score_per_bone).item()
    best_iou  = iou[best_idx].item()
    best_dice = dice[best_idx].item()
    best_score = score_per_bone[best_idx].item()

    # predicted bone class for that best_idx
    bone_logits = cls[best_idx]              # (19,)
    bone_probs  = bone_logits.softmax(dim=0)
    cls_id = int(torch.argmax(bone_probs).item())
    bone_name = CLASS_NAMES[cls_id]

    return best_score, bone_name, best_iou, best_dice


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print(f"\n[INFO] Running ENERGY + SAM2Rad on: {DICOM_PATH}")

    # ---------------------------------------------------------
    # 1) Get bone_intervals from energy_intervals.py
    # ---------------------------------------------------------
    # NOTE: energy_intervals.py itself prints for its own hardcoded dicom
    #       when imported. Ignore that; THIS call uses our DICOM_PATH.
    bar_mask_final, bar_intervals, bone_intervals = detect_bar1_regions_debug(DICOM_PATH)

    print("\n[INFO] Detected Bone Intervals (from energy_intervals):")
    for i, (s, e) in enumerate(bone_intervals, 1):
        print(f"  Bone {i}: {s}–{e}")

    # ---------------------------------------------------------
    # 2) Load DICOM frames
    # ---------------------------------------------------------
    ds = pydicom.dcmread(DICOM_PATH)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]   # (N,H,W)
    N, H, W = arr.shape

    # ---------------------------------------------------------
    # 3) Load SAM2Rad model
    # ---------------------------------------------------------
    model = load_model()

    final_results = []

    # ---------------------------------------------------------
    # 4) LOOP THROUGH BONE INTERVALS
    # ---------------------------------------------------------
    for (s, e) in bone_intervals:
        best_f = None
        best_score = -1.0
        best_bone = None

        print(f"\n=== Interval {s}–{e} ===")

        # frame indices in DICOM are 0-based in arr, 1-based in intervals
        for f in range(s, e + 1):
            idx = f - 1
            if idx < 0 or idx >= N:
                print(f"  [WARN] Frame {f} out of range, skipping.")
                continue

            frame = arr[idx]

            score, bone, iou, dice = run_frame(model, frame)

            print(
                f"Interval {s:3d}–{e:3d} | Frame {f:4d} | "
                f"IoU={iou:.3f} | Dice={dice:.3f} | Score={score:.3f} | Bone={bone}"
            )

            if score > best_score:
                best_score = score
                best_f = f
                best_bone = bone

        final_results.append((s, e, best_f, best_bone, best_score))

    # ---------------------------------------------------------
    # 5) FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n==============================")
    print(" FINAL SELECTION SUMMARY")
    print("==============================\n")

    for (s, e, bf, bname, sc) in final_results:
        print(f"Interval {s}–{e} → Best Frame {bf} → Bone: {bname} (Score={sc:.3f})")


if __name__ == "__main__":
    main()