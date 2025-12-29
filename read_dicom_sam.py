# read_dicom_sam.py
import os
import torch
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

from sam2rad_model import SAM2RadModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from OR_auto_calculator import (
    compute_OR_from_components,
    MIN_COMP_AREA
)
def get_subject_name(dicom_path):
    return os.path.splitext(os.path.basename(dicom_path))[0]

# -------------------------------------------------
# CLASS MAP (must match training)
# -------------------------------------------------
CLASS_MAP = {
    0: "PD4",
    1: "PD3",
    2: "PD2",
    3: "PD5",
    4: "PM5",
    5: "PM4",
    6: "PM3",
    7: "PM2",
    8: "PP5",
    9: "MC5",
    10: "PP4",
    11: "MC4",
    12: "PP3",
    13: "MC3",
    14: "PP2",
    15: "MC2",
    16: "PD1",
    17: "PP1",
    18: "or",
}

# -------------------------------------------------
# PATHS
# -------------------------------------------------
SAM_CKPT = "/home/ds/Desktop/sam_hand/sam_hand/sam_vit_b_01ec64.pth"
RAD_CKPT = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad_train_sam_b_2/checkpoints/best_model.pth"


# -------------------------------------------------
# Load model
# -------------------------------------------------
def load_model():
    print("Loading SAM2Rad checkpoint...")

    model = SAM2RadModel(
        sam_checkpoint=SAM_CKPT,
        lora_rank=4
    )

    state = torch.load(RAD_CKPT, map_location="cpu")
    if "model" in state:
        state = state["model"]

    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print(f"Loaded checkpoint.")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    print("Model ready.\n")

    model.eval()
    model.to(DEVICE)
    return model


# -------------------------------------------------
# DICOM Frame Loader
# -------------------------------------------------
def load_dicom_frame(dicom_path, frame_idx):
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    frame = arr[frame_idx]

    frame = frame.astype(np.float32)
    frame -= frame.min()
    frame /= max(frame.max(), 1e-5)
    frame = (frame * 255).astype(np.uint8)
    return frame


# -------------------------------------------------
# Preprocess for SAM2Rad
# -------------------------------------------------
def preprocess_frame(frame_uint8):
    frame_resized = cv2.resize(frame_uint8, (1024, 1024))
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
    return to_tensor(rgb).unsqueeze(0).to(DEVICE), rgb


# -------------------------------------------------
# Inference + OR
# -------------------------------------------------
#def run_inference(model, dicom_path, frame_idx, save_images=True):

    frame = load_dicom_frame(dicom_path, frame_idx)
    tensor, rgb = preprocess_frame(frame)

    with torch.no_grad():
        out = model(tensor, num_bones=5)
        pts = out["ppn_coords"]
        max_pts = pts.shape[1]         # (7*N)
        num_bones = max_pts // 7
        if num_bones < 1:
            num_bones = 1              # safety
        out = model(tensor, num_bones=num_bones)
    print(f"[INFERENCE] num_bones={num_bones}")

    # logits 256×256
    pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
    # -----------------------------------------------------
    # Phase-5: predicted bone class IDs
    # -----------------------------------------------------
    # shape (1, num_bones, num_classes) -> (num_bones,)
    pred_logits = out["pred_class_logits"][0]  # (num_bones, 19)
    pred_ids = pred_logits.argmax(-1).cpu().numpy()

    # filter invalid IDs
    pred_ids = [pid if (0 <= pid < len(CLASS_MAP)) else -1 for pid in pred_ids]

    bone_names = []
    for pid in pred_ids:
        if pid == -1:
            bone_names.append("UNKNOWN")
        else:
            bone_names.append(CLASS_MAP[pid])

    print("\n===== CLASS PREDICTION DEBUG =====")
    print("pred_ids:", pred_ids)
    print("bone_names:", bone_names)
    print("==================================\n")


    # upscale to 1024
    pmask_big = cv2.resize(
        pmask, (1024, 1024),
        interpolation=cv2.INTER_NEAREST
    )

    # ============================
    # Prepare overlay at full size
    # ============================
    overlay = np.zeros((1024, 1024, 3), dtype=np.uint8)
    overlay[..., 1] = pmask_big  # green channel shows mask
    overlay = (overlay * 255).astype(np.uint8)


    # threshold (keep same threshold you chose earlier)
    #bin_mask = (pmask_big > 0.5).astype(np.uint8)
    #for thr in [0.25, 0.3, 0.35, 0.45, 0.55]:
    for thr in [0.65]:
        bin_mask = (pmask_big > thr).astype(np.uint8)
        num,_ = cv2.connectedComponents(bin_mask)
        print("threshold",thr,"→ components:",num-1)


    # connected components
    num_labels, lab = cv2.connectedComponents(bin_mask)

    print("\n================== MASK DEBUG ==================")
    print(f"Raw components from connectedComponents: {num_labels - 1}")

    comps = []
    for cid in range(1, num_labels):
        comp = (lab == cid).astype(np.uint8)
        area = int(comp.sum())

        print(f" - comp {cid}: area={area}")

        if area >= MIN_COMP_AREA:
            print("   -> accepted as bone")
            comps.append(comp)
        else:
            print("   -> rejected (small area)")

        # ------------------------------------------
        # Predict centroid and annotate bone name
        # ------------------------------------------
        ys, xs = np.where(comp > 0)
        if len(xs) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
        else:
            cx, cy = 0, 0

        # bone index to read class for:
        comp_idx = cid - 1  # cid=1 -> index0, cid=2 -> index1 ...

        label_txt = bone_names[comp_idx] if comp_idx < len(bone_names) else "?"
        cv2.putText(
            overlay,  # use same overlay where yellow points appear OR define below
            label_txt,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                # scale
            (0, 255, 255),      # yellow
            6,                  # thickness
            cv2.LINE_AA
        )

        print(f"   -> predicted name: {label_txt} @ ({cx},{cy})")


    print(f"\nVALID bones passed to OR module: {len(comps)}")
    print("================================================\n")

    # OR calc (unchanged)
    or_out = compute_OR_from_components(
        comps=comps,
        rgb=rgb,
        out_path=f"output_or_1.png" if save_images else None
    )

    # PPN points for visualization
    coords = out["ppn_coords"][0].cpu().numpy()
    px = (coords[:, 0] * 1024).astype(int)
    py = (coords[:, 1] * 1024).astype(int)

    # prepare a blank RGB overlay at DICOM resolution
    overlay = np.zeros((bin_mask.shape[0], bin_mask.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = pmask_big
    overlay = (overlay * 255).astype(np.uint8)

    # Save only if enabled
    if save_images:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(rgb, cmap="gray")
        ax[0].scatter(px, py, c="yellow", s=40)
        ax[0].set_title(f"Frame {frame_idx}")

        ax[1].imshow(pmask_big, cmap="gray")
        ax[1].set_title("Predicted Mask")

        ax[2].imshow(overlay)
        ax[2].set_title("Overlay")

        for a in ax:
            a.axis("off")

        out_path = f"output_frame_2.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Image saved: {out_path}")

        if or_out and save_images:
            print(f"OR Debug saved as output_or.png")

    print("\n======== OR RESULTS ========")
    for k, v in or_out.items():
        print(f"{k}: {v}")
    print("============================\n")

    return or_out

#def run_inference(model, dicom_path, frame_idx, save_images=True):
    print("\n=============== SAM2Rad RUN ===============")
    print(f"Running OR on EXACT frame index: {frame_idx}")
    print("===========================================\n")

    # -------------------------------------------------
    # LOAD ONLY ONE FRAME
    # -------------------------------------------------
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    frame = arr[frame_idx].astype(np.uint8)   # <-- EXACTLY 1 FRAME

    # normalize
    f = frame.astype(np.float32)
    f = (f - f.min()) / max(f.max() - f.min(), 1e-6)
    f = (f * 255).astype(np.uint8)

    # resize + to RGB
    resized = cv2.resize(f, (1024, 1024))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    tensor = to_tensor(rgb).unsqueeze(0).to(DEVICE)

    # -------------------------------------------------
    # INFERENCE
    # -------------------------------------------------
    with torch.no_grad():
        # initial forward to guess number of bones
        out = model(tensor, num_bones=5)
        pts = out["ppn_coords"]
        max_pts = pts.shape[1]
        nb = max(1, max_pts // 7)

        # final forward
        out = model(tensor, num_bones=nb)

    # -------------------------------------------------
    # MASK + COMPONENTS
    # -------------------------------------------------
    pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
    pmask_big = cv2.resize(pmask, (1024,1024), interpolation=cv2.INTER_NEAREST)

    bin_mask = (pmask_big > 0.5).astype(np.uint8)
    num_labels, lab = cv2.connectedComponents(bin_mask)

    comps = []
    for cid in range(1, num_labels):
        comp = (lab == cid).astype(np.uint8)
        if comp.sum() >= MIN_COMP_AREA:
            comps.append(comp)

    # -------------------------------------------------
    # OR calculation
    # -------------------------------------------------
    or_res = compute_OR_from_components(comps, rgb, None)

    # attach bone count
    or_res["bone_count"] = len(comps)

    print("\n===== FINAL OR RESULT =====")
    print(or_res)
    print("===========================\n")

    return or_res
def run_inference(model, dicom_path, frame_idx, save_images=True):

    subject = get_subject_name(dicom_path)

    print("\n=============== SAM2Rad RUN ===============")
    print(f"Subject: {subject}")
    print(f"Frame: {frame_idx}")
    print("===========================================\n")

    # -------------------------------------------------
    # LOAD FRAME
    # -------------------------------------------------
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    frame = arr[frame_idx].astype(np.uint8)

    # normalize
    f = frame.astype(np.float32)
    f = (f - f.min()) / max(f.max() - f.min(), 1e-6)
    f = (f * 255).astype(np.uint8)

    resized = cv2.resize(f, (1024, 1024))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    tensor = to_tensor(rgb).unsqueeze(0).to(DEVICE)

    # -------------------------------------------------
    # INFERENCE
    # -------------------------------------------------
    with torch.no_grad():
        out = model(tensor, num_bones=5)
        max_pts = out["ppn_coords"].shape[1]
        nb = max(1, max_pts // 7)
        out = model(tensor, num_bones=nb)

    # -------------------------------------------------
    # MASK + COMPONENTS
    # -------------------------------------------------
    pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
    pmask_big = cv2.resize(pmask, (1024,1024), interpolation=cv2.INTER_NEAREST)

    bin_mask = (pmask_big > 0.5).astype(np.uint8)
    num_labels, lab = cv2.connectedComponents(bin_mask)

    comps = []
    for cid in range(1, num_labels):
        comp = (lab == cid).astype(np.uint8)
        if comp.sum() >= MIN_COMP_AREA:
            comps.append(comp)

    # -------------------------------------------------
    # OR calculation + SAVE OR IMAGE
    # -------------------------------------------------
    or_img_path = None
    if save_images:
        or_img_path = f"{subject}_frame_{frame_idx}_or.png"

    or_res = compute_OR_from_components(
        comps=comps,
        rgb=rgb,
        out_path=or_img_path
    )

    # -------------------------------------------------
    # SAVE DEBUG IMAGE
    # -------------------------------------------------
    if save_images:
        debug_path = f"{subject}_frame_{frame_idx}.png"

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(rgb, cmap="gray")
        ax[0].set_title("Input Frame")

        ax[1].imshow(pmask_big, cmap="gray")
        ax[1].set_title("Predicted Mask")

        for a in ax:
            a.axis("off")

        plt.savefig(debug_path, dpi=150)
        plt.close()

        print(f"[SAVED] {debug_path}")
        if or_img_path:
            print(f"[SAVED] {or_img_path}")

    or_res["bone_count"] = len(comps)
    return or_res

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    DICOM_PATH = "/home/ds/Desktop/Hand_dicom/K02.dcm"
    FRAME     =  78

    model = load_model()
    run_inference(model, DICOM_PATH, FRAME, save_images=True)