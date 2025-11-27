#read_dicom_sam.py
import os
import torch
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

from sam2rad_model import SAM2RadModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# PATHS
# -----------------------
SAM_CKPT = "/home/ds/Desktop/sam_hand/sam_hand/sam_vit_h_4b8939.pth"
RAD_CKPT = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad/checkpoints/best_model.pth"


# -----------------------
# Load SAM2Rad model
# -----------------------
def load_model():

    print("Loading SAM2Rad checkpoint...")

    # Build model
    model = SAM2RadModel(
        sam_checkpoint=SAM_CKPT,
        lora_rank=8
    )

    # Load trained weights (PPN + LoRA)
    state = torch.load(RAD_CKPT, map_location="cpu")

    if "model" in state:
        state = state["model"]

    # Remove DataParallel prefixes
    clean_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print("Loaded. Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()
    model.to(DEVICE)
    return model


# -----------------------
# Load specific DICOM frame
# -----------------------
def load_dicom_frame(dicom_path, frame_number):
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array  # (F,H,W)

    frame = arr[frame_number]

    # Normalize and convert to uint8
    frame = frame.astype(np.float32)
    frame -= frame.min()
    frame /= frame.max()
    frame = (frame * 255).astype(np.uint8)

    return frame


# -----------------------
# Preprocess for SAM2Rad
# -----------------------
def preprocess_frame(frame):

    # Resize to 1024×1024
    frame_resized = cv2.resize(frame, (1024, 1024))

    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)

    tensor = to_tensor(rgb).unsqueeze(0)  # [1,3,H,W]
    return tensor.to(DEVICE), rgb


# -----------------------
# Inference + Visualization
# -----------------------
def run_inference(model, dicom_path, frame_number):

    frame = load_dicom_frame(dicom_path, frame_number)
    tensor, rgb = preprocess_frame(frame)

    with torch.no_grad():
        out = model(tensor)

    # Pred mask (256×256)
    pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()

    # Upscale to 1024×1024
    pmask_big = cv2.resize(pmask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # PPN Points
    coords = out["ppn_coords"][0].cpu().numpy()
    H, W = 1024, 1024
    px = (coords[:, 0] * W).astype(int)
    py = (coords[:, 1] * H).astype(int)

    # Overlay mask
    overlay = np.zeros((1024, 1024, 3), dtype=np.float32)
    overlay[..., 1] = pmask_big  # green
    overlay = (overlay * 255).astype(np.uint8)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(rgb, cmap="gray")
    ax[0].scatter(px, py, c="yellow", s=40)
    ax[0].set_title(f"Frame {frame_number}")

    ax[1].imshow(pmask_big, cmap="gray")
    ax[1].set_title("Predicted Mask")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")

    for a in ax:
        a.axis("off")

    out_path = f"output_frame_{frame_number:04d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved: {out_path}\n")


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    DICOM_PATH = "/home/ds/Desktop/Hand_dicom/H006.dcm"
    FRAME_NUM = 166

    model = load_model()
    run_inference(model, DICOM_PATH, FRAME_NUM)