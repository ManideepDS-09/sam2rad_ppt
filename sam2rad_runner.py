# sam2rad_runner.py
import torch
import numpy as np
import pydicom
import cv2

from sam2rad_model import SAM2RadModel
from OR_auto_calculator import compute_OR_from_components, MIN_COMP_AREA

# same CLASS_MAP you used in read_dicom_sam.py
CLASS_MAP = {
    0: "PD4", 1: "PD3", 2: "PD2", 3: "PD5",
    4: "PM5", 5: "PM4", 6: "PM3", 7: "PM2",
    8: "PP5", 9: "MC5", 10: "PP4", 11: "MC4",
    12: "PP3", 13: "MC3", 14: "PP2", 15: "MC2",
    16: "PD1", 17: "PP1", 18: "or",
}

SAM_CKPT = "/home/ds/Desktop/sam_hand/sam_hand/sam_vit_h_4b8939.pth"
RAD_CKPT = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad/checkpoints/epoch_0009.pth"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_GLOBAL_MODEL = None


def _load_model_once():
    global _GLOBAL_MODEL

    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL

    model = SAM2RadModel(
        sam_checkpoint=SAM_CKPT,
        lora_rank=4
    )
    state = torch.load(RAD_CKPT, map_location="cpu")
    if "model" in state:
        state = state["model"]

    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.eval()
    model.to(_DEVICE)

    _GLOBAL_MODEL = model
    return _GLOBAL_MODEL


def _load_frame_uint8(dicom_path, frame_idx):
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    frame = arr[frame_idx]
    frame = frame.astype(np.float32)
    frame -= frame.min()
    frame /= max(frame.max(), 1e-5)
    return (frame * 255).astype(np.uint8)


def _preprocess_1024(frame_u8):
    f = cv2.resize(frame_u8, (1024, 1024))
    rgb = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return t.unsqueeze(0).to(_DEVICE), rgb


@torch.no_grad()
def run_sam2rad_on_frame(dicom_path, frame_idx):
    model = _load_model_once()

    frame = _load_frame_uint8(dicom_path, frame_idx)
    tensor, rgb = _preprocess_1024(frame)

    # infer bones count via PPN
    out = model(tensor, num_bones=5)
    pts = out["ppn_coords"]
    max_pts = pts.shape[1]
    num_bones = max(max_pts // 7, 1)

    # refine inference with correct bone count
    out = model(tensor, num_bones=num_bones)

    # --- mask
    pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
    pmask_big = cv2.resize(pmask, (1024,1024), interpolation=cv2.INTER_NEAREST)

    # --- predicted classes
    pred = out["pred_class_logits"][0]  # (num_bones, 19)
    pred_ids = pred.argmax(-1).cpu().numpy()

    bone_names = []
    for pid in pred_ids:
        bone_names.append(CLASS_MAP.get(int(pid), "UNKNOWN"))

    # --- connected components
    thr = 0.55
    bin_mask = (pmask_big > thr).astype(np.uint8)
    num_labels, lab = cv2.connectedComponents(bin_mask)

    comps = []
    for cid in range(1, num_labels):
        comp = (lab == cid).astype(np.uint8)
        if int(comp.sum()) >= MIN_COMP_AREA:
            comps.append(comp)

    # TRUE bone count
    n_bones = len(comps)

    # OR calculation
    or_out = compute_OR_from_components(
        comps=comps,
        rgb=rgb,
        out_path=None
    )

    return {
        "best_frame": frame_idx,
        "n_bones": n_bones,
        "bone_names": bone_names,
        "or1": or_out.get("OR1", None),
        "or2": or_out.get("OR2", None),
        "coords": out["ppn_coords"][0].cpu().numpy(),
    }