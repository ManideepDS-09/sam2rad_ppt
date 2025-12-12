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

SAM_CKPT = "/home/ds/Desktop/sam_hand/sam_hand/sam_vit_b_01ec64.pth"
RAD_CKPT = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad_train_sam_b_2/checkpoints/best_model.pth"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_GLOBAL_MODEL = None
_FRAME_CACHE = None


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
    global _FRAME_CACHE

    if _FRAME_CACHE is None:
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array

        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[-1] > 1:
            arr = arr[..., 0]

        arr = arr.astype(np.float32)
        mn = arr.min()
        arr -= mn
        mx = arr.max()
        arr /= max(mx, 1e-5)
        arr = (arr * 255).astype(np.uint8)

        _FRAME_CACHE = arr

    return _FRAME_CACHE[frame_idx]


def _preprocess_1024(frame_u8, fast=False):
    size = 1024  # force always 1024 for ViT-H
    f = cv2.resize(frame_u8, (size, size))
    rgb = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(rgb.transpose(2,0,1)).float() / 255.0
    return t.unsqueeze(0).to(_DEVICE), rgb

@torch.no_grad()
def run_sam2rad_on_frame(dicom_path, frame_idx, fast_mode=False):
    model = _load_model_once()

    frame = _load_frame_uint8(dicom_path, frame_idx)
    tensor, rgb = _preprocess_1024(frame, fast=fast_mode)

    # Single forward ONLY
    out = model(tensor, num_bones=5)

    pts = out["ppn_coords"]
    num_bones = max(pts.shape[1] // 7, 1)

    if fast_mode:
        # only PPN + class confidence
        pred = out["pred_class_logits"][0]
        probs = pred.softmax(dim=-1)
        conf = float(probs.max().cpu().item())

        return {
            "n_bones": int(num_bones),
            "score": conf,
        }

    # ---------- FULL MODE ----------
    # second forward pass with refined count
    out = model(tensor, num_bones=num_bones)

    mask_logits = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
    pmask = cv2.resize(mask_logits, (1024,1024), interpolation=cv2.INTER_NEAREST)

    pred = out["pred_class_logits"][0]
    pred_ids = pred.argmax(-1).cpu().numpy()

    bone_names = [CLASS_MAP.get(int(pid), "UNK") for pid in pred_ids]

    bin_mask = (pmask > 0.55).astype(np.uint8)
    num_labels, lab = cv2.connectedComponents(bin_mask)

    comps = []
    for cid in range(1, num_labels):
        comp = (lab == cid).astype(np.uint8)
        if int(comp.sum()) >= MIN_COMP_AREA:
            comps.append(comp)

    or_out = compute_OR_from_components(comps=comps, rgb=rgb, out_path=None)

    return {
        "n_bones": len(comps),
        "bone_names": bone_names,
        "or1": or_out.get("OR1"),
        "or2": or_out.get("OR2"),
        "coords": out["ppn_coords"][0].cpu().numpy(),
    }