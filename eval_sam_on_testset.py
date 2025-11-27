import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

from sam2rad_model import SAM2RadModel

# -----------------------------
# Config paths
# -----------------------------
TEST_IMG_DIR = "/home/ds/Desktop/sam_hand/sam_hand/dataset/test"
TEST_JSON = "/home/ds/Desktop/sam_hand/sam_hand/dataset/test/US_hand_test_coco.json"
CHECKPOINT = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad/checkpoints/best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------
# COCO Utilities
# --------------------------------------------------------
def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    ann_map = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    return images, ann_map


def polygon_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    if isinstance(segmentation, list):
        pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    else:
        from pycocotools import mask as maskUtils
        mask = maskUtils.decode(segmentation)

    return mask


# --------------------------------------------------------
# Load the trained model
# --------------------------------------------------------
def load_trained_model():
    print("Loading trained SAM2Rad model...")

    # Build model
    model = SAM2RadModel(
        sam_checkpoint="/home/ds/Desktop/sam_hand/sam_hand/sam_vit_h_4b8939.pth",
        lora_rank=8
    )

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    clean_state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(clean_state, strict=True)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.\n")
    return model


# --------------------------------------------------------
# Dice / IoU metrics
# --------------------------------------------------------
def dice(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    inter = (pred * gt).sum()
    return (2*inter) / (pred.sum() + gt.sum() + 1e-6)


def iou(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return inter / (union + 1e-6)


# --------------------------------------------------------
# Run model on test set
# --------------------------------------------------------
def evaluate(model):
    images, ann_map = load_coco_annotations(TEST_JSON)

    dice_scores = []
    iou_scores = []

    for img_id, img_info in images.items():

        fname = img_info["file_name"]
        path = os.path.join(TEST_IMG_DIR, fname)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠ Could not read image: {path}")
            continue

        H, W = img.shape
        img_resized = cv2.resize(img, (1024, 1024))

        # Convert to tensor
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        tensor = to_tensor(rgb).unsqueeze(0).to(DEVICE)

        # GT mask
        anns = ann_map.get(img_id, [])
        if len(anns) == 0:
            continue

        gt_mask = polygon_to_mask(anns[0]["segmentation"], H, W)
        gt_mask_resized = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Inference
        with torch.no_grad():
            out = model(tensor)

        pred = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
        pred = cv2.resize(pred, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Metrics
        d = dice(pred, gt_mask_resized)
        j = iou(pred, gt_mask_resized)

        dice_scores.append(d)
        iou_scores.append(j)

        # -------------------------
        # Visualization save
        # -------------------------
        overlay = np.zeros((1024, 1024, 3), dtype=np.uint8)
        overlay[..., 1] = (pred * 255).astype(np.uint8)
        overlay[..., 0] = (gt_mask_resized * 255).astype(np.uint8)

        out_path = os.path.join(OUT_DIR, f"{img_id:04d}_overlay.png")

        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        ax[0].imshow(rgb, cmap="gray")
        ax[0].set_title("Image")

        ax[1].imshow(pred, cmap="gray")
        ax[1].set_title("Predicted Mask")

        ax[2].imshow(overlay)
        ax[2].set_title("GT (Red) / Pred (Green)")

        for a in ax:
            a.axis("off")

        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Processed {fname}  Dice={d:.4f}, IoU={j:.4f}")

    # -------------------------
    # Final report
    # -------------------------
    print("\n======================")
    print(" EVALUATION RESULTS")
    print("======================")
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU : {np.mean(iou_scores):.4f}")
    print("======================\n")


# --------------------------------------------------------
# Main
# --------------------------------------------------------
if __name__ == "__main__":
    model = load_trained_model()
    evaluate(model)