import os
import json
import cv2
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor

from sam2rad_model import SAM2RadModel

# --------------------------------------
# Config
# --------------------------------------
TEST_IMG_DIR = "/home/ds/Desktop/sam_hand/sam_hand/dataset/test"
TEST_JSON    = "/home/ds/Desktop/sam_hand/sam_hand/dataset/test/US_hand_test_coco.json"
CHECKPOINT   = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad/checkpoints/best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------
# COCO utilities
# --------------------------------------
def load_coco_annotations(path):
    with open(path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    ann_map = {}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_map.setdefault(img_id, []).append(ann)

    return images, ann_map


def polygon_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    if isinstance(segmentation, list):   # polygon
        pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    else:
        from pycocotools import mask as maskUtils
        mask = maskUtils.decode(segmentation)

    return mask


# --------------------------------------
# Load trained SAM2Rad model
# --------------------------------------
def load_trained_model():
    print("Loading SAM2Rad model for evaluation...")

    model = SAM2RadModel(
        sam_checkpoint="/home/ds/Desktop/sam_hand/sam_hand/sam_vit_h_4b8939.pth",
        lora_rank=4
    )

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=True)

    model.to(DEVICE)
    model.eval()
    print("Done.\n")
    return model


# --------------------------------------
# Metrics
# --------------------------------------
def dice(pred, gt):
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    inter = (p * g).sum()
    return (2 * inter) / (p.sum() + g.sum() + 1e-6)


def iou(pred, gt):
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return inter / (union + 1e-6)


# --------------------------------------
# Evaluation loop
# --------------------------------------
def evaluate(model):

    images, ann_map = load_coco_annotations(TEST_JSON)

    # class-wise accumulators
    class_dice = {}
    class_iou  = {}
    class_count = {}

    # Loop over test images
    for img_id, info in images.items():

        fname = info["file_name"]
        img_path = os.path.join(TEST_IMG_DIR, fname)

        # extract joint name from filename before "_"
        joint_name = fname.split("_")[0]

        # prepare buckets
        class_dice.setdefault(joint_name, [])
        class_iou.setdefault(joint_name, [])
        class_count.setdefault(joint_name, 0)

        # load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        H, W = img.shape
        img_resized = cv2.resize(img, (1024, 1024))

        rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        tensor = to_tensor(rgb).unsqueeze(0).to(DEVICE)

        # load GT mask
        anns = ann_map.get(img_id, [])
        if len(anns) == 0:
            continue

        gt_mask = polygon_to_mask(anns[0]["segmentation"], H, W)
        gt_resized = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # inference
        with torch.no_grad():
            out = model(tensor)

        pred = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()
        pred = cv2.resize(pred, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # compute metrics
        d = dice(pred, gt_resized)
        j = iou(pred, gt_resized)

        class_dice[joint_name].append(d)
        class_iou[joint_name].append(j)
        class_count[joint_name] += 1

    # --------------------------------------
    # Final summary
    # --------------------------------------
    print("\n======================")
    print(" PER-JOINT METRICS")
    print("======================")
    joint_names = sorted(class_dice.keys())

    total_dice = []
    total_iou  = []

    for jn in joint_names:
        if class_count[jn] == 0:
            continue
        mean_d = np.mean(class_dice[jn])
        mean_i = np.mean(class_iou[jn])

        total_dice.append(mean_d)
        total_iou.append(mean_i)

        print(f"{jn:6s} | Dice={mean_d:.4f} | IoU={mean_i:.4f} | N={class_count[jn]}")

    print("\n======================")
    print(" OVERALL RESULTS")
    print("======================")
    print(f"Mean Dice: {np.mean(total_dice):.4f}")
    print(f"Mean IoU : {np.mean(total_iou):.4f}")
    print("======================\n")


# --------------------------------------
if __name__ == "__main__":
    model = load_trained_model()
    evaluate(model)