# data.py
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import random
import cv2


# ============================
#   Utility: grayscale → RGB
# ============================
def gray_to_rgb(img):
    return img.convert("RGB")


# ============================
#   Utility: apply augmentations
# ============================
import random
import torch
import torchvision.transforms.functional as TF

def apply_augmentations(img, masks): # followed this page for augmentation -- https://medium.com/@benjybo7/7-game-changing-data-augmentations-you-need-to-try-a3ec68a2cb62
    # ---------- 1. Horizontal flip ----------
    if random.random() < 0.5:
        img = TF.hflip(img)
        masks = masks.flip(dims=[2])

    # ---------- 2. Vertical flip (rare but helps robustness) ----------
    if random.random() < 0.1:
        img = TF.vflip(img)
        masks = masks.flip(dims=[1])

    # ---------- 3. Rotation ----------
    if random.random() < 0.5:
        angle = random.uniform(-12, 12)
        img = TF.rotate(img, angle, fill=0)
        masks = TF.rotate(masks, angle, fill=0)

    # ---------- 4. Random scaling ----------
    if random.random() < 0.4:
        scale = random.uniform(0.85, 1.15)
        h, w = img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        img = TF.resize(img, [new_h, new_w])
        masks = TF.resize(masks, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        masks = TF.center_crop(masks, [h, w])

    # ---------- 5. Gaussian blur ----------
    if random.random() < 0.3:
        k = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.2)
        img = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)

    # ---------- 6. Speckle Noise (Ultrasound style) ----------
    if random.random() < 0.4:
        noise = torch.randn_like(img) * 0.15
        img = img + img * noise
        img = torch.clamp(img, 0.0, 1.0)

    # ---------- 7. Intensity Jitter ----------
    if random.random() < 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        img = TF.adjust_gamma(img, random.uniform(0.7, 1.4))

    # Random Cutout (SpecAugment-style)
    if random.random() < 0.4:
        C, H, W = img.shape
        drop_w = int(W * random.uniform(0.05, 0.15))
        drop_h = int(H * random.uniform(0.05, 0.15))
        x0 = random.randint(0, max(1, W - drop_w))
        y0 = random.randint(0, max(1, H - drop_h))
        img[:, y0:y0+drop_h, x0:x0+drop_w] *= random.uniform(0.0, 0.3)
        masks[:, y0:y0+drop_h, x0:x0+drop_w] = 0


    return img, masks

# ==========================================
#   NEW — Compute PPN Targets from mask
# ==========================================
#def compute_ppn_targets(mask):
    """
    mask: Tensor [H,W] binary (0/1)
    Returns:
        pts_norm: (7,2) normalized to [0,1]
        labels: (7,1)  1 for FG, 0 for BG
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    H, W = mask_np.shape

    ys, xs = np.where(mask_np > 0)

    # If mask is empty → return zeros
    if len(xs) < 10:
        pts = np.zeros((7, 2), dtype=np.float32)
        labels = np.zeros((7, 1), dtype=np.float32)
        return pts, labels

    # ----- 1. Center of mass -----
    cx = xs.mean()
    cy = ys.mean()

    # ----- 2. PCA for major/minor axes -----
    coords = np.column_stack([xs, ys])
    coords_centered = coords - coords.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(coords_centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]

    # Project points
    proj1 = coords_centered @ pc1
    proj2 = coords_centered @ pc2

    idx_min1 = np.argmin(proj1)
    idx_max1 = np.argmax(proj1)
    idx_min2 = np.argmin(proj2)
    idx_max2 = np.argmax(proj2)

    major1 = coords[idx_min1]
    major2 = coords[idx_max1]
    minor1 = coords[idx_min2]
    minor2 = coords[idx_max2]

    # ----- 3. Two random background samples -----
    bg_pts = []
    bg_mask = (mask_np == 0)
    bys, bxs = np.where(bg_mask)
    if len(bxs) > 0:
        for _ in range(2):
            k = random.randint(0, len(bxs) - 1)
            bg_pts.append([bxs[k], bys[k]])
    else:
        bg_pts = [[0, 0], [W - 1, H - 1]]

    # ORDER: [center, maj1, maj2, min1, min2, bg1, bg2]
    pts = np.array([
        [cx, cy],
        major1, major2,
        minor1, minor2,
        bg_pts[0], bg_pts[1]
    ], dtype=np.float32)

    # Normalize to [0,1]
    pts_norm = pts.copy()
    pts_norm[:, 0] /= W
    pts_norm[:, 1] /= H

    # Labels: 1 for first 5 points, 0 for last 2
    labels = np.array([[1], [1], [1], [1], [1], [0], [0]], dtype=np.float32)
    return pts_norm, labels

# ============================
#        COCO Dataset
# ============================
class CocoUSHandDataset(Dataset):

    def __init__(self, image_dir, ann_path, augment=False, image_size=1024):
        self.image_dir = image_dir
        self.augment = augment
        self.image_size = image_size

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

        self.ann_map = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.ann_map:
                self.ann_map[img_id] = []
            self.ann_map[img_id].append(ann)


    def __getitem__(self, idx):
        img_info = self.images[idx]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.image_dir, file_name)

        img = Image.open(img_path).convert("L")
        img = gray_to_rgb(img)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = TF.to_tensor(img)  # [3,H,W]

        # ===========================
        # Load instance mask
        # ===========================
        img_id = img_info["id"]
        anns = self.ann_map.get(img_id, [])

        mask_list = []
        class_id_list = []              # >>> NEW: CLASS ID <<<

        for ann in anns:
            if "segmentation" in ann:
                mask = self._seg_to_mask(
                    ann["segmentation"],
                    self.image_size,
                    self.image_size,
                    orig_h=img_info["height"],
                    orig_w=img_info["width"]
                )
                mask_list.append(mask)

                #  ------------ NEW HERE ------------
                # Coco format always provides "category_id".
                cid = ann.get("category_id", None)
                if cid is None:
                    cid = -1              # safety fallback
                class_id_list.append(cid)  # store class id

        
        # -------------------------------------------------------
        # NEW: Compute 7 PPN GT points for each bone mask
        # -------------------------------------------------------
        ppn_pts_list = []
        ppn_lbl_list = []

        for mask in mask_list:
            pts_xy, pts_lbl = self._compute_ppn_points(mask.numpy())
            ppn_pts_list.append(pts_xy)   # (7,2)
            ppn_lbl_list.append(pts_lbl)  # (7,1)   

        if len(mask_list) == 0:
           #masks = torch.zeros(
           #    (1, self.image_size, self.image_size),
           #    dtype=torch.float32
           #)
           masks = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
           # ensure at least one dummy PPN target so collate doesn't break
           dummy_pts = np.zeros((7, 2), dtype=np.float32)
           dummy_lbl = np.zeros((7, 1), dtype=np.float32)
           ppn_pts_list = [dummy_pts]
           ppn_lbl_list = [dummy_lbl]
           class_id_list = [-1] # >>> NEW: ensure dummy class id <<<
        else:
            #asks = mask_list[0].unsqueeze(0).float()
           #merged = torch.sum(torch.stack(mask_list, dim=0), dim=0)
           #merged = (merged > 0).float()  # binarize
           #masks = merged.unsqueeze(0) 
           masks = torch.stack(mask_list, dim=0).float()   # keep ALL bones

        # ===========================
        # Augmentation
        # ===========================
        if self.augment:
            img, masks = apply_augmentations(img, masks)

        # ===========================
        # NEW — Compute PPN GT HERE
        # ===========================
       #fg_mask = masks[0]       # [H,W]
       #pts, lbl = compute_ppn_targets(fg_mask)

        return {
            "image": img,                     
            "masks": masks,
            "class_ids": torch.tensor(class_id_list, dtype=torch.long),  # >>> NEW <<<                   
#           "ppn_points": torch.tensor(pts, dtype=torch.float32),   # (7,2)
#           "ppn_labels": torch.tensor(lbl, dtype=torch.float32),   # (7,1)
            "ppn_points": ppn_pts_list,
            "ppn_labels": ppn_lbl_list,
            "image_id": img_id,
            "file_name": file_name
        }


    def __len__(self):
        return len(self.images)


    def _seg_to_mask(self, segmentation, h, w, orig_h, orig_w):
        mask = np.zeros((h, w), dtype=np.uint8)

        if isinstance(segmentation, list):
            pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.float32)

            scale_x = w / orig_w
            scale_y = h / orig_h

            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y

            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

        elif isinstance(segmentation, dict) and segmentation.get("counts"):
            m = self._decode_rle(segmentation)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = m.astype(np.uint8)

        return torch.from_numpy(mask)
    
    def _compute_ppn_points(self, mask):
        """
        New Phase-2 PPN GT:
        - 3 FG points: center, major_min, major_max
        - 4 BG points: top, bottom, left, right background samples
        """

        H, W = mask.shape
        ys, xs = np.where(mask > 0)

        # Mask empty → return zeros
        if len(xs) < 10:
            pts = np.zeros((7, 2), dtype=np.float32)
            lbl = np.zeros((7, 1), dtype=np.float32)
            return pts, lbl

        coords = np.column_stack([xs, ys]).astype(np.float32)

        # ---------- 1. Centroid ----------
        cx = coords[:, 0].mean()
        cy = coords[:, 1].mean()

        # ---------- 2. PCA for the major axis only ----------
        centered = coords - np.array([[cx, cy]])
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = Vt[0]  # major axis

        proj1 = centered @ pc1
        major_min = coords[np.argmin(proj1)]
        major_max = coords[np.argmax(proj1)]

        # ---------- 3. Construct 4 structured background points ----------
        bg_pts = []

        # pick ANY zero mask pixel
        bg_mask = (mask == 0)
        by, bx = np.where(bg_mask)

        if len(bx) == 0:
            # fallback
            bg_pts = [
                [0, 0],               # top-left
                [W - 1, 0],           # top-right
                [0, H - 1],           # bottom-left
                [W - 1, H - 1]        # bottom-right
            ]
        else:
            # Four deterministic BG points (no randomness)
            bg_pts = []
            k = np.argsort(bx + by)  # simple stable ordering
            sel = [
                len(k) // 8,
                len(k) // 4,
                len(k) // 2,
                3 * len(k) // 4
            ]
            for idx in sel:
                i = k[idx]
                bg_pts.append([bx[i], by[i]])

        # ---------- 4. Pack FG (3) + BG (4) ----------
        pts = np.array([
            [cx, cy],         # 0 FG center
            major_min,        # 1 FG major-
            major_max,        # 2 FG major+
            bg_pts[0],        # BG1
            bg_pts[1],        # BG2
            bg_pts[2],        # BG3
            bg_pts[3],        # BG4
        ], dtype=np.float32)

        # ---------- 5. Normalize to [0,1] ----------
        pts_norm = pts.copy()
        pts_norm[:, 0] /= W
        pts_norm[:, 1] /= H

        # ---------- 6. Labels (3 FG + 4 BG) ----------
        labels = np.array([
            [1], [1], [1],  # 3 FG
            [0], [0], [0], [0]  # 4 BG
        ], dtype=np.float32)

        return pts_norm, labels


def collate_with_ppn(batch):
    """
    batch = list of dataset items, each item is a dict.
    We flatten per-image PPN GT lists into a single tensor.
    """
    images = []
    masks = []
    ppn_points = []
    ppn_labels = []
    image_ids = []
    file_names = []
    class_ids = []  # >>> NEW <<<

    for item in batch:
        images.append(item["image"])
        masks.append(item["masks"])
        image_ids.append(item["image_id"])
        file_names.append(item["file_name"])
        class_ids.append(item["class_ids"])      # <<< FIXED LOCATION


        # item["ppn_points"] = list of (7,2)
        # flatten:
        pts = [torch.tensor(p, dtype=torch.float32) for p in item["ppn_points"]]
        lbl = [torch.tensor(l, dtype=torch.float32) for l in item["ppn_labels"]]

        pts = torch.cat(pts, dim=0)   # (7*num_bones, 2)
        lbl = torch.cat(lbl, dim=0)   # (7*num_bones, 1)

        ppn_points.append(pts)
        ppn_labels.append(lbl)

    # stack images: (B,3,H,W)
    images = torch.stack(images, dim=0)
    # -----------------------------------------------------
    # PAD MASKS so all images have the same #instances
    # -----------------------------------------------------
    max_inst = max([m.shape[0] for m in masks])   # max bones in the batch
    padded_masks = []

    for m in masks:
        inst = m.shape[0]
        if inst < max_inst:
            # pad with empty masks
            pad = torch.zeros((max_inst - inst, m.shape[1], m.shape[2]), dtype=m.dtype)
            m = torch.cat([m, pad], dim=0)
        padded_masks.append(m)

    masks = torch.stack(padded_masks, dim=0)   # (B, max_inst, H, W)

   #masks = torch.stack(masks, dim=0)

    # pad PPN to max points across batch
    max_pts = max([p.shape[0] for p in ppn_points])

    padded_pts = []
    padded_lbl = []

    for p, l in zip(ppn_points, ppn_labels):
        pad_n = max_pts - p.shape[0]
        if pad_n > 0:
            p = torch.cat([p, torch.zeros(pad_n, 2)], dim=0)
            l = torch.cat([l, torch.zeros(pad_n, 1)], dim=0)
        padded_pts.append(p)
        padded_lbl.append(l)

    ppn_points = torch.stack(padded_pts, dim=0)    # (B, max_pts, 2)
    ppn_labels = torch.stack(padded_lbl, dim=0)    # (B, max_pts, 1)

    # >>> NEW: pad class ids so shapes match across batch
    max_inst = masks.shape[1]  # already computed above

    padded_cls = []
    for c in class_ids:
        inst = c.shape[0]
        if inst < max_inst:
            pad = torch.full((max_inst - inst,), -1, dtype=torch.long)
            c = torch.cat([c, pad], dim=0)
        padded_cls.append(c)

    class_ids = torch.stack(padded_cls, dim=0)  # (B, max_inst)


    return {
        "image": images,
        "masks": masks,
        "class_ids": class_ids,
        "ppn_points": ppn_points,
        "ppn_labels": ppn_labels,
        "image_id": image_ids,
        "file_name": file_names,
    }

# ===========================
# Dataloaders
# ===========================
def build_dataloaders(cfg):
    train_set = CocoUSHandDataset(
        image_dir=cfg["data"]["train_image_dir"],
        ann_path=cfg["data"]["train_ann_path"],
        augment=True,
        image_size=cfg["data"]["image_size"],
    )

    val_set = CocoUSHandDataset(
        image_dir=cfg["data"]["val_image_dir"],
        ann_path=cfg["data"]["val_ann_path"],
        augment=False,
        image_size=cfg["data"]["image_size"],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_with_ppn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_with_ppn
    )

    return train_loader, val_loader