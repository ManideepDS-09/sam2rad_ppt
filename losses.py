# losses.py
# Phase-5 Losses:
#   1) Dice + Focal mask loss
#   2) SmoothL1 point regression
#   3) BCE point label classification
#   4) NEW: Bone-Class Cross Entropy loss

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Dice Loss (for binary masks)
# --------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        num = 2 * (pred * target).sum(dim=[1,2,3])
        den = pred.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3]) + self.eps

        dice = 1 - (num / den)
        return dice.mean()


# --------------------------------------------------
# Focal Loss (binary)
# --------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)

        p_t = prob * target + (1 - prob) * (1 - target)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce

        return focal.mean()


# --------------------------------------------------
# Combined Mask Loss = Dice + Focal
# --------------------------------------------------
class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, pred_mask_logits, gt_mask):
        return self.dice(pred_mask_logits, gt_mask) + self.focal(pred_mask_logits, gt_mask)


# --------------------------------------------------
# SmoothL1 for PPN coordinates
# pred: (B,7N,2)
# gt:   (B,7N,2)
# --------------------------------------------------
class PPNCoordLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_points, gt_points):
        B = pred_points.shape[0]
        pred = pred_points.reshape(B, -1, 2)
        gt = gt_points.reshape(B, -1, 2)
        return self.loss_fn(pred, gt)


# --------------------------------------------------
# BCE for PPN point-labels
# pred_logits: (B,7N,1)
# gt_labels:   (B,7N,1)
# --------------------------------------------------
class PPNLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, gt_labels):
        B = pred_logits.shape[0]
        pred = pred_logits.reshape(B, -1)
        gt   = gt_labels.reshape(B, -1)
        return self.loss_fn(pred, gt)


# --------------------------------------------------
# NEW: Class CrossEntropy per-bone
#
# pred_class_logits: (B,N,19)
# gt_class_ids:      (B,N)
#
# Supports ignored index -1
# --------------------------------------------------
class BoneClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pred_logits, gt_class_ids):
        """
        pred_logits: (..., num_classes)
        gt_class_ids: same leading shape as pred_logits except last dim
                      (e.g. (B, N), (B, N, K), etc.)
        """
        # last dim is num_classes
        num_classes = pred_logits.shape[-1]

        # DEBUG (keep for now, you can comment later)
        print("BoneClassLoss pred_logits.shape:", pred_logits.shape)
        print("BoneClassLoss gt_class_ids.shape:", gt_class_ids.shape)
        print("\n--- BoneClassLoss DEBUG ---")
        print("pred_logits.shape:", pred_logits.shape)
        print("gt_class_ids.shape:", gt_class_ids.shape)
        print("pred unique:", torch.unique(pred_logits.argmax(-1)))
        print("gt unique:", torch.unique(gt_class_ids))
        print("---------------------------\n")


        # flatten everything except class dimension
        logits_flat = pred_logits.reshape(-1, num_classes)   # (M, num_classes)
        labels_flat = gt_class_ids.reshape(-1)               # (M,)

        return self.loss(logits_flat, labels_flat)


# --------------------------------------------------
# Total Phase-5 Loss
# --------------------------------------------------
class SAM2RadLoss(nn.Module):
    def __init__(self,
                 mask_weight=1.0,
                 coord_weight=0.3,
                 label_weight=0.3,
                 class_weight=0.5):   # NEW class loss weight
        super().__init__()

        self.mask_loss  = MaskLoss()
        self.coord_loss = PPNCoordLoss()
        self.label_loss = PPNLabelLoss()
        self.class_loss = BoneClassLoss()

        self.w_mask  = mask_weight
        self.w_coord = coord_weight
        self.w_label = label_weight
        self.w_class = class_weight     # NEW


    def forward(self,
                pred_mask_logits,
                gt_mask,
                pred_points_xy,
                gt_points_xy,
                pred_point_logits,
                gt_point_labels,
                pred_class_logits,       # NEW
                gt_class_ids):           # NEW

        # 1) segmentation loss
        L_mask  = self.mask_loss(pred_mask_logits, gt_mask)

        # 2) keypoint regression
        L_coord = self.coord_loss(pred_points_xy, gt_points_xy)

        # 3) point label classification
        L_label = self.label_loss(pred_point_logits, gt_point_labels)

        # 4) bone-class prediction loss
        L_class = self.class_loss(pred_class_logits, gt_class_ids)

        # total
        L_total = (
            self.w_mask  * L_mask  +
            self.w_coord * L_coord +
            self.w_label * L_label +
            self.w_class * L_class
        )

        return {
            "total_loss": L_total,
            "mask_loss":  L_mask,
            "coord_loss": L_coord,
            "label_loss": L_label,
            "class_loss": L_class
        }