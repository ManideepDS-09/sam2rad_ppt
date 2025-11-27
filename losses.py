# losses.py
# Loss functions for SAM2Rad:
#   - Dice + Focal mask loss
#   - PPN coordinate regression loss (SmoothL1)
#   - PPN label classification loss (BCEWithLogits)

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
# PPN Coordinate Regression Loss (SmoothL1)
# DYNAMIC: supports variable #points (7*N)
# --------------------------------------------------
class PPNCoordLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_points, gt_points):
        """
        pred_points: (B, 7*N, 2)
        gt_points:   (7*N, 2) or (B, 7*N, 2)

        We flatten so shapes always match.
        """

        B = pred_points.shape[0]

        pred = pred_points.reshape(B, -1, 2)     # (B,7N,2)
        gt = gt_points.reshape(B, -1, 2)         # (B,7N,2)

        return self.loss_fn(pred, gt)


# --------------------------------------------------
# PPN Label Loss (BCEWithLogits)
# DYNAMIC: supports variable #points (7*N)
# --------------------------------------------------
class PPNLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, gt_labels):
        """
        pred_logits: (B, 7*N, 1)
        gt_labels:   (7*N, 1) or (B, 7*N, 1)
        """

        B = pred_logits.shape[0]

        pred = pred_logits.reshape(B, -1)        # (B,7N)
        gt = gt_labels.reshape(B, -1)            # (B,7N)

        return self.loss_fn(pred, gt)


# --------------------------------------------------
# Total SAM2Rad Loss Wrapper
# --------------------------------------------------
class SAM2RadLoss(nn.Module):
    def __init__(self,
                 mask_weight=1.0,
                 coord_weight=0.3,
                 label_weight=0.1):
        super().__init__()

        self.mask_loss = MaskLoss()
        self.coord_loss = PPNCoordLoss()
        self.label_loss = PPNLabelLoss()

        self.w_mask = mask_weight
        self.w_coord = coord_weight
        self.w_label = label_weight

    def forward(self,
                pred_mask_logits,
                gt_mask,
                pred_points_xy,
                gt_points_xy,
                pred_point_logits,
                gt_point_labels):

        # --------- mask loss ----------
        L_mask = self.mask_loss(pred_mask_logits, gt_mask)

        # --------- point coord loss ----
        L_coord = self.coord_loss(pred_points_xy, gt_points_xy)

        # --------- label classification loss ----
        L_label = self.label_loss(pred_point_logits, gt_point_labels)

        L_total = (
            self.w_mask * L_mask +
            self.w_coord * L_coord +
            self.w_label * L_label
        )

        return {
            "total_loss": L_total,
            "mask_loss": L_mask,
            "coord_loss": L_coord,
            "label_loss": L_label
        }