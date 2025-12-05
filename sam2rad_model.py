# sam2rad_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_backbone import SamBackbone
from ppn import PPN

NUM_CLASSES = 19


class SAM2RadModel(nn.Module):
    """
    SAM backbone + PPN (7 pts per bone)
    Per-instance SAM decoding (Option-A)
    + Phase-5 pooled per-bone classifier
    """

    def __init__(self, sam_checkpoint, lora_rank=8):
        super().__init__()

        self.backbone = SamBackbone(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank
        )

        # PPN predicts 7 points × N bones
        self.ppn = PPN(in_channels=256, num_points=7)

        # ------------------------------------------------------------------
        # AUTO-DETECT ACTUAL FEATURE DIM
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.randn(1, 3, 1024, 1024)
            f = self.backbone.sam.image_encoder(dummy)
        self.feat_dim = f.shape[1]
        print("SAM feature channels =", self.feat_dim)

        self.num_classes = NUM_CLASSES

        # FINAL SINGLE CLASS HEAD
        # pooled bone embedding (B,N,feat_dim) → (B,N,classes)
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.num_classes)
        )


    def forward(self, images, num_bones):
        if images is None:
            raise RuntimeError("images tensor missing in forward")

        B, C, H, W = images.shape

        if C == 1:
            images = images.repeat(1, 3, 1, 1)

        # ===== 1) SAM encoder features =====
        feats = self.backbone.sam.image_encoder(images)   # (B,C,h,w)
        dense_pe = self.backbone.sam.prompt_encoder.get_dense_pe()

        # ===== 2) Predict 7*N PPN points =====
        ppn_out = self.ppn(feats, num_bones=num_bones)     # (B,7N,3)

        pred_coords = ppn_out[:, :, :2]                    # (B,7N,2)
        pred_logits = ppn_out[:, :, 2:]                    # (B,7N,1)
        pred_labels = (pred_logits > 0).long().squeeze(-1) # (B,7N)

        all_masks = []
        all_ious  = []

        # ===== 3) decode per-instance masks =====
        for k in range(num_bones):
            p0 = k * 7
            p1 = p0 + 7

            pts_xy  = pred_coords[:, p0:p1, :]
            pts_lbl = pred_labels[:, p0:p1]

            s_emb, d_emb = self.backbone.sam.prompt_encoder(
                points=(pts_xy, pts_lbl),
                boxes=None,
                masks=None
            )

            dec = self.backbone.sam.mask_decoder(
                image_embeddings=feats,
                image_pe=dense_pe,
                sparse_prompt_embeddings=s_emb,
                dense_prompt_embeddings=d_emb,
                multimask_output=False
            )

            if len(dec) == 2:
                low_res, iou = dec
                mask_logits = low_res
            elif len(dec) == 3:
                low_res, iou, mask_logits = dec
            else:
                raise RuntimeError(f"Unexpected SAM decoder output count: {len(dec)}")

            all_masks.append(mask_logits)   # (B,1,H,W)
            all_ious.append(iou)            # (B,1)

        # ===== 4) stack masks =====
        mask_logits = torch.stack(all_masks, dim=1).squeeze(2)   # (B,N,H,W)
        iou_pred    = torch.stack(all_ious, dim=1).squeeze(-1)   # (B,N)

        # =======================================================
        # ===== 5) Pooled local class-embedding (Phase-5) =======
        # =======================================================
        B, C, h_feat, w_feat = feats.shape

        # (B,N,H,W) → flatten BN so we can resize masks
        B, N, Hm, Wm = mask_logits.shape
        mask_flat = mask_logits.reshape(B*N, 1, Hm, Wm)

        # resize to feature map size
        mask_resized = F.interpolate(
            mask_flat,
            size=(h_feat, w_feat),
            mode='bilinear',
            align_corners=False
        ).reshape(B, N, h_feat, w_feat)  # (B,N,h_feat,w_feat)

        # pool features inside mask region
        feats_expand = feats.unsqueeze(1)  # (B,1,C,h_feat,w_feat)
        numer = (feats_expand * mask_resized.unsqueeze(2)).sum(dim=[3,4])  # (B,N,C)
        denom = mask_resized.sum(dim=[2,3], keepdim=True) + 1e-6           # (B,N,1)

        pooled = numer / denom   # (B,N,C)

        # final logits
        bone_class_logits = self.class_head(pooled)  # (B,N,19)


        print("FINAL pred_class_logits:", bone_class_logits.shape)
        print("FINAL pred_class_logits:", bone_class_logits.shape)

        # If we accidentally produced (B,N,K,19), collapse K
        if bone_class_logits.dim() == 4:
            print("WARNING: collapsing extra dimension in pred_class_logits")
            bone_class_logits = bone_class_logits.mean(dim=2)

        print("FIXED pred_class_logits:", bone_class_logits.shape)

        # ===== RETURN =====
        return {
            "ppn_out": ppn_out,
            "mask_logits": mask_logits,
            "iou_pred": iou_pred,

            "ppn_coords": pred_coords,
            "ppn_labels": pred_labels,
            "pred_point_logits": pred_logits,

            "pred_class_logits": bone_class_logits,
        }