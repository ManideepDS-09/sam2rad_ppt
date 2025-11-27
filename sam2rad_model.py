# sam2rad_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_backbone import SamBackbone
from ppn import PPN


class SAM2RadModel(nn.Module):
    """
    Combined model:
        SAM backbone (ViT-H + LoRA)
        PPN (7-point predictor)
    """

    def __init__(self, sam_checkpoint, lora_rank=8):
        super().__init__()

        # SAM backbone with LoRA
        self.backbone = SamBackbone(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank
        )

        # PPN module — takes image embeddings from SAM
        self.ppn = PPN(in_channels=256, num_points=7)


    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    # images: [B,1,H,W] grayscale
    # gt_ppn (optional): dict containing keys:
    #    coords: [B,7,2]
    #    labels: [B,7]
    # ------------------------------------------------------------
    def forward(self, images, *, num_bones=1, gt_ppn=None):
        if images is None:
            raise RuntimeError("Model received no images tensor — DataParallel arg mismatch.")
        """
        Returns dictionary with:
            A) paper-style outputs
            B) expanded debug outputs
        """

        # --------------------------
        # 1. Image encoder
        # --------------------------
        # Ensure 4D
        assert images.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {images.shape}"
        # Ensure RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        feats = self.backbone.sam.image_encoder(images)

        # --------------------------
        # 2. PPN prediction
        # --------------------------
        ppn_out = self.ppn(feats, num_bones=num_bones)  # (B, 7*num_bones, 3)

        # split ppn_out
        pred_coords = ppn_out[:, :, :2]     # (B, 7*num_bones, 2)
        pred_logits = ppn_out[:, :, 2:]     # (B, 7*num_bones, 1)
        pred_labels = (pred_logits > 0).long().squeeze(-1)   # (B, 7*num_bones)

        # --------------------------
        # 3. SAM prompt encoder
        # --------------------------
        sparse_emb, dense_emb = self.backbone.sam.prompt_encoder(
            points=(pred_coords, pred_labels),
            boxes=None,
            masks=None
        )

        # --------------------------
        # 4. Mask decoder  (SAM returns only 2 outputs)
        # --------------------------
        decoder_out = self.backbone.sam.mask_decoder(
            image_embeddings=feats,
            image_pe=self.backbone.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False
        )

        # SAM version check (your version returns 2)
        if len(decoder_out) == 2:
            low_res, iou_pred = decoder_out
            mask_logits = low_res          # fallback: use low_res as logits
        elif len(decoder_out) == 3:
            low_res, iou_pred, mask_logits = decoder_out
        else:
            raise RuntimeError(f"Unexpected number of decoder outputs: {len(decoder_out)}")

        # --------------------------
        # RETURN HYBRID OUTPUT
        # --------------------------
        return {
            # ======== A: Paper-style outputs ========
            "ppn_out": ppn_out,
            "mask_logits": mask_logits,
            "iou_pred": iou_pred,

            # ======== B: Debug outputs ========
            "ppn_coords": pred_coords,
            "ppn_labels": pred_labels,
            "low_res_masks": low_res,
            "raw_sparse_emb": sparse_emb,
            "raw_dense_emb": dense_emb,
            "image_embeddings": feats,
        }