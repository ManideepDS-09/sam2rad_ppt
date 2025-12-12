# sam_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry


# ==============================
#   LoRA Linear Layer
# ==============================
class LoRALinear(nn.Module):
    """
    For replacing Linear layers (qkv in SAM ViT blocks).
    W + BA where W is frozen.
    """
    def __init__(self, linear, rank=8, scale=1.0):
        super().__init__()
        self.linear = linear  # Frozen base Linear
        self.rank = rank
        self.scale = scale

        in_dim = linear.in_features
        out_dim = linear.out_features

        # LoRA parameters
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=0.1)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale


# ============================================================
#      Inject LoRA ONLY into qkv inside image_encoder blocks
# ============================================================
def insert_lora_into_sam(model, rank=8):
    """
    Only modify qkv inside ViT blocks.
    patch_embed.proj stays untouched (fixes the dimension explosion).
    """
    for blk in model.image_encoder.blocks:
        old_qkv = blk.attn.qkv
        blk.attn.qkv = LoRALinear(old_qkv, rank=rank)

    return model


# ============================================================
#      Freeze encoder except LoRA modules
# ============================================================
def freeze_image_encoder(model):
    for name, p in model.named_parameters():
        if name.startswith("image_encoder"):
            p.requires_grad = False

    # Re-enable LoRA trainable parameters
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


# ============================================================
#               Wrapper Backbone for SAM
# ============================================================
class SamBackbone(nn.Module):
    def __init__(self, sam_checkpoint, lora_rank=8):
        super().__init__()

        # Load original SAM-ViT-H
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)

        # Add LoRA to qkv
        insert_lora_into_sam(self.sam, rank=lora_rank)

        # Freeze encoder except LoRA params
        freeze_image_encoder(self.sam)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, images, points=None):

        # SAM expects 3-channel; dataset already converts.
        # images: [B,3,H,W]

        # 1. Image embeddings
        image_embeddings = self.sam.image_encoder(images)

        # 2. Prompt encoder
        if points is None:
            sparse, dense = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )
        else:
            sparse, dense = self.sam.prompt_encoder(
                points=(points["coords"], points["labels"]),
                boxes=None, masks=None
            )

        # 3. Mask decoder
        low_res_masks, iou_pred, mask_logits = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )

        return {
            "low_res_masks": low_res_masks,
            "mask_logits": mask_logits,
            "iou_pred": iou_pred,
            "image_embeddings": image_embeddings
        }