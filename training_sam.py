# training_sam.py
import os
import argparse
import yaml
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import build_dataloaders
from losses import SAM2RadLoss
from sam2rad_model import SAM2RadModel
import cv2
import pandas as pd


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


# ================================================================
# SAVE SAMPLE VISUALIZATION
# ================================================================
def save_sample_visual(model, val_loader, device, out_dir, epoch):
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    core_model.eval()

    with torch.no_grad():
        for batch in val_loader:

            img = batch["image"].to(device)
            masks = batch["masks"].to(device)

            # GT points
            gt_pts = batch["ppn_points"][0].cpu().numpy()   # (max_pts, 2)
            gt_labels = batch["ppn_labels"][0].cpu().numpy()

            # number of bones = max_pts / 7
            max_pts = gt_pts.shape[0]
            N = max_pts // 7

            out = core_model(img, num_bones=N)

            # predicted mask
            pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()

            gt_mask = masks.float()   # (B, N, H, W)
            gmask = gt_mask[0, 0].cpu().numpy()

            # resize pred mask
            H_gt, W_gt = gmask.shape
            pmask_big = cv2.resize(pmask, (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)

            # predicted PPN coords
            pred_coords = out["ppn_coords"][0].cpu().numpy()  # (7*N,2)

            # map to pixel coords
            H, W = img.shape[2], img.shape[3]
            px_pred = (pred_coords[:, 0] * W).astype(int)
            py_pred = (pred_coords[:, 1] * H).astype(int)

            # GT mapped to pixel coords
            px_gt = (gt_pts[:, 0] * W).astype(int)
            py_gt = (gt_pts[:, 1] * H).astype(int)

            # overlay masks
            overlay = np.zeros((H_gt, W_gt, 3), dtype=np.float32)
            overlay[..., 0] = gmask      # red = GT
            overlay[..., 1] = pmask_big  # green = predicted

            fig, ax = plt.subplots(1, 4, figsize=(22, 6))

            # ---------------------------------------------------
            # 1. Input image with GT red points + Pred yellow points
            # ---------------------------------------------------
            ax[0].imshow(img[0].cpu().permute(1, 2, 0))
            ax[0].scatter(px_pred, py_pred, c="yellow", s=40, label="Pred")
            ax[0].scatter(px_gt, py_gt, c="red", s=40, label="GT")
            ax[0].legend(loc="lower right")
            ax[0].set_title("Input + PPN (Pred=Yellow, GT=Red)")

            # ---------------------------------------------------
            # 2. predicted mask
            # ---------------------------------------------------
            ax[1].imshow(pmask, cmap="gray")
            ax[1].set_title("Pred Mask")

            # ---------------------------------------------------
            # 3. GT mask
            # ---------------------------------------------------
            ax[2].imshow(gmask, cmap="gray")
            ax[2].set_title("GT Mask")

            # ---------------------------------------------------
            # 4. Overlay
            # ---------------------------------------------------
            ax[3].imshow(overlay)
            ax[3].set_title("Overlay (GT=Red, Pred=Green)")

            for a in ax:
                a.axis("off")

            fname = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            break

def save_loss_plot(history, out_dir):

    # ---------------------------
    # 1. Total Loss (train + val)
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_total"], label="train_total")
    plt.plot(history["val_total"], label="val_total")
    plt.grid(True)
    plt.legend()
    plt.title("Total Loss")
    plt.savefig(os.path.join(out_dir, "loss_total.png"), dpi=150)
    plt.close()

    # ---------------------------
    # 2. Mask Loss
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_mask"], label="train_mask")
    plt.plot(history["val_mask"], label="val_mask")
    plt.grid(True)
    plt.legend()
    plt.title("Mask Loss")
    plt.savefig(os.path.join(out_dir, "loss_mask.png"), dpi=150)
    plt.close()

    # ---------------------------
    # 3. PPN Coord Loss
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_coord"], label="train_coord")
    plt.plot(history["val_coord"], label="val_coord")
    plt.grid(True)
    plt.legend()
    plt.title("PPN Coord Loss")
    plt.savefig(os.path.join(out_dir, "loss_coord.png"), dpi=150)
    plt.close()

    # ---------------------------
    # 4. PPN Label Loss
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_label"], label="train_label")
    plt.plot(history["val_label"], label="val_label")
    plt.grid(True)
    plt.legend()
    plt.title("PPN Label Loss")
    plt.savefig(os.path.join(out_dir, "loss_label.png"), dpi=150)
    plt.close()

    # ---------------------------
    # 5. Metrics: Dice + IoU
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(history["dice"], label="dice")
    plt.plot(history["iou"], label="iou")
    plt.grid(True)
    plt.legend()
    plt.title("Dice & IoU")
    plt.savefig(os.path.join(out_dir, "metrics_dice_iou.png"), dpi=150)
    plt.close()

# ================================================================
# TRAINING LOOP
# ================================================================
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Total training images: {len(train_loader.dataset)}")
    print(f"Total validation images: {len(val_loader.dataset)}")

    model = SAM2RadModel(
        sam_checkpoint=cfg["model"]["sam_checkpoint"],
        lora_rank=cfg["model"]["lora_rank"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(" Using DataParallel across GPUs:", torch.cuda.device_count())
        model = nn.DataParallel(model)

    criterion = SAM2RadLoss()

    # ----------------------------
    # Optimizer / LR Groups
    # ----------------------------
    lora_params = []
    decoder_params = []
    ppn_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            name_low = name.lower()

            if "lora" in name_low:
                lora_params.append(p)
            elif "ppn" in name_low:
                ppn_params.append(p)
            else:
                decoder_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": lora_params, "lr": cfg["training"]["lr"]["lora"]},
            {"params": decoder_params, "lr": cfg["training"]["lr"]["mask_decoder"]},
            {"params": ppn_params, "lr": cfg["training"]["lr"]["ppn"]},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["num_epochs"])

    # ----------------------------
    # Folders
    # ----------------------------
    ckpt_dir = os.path.join(cfg["save"]["output_dir"], "checkpoints")
    print("[DEBUG] checkpoint directory:", ckpt_dir)
    print("[DEBUG] directory exists?", os.path.exists(ckpt_dir))
    sample_dir = os.path.join(cfg["save"]["output_dir"], "samples")
    log_dir = os.path.join(cfg["save"]["output_dir"], "logs")
    mkdir(ckpt_dir); mkdir(sample_dir); mkdir(log_dir)

    # ----------------------------
    # Early stopping
    # ----------------------------
    best_val = float("inf")
    patience = cfg["training"].get("patience", 12)
    no_improve = 0

    history = {
        "train_total": [], "val_total": [],
        "train_mask": [], "val_mask": [],
        "train_coord": [], "val_coord": [],
        "train_label": [], "val_label": [],
        "dice": [], "iou": [],
    }

    # ============================================================
    # EPOCH LOOP
    # ============================================================
    for epoch in range(1, cfg["training"]["num_epochs"] + 1):

        # ============================
        # TRAINING
        # ============================
        model.train()
        running_total = 0
        mask_total = 0
        coord_total = 0
        label_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['num_epochs']}")

        for batch in pbar:
            img = batch["image"].to(device)
            masks = batch["masks"].to(device)
           # merge all masks into a single binary mask
            # masks: (B, num_inst, H, W)
            gt_mask = masks.float()   # (B, N, H, W)


            # -----------------------------------------------------
            #  MULTI-BONE PPN GT PREPARATION
            # -----------------------------------------------------
           #ppn_list = batch["ppn_points"]         # list of N tensors (7,2)
           #lbl_list = batch["ppn_labels"]         # list of N tensors (7,1)
           #N = len(ppn_list)                      # number of bones flatten: (N,7,2) → (N*7,2)
           #gt_pts = torch.tensor(ppn_list).reshape(-1, 2).float().to(device)
           #gt_labels = torch.tensor(lbl_list).reshape(-1, 1).float().to(device)

            gt_pts = batch["ppn_points"].to(device)
            gt_labels = batch["ppn_labels"].to(device)   # (B,max_pts,1)
            # number of bones = max_pts / 7
            max_pts = gt_pts.shape[1]
            num_bones = max_pts // 7

            optimizer.zero_grad()

            # pass number of bones to PPN
            out = model(img, num_bones=num_bones)

            
            pred_mask_logits = out["mask_logits"]
            pred_pts = out["ppn_coords"]
            pred_logits = out["ppn_out"][:, :, 2:]

            # Resize mask to GT resolution
            pred_mask_logits = torch.nn.functional.interpolate(
                pred_mask_logits,
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            print("pred_class_logits shape:", out["pred_class_logits"].shape)
            print("gt_class_ids shape:", batch["class_ids"].shape)
            # --------------------------
            # MAP COCO class_ids → [0..18]
            # keep -1 untouched
            # --------------------------
            gt_class_ids = batch["class_ids"].to(device)
            gt_class_ids = torch.where(
                gt_class_ids >= 0,
                gt_class_ids - 1,
                gt_class_ids
            )

            # Debug
            print("gt_class_ids (final):", torch.unique(gt_class_ids))

            loss_dict = criterion(
                pred_mask_logits,
                gt_mask,
                pred_pts,
                gt_pts,
                pred_logits,
                gt_labels,
                out["pred_class_logits"],   # NEW
                gt_class_ids,               # FIXED
            )

            mask_loss = loss_dict["mask_loss"].item()
            coord_loss = loss_dict["coord_loss"].item()
            label_loss = loss_dict["label_loss"].item()

            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            mask_total += mask_loss
            coord_total += coord_loss
            label_total += label_loss

            running_total += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}",
                             "mask": f"{mask_loss:.3f}",
                             "coord": f"{coord_loss:.3f}",
                             "label": f"{label_loss:.3f}"})

        train_total = running_total / len(train_loader)

        # ============================
        # VALIDATION
        # ============================
        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        eval_model.eval()
        val_total = 0
        val_dice = 0
        val_iou = 0
        count = 0
        mask_val_total = 0
        coord_val_total = 0
        label_val_total = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Val {epoch}")
            for batch in pbar_val:
                img = batch["image"].to(device)
                masks = batch["masks"].to(device)
                gt_mask = masks.float()   # (B, N, H, W)

                # MULTI-BONE GT for validation
                #pn_list = batch["ppn_points"]
                #bl_list = batch["ppn_labels"]
                # = len(ppn_list)

               #gt_pts = torch.tensor(ppn_list).reshape(-1, 2).float().to(device)
               #gt_labels = torch.tensor(lbl_list).reshape(-1, 1).float().to(device)

                gt_pts = batch["ppn_points"].to(device)
                gt_labels = batch["ppn_labels"].to(device)
                max_pts = gt_pts.shape[1]
                num_bones = max_pts // 7
                # pass num_bones
                out = eval_model(img, num_bones=num_bones)

                pred_mask_logits = out["mask_logits"]
                pred_mask_logits = torch.nn.functional.interpolate(
                    pred_mask_logits,
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                pmask = pred_mask_logits.sigmoid()

                pred_pts = out["ppn_coords"]
                pred_logits = out["ppn_out"][:, :, 2:]

                gt_class_ids = batch["class_ids"].to(device)
                gt_class_ids = torch.where(
                    gt_class_ids >= 0,
                    gt_class_ids - 1,
                    gt_class_ids
                )

                loss_dict = criterion(
                    pred_mask_logits,
                    gt_mask,
                    pred_pts,
                    gt_pts,
                    pred_logits,
                    gt_labels,
                    out["pred_class_logits"],
                    gt_class_ids,
                )

                mask_loss = loss_dict["mask_loss"].item()
                coord_loss = loss_dict["coord_loss"].item()
                label_loss = loss_dict["label_loss"].item()

                mask_val_total += mask_loss
                coord_val_total += coord_loss
                label_val_total += label_loss


                #val_total += loss_dict["total_loss"].item()
                total_val_loss = loss_dict["total_loss"]
                if not torch.isnan(total_val_loss):
                    val_total += total_val_loss.item()
                    count += 1
                else:
                    print("Warning -> skipping NAN validation batch") 
                val_dice += dice_score(pmask, gt_mask).item()
                val_iou += iou_score(pmask, gt_mask).item()
                #count += 1

        #val_total /= count
        if count == 0:
            val_total = float("inf")
        else:
            val_total /= count
        val_dice /= count
        val_iou /= count

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["dice"].append(val_dice)
        history["iou"].append(val_iou)
        history["train_mask"].append(mask_total / len(train_loader))
        history["train_coord"].append(coord_total / len(train_loader))
        history["train_label"].append(label_total / len(train_loader))
        history["val_mask"].append(mask_val_total / count)
        history["val_coord"].append(coord_val_total / count)
        history["val_label"].append(label_val_total / count)


        # ============================
        # PRINT EPOCH SUMMARY
        # ============================
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"\nEpoch {epoch}: LR={lrs} | "
            f"Train={train_total:.4f} | "
            f"Val={val_total:.4f} | "
            f"Dice={val_dice:.4f} | IoU={val_iou:.4f}\n"
        )
        print(
            f"Mask: Train={history['train_mask'][-1]:.4f}, Val={history['val_mask'][-1]:.4f} | "
            f"Coord: Train={history['train_coord'][-1]:.4f}, Val={history['val_coord'][-1]:.4f} | "
            f"Label: Train={history['train_label'][-1]:.4f}, Val={history['val_label'][-1]:.4f}"
        )

        # ============================
        # EARLY STOPPING
        # ============================
        print("\n[DEBUG CHECKPOINT]")
        print("val_total =", val_total, "   (type:", type(val_total), ")")
        print("best_val BEFORE =", best_val)
        print("val_total < best_val ?", val_total < best_val)
        print("is NaN?", torch.isnan(torch.tensor(val_total)))
        print("ckpt_dir =", ckpt_dir)
        print("------------")

        if val_total < best_val:
            best_val = val_total
            no_improve = 0
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            print("[DEBUG] ENTERED CHECKPOINT SAVE")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict()
                },
                ckpt_path
            )
            print("[DEBUG] best_val UPDATED TO:", best_val)
        else:
            no_improve += 1

        if no_improve >= patience:
            print("\nEarly stopping triggered.")
            break

        # ----------------------------------
        # LR adjust for SAM (LoRA parameters)
        # ----------------------------------
        if epoch == 10:
            for pg in optimizer.param_groups:
                # the LoRA group is group index 0
                if pg["lr"] == cfg["training"]["lr"]["lora"]:
                    old_lr = pg["lr"]
                    pg["lr"] = old_lr * 0.33
                    print(f"\n[LR UPDATE] Reduced LoRA LR from {old_lr} → {pg['lr']}\n")

        scheduler.step()

        # Save sample visualization
        save_sample_visual(model, val_loader, device, sample_dir, epoch)

        # Save loss & metrics plot
        save_loss_plot(history, log_dir)
        #pd.DataFrame(history).to_csv(os.path.join(log_dir, "loss_metrics.csv"), index=False)
        df = pd.DataFrame([{
            "epoch": epoch,
            "train_total": train_total,
            "val_total": val_total,
            "train_mask": history["train_mask"][-1],
            "val_mask": history["val_mask"][-1],
            "train_coord": history["train_coord"][-1],
            "val_coord": history["val_coord"][-1],
            "train_label": history["train_label"][-1],
            "val_label": history["val_label"][-1],
            "dice": val_dice,
            "iou": val_iou
        }])
        csv_path = os.path.join(log_dir, "loss_metrics.csv")
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

        # Save last epoch checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
        print("Saved:", ckpt_path)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sam2rad.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()