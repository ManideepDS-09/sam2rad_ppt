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
            img = batch["image"].to(device)   # (1,3,H,W)
            masks = batch["masks"].to(device)

            # ✔ pass num_bones to model
            #N=len(batch["ppn_points"]) 
            max_pts = batch["ppn_points"].shape[1]
            N = max_pts // 7
            out = core_model(img, num_bones=N)

            pmask = out["mask_logits"].sigmoid()[0, 0].cpu().numpy()

           #if masks.shape[0] > 0:
           #    gmask = masks[0].cpu().numpy()
           #    if gmask.ndim == 3:
           #        gmask = gmask[0]
           #else:
           #    gmask = np.zeros_like(pmask)
            gt_mask = (masks.sum(dim=1, keepdim=True) > 0).float()
            gmask = gt_mask[0, 0].cpu().numpy()

            # Resize pred mask to GT resolution
            H_gt, W_gt = gmask.shape
            pmask_big = cv2.resize(pmask, (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)

            # Overlay
            overlay = np.zeros((H_gt, W_gt, 3), dtype=np.float32)
            overlay[..., 0] = gmask
            overlay[..., 1] = pmask_big

            # PPN points
            coords = out["ppn_coords"][0].cpu().numpy()
            H, W = img.shape[2], img.shape[3]
            px = (coords[:, 0] * W).astype(int)
            py = (coords[:, 1] * H).astype(int)

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            ax[0].imshow(img[0].cpu().permute(1, 2, 0))
            ax[0].scatter(px, py, c="yellow", s=40)
            ax[0].set_title("Input + PPN")

            ax[1].imshow(pmask, cmap="gray")
            ax[1].set_title("Pred Mask")

            ax[2].imshow(gmask, cmap="gray")
            ax[2].set_title("GT Mask")

            ax[3].imshow(overlay)
            ax[3].set_title("Overlay (Pred=Green, GT=Red)")

            for a in ax:
                a.axis("off")

            fname = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            break

def save_loss_plot(history, out_dir):
    plt.title("Loss Breakdown Over Epochs")
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_total"], label="train_total")
    plt.plot(history["val_total"], label="val_total")
    plt.plot(history["dice"], label="dice")
    plt.plot(history["iou"], label="iou")
    plt.plot(history["train_mask"], label="train_mask")
    plt.plot(history["val_mask"], label="val_mask")
    plt.plot(history["train_coord"], label="train_coord")
    plt.plot(history["val_coord"], label="val_coord")
    plt.plot(history["train_label"], label="train_label")
    plt.plot(history["val_label"], label="val_label")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_plot.png"), dpi=150)
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
    sample_dir = os.path.join(cfg["save"]["output_dir"], "samples")
    log_dir = os.path.join(cfg["save"]["output_dir"], "logs")
    mkdir(ckpt_dir); mkdir(sample_dir); mkdir(log_dir)

    # ----------------------------
    # Early stopping
    # ----------------------------
    best_val = float("inf")
    patience = cfg["training"].get("patience", 10)
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
            gt_mask = (masks.sum(dim=1, keepdim=True) > 0).float()


            # -----------------------------------------------------
            # ✔ MULTI-BONE PPN GT PREPARATION
            # -----------------------------------------------------
           #ppn_list = batch["ppn_points"]         # list of N tensors (7,2)
           #lbl_list = batch["ppn_labels"]         # list of N tensors (7,1)
           #N = len(ppn_list)                      # number of bones

            # flatten: (N,7,2) → (N*7,2)
           #gt_pts = torch.tensor(ppn_list).reshape(-1, 2).float().to(device)
           #gt_labels = torch.tensor(lbl_list).reshape(-1, 1).float().to(device)

            gt_pts = batch["ppn_points"].to(device)
            gt_labels = batch["ppn_labels"].to(device)   # (B,max_pts,1)
            # number of bones = max_pts / 7
            max_pts = gt_pts.shape[1]
            num_bones = max_pts // 7

            optimizer.zero_grad()

            # ✔ pass number of bones to PPN
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

            loss_dict = criterion(
                pred_mask_logits,
                gt_mask,
                pred_pts,
                gt_pts,
                pred_logits,
                gt_labels,
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
                gt_mask = (masks.sum(dim=1, keepdim=True) > 0).float()

                # ✔ MULTI-BONE GT for validation
                #pn_list = batch["ppn_points"]
                #bl_list = batch["ppn_labels"]
                # = len(ppn_list)

               #gt_pts = torch.tensor(ppn_list).reshape(-1, 2).float().to(device)
               #gt_labels = torch.tensor(lbl_list).reshape(-1, 1).float().to(device)

                gt_pts = batch["ppn_points"].to(device)
                gt_labels = batch["ppn_labels"].to(device)
                max_pts = gt_pts.shape[1]
                num_bones = max_pts // 7
                # ✔ pass num_bones
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

                loss_dict = criterion(
                    pmask,
                    gt_mask,
                    pred_pts,
                    gt_pts,
                    pred_logits,
                    gt_labels,
                )
                mask_loss = loss_dict["mask_loss"].item()
                coord_loss = loss_dict["coord_loss"].item()
                label_loss = loss_dict["label_loss"].item()

                mask_val_total += mask_loss
                coord_val_total += coord_loss
                label_val_total += label_loss


                val_total += loss_dict["total_loss"].item()
                val_dice += dice_score(pmask, gt_mask).item()
                val_iou += iou_score(pmask, gt_mask).item()
                count += 1

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
        if val_total < best_val:
            best_val = val_total
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
        else:
            no_improve += 1

        if no_improve >= patience:
            print("\nEarly stopping triggered.")
            break

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