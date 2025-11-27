# debug_ppn_gt.py
import matplotlib.pyplot as plt
import torch
import numpy as np
from data import CocoUSHandDataset   # your dataset
import yaml


def visualize_ppn_points(img, mask, pts, labels, save_path=None):
    """
    img: [3,H,W] torch tensor 0-1
    mask: [H,W] binary
    pts: (7,2) normalized coords
    labels: (7,1)
    """
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    H, W = mask_np.shape

    # Convert normalized → pixel coords
    xs = (pts[:, 0] * W).astype(int)
    ys = (pts[:, 1] * H).astype(int)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # --- 1. Raw image ---
    ax[0].imshow(img_np)
    ax[0].set_title("Input Image")
    ax[0].scatter(xs, ys, c="yellow", s=40)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax[0].text(x, y, f"{i}", color="red")

    ax[0].axis("off")

    # --- 2. GT mask ---
    ax[1].imshow(mask_np, cmap="gray")
    ax[1].set_title("GT Mask")
    ax[1].axis("off")

    # --- 3. Overlay ---
    overlay = img_np.copy()
    overlay[..., 1] += mask_np * 0.5  # green overlay for mask
    overlay = np.clip(overlay, 0, 1)

    ax[2].imshow(overlay)
    ax[2].set_title("Image + GT Mask + PPN Points")
    ax[2].scatter(xs, ys, c="yellow", s=40)

    for i, (x, y) in enumerate(zip(xs, ys)):
        lbl = int(labels[i][0])
        ax[2].text(x, y, f"{i}:{lbl}", color="red")

    ax[2].axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150)
        print("Saved:", save_path)

    plt.show()


def main():
    # load config
    with open("configs/sam2rad.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset = CocoUSHandDataset(
        image_dir=cfg["data"]["train_image_dir"],
        ann_path=cfg["data"]["train_ann_path"],
        augment=False,
        image_size=cfg["data"]["image_size"],
    )

    print("Dataset loaded. Total images:", len(dataset))

    idx = 0  # choose which image
    sample = dataset[idx]

    img = sample["image"]
    masks = sample["masks"]          # [N,H,W]
    ppn_pts_list = sample["ppn_points"]   # list of (7,2)
    ppn_lbl_list = sample["ppn_labels"]   # list of (7,1)

    print(f"Image {idx} has {len(masks)} bones.")

    # loop over each bone
    for b in range(len(masks)):
        pts = np.array(ppn_pts_list[b])
        labels = np.array(ppn_lbl_list[b])
        gmask = masks[b]

        print(f"\nBone {b}:")
        print("PPN Points:\n", pts)
        print("Labels:\n", labels)

        save_name = f"debug_ppn_bone_{b}.png"

        visualize_ppn_points(
            img,
            gmask,
            pts,
            labels,
            save_path=save_name
        )


if __name__ == "__main__":
    main()