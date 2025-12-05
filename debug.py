#import torch

#ckpt_path = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad/checkpoints/best_model.pth"
#print("Loading checkpoint...")
#ckpt = torch.load(ckpt_path, map_location="cpu")

#print("\n all the keys ")
#print(list(ckpt.keys()))

# If checkpoint has "model", inspect that
#model_dict = ckpt.get("model", ckpt)

#print("\n -- all the blocks -- ")
#for k in list(model_dict.keys())[:-1]:
#    print(k)

#from segment_anything import sam_model_registry
#sam = sam_model_registry["vit_h"](checkpoint="/home/ds/Desktop/sam_hand/sam_hand/sam_vit_h_4b8939.pth")

#print(sam.image_encoder.blocks[0])

import json
from collections import defaultdict

# MODIFY THIS PATH
JSON_PATH = "/home/ds/Desktop/sam_hand/sam_hand/dataset/train/US_hand_train_coco.json"

def inspect_coco(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annos  = data.get("annotations", [])
    cats   = data.get("categories", [])

    print(f"\nLoaded JSON:")
    print(f"  images: {len(images)}")
    print(f"  annotations: {len(annos)}")
    print(f"  categories: {len(cats)}")

    # ---------------
    # 1) gather bone labels
    # ---------------
    bone_names = set()

    anno_by_img = defaultdict(list)
    for a in annos:
        img_id = a.get("image_id")
        anno_by_img[img_id].append(a)

        # Try the common places bone names live
        bone = None

        # A) categories ID → name lookup
        cat_id = a.get("category_id")
        if cat_id is not None:
            for c in cats:
                if c["id"] == cat_id:
                    bone = c.get("name")
                    break

        # B) region attributes (LabelMe/VIA style)
        if bone is None:
            ra = a.get("region_attributes" ,{})
            if isinstance(ra, dict):
                # guess possible keys
                for key in ["bone", "bone_name", "Bone", "label"]:
                    if key in ra:
                        bone = ra[key]
                        break

        # C) segmentation_meta
        if bone is None:
            meta = a.get("segmentation_meta", {})
            if isinstance(meta, dict):
                for key in ["bone", "bone_name", "label"]:
                    if key in meta:
                        bone = meta[key]
                        break

        if bone:
            bone_names.add(bone)

    # ---------------
    # 2) print sample entries
    # ---------------
    print("\n=== UNIQUE BONE NAMES FOUND ===")
    for b in sorted(bone_names):
        print("  ", b)

    # ---------------
    # 3) pick 1 image and dump its anno masks
    # ---------------
    if not images:
        return

    sample_img = images[0]
    img_id     = sample_img["id"]
    annos_here = anno_by_img.get(img_id, [])

    print("\n=== SAMPLE IMAGE ===")
    print("filename:", sample_img.get("file_name"))
    print("width:", sample_img.get("width"))
    print("height:", sample_img.get("height"))
    print(f"annotations found: {len(annos_here)}")

    for a in annos_here:
        cid = a.get("category_id")
        seg = a.get("segmentation", None)

        # find bone class name again
        bone = None
        for c in cats:
            if c["id"] == cid:
                bone = c.get("name")
                break

        print("\n-------------------------")
        print("annotation_id:", a.get("id"))
        print("category_id:", cid)
        print("bone_name:", bone)
        print("segmentation type:", type(seg))
        if isinstance(seg, list):
            print("polygon count:", len(seg))
        elif isinstance(seg, dict):
            print("seg keys:", seg.keys())
        else:
            print("unknown segmentation format")

        # also print attributes if present
        attrs = a.get("region_attributes")
        if attrs:
            print("region_attributes:", attrs)

        meta = a.get("segmentation_meta")
        if meta:
            print("segmentation_meta:", meta)


if __name__ == "__main__":
    inspect_coco(JSON_PATH)
