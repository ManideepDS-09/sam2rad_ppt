import torch

ckpt_path = "/home/ds/Desktop/sam2rad_ppt/checkpoints_sam2rad/checkpoints/best_model.pth"
print("Loading checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n all the keys ")
print(list(ckpt.keys()))

# If checkpoint has "model", inspect that
model_dict = ckpt.get("model", ckpt)

print("\n -- all the blocks -- ")
for k in list(model_dict.keys())[:-1]:
    print(k)