# ppn.py
# Prompt Prediction Network (PPN) for SAM2Rad
# 7-point regression: 5 geometric foreground + 2 background distractors

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPN(nn.Module):
    """
    Predicts 7 keypoints for each instance:
        - 5 foreground geometric points
            (center, major-min, major-max, minor-min, minor-max)
        - 2 background distractor points
    Each point = (x, y, label)
        x,y: normalized coords 0..1
        label: 1 for FG, 0 for BG
    Output shape: (B, 7 * num_bones, 3)
    """

    def __init__(self, in_channels=256, num_points=7):
        super().__init__()

        self.num_points = num_points

        # ------------------------------
        # 3-layer lightweight CNN head
        # ------------------------------
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        # global pooled vector → (B, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------
        # MLP heads:
        #    point regression: (B, num_points*2)
        #    point labels:     (B, num_points*1)
        # ------------------------------
        self.mlp_points = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 2)   # x,y per point
        )

        self.mlp_labels = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 1)   # label logit per point
        )
        
    
    def forward(self, img_feats, num_bones=1):
        """
        img_feats: (B,256,H`,W`)
        num_bones: #instances for this image
        """

        # CNN encoder for PPN
        x = F.relu(self.bn1(self.conv1(img_feats)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global descriptor (B,256)
        x = self.pool(x).squeeze(-1).squeeze(-1)

        # -------------------------------------------------------
        # Predict UNIQUE 7 points for EACH bone
        # instead of duplicating the same 7 points N times.
        # -------------------------------------------------------
        B = x.shape[0]
        P = self.num_points     # 7

        # Expand encoder feature → per-bone embedding
        # (B, 256) → (B, num_bones, 256)
        x_rep = x.unsqueeze(1).repeat(1, num_bones, 1)

        # Flatten to feed MLP
        # (B * num_bones, 256)
        x_flat = x_rep.reshape(B * num_bones, 256)

        # Predict 7*2 coords per bone
        coords = self.mlp_points(x_flat)                 # (B*num_bones, 14)
        coords = coords.reshape(B, num_bones, P, 2)      # (B, num_bones, 7, 2)
        coords = torch.sigmoid(coords)
        coords = coords.reshape(B, num_bones * P, 2)     # (B,7*num_bones,2)

        # Predict 7 labels per bone
        labels = self.mlp_labels(x_flat)                 # (B*num_bones, 7)
        labels = labels.reshape(B, num_bones * P, 1)     # (B,7*num_bones,1)

        # Final: (B, 7*num_bones, 3)
        out = torch.cat([coords, labels], dim=-1)
        return out