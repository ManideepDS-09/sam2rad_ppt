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
        img_feats: SAM encoder output
            shape (B, 256, H', W')
        """

        x = F.relu(self.bn1(self.conv1(img_feats)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling
        x = self.pool(x)         # (B,256,1,1)
        x = x.squeeze(-1).squeeze(-1)  # (B,256)
        total_points = self.num_points * num_bones

        # Predict per-bone, unique 7 points
        point_xy = self.mlp_points(x)                     # (B, 14)
        point_xy = point_xy.unsqueeze(1).repeat(1, num_bones, 1)  
        point_xy = point_xy.reshape(-1, num_bones, 7, 2)  # (B, num_bones, 7, 2)
        point_xy = point_xy.reshape(-1, total_points, 2)  # (B, 7*num_bones, 2)
        point_xy = torch.sigmoid(point_xy)

        # Predict labels (foreground=1, background=0)
        point_logits = self.mlp_labels(x)        # (B, num_points)
        point_logits = point_logits.unsqueeze(1).repeat(1, num_bones, 1)
        point_logits = point_logits.view(-1, total_points, 1)

        # Concatenate coords + label logit
        """
        output[:, i] = [x_i, y_i, logit_label_i]
        """
        output = torch.cat([point_xy, point_logits], dim=-1)  # (B, 7*num_bones, 3)

        return output