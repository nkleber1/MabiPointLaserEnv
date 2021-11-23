"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
This code is inspired Fei Xia's implementation (https://github.com/fxia22/pointnet.pytorch)
"""
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, args):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.mlp = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )

        if args.pooling == 'avg':
            self.pooling = nn.AvgPool1d(args.num_points)
        if args.pooling == 'max':
            self.pooling = nn.MaxPool1d(args.num_points)

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, pts):
        # encoder
        pts = pts.transpose(2, 1)
        pts = F.relu(self.bn1(self.conv1(pts)))
        pts = F.relu(self.bn1(self.conv2(pts)))
        pts = F.relu(self.bn2(self.conv3(pts)))
        pts = F.relu(self.bn3(self.conv4(pts)))

        # do global pooling
        pts = self.pooling(pts)

        pts = self.mlp(pts)
        feat = pts.transpose(2, 1)
        return feat
