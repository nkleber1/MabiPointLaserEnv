import argparse

import torch.nn as nn
import torch.nn.functional as F

from point_clouds.embedding.foldingnet.dataset import PointCloudDataset


class DenseEncoder(nn.Module):
    def __init__(self, args):
        super(DenseEncoder, self).__init__()
        self.num_points = args.num_points
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=8, kernel_size=1)
        self.fc1 = nn.Linear(in_features=8 * args.num_points, out_features=4 * args.num_points)
        self.fc2 = nn.Linear(in_features=4 * args.num_points, out_features=1024)

        self.mlp = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4 * args.num_points)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, pts):
        # encoder
        pts = pts.transpose(2, 1)
        pts = F.relu(self.bn1(self.conv1(pts)))
        pts = F.relu(self.bn2(self.conv2(pts)))
        pts = F.relu(self.bn3(self.conv3(pts)))
        pts = pts.view(-1, self.num_points * 8)
        pts = F.relu(self.bn4(self.fc1(pts)))
        pts = F.relu(self.bn5(self.fc2(pts)))
        pts = pts.unsqueeze(2)
        pts = self.mlp(pts)
        feat = pts.transpose(2, 1)
        return feat


# def get_parser():
#     parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
#     parser.add_argument('--num_points', type=int, default=1024,
#                         help='Num of points to use')
#     parser.add_argument('--feat_dims', type=int, default=128, metavar='N',
#                         help='Number of dims for feature ')
#     args = parser.parse_args()
#     return args
#
# args = get_parser()
# data = PointCloudDataset('uniform_density')[:5]
# print(data.shape)
# encoder = DenseEncoder(args)
# feat = encoder(data)
# print(feat.shape)

