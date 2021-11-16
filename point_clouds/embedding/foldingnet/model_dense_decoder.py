import argparse
import torch.nn as nn
import torch.nn.functional as F
from point_clouds.embedding.foldingnet import DenseEncoder
from point_clouds.embedding.foldingnet.dataset import PointCloudDataset


class DenseDecoder(nn.Module):
    def __init__(self, args):
        super(DenseDecoder, self).__init__()
        self.m = args.num_points
        self.fc1 = nn.Linear(in_features=args.feat_dims, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=int(args.num_points/2))
        self.fc3 = nn.Linear(in_features=int(args.num_points/2), out_features=args.num_points)
        self.fc4 = nn.Linear(in_features=args.num_points, out_features=int(args.num_points * 2))

        # batch norm
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(int(args.num_points/2))
        self.bn2 = nn.BatchNorm1d(args.num_points)

    def forward(self, feat):
        B, _, _ = feat.shape
        # decoder
        feat = feat.squeeze()
        pts = F.relu(self.bn0(self.fc1(feat)))
        pts = F.relu(self.bn1(self.fc2(pts)))
        pts = F.relu(self.bn2(self.fc3(pts)))
        pts = self.fc4(pts)

        # do reshaping
        return pts.reshape(B, -1, self.m)


# def get_parser():
#     parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
#     parser.add_argument('--num_points', type=int, default=1024,
#                         help='Num of points to use')
#     parser.add_argument('--feat_dims', type=int, default=128, metavar='N',
#                         help='Number of dims for feature ')
#     args = parser.parse_args()
#     return args
#
#
# args = get_parser()
# data = PointCloudDataset('uniform_density')[:5]
# print(data.shape)
# encoder = DenseEncoder(args)
# decoder = DenseDecoder(args)
# feat = encoder(data)
# print(feat.shape)
# re = decoder(feat)
# print(re.shape)

