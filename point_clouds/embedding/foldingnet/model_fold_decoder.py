import torch
import torch.nn as nn
import numpy as np
import math
# TODO Make usable for all 1D, 2D, and 3D


class Fold_Decoder(nn.Module):
    def __init__(self, args):
        super(Fold_Decoder, self).__init__()
        self.m = args.num_points  # 360  # TODO change if 3D  --> 2025  # 45 * 45.
        self.args = args
        if args.shape == '1d':
            point_dim = 1
        else:  # eventually add case for 3D
            point_dim = 2
        self.folding1 = nn.Sequential(
            nn.Conv1d(args.feat_dims + point_dim, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 2, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims + 2, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 2, 1),  # changed to 2D
        )

    def build_grid(self, batch_size):
        n_points = self.args.num_points
        if self.args.shape == '1d':
            points = np.linspace(0, 1, n_points)
            points = points[np.newaxis, ...]
        elif self.args.shape == 'diagonal':
            xy = np.linspace(0, 1, n_points)
            points = np.vstack((xy, xy))
        elif self.args.shape == 'circle':
            x = list()
            y = list()
            for i in range(n_points):
                i = 2 / n_points * i * math.pi
                sin = math.sin(i) / 2 + 0.5
                cos = math.cos(i) / 2 + 0.5
                x.append(sin)
                y.append(cos)
            points = np.array([x, y])
        elif self.args.shape == 'square':
            n_points = n_points / 4
            x = np.linspace(0, 1, n_points + 2)
            x = x[1:n_points+1]
            x0 = np.zeros(n_points)
            x1 = np.ones(n_points)
            e0 = np.vstack((x0, x))
            e1 = np.vstack((x, x1))
            x = np.flip(x)
            e2 = np.vstack((x1, x))
            e3 = np.vstack((x, x0))
            points = np.hstack((e0, e1, e2, e3))
        elif self.args.shape == 'gaussian':
            x = np.random.normal(loc=0.5, scale=0.2, size=n_points * 2)
            points = np.reshape(x, (2, -1))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)  # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0])   # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points),
                         dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)  # (batch_size, 514, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)
