#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: model.py
@Time: 2020/3/23 5:39 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .loss import ChamferLoss


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)  # (batch_size, 2, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 2)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*k, 2)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 2)

    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(
        2))  # (batch_size, num_points, 2, 1) * (batch_size, num_points, 1, 2) -> (batch_size, num_points, 2, 2)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, num_dims**2).transpose(2, 1)  # (batch_size, 4, num_points)
    x = torch.cat((pts, x), dim=1)  # (batch_size, 6, num_points)
    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)

    return x


class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 360  # input point cloud size  # TODO !!!
        self.mlp1 = nn.Sequential(
            nn.Conv1d(6, 64, 1),  # TODO 12 if 3D
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)  # (batch_size, 2, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)  # (batch_size, 2, num_points) -> (batch_size, 6, num_points])
        x = self.mlp1(x)  # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)  # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)  # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2, 1)  # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat  # (batch_size, 1, feat_dims)


class FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Decoder, self).__init__()
        self.m = args.num_points  # 360  # TODO change if 3D  --> 2025  # 45 * 45.
        self.args = args
        self.folding1 = nn.Sequential(
            nn.Conv1d(args.feat_dims + 1, args.feat_dims, 1),  # TODO +2 if 2D points
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 2, 1),  # changed to 2D
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims + 2, args.feat_dims, 1),  # TODO +2 if 2D points
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 2, 1),  # changed to 2D
        )

    def build_grid(self, batch_size):
        n_points = self.args.points
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

    # TODO make 2D
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


class ReconstructionNet(nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        self.encoder = FoldNet_Encoder(args)
        self.decoder = FoldNet_Decoder(args)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return output, feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)