#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: model.py
@Time: 2020/3/23 5:39 PM
"""

import torch.nn as nn
from . import GraphEncoder, PointNet2Encoder, PointNetEncoder, FoldDecoder
from .loss import ChamferLoss
# TODO Make encoder/decoder model selectable


class ReconstructionNet(nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        if args.encoder == 'graph':
            self.encoder = GraphEncoder(args)
        elif args.encoder == 'pointnet++':
            self.encoder = PointNet2Encoder(args)
        elif args.encoder == 'pointnet':
            self.encoder = PointNetEncoder(args)
        elif args.encoder == 'dense':
            pass
        if args.decoder == 'fold':
            self.decoder = FoldDecoder(args)
        elif args.decoder == 'upsampling':
            pass
        elif args.decoder == 'dense':
            pass
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