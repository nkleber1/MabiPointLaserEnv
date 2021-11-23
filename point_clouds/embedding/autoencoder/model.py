import torch.nn as nn
from . import GraphEncoder, PointNet2Encoder, PointNetEncoder, DenseEncoder, FoldDecoder, DenseDecoder
from .loss import ChamferLoss


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
            self.encoder = DenseEncoder(args)
        if args.decoder == 'fold':
            self.decoder = FoldDecoder(args)
        elif args.decoder == 'upsampling':
            pass
        elif args.decoder == 'dense':
            self.decoder = DenseDecoder(args)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return output, feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        return self.loss(input, output)