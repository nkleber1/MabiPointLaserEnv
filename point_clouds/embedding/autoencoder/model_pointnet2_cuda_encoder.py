import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from typing import List, Optional, Tuple
import pytorch_lightning as pl

# try:
#     import pointnet2_ops._ext as _ext
# except ImportError:
#     from torch.utils.cpp_extension import load
#     import glob
#     import os.path as osp
#     import os
#
#     warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")
#
#     _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
#     _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
#         osp.join(_ext_src_root, "src", "*.cu")
#     )
#     _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))
#
#     os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
#     _ext = load(
#         "_ext",
#         sources=_ext_sources,
#         extra_include_paths=[osp.join(_ext_src_root, "include")],
#         extra_cflags=["-O3"],
#         extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
#         with_cuda=True,
#     )
#
# lr_clip = 1e-5
# bnm_clip = 1e-2
#
# class BallQuery(Function):
#     @staticmethod
#     def forward(ctx, radius, nsample, xyz, new_xyz):
#         # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
#         r"""
#         Parameters
#         ----------
#         radius : float
#             radius of the balls
#         nsample : int
#             maximum number of features in the balls
#         xyz : torch.Tensor
#             (B, N, 3) xyz coordinates of the features
#         new_xyz : torch.Tensor
#             (B, npoint, 3) centers of the ball query
#         Returns
#         -------
#         torch.Tensor
#             (B, npoint, nsample) tensor with the indicies of the features that form the query balls
#         """
#         output = _ext.ball_query(new_xyz, xyz, radius, nsample)
#
#         ctx.mark_non_differentiable(output)
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         return ()
#
#
# ball_query = BallQuery.apply
#
#
# class GroupingOperation(Function):
#     @staticmethod
#     def forward(ctx, features, idx):
#         # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
#         r"""
#         Parameters
#         ----------
#         features : torch.Tensor
#             (B, C, N) tensor of features to group
#         idx : torch.Tensor
#             (B, npoint, nsample) tensor containing the indicies of features to group with
#         Returns
#         -------
#         torch.Tensor
#             (B, C, npoint, nsample) tensor
#         """
#         ctx.save_for_backward(idx, features)
#
#         return _ext.group_points(features, idx)
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
#         r"""
#         Parameters
#         ----------
#         grad_out : torch.Tensor
#             (B, C, npoint, nsample) tensor of the gradients of the output from forward
#         Returns
#         -------
#         torch.Tensor
#             (B, C, N) gradient of the features
#         None
#         """
#         idx, features = ctx.saved_tensors
#         N = features.size(2)
#
#         grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
#
#         return grad_features, torch.zeros_like(idx)
#
#
# grouping_operation = GroupingOperation.apply
#
#
# class QueryAndGroup(nn.Module):
#     r"""
#     Groups with a ball query of radius
#     Parameters
#     ---------
#     radius : float32
#         Radius of ball
#     nsample : int32
#         Maximum number of features to gather in the ball
#     """
#
#     def __init__(self, radius, nsample, use_xyz=True):
#         # type: (QueryAndGroup, float, int, bool) -> None
#         super(QueryAndGroup, self).__init__()
#         self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
#
#     def forward(self, xyz, new_xyz, features=None):
#         # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
#         r"""
#         Parameters
#         ----------
#         xyz : torch.Tensor
#             xyz coordinates of the features (B, N, 3)
#         new_xyz : torch.Tensor
#             centriods (B, npoint, 3)
#         features : torch.Tensor
#             Descriptors of the features (B, C, N)
#         Returns
#         -------
#         new_features : torch.Tensor
#             (B, 3 + C, npoint, nsample) tensor
#         """
#
#         idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
#         xyz_trans = xyz.transpose(1, 2).contiguous()
#         grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
#         grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
#
#         if features is not None:
#             grouped_features = grouping_operation(features, idx)
#             if self.use_xyz:
#                 new_features = torch.cat(
#                     [grouped_xyz, grouped_features], dim=1
#                 )  # (B, C + 3, npoint, nsample)
#             else:
#                 new_features = grouped_features
#         else:
#             assert (
#                 self.use_xyz
#             ), "Cannot have not features and not use xyz as a feature!"
#             new_features = grouped_xyz
#
#         return new_features
#
#
# class GroupAll(nn.Module):
#     r"""
#     Groups all features
#     Parameters
#     ---------
#     """
#
#     def __init__(self, use_xyz=True):
#         # type: (GroupAll, bool) -> None
#         super(GroupAll, self).__init__()
#         self.use_xyz = use_xyz
#
#     def forward(self, xyz, new_xyz, features=None):
#         # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
#         r"""
#         Parameters
#         ----------
#         xyz : torch.Tensor
#             xyz coordinates of the features (B, N, 3)
#         new_xyz : torch.Tensor
#             Ignored
#         features : torch.Tensor
#             Descriptors of the features (B, C, N)
#         Returns
#         -------
#         new_features : torch.Tensor
#             (B, C + 3, 1, N) tensor
#         """
#
#         grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
#         if features is not None:
#             grouped_features = features.unsqueeze(2)
#             if self.use_xyz:
#                 new_features = torch.cat(
#                     [grouped_xyz, grouped_features], dim=1
#                 )  # (B, 3 + C, 1, N)
#             else:
#                 new_features = grouped_features
#         else:
#             new_features = grouped_xyz
#
#         return new_features
#
#
# class GatherOperation(Function):
#     @staticmethod
#     def forward(ctx, features, idx):
#         # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
#         r"""
#         Parameters
#         ----------
#         features : torch.Tensor
#             (B, C, N) tensor
#         idx : torch.Tensor
#             (B, npoint) tensor of the features to gather
#         Returns
#         -------
#         torch.Tensor
#             (B, C, npoint) tensor
#         """
#
#         ctx.save_for_backward(idx, features)
#
#         return _ext.gather_points(features, idx)
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         idx, features = ctx.saved_tensors
#         N = features.size(2)
#
#         grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
#         return grad_features, None
#
#
# gather_operation = GatherOperation.apply
#
#
# class FurthestPointSampling(Function):
#     @staticmethod
#     def forward(ctx, xyz, npoint):
#         # type: (Any, torch.Tensor, int) -> torch.Tensor
#         r"""
#         Uses iterative furthest point sampling to select a set of npoint features that have the largest
#         minimum distance
#         Parameters
#         ----------
#         xyz : torch.Tensor
#             (B, N, 3) tensor where N > npoint
#         npoint : int32
#             number of features in the sampled set
#         Returns
#         -------
#         torch.Tensor
#             (B, npoint) tensor containing the set
#         """
#         out = _ext.furthest_point_sampling(xyz, npoint)
#
#         ctx.mark_non_differentiable(out)
#
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         return ()
#
#
# furthest_point_sample = FurthestPointSampling.apply
#
#


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class PointnetSAModule(nn.Module):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    n_point : int
        Number of features
    radius : float
        Radius of ball
    n_sample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, n_point=None, radius=None, n_sample=None, bn=True, use_xyz=True):  # TODO what does use_xyz doe
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__()

        self.n_point = n_point
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        self.groupers.append(
            pointnet2_utils.QueryAndGroup(radius, n_sample, use_xyz=use_xyz)
            if n_point is not None
            else pointnet2_utils.GroupAll(use_xyz)
        )
        if use_xyz:
            mlp[0] += 3
        self.mlps.append(build_shared_mlp(mlp, bn))

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2ClassificationSSG(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModule(
                n_point=512,
                radius=0.2,
                n_sample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],  # TODO use args
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                n_point=128,
                radius=0.4,
                n_sample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.hparams["model.use_xyz"]
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

    def forward(self, pointcloud):
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))


model = PointNet2ClassificationSSG(None)

