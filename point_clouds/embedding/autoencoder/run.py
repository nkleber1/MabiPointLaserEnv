"""
This code is based on the implementation of An Tao (ta19@mails.tsinghua.edu.cn)
"""

import argparse
from point_clouds.embedding.autoencoder.reconstruction import Reconstruction


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--exp_name', type=str, default='graph_1d_fold_uniform_k16', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--encoder', type=str, default='graph', metavar='N',
                        choices=['graph', 'pointnet++', 'pointnet', 'dense'],
                        help='Encoder architecture used, [graph, pointnet++, pointnet, dense]')
    parser.add_argument('--decoder', type=str, default='fold', metavar='N',
                        choices=['fold', 'upsampling', 'dense'],
                        help='Decoder architecture used, [fold, upsampling, dense]')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')  # TODO Wo?
    parser.add_argument('--num_points', type=int, default=1024,  # TODO automatic
                        help='Num of points to use')
    parser.add_argument('--feat_dims', type=int, default=128, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=16, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=10240, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N',
                        help='Save snapshot interval')
    parser.add_argument('--shape', type=str, default='1d', metavar='N',
                        choices=['1d', 'diagonal', 'circle', 'square', 'gaussian'],
                        help='Shape of points to input decoder, [1d, diagonal, circle, square, gaussian]')
    parser.add_argument('--pooling', type=str, default='avg', metavar='N',
                        choices=['avg', 'max'],
                        help='Pooling type used for PointNet, [avg, max]')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--model_path', type=str, default='C:/Users/nilsk/Projects/MabiPointLaserEnv/point_clouds/embedding/autoencoder/snapshot/Reconstruct_graph_1d_fold_uniform_k16\models/uniform_density_4210.pkl',
                        metavar='N', help='Path to load model')
    parser.add_argument('--dataset', type=str, default='uniform_density', metavar='N',
                        choices=['lidar', 'uniform_density', 'regular_distances'],
                        help='Encoder to use, [lidar, uniform_density, regular_distances]')
    parser.add_argument('--rotate', action='store_true',
                        help='rotate point clouds during training')
    # parser.add_argument('--use_rotate', action='store_true',
    #                     help='Rotate the pointcloud before training')
    # parser.add_argument('--use_translate', action='store_true',
    #                     help='Translate the pointcloud before training')
    # parser.add_argument('--use_jitter', action='store_true',
    #                     help='Jitter the pointcloud before training')
    # parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    # TODO make GPU availlebel
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    parser.add_argument('--no_cuda', type=bool, default=True, help='Enables CUDA training')
    # parser.add_argument('--no_cuda', action='store_true',
    #                     help='Enables CUDA training')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    if args.eval == False:
        reconstruction = Reconstruction(args)


        reconstruction.run()
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        data = np.load(
            'C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/point_clouds/2500_point_clouds_360_norm.npy')
        x = data[44, :, :]
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        x = x.permute(0, 2, 1)

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        x = x.transpose(2, 1).float()

        reconstruction.model.eval()
        embedding = reconstruction.model.encoder.forward(x)
        reconstruction = reconstruction.model.decoder.forward(embedding)
        print(reconstruction.shape)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = x.detach().numpy()
        ax1.scatter(x[0, :, 0], x[0, :, 1], s=10, c='b', marker="s", label='true')
        reconstruction = reconstruction.detach().numpy()
        print(reconstruction.shape)
        ax1.scatter(reconstruction[0, :, 0], reconstruction[0, :, 1], s=10, c='r', marker="o", label='reconstruction')
        plt.legend(loc='upper left')
        plt.show()


