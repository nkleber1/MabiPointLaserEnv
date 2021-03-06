# MabiPointLaserEnv
This project is part of my master's thesis. 
It contains a simulated environment for learning active localisation 
using a reinforcement learning agent. 
The environment follows the openAI Gym framework.
## point_clouds
### Encoder
- GraphEncoder from FoldingNet (
[file](https://github.com/nkleber1/MabiPointLaserEnv/blob/main/point_clouds/embedding/autoencoder/model_graph_encoder.py) |
[paper](https://arxiv.org/abs/1712.07262) | 
[source](https://github.com/AnTao97/UnsupervisedPointCloudReconstruction) )
- PointNet++ (
[file](https://github.com/nkleber1/MabiPointLaserEnv/blob/main/point_clouds/embedding/autoencoder/model_pointnet2_encoder.py) | 
[paper](https://arxiv.org/abs/1706.02413) | 
[source](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) )
- Pointnet (
[file](https://github.com/nkleber1/MabiPointLaserEnv/blob/main/point_clouds/embedding/autoencoder/model_pointnet_encoder.py) | 
[paper](https://arxiv.org/abs/1612.00593) | 
[source](https://github.com/fxia22/pointnet.pytorch) )
- Simple (
[file](https://github.com/nkleber1/MabiPointLaserEnv/blob/main/point_clouds/embedding/autoencoder/model_dense_encoder.py) )
### Decoder
- FoldEncoder from FoldingNet (
[file](https://github.com/nkleber1/MabiPointLaserEnv/blob/main/point_clouds/embedding/autoencoder/model_fold_decoder.py) | 
[paper](https://arxiv.org/abs/1712.07262) | 
[source](https://github.com/AnTao97/UnsupervisedPointCloudReconstruction) )
- Upconv (file | 
[source](https://github.com/charlesq34/pointnet-autoencoder))
- Simple ( [file](https://github.com/nkleber1/MabiPointLaserEnv/blob/main/point_clouds/embedding/autoencoder/model_dense_decoder.py) )