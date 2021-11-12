import random
import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from point_clouds.embedding.models.simpel_ae import PointCloudAutoEncoder
from pytorch3d.loss.chamfer import chamfer_distance

EMBEDDING_SIZE = 32
N_EPOCHS = 1024


dataset = PointCloudDataset('C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/point_clouds/2500_point_clouds_360_norm.npy')
train_data = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# Seed the Randomness
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# determine the device to run the network on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Make AutoEncoder (ae)
ae = PointCloudAutoEncoder(point_dim=dataset.point_dim, num_points=dataset.n_points, embedding_size=EMBEDDING_SIZE)
ae.to(device)
ae.double()
optimizer = optim.Adam(ae.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

l1 = torch.nn.L1Loss()

loss_hist = list()
for epoch in range(N_EPOCHS):
    for i, points in enumerate(train_data):
        points = points.permute(0, 2, 1)
        points = points.to(device)
        optimizer.zero_grad()
        reconstruction, latent_vector = ae(points)  # perform training

        loss, _ = chamfer_distance(points.float(), reconstruction.float())
        # loss = l1(points.float(), reconstruction.float())
        # loss = torch.norm(reconstruction - points)
        loss_hist.append(loss.item())
        print(loss.item())
        loss.backward()  # Calculate the gradients using Back Propogation
        optimizer.step()  # Update the weights and biases
    scheduler.step()

import matplotlib.pyplot as plt

#plt.plot(loss_hist)
#plt.ylabel('chamfer distance loss')
#plt.ylabel('network update')
#plt.show()


data = np.load('C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/point_clouds/2500_point_clouds_360_norm.npy')
x = data[44, :, :]
x = torch.from_numpy(x)
x = torch.unsqueeze(x, 0)
x = x.permute(0, 2, 1)

batch_size = x.shape[0]
point_dim = x.shape[1]
num_points = x.shape[2]

ae.eval()
embedding = ae.decoder.forward(x)
reconstruction = ae.encoder.forward(embedding, batch_size, point_dim, num_points)
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = x.detach().numpy()
ax1.scatter(x[0, 0, :], x[0, 1, :], s=10, c='b', marker="s", label='true')
reconstruction = reconstruction.detach().numpy()
print(reconstruction.shape)
ax1.scatter(reconstruction[0, 0, :], reconstruction[0, 1, :], s=10, c='r', marker="o", label='reconstruction')
plt.legend(loc='upper left')
plt.show()




