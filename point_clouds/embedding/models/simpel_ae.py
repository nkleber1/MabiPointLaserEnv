import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(nn.Module):
    def __init__(self, point_dim, num_points, embedding_size):
        super(LinearDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_points = num_points

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)

        self.fc1 = nn.Linear(in_features=8*num_points, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=embedding_size)

        # batch norm
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))


        # do max pooling
        x = x.view(-1, self.num_points*8)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = x.view(-1, self.embedding_size)
        # return the global embedding
        return x


class PointCloudDecoder(nn.Module):
    def __init__(self, point_dim, num_points, embedding_size):
        super(PointCloudDecoder, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=embedding_size)

        self.avg_pool = nn.AvgPool1d(num_points)

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))

        # do max pooling
        # x = torch.max(x, 2, keepdim=True)[0]
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = x.view(-1, self.embedding_size)
        # return the global embedding
        return x


class PointCloudEncoder(nn.Module):
    def __init__(self, point_dim, num_points, embedding_size):
        super(PointCloudEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=4096)
        self.fc4 = nn.Linear(in_features=4096, out_features=num_points * point_dim)

        # batch norm
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(4096)

    def forward(self, x, batch_size, point_dim, num_points):
        # decoder
        x = F.relu(self.bn0(self.fc1(x)))
        x = F.relu(self.bn1(self.fc2(x)))
        x = F.relu(self.bn2(self.fc3(x)))
        x = self.fc4(x)

        # do reshaping
        return x.reshape(batch_size, point_dim, num_points)


class PointCloudAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud
    Input:
    Output:
    """

    def __init__(self, point_dim, num_points, embedding_size):
        super(PointCloudAutoEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.decoder = LinearDecoder(point_dim, num_points, embedding_size)
        self.encoder = PointCloudEncoder(point_dim, num_points, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        embedding = self.decoder(x)
        reconstruction = self.encoder(embedding, batch_size, point_dim, num_points)
        return reconstruction, embedding