from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os

num_meshes = len(os.listdir('C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/meshes/'))
num_positions = 10
num_points = 512

def load_room(index):
    # load a room (code form mesh.py Mesh.load_stl(file))
    mesh_file = 'C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/meshes/r_map{}.stl'.format(index)
    triangle_mesh = o3d.io.read_triangle_mesh(mesh_file)
    room = PyntCloud.from_instance("open3d", triangle_mesh)
    return room

def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0,1]] = pointcloud[:, [0,1]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud

# START SCRIPT
dataset = None
for i in range(1, num_meshes+1):
    room = load_room(i)
    room = room.get_sample("mesh_random", n=num_points*4, rgb=False, normals=False, as_PyntCloud=True)
    # room.plot(backend="matplotlib")
    for num in range(num_positions):
        df = room.points
        max_z = df['z'].max()
        df = df[df['z'] > 0]
        df = df[df['z'] < max_z]
        pts = df.to_numpy()[:, :2]
        pts = pts[np.random.choice(pts.shape[0], num_points, replace=False)]
        # plt.scatter(pts[:, 0], pts[:, 1])
        # plt.show()
        if num is not 0:
            point_cloud = rotate_pointcloud(point_cloud)
        point_cloud = pts - pts.min()
        pts = pts / pts.max()
        pts = np.expand_dims(pts, 0)
        if dataset is None:
            dataset = pts
        else:
            dataset = np.concatenate([dataset, pts], axis=0)
print(dataset.shape)
np.save('../../../meshes/train_data/point_clouds/uniform_density_512_big.npy', dataset)




