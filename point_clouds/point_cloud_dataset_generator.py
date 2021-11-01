import os
from vtk_pointlaser.config import Config
import numpy as np
import cv2 as cv
import vtk
from scipy.spatial.transform import Rotation as R
from stl import mesh
from vtk_pointlaser.local_info import get_local_info, correct_rotations_m1

num_meshes = len(os.listdir('C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/meshes/'))
num_positions = 5
num_points = 365
tol = 1  # ray cast tolerance
p1 = None
ray = np.array([99999, 0, 0])  # end of ray
r = R.from_euler('z', 360/num_points, degrees=True).as_matrix()


def load_room(index):
    # load a room (code form mesh.py Mesh.load_stl(file))
    mesh_file = 'C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/meshes/r_map{}.stl'.format(index)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    room = reader.GetOutput()

    # Build BSP tree for ray-casting
    bsp_tree = vtk.vtkModifiedBSPTree()
    bsp_tree.SetDataSet(room)
    bsp_tree.BuildLocator()

    return room, bsp_tree


def sample_position(room):
    bounds = room.GetBounds()
    min_bounds = np.array(bounds[::2])
    max_bounds = np.array(bounds[1::2])

    # Implicit function to find if point is inside/outside surface and at what distance
    implicit_function = vtk.vtkImplicitPolyDataDistance()
    implicit_function.SetInput(room)

    while True:
        pos = np.random.uniform(min_bounds + 100, max_bounds - 100)
        signed_dist = implicit_function.FunctionValue(pos)
        if 100 < -signed_dist < 5000.:
            break
    pos = pos
    pos[2] = 0
    return pos


# START SCRIPT
dataset = None
for i in range(1, num_meshes+1):
    room, bsp_tree = load_room(i)

    for _ in range(num_positions):
        p1 = sample_position(room)
        # Perform intersection
        point_cloud = None
        points = vtk.vtkPoints()

        for _ in range(num_points):
            p2 = p1 + ray
            code = bsp_tree.IntersectWithLine(p1, p2, tol, points, None)
            if code is not 0:
                new_point = np.array(points.GetData().GetTuple3(0))[:2]
                if point_cloud is None:
                    point_cloud = new_point
                else:
                    point_cloud = np.vstack([point_cloud, new_point])
            ray = ray.dot(r)

        point_cloud = np.expand_dims(point_cloud, 0)
        if dataset is None:
            dataset = point_cloud
        else:
            dataset = np.concatenate([dataset, point_cloud], axis=0)

print(dataset.shape)
np.save('../meshes/train_data/point_clouds/point_cloud_360.npy', dataset)
import matplotlib.pyplot as plt

data = np.load('../meshes/train_data/point_clouds/point_cloud_360.npy')
plt.scatter(data[0, :, 0], data[0, :, 1])
plt.show()
