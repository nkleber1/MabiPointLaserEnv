from openmesh import *
import os

num_meshes = len(os.listdir('C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/eval_data/meshes/'))

mesh = TriMesh()
for i in range(num_meshes):
    i = i+1
    stl_source_file = 'C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/eval_data/meshes/r_map{}.stl'.format(i)
    ply_source_file = 'C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/eval_data/ply/r_map{}.ply'.format(i)

    mesh = TriMesh()
    read_trimesh(stl_source_file)
    write_mesh(ply_source_file, mesh)