from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

mesh_file = 'C:/Users/nilsk/Projects/MabiPointLaserEnv/meshes/train_data/meshes/r_map9.stl'

triangle_mesh = o3d.io.read_triangle_mesh(mesh_file)

room = PyntCloud.from_instance("open3d", triangle_mesh)
# room.plot(mesh=True, backend="matplotlib")
room = room.get_sample("mesh_random", n=10000, rgb=False, normals=False, as_PyntCloud=True)
# room.plot(backend="matplotlib")
df = room.points
max_z = df['z'].max()
df = df[df['z'] > 0]
df = df[df['z'] < max_z]

pts = df.to_numpy()[:, :2]

pts = pts[np.random.choice(pts.shape[0], 5000, replace=False)]
print(pts.shape)

plt.scatter(pts[:, 0], pts[:, 1])
plt.show()




