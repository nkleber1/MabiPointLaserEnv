import numpy as np
import matplotlib.pyplot as plt

data = np.load('../meshes/train_data/point_clouds/lidar.npy')
for i in range(79, 100):
    plt.scatter(data[i, :, 0], data[i, :, 1])
    print(i)
    plt.show()