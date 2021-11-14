import numpy as np
import matplotlib.pyplot as plt

data = np.load('../meshes/train_data/point_clouds/uniform_density_2048.npy')
for i in range(100):
    plt.scatter(data[i, :, 0], data[i, :, 1])
    print(i)
    plt.show()