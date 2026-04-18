import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
data = np.column_stack((X, y))

np.savetxt("data/spiral.csv", data, delimiter=",", fmt=["%.5f", "%.5f", "%d"])
