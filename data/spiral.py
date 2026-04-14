import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# Save to one file
np.savetxt("spiral.csv", X, delimiter=", ", fmt="%.5f")
