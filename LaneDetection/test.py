import numpy as np

rows, height, width = 3, 240, 480

hor = np.empty((rows,height,width,3), np.uint8)
print(hor.shape)