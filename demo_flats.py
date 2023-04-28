import numpy as np
from flats import flats,bresenham,fill_triangle
import matplotlib.pyplot as plt
# The vertexes of the triangles. Size of K x 2 and contains the X and Y coordinates of each edge.
verts2d = np.load("h1.npy", allow_pickle=True).tolist()['verts2d']


M = 512
N = 512

verts2d[:,0] = M - verts2d[:,0]
# verts2d[:,1] = N - verts2d[:,1]
# The colors of the edges. Size of K x 3 and contains the RGB values for each i-th edge.
vcolors = np.load("h1.npy", allow_pickle=True).tolist()['vcolors']

# Multiply vcolors by 255 so that the values are in range [0,255]
vcolors = np.floor(vcolors*255)

# Contains the faces of the triangles as a L x 3 array. Each of the K rows contains the indexes for the verts2d
# triangle edges.
faces = np.load("h1.npy", allow_pickle=True).tolist()['faces']

# Contains the depth of each edge. Size of K x 1
depth = np.load("h1.npy", allow_pickle=True).tolist()['depth']


M = 512
N = 512

canvas = 255 * np.ones([M, N, 3], dtype=np.uint8)

canvas = flats(canvas,verts2d,vcolors,faces,depth)
plt.imshow(canvas, cmap='gray')
plt.show()

breakpoint()