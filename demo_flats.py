import numpy as np
import cv2,time
import matplotlib.pyplot as plt
from render import render

M = 512
N = 512

verts2d = np.load("h1.npy", allow_pickle=True).tolist()['verts2d']
vcolors = np.load("h1.npy", allow_pickle=True).tolist()['vcolors']
faces = np.load("h1.npy", allow_pickle=True).tolist()['faces']
depth = np.load("h1.npy", allow_pickle=True).tolist()['depth']
# verts2d[:,0] = M - verts2d[:,0]

# Multiply vcolors by 255 so that the values are in range [0,255]
vcolors = np.floor(vcolors*255)

# Initialize canvas with rgb values equal to 255 so that the background is white
canvas = 255 * np.ones([M, N, 3], dtype=np.uint8)

shade_t = "flats"
start_time = time.time()
canvas = render(canvas,verts2d,faces,vcolors,depth,shade_t)
end_time = time.time()

print(" Rendering with flat shading complete after {:.2f} seconds".format(end_time - start_time))
plt.imshow(canvas)
plt.show()
cv2.imwrite('flats_img.jpg', canvas)
