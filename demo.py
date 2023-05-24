import numpy as np
from camera import renderObject,plot_im
from transformation import rotateTranslate

p3d = np.load("h2.npy", allow_pickle=True).tolist()['verts3d']
faces = np.load("h2.npy", allow_pickle=True).tolist()['faces']
vcolors = np.load("h2.npy", allow_pickle=True).tolist()['vcolors']
vcolors = np.floor(vcolors*255)

# Axis along which we rotate
u = np.load("h2.npy", allow_pickle=True).tolist()['u']
# The coordinates of the target point
ck = np.load("h2.npy", allow_pickle=True).tolist()['c_lookat']
# The coordinates of the up vector of the camera
c_up = np.load("h2.npy", allow_pickle=True).tolist()['c_up']
# The coordinates of the center of the camera
cv = np.load("h2.npy", allow_pickle=True).tolist()['c_org']
# The focal length of the lens
focal_length = np.load("h2.npy", allow_pickle=True).tolist()['focal']

# The displacement vectors
t0 = np.array([0,0,0])
t1 =  np.load("h2.npy", allow_pickle=True).tolist()['t_1']
t2 =  np.load("h2.npy", allow_pickle=True).tolist()['t_2']

# The angle of the rotation
phi =  np.load("h2.npy", allow_pickle=True).tolist()['phi']
# Point of rotation axis
A = np.array([0,0,0])

# Image and camera height and width
img_W = 512
img_H = 512
cam_W = 15
cam_H = 15

# Initial image without displacement and rotation
I = renderObject(p3d,faces,vcolors,cam_W,cam_H,img_H,img_W,focal_length, cv, ck, c_up)
plot_im(I,0)

# Displace image by t1
p3d = rotateTranslate(p3d,0,u,A,t1)
I = renderObject(p3d, faces, vcolors, cam_W, cam_H, img_H, img_W, focal_length, cv, ck, c_up)
plot_im(I,1)

# Rotate image by phi
p3d = rotateTranslate(p3d,phi,u,A,t0)
I = renderObject(p3d, faces, vcolors, cam_W, cam_H, img_H, img_W, focal_length, cv, ck, c_up)
plot_im(I,2)

# Displace image by t1
p3d = rotateTranslate(p3d,0,u,A,t2)
I = renderObject(p3d, faces, vcolors, cam_W, cam_H, img_H, img_W, focal_length, cv, ck, c_up)
plot_im(I,3)

breakpoint()

