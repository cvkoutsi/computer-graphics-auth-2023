import numpy as np
import matplotlib.pyplot as plt
from render import render
import cv2

def pinHole(f, cv, cx, cy, cz, p3d):
    """""
    @:param f: The focal length
    @:param cv,cx,cy,cz: The coordinates of the center and the unit vectors of the camera (WCS)
    @:param p3d: (K,3) array that contains the [x,y,z] coordinates of each vertix. (CCS)
    
    @:returns p2d: (K,2) array that contains the projected [x,y] coordinates of each vertix. (CCS)
    @:returns depth: An array with the depth of the points
    """""

    # Transform coordinates from WCS to CCS
    R = np.hstack((cx,cy,cz))
    distance = p3d - cv.reshape(3,1)
    p_ccs = np.dot(np.transpose(R),distance)
    px_ccs = p_ccs[0, :]
    py_ccs = p_ccs[1, :]
    pz_ccs = p_ccs[2, :]

    # Calculate the projected coordinates in the CCS
    p2d = f * np.hstack((np.expand_dims(px_ccs/pz_ccs,axis=1),np.expand_dims(py_ccs/pz_ccs,axis=1)))
    depth = pz_ccs

    return p2d, depth

def cameraLookingAt(f, cv, cK, cup, p3d):
    """""
    @:param f: The focal length
    @:param cv: The coordinates of the center of the camera (WCS)
    @:param cK: The coordinates of the target point K (WCS)
    @:param cup: The coordinates of the up vector of the camera (WCS)
    @:param p3d: (K,3) array that contains the [x,y,z] coordinates of each vertix. (CCS)
    
    @:returns p2d: (K,2) array that contains the projected [x,y] coordinates of each vertix. (CCS)
    @:returns depth: An array with the depth of the points
    where K is the number of vertices
    """""
    # Calculate the unit vectors of the camera
    zc = cK - cv / np.linalg.norm(cK - cv)
    t = cup - np.dot(np.transpose(zc),cup)*zc
    yc = t/ np.linalg.norm(t)
    xc = np.cross(np.squeeze(yc),np.squeeze(zc))

    xc, yc, zc = xc.reshape(3,1),yc.reshape(3,1),zc.reshape(3,1)

    p2d,depth = pinHole(f,cv,xc,yc,zc,p3d)
    return p2d,depth

def rasterize(p2d,Rows,Columns,H,W):
    """""
    @:param p2d: (K,2) array that contains the projected [x,y] coordinates of each vertix. (CCS)
    @:param Rows,Columns: The rows and columns of the final (projectes) image
    @:param H,W: The height and width of the image projected in the camera

    @:returns n2d: (K,2) array that contains the [x,y] coordinates in the coordinate system of the image
    where K is the number of vertices
    """""
    n2d = np.zeros(p2d.shape, dtype=np.int)
    # Calculate scaling factors
    img_width = Columns - 1
    img_height = Rows - 1
    x_scale = img_width/W
    y_scale = img_height/H

    # Apply the scaling factors to the coordinates
    n2d[:,1] = np.round(p2d[:,0] * x_scale)
    n2d[:,0] = np.round(p2d[:,1] * y_scale)

    # Move (0,0) to the top left corner
    n2d[:,0] =img_width/2 - n2d[:,0]
    n2d[:,1] = img_height/2 - n2d[:,1]

    n2d[:,1] = Columns - n2d[:,1]
    return n2d

def renderObject(p3d, faces, vcolors,H,W,Rows,Columns, f, cv, cK, cup):
    """""
    @:param p3d: (K,3) array that contains the [x,y,z] coordinates of each vertix. (CCS)
    @:param faces: (L,3) array that contains 3 indexes to the vertices array that define the vertices of the triangles.
    @:param vcolors: (K,3) array that contains the [r,g,b] values of each vertix
    @:param H,W: The height and width of the image projected in the camera
    @:param Rows,Columns: The rows and columns of the final (projectes) image
    @:param f: The focal length
    @:param cv: The coordinates of the center of the camera (WCS)
    @:param cK: The coordinates of the target point K (WCS)
    @:param cup: The coordinates of the up vector of the camera (WCS)
    where K is the number of vertices and L is the number of triangles
    
    @:returns I: the final [r,g,b] image
    """""
    # Initialize image and set white background
    I = 255 * np.ones([Rows,Columns,3], dtype=np.uint8)

    # Calculate 2d projections and depth
    p2d, depth = cameraLookingAt(f, cv, cK, cup, p3d)

    # Rasterize image
    n2d = rasterize(p2d, Rows, Columns, H, W)

    # Render image with gourauds shading
    I = render(I, n2d, faces, vcolors, depth, "gourauds")

    return I

def plot_im(I,i):
    plt.imshow(I)
    plt.show()
    path = 'results/'
    filename = 'position' + str(i) + '.jpg'
    cv2.imwrite(path + filename,I)