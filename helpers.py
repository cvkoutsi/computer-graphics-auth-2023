import numpy as np
from matplotlib import pyplot as plt
def load_data(file):
    data = np.load(file, allow_pickle=True).tolist()
    #  (K,3) array that contains the [x,y,z] coordinates of each vertix. K is the number of vertices
    verts = data['verts']
    # (K,3) array that contains the [r,g,b] values of each vertix in range [0,1]. K is the number of vertices
    vertex_colors = data['vertex_colors']
    # (L,3) array that contains 3 indexes to the vertices array that define the vertices of the triangles. L is the number of triangles
    face_indices = data['face_indices']

    # Coordinates of the center of the camera
    cam_eye = data['cam_eye']
    # Coordinates of the up vector
    cam_up = data['cam_up']
    # Coordinates of the target point
    cam_lookat = data['cam_lookat']

    # Coefficient of diffused light from the environment
    ka = data['ka']
    # Diffuse reflection coefficient
    kd = data['kd']
    # Specular reflection coefficient
    ks = data['ks']
    # Phong coefficient
    n = data['n']
    # Position of the light sources
    light_positions = data['light_positions']
    # Intensity of the light sources
    light_intensities = data['light_intensities']
    # Ambient light
    Ia = data['Ia']
    # Height and width of the final image
    M = data['M']
    N = data['N']
    # Height and width of the sensor
    W = data['W']
    H = data['H']
    bg_color = data['bg_color']
    # Focal length
    focal = data['focal']
    return  verts,vertex_colors,face_indices,cam_eye,cam_up,cam_lookat,ka,kd,ks,n,light_positions, light_intensities,Ia,M,N,W,H,bg_color,focal

def interpolate_vectors(p1,p2,V1,V2,xy,dim):
    """"" 
    :param V1,V2: values of the vectors V1,V2
    :param p1,p2: coordinates of vectors V1,V2
    :param xy: takes the value of x or y, if dim = 1 or dim = 2
    :param dim: dimension along which we interpolate
    
    :returns V: value of vector with coordinates x,y
    """""
    x1,y1 = p1
    x2,y2 = p2
    if dim == 1:
        x = xy
        if x > max(x1,x2):
            print("Given coordinates are not in range [x1,x2]")
            exit(1)
        if abs(x2-x1) < 1e-3:
            if abs(x-x1) < abs(x-x2):
                return x1
            else:
                return x2
        v1_coeff = np.abs(x2 - x) / np.abs(x2 - x1)
        v2_coeff = np.abs(x - x1) / np.abs(x2 - x1)
        return V1*v1_coeff + V2*v2_coeff
    elif dim == 2:
        y = xy

        if y > max(y1,y2):
            print("Given coordinates are not in range [y1,y2]")
            exit(1)

        if abs(y2-y1) < 1e-3:
            if abs(y-y1) < abs(y-y2):
                return y1
            else:
                return y2
        v1_coeff = np.abs(y2 - y) / np.abs(y2 - y1)
        v2_coeff = np.abs(y - y1) / np.abs(y2 - y1)
        return V1*v1_coeff + V2*v2_coeff


def bresenham(vertex_point1, vertex_point2, axis):
    """"" 
    Implements the bresenham line drawing algorithm for two vertices
    :param vertex_point1: coordinates [x,y] of vertex point 1
    :param vertex_point2: coordinates [x,y] of vertex point 2
    :param axis: the axis along which we calculate the line

    :returns edge_idx: (M,2) array that contains the coordinates [x,y] of the pixels that belong to the line     
    """""

    # Bresenham in y axis
    if axis == 1:
        # Find starting and ending point
        if vertex_point1[1] <= vertex_point2[1]:
            x0, y0 = vertex_point1
            x1, y1 = vertex_point2
        else:
            x0, y0 = vertex_point2
            x1, y1 = vertex_point1

        # Compute deltaX and deltaY
        deltaY = 2 * (y1 - y0)
        deltaX = 2 * np.abs(x1 - x0)
        f = -deltaX + deltaY / 2
        x = x0
        y = y0
        edge_idx = np.array([x, y])
        for y in range(y0 + 1, y1):
            if f < 0:
                if x0 < x1:
                    x = x + 1
                else:
                    x = x - 1
                f = f + deltaY
            f = f - deltaX
            edge_idx = np.vstack((edge_idx, np.array([x, y])))
        edge_idx = np.vstack((edge_idx, np.array([x1, y1])))

    # Bresenham in x axis
    elif axis == 0:
        # Find starting and ending point
        if vertex_point1[0] < vertex_point2[0]:
            x0, y0 = vertex_point1
            x1, y1 = vertex_point2
        else:
            x0, y0 = vertex_point2
            x1, y1 = vertex_point1

        # Compute deltaX and deltaY
        deltaX = 2 * (x1 - x0)
        deltaY = 2 * np.abs(y1 - y0)
        f = -deltaY + deltaX / 2
        x = x0
        y = y0
        edge_idx = np.array([x, y])
        for x in range(x0 + 1, x1):
            if f < 0:
                if y0 < y1:
                    y = y + 1
                else:
                    y = y - 1
                f = f + deltaX
            f = f - deltaY
            edge_idx = np.vstack((edge_idx, np.array([x, y])))
        edge_idx = np.vstack((edge_idx, np.array([x1, y1])))
    return edge_idx


def fill_line(vertex_point1, vertex_point2):
    """"" 
    Draws the line for two given vertices
    :param vertex_point1: coordinates [x,y] of vertex point 1
    :param vertex_point2: coordinates [x,y] of vertex point 2

    :returns edge_idx: (M,2) array that contains the coordinates [x,y] of the pixels that belong to the line     
    """""

    # If the two points are in the same line
    if vertex_point1[0] == vertex_point2[0]:
        x = vertex_point1[0]
        start = min(vertex_point1[1], vertex_point2[1])
        end = max(vertex_point1[1], vertex_point2[1])

        edge_idx = np.array([x, start])
        for y in range(start + 1, end + 1):
            edge_idx = np.vstack((edge_idx, np.array([x, y])))

    # If the two points are in the same column
    if vertex_point1[1] == vertex_point2[1]:
        y = vertex_point1[1]
        start = min(vertex_point1[0], vertex_point2[0])
        end = max(vertex_point1[0], vertex_point2[0])

        edge_idx = np.array([start, y])
        for x in range(start + 1, end + 1):
            edge_idx = np.vstack((edge_idx, np.array([x, y])))

    # If the two points are neither in the same column and line, perform bresenham in x or y axis, depending on the slope
    # If slope < 1 -> bresenham in y axis, else -> bresenham in x axis
    else:
        # Find slope
        slope = (vertex_point1[0] - vertex_point2[0]) / (vertex_point2[1] - vertex_point1[1])
        if np.abs(slope) < 1:
            # Bresenham in y axis (axis = 1)
            edge_idx = bresenham(vertex_point1, vertex_point2, axis=1)
        else:
            # Bresenham in x axis
            edge_idx = bresenham(vertex_point1, vertex_point2, axis=0)
    return edge_idx

def plot_im(I,i):
    plt.imshow(I)
    plt.show()
    filename = str(i) + '.jpg'
