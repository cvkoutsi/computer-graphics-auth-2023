import numpy as np
from interpolate_vectors import interpolate_vectors
import matplotlib.pyplot as plt

def bresenham(vertex_point1, vertex_point2, axis):
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

    # If the two points are neither in the same column and line
    else:
        #Find slope
        slope = (vertex_point1[0] - vertex_point2[0]) / (vertex_point2[1] - vertex_point1[1])
        if np.abs(slope) < 1:
            # Bresenham in y axis (axis = 1)
            edge_idx = bresenham(vertex_point1,vertex_point2,axis=1)
        else:
            # Bresenham in x axis
            edge_idx = bresenham(vertex_point1,vertex_point2,axis=0)
    return edge_idx

def flats(vertex_point1,v1_color,vertex_point2,v2_color,vertex_point3,v3_color):
    # Find the three edges of the triangle using bresenham
    edge1 = fill_line(vertex_point1, vertex_point2)
    edge2 = fill_line(vertex_point1, vertex_point3)
    edge3 = fill_line(vertex_point2, vertex_point3)
    edges = np.vstack((np.squeeze(edge1), np.squeeze(edge2), np.squeeze(edge3)))

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

    # Bounding box of the triangle
    min_x = min(vertex_point1[0], vertex_point2[0], vertex_point3[0])
    max_x = max(vertex_point1[0], vertex_point2[0], vertex_point3[0])

    pixels_in_triangle = edges
    for x in range(min_x, max_x):
        idx = np.where(edges[:, 0] == x)

        # Check if the line has one edge or none
        if idx[0].size == 1 or idx[0].size == 0:
            continue

        edges_in_line = np.squeeze(edges[idx, :])
        # Check if the line has more than two edges, i.e. consecutive lines
        if edges_in_line.shape[0] > 2:
            # Remove all edges that are side by side in the y axis (limit edges)
            idx = np.where(np.abs(np.diff(edges_in_line[:, 1]) > 1))
            idx = idx[0].tolist()
            if len(idx) == 0:
                continue
            idx.append(idx[0] + 1)
            edges_in_line = edges_in_line[idx]

        for y in range(edges_in_line[0, 1] + 1, edges_in_line[1, 1]):
            pixels_in_triangle = np.vstack((pixels_in_triangle, np.array([x, y])))

    flats_color = np.mean(np.vstack((v1_color, v2_color, v3_color)), axis=0)
    flats_color = np.round(flats_color).reshape((1,3))

    return pixels_in_triangle,flats_color

def gourauds(vertex_point1,v1_color,vertex_point2,v2_color,vertex_point3,v3_color):
    edge1 = fill_line(vertex_point1, vertex_point2)
    edge2 = fill_line(vertex_point1, vertex_point3)
    edge3 = fill_line(vertex_point2, vertex_point3)

    if np.all(vertex_point1 == edge1[0]):
        v12_color = np.vstack((v1_color,v2_color))
    else:
        v12_color = np.vstack((v2_color,v1_color))

    if np.all(vertex_point2 == edge2[0]):
        v13_color = np.vstack((v1_color,v3_color))
    else:
        v13_color = np.vstack((v3_color,v1_color))

    if np.all(vertex_point3 == edge3[0]):
        v23_color = np.vstack((v2_color,v3_color))
    else:
        v23_color = np.vstack((v3_color,v2_color))

    edges =  [edge1, edge2, edge3]
    vertex_colors = [v12_color, v13_color, v23_color]

    # Remove edges with only 1 pixel and duplicate edges
    k = 0
    for i in range(len(edges)):
        if len(edges[i-k].shape) == 1:
            del edges[i-k]
            del vertex_colors[i-k]
            k +=1

        # If all edges have 1 pixel, return the position of the pixel and the color of the vertex
        if len(edges) == 1:
            return edges[0].reshape((1,2)),vertex_colors[0][0,:].reshape((1,3))

    for i, edge in enumerate(edges):
        for j in range(i+1, len(edges)):
            if np.all(edge == edges[j]):
                del edges[i]
                del vertex_colors[i]

    edge_colors = []
    for i in range(len(edges)):
        edge = edges[i]
        v_colors = vertex_colors[i]

        e_colors = np.array([v_colors[0,:]])
        # If the colors in the vertex points are the same, dont interpolate
        if np.all(v_colors[0, :] == v_colors[1,:]):
            for i in range(1,len(edge)):
                e_colors = np.vstack((e_colors, v_colors[0,:]))

        # Else, interpolate between vertex colors to find the color on edges
        else:
            # Decide if we interpolate in axis x or y to find color on edges
            dx1 = max(edge[:, 0]) - min(edge[:, 0])
            dy1 = max(edge[:, 1]) - min(edge[:, 1])
            if dx1 > dy1:
                dim = 1
                xy = edge[:, 0]
            else:
                dim = 2
                xy = edge[:, 1]

            for i in range(1,len(edge)):
                r = round(interpolate_vectors(edge[0,:],edge[-1,:],v_colors[0,0],v_colors[1,0],xy[i],dim))
                g = round(interpolate_vectors(edge[0,:],edge[-1,:],v_colors[0,1],v_colors[1,1],xy[i],dim))
                b = round(interpolate_vectors(edge[0,:],edge[-1,:],v_colors[0,2],v_colors[1,2],xy[i],dim))
                e_colors = np.vstack((e_colors, np.array([r, g, b])))
        edge_colors.append(e_colors)

    edges = np.vstack(edges)
    edge_colors = np.vstack(edge_colors)


    # Remove duplicate edges
    edges, idx = np.unique(edges, axis=0, return_index=True)
    edge_colors = edge_colors[idx]
    # return edges.reshape((-1, 2)), edge_colors.reshape((-1, 3))

    min_x = min(vertex_point1[0], vertex_point2[0], vertex_point3[0])
    max_x = max(vertex_point1[0], vertex_point2[0], vertex_point3[0])

    # Interpolate in the y axis to fill the triangle
    pixels_in_triangle = edges
    pixel_colors = edge_colors

    for x in range(min_x, max_x):
        idx = np.where(edges[:, 0] == x)

        # Check if the line has one edge or none
        if idx[0].size == 1 or idx[0].size == 0:
            continue

        edges_in_line = np.squeeze(edges[idx, :])
        edge_color_in_line = np.squeeze(edge_colors[idx, :])

        # Check if the line has more than two edges, i.e. consecutive lines
        if edges_in_line.shape[0] > 2:
            # Remove all edges that are side by side in the y axis (limit edges)
            idx = np.where(np.abs(np.diff(edges_in_line[:, 1])) > 1)
            idx = idx[0].tolist()
            if len(idx) == 0:
                continue
            idx.append(idx[0] + 1)

            edges_in_line = edges_in_line[idx]
            edge_color_in_line = edge_color_in_line[idx]


        for y in range(edges_in_line[0, 1] + 1, edges_in_line[1, 1]):
            pixels_in_triangle = np.vstack((pixels_in_triangle, np.array([x, y])))
            r = round(interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_color_in_line[0, 0], edge_color_in_line[1, 0], y, dim=2))
            g = round(interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_color_in_line[0, 1], edge_color_in_line[1, 1], y, dim=2))
            b = round(interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_color_in_line[0, 2], edge_color_in_line[1, 2], y, dim=2))
            pixel_colors = np.vstack((pixel_colors, np.array([r, g, b])))

    return pixels_in_triangle,pixel_colors
def shade_triangle(vertex_point1,v1_color,vertex_point2,v2_color,vertex_point3,v3_color,shade_t="flat"):
    if shade_t == "flats":
        pixels_in_triangle,flats_color = flats(vertex_point1,v1_color,vertex_point2,v2_color,vertex_point3,v3_color)
    elif shade_t == "gourauds":
        pixels_in_triangle,flats_color = gourauds(vertex_point1,v1_color,vertex_point2,v2_color,vertex_point3,v3_color)

    return pixels_in_triangle,flats_color

def render(canvas,vertices,faces,vcolors,depth,shade_t):
    triangle_depth = [np.mean(depth[faces[i]]) for i in range(faces.shape[0])]
    # Get the indexes after sorting the triangle depth array with descending order
    sorted_triangle_idx = np.argsort(triangle_depth, kind='quicksort')[::-1]

    i = 0
    for triangle_idx in sorted_triangle_idx:
        print(triangle_idx)
        i +=1
        print(i)
        # # triangle_idx = 9619
        vertex_idx = faces[triangle_idx]
        vertex_point1 = vertices[vertex_idx[0]]
        vertex_point2 = vertices[vertex_idx[1]]
        vertex_point3 = vertices[vertex_idx[2]]

        v1_color = vcolors[vertex_idx[0]]
        v2_color = vcolors[vertex_idx[1]]
        v3_color = vcolors[vertex_idx[2]]

        pixels_in_triangle, flats_color = shade_triangle(vertex_point1,v1_color,vertex_point2,v2_color,vertex_point3,v3_color,shade_t)

        canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 0] = flats_color[:,0]
        canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 1] = flats_color[:,1]
        canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 2] = flats_color[:,2]
        plt.imshow(canvas, cmap='gray')
        plt.show()
        breakpoint()
    return canvas

