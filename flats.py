import numpy as np
from interpolate_vectors import interpolate_vectors
import matplotlib.pyplot as plt

def bresenham(vertex_point1, vertex_point2):

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
            # Bresenham in y axis
            # Find starting and ending point
            if vertex_point1[1] <= vertex_point2[1]:
                x0,y0 = vertex_point1
                x1,y1 = vertex_point2
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
        else:
            # Bresenham in x axis
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

def fill_triangle(vertex_point1, vertex_point2,vertex_point3):
    # Find the three edges of the triangle using bresenham
    edge1 = bresenham(vertex_point1, vertex_point2)
    edge2 = bresenham(vertex_point1, vertex_point3)
    edge3 = bresenham(vertex_point2, vertex_point3)
    edges = np.vstack((np.squeeze(edge1), np.squeeze(edge2), np.squeeze(edge3)))

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

    # Bounding box of the triangle
    min_x = min(vertex_point1[0], vertex_point2[0], vertex_point3[0])
    max_x = max(vertex_point1[0], vertex_point2[0], vertex_point3[0])

    pixels_in_triangle = edges
    for x in range(min_x, max_x):
        idx = np.where(edges[:, 0] == x)

        #Check if the line one edge or none
        if idx[0].size == 1 or idx[0].size == 0:
            continue

        edges_in_line = np.squeeze(edges[idx, :])
        # Check if the line has more than to edges, i.e. consecutive lines
        if edges_in_line.shape[0] > 2:
            # Remove all edges that are side by side in the y axis (limit edges)
            idx = np.where(np.diff(edges_in_line[:, 1]) > 1)
            idx = idx[0].tolist()
            if len(idx) == 0:
                continue
            idx.append(idx[0]+1)
            edges_in_line = edges_in_line[idx]


        for y in range(edges_in_line[0, 1] + 1, edges_in_line[1, 1]):
            pixels_in_triangle = np.vstack((pixels_in_triangle, np.array([x, y])))

    return pixels_in_triangle

def flats(canvas,vertices,vcolors,faces,depth):
    triangle_depth = [np.mean(depth[faces[i]]) for i in range(faces.shape[0])]
    # Get the indexes after sorting the triangle depth array with descending order
    sorted_triangle_idx = np.argsort(triangle_depth, kind='quicksort')[::-1]

    for triangle_idx in sorted_triangle_idx:
        vertex_idx = faces[triangle_idx]
        vertex_point1 = vertices[vertex_idx[0]]
        vertex_point2 = vertices[vertex_idx[1]]
        vertex_point3 = vertices[vertex_idx[2]]

        v1_color = vcolors[vertex_idx[0]]
        v2_color = vcolors[vertex_idx[1]]
        v3_color = vcolors[vertex_idx[2]]

        flats_color = np.mean(np.vstack((v1_color, v2_color, v3_color)),axis=0)
        flats_color = np.round(flats_color)

        pixels_in_triangle = fill_triangle(vertex_point1, vertex_point2,vertex_point3)

        canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 0] = flats_color[0]
        canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 1] = flats_color[1]
        canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 2] = flats_color[2]
        # plt.imshow(canvas)
        # plt.show()
        # breakpoint()
    return canvas
#
# def gourauds(canvas,vertices,vcolors,faces,depth):
#     triangle_depth = [np.mean(depth[faces[i]]) for i in range(faces.shape[0])]
#     # Get the indexes after sorting the triangle depth array with descending order
#     sorted_triangle_idx = np.argsort(triangle_depth, kind='quicksort')[::-1]
#
#     for triangle_idx in sorted_triangle_idx:
#         vertex_idx = faces[triangle_idx]
#         vertex_point1 = vertices[vertex_idx[0]]
#         vertex_point2 = vertices[vertex_idx[1]]
#         vertex_point3 = vertices[vertex_idx[2]]
#
#         v1_color = vcolors[vertex_idx[0]]
#         v2_color = vcolors[vertex_idx[1]]
#         v3_color = vcolors[vertex_idx[2]]
#
#         edge1 = bresenham(vertex_point1, vertex_point2)
#         edge2 = bresenham(vertex_point1, vertex_point3)
#         edge3 = bresenham(vertex_point2, vertex_point3)
#
#
#         # Fill the edges of triangle
#         for pixels in edge2:
#             val1 = interpolate_vectors(vertex_point1,vertex_point2, v1_color[0], v2_color[0], pixels[1],dim=2)
#             breakpoint()
#
#
#         pixels_in_triangle = fill_triangle(vertex_point1, vertex_point2, vertex_point3)
#         canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 0] = flats_color[0]
#         canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 1] = flats_color[1]
#         canvas[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 2] = flats_color[2]
#
#     return canvas

