import numpy as np
import camera,light
import render
import helpers

def calculate_normals(verts, faces):
    triangle_normals = np.zeros(faces.shape)
    normals = np.zeros(verts.shape)

    # Find the norma vector for each triangle
    for i in range(faces.shape[1]):
        verts_idx = faces[:,i]
        v1 = verts[:,verts_idx[0]]
        v2 = verts[:,verts_idx[1]]
        v3 = verts[:,verts_idx[2]]
        edge1 = v2 - v1
        edge2 = v3 - v1
        triangle_normals[:,i] = np.cross(edge1,edge2)

    # Find the normal vector of each vertix by taking the mean of the normal vectors from the triangles it belongs
    for i in range(verts.shape[1]):
        _,j = np.where(faces == i)
        t = np.mean(triangle_normals[:, j],axis=1)
        normals[:, i] = t / np.linalg.norm(t)
    return normals

def render_object(shader, focal, eye, lookat, up, bg_color,M,N,H,W,verts, vert_colors, faces,mat, lights, light_amb):

    # Create canvas of size MxN with the given background color
    X = np.ones([M, N, 3]) * 255* bg_color
    # Calculate the normal vectors for each vertix
    Nn = calculate_normals(verts,faces)
    # Find the projected coordinates ana the depth for each vertix
    p2d,depth = camera.cameraLookingAt(focal, eye, lookat, up, verts)
    # Rasterize the projected coordinates
    n2d = camera.rasterize(p2d, M, N, H, W)

    # Calculate the triangle depth by taking the mean of the depth of the triangle vertices
    triangle_depth = np.mean(depth[faces],axis=0)
    # Sort the triangles in order of descending depth
    sorted_triangle_idx = np.argsort(triangle_depth, kind='quicksort')[::-1]

    # Shade each triangle
    for i in range(len(sorted_triangle_idx)):
        triangle_idx = sorted_triangle_idx[i]
        vertex_idx = faces[:,triangle_idx]
        vertsp = n2d[vertex_idx].transpose() # Coordinates of the vertices
        vertsn = Nn[:, vertex_idx] # Normal vectors of the vertices
        vertsc = vert_colors[:,vertex_idx] # Colors of the vertices
        bcoords = np.sum(verts[:,vertex_idx],axis=1)/3 # Barycenter of the triangle

        if shader == 'gouraud':
            X = shade_gouraud(vertsp, vertsn, vertsc, bcoords, eye, mat, lights, light_amb,X)

        if shader == 'phong':
            X = shade_phong(vertsp, vertsn, vertsc, bcoords, eye, mat, lights, light_amb,X)

    return X.astype(np.uint8)

def shade_gouraud(vertsp, vertsn, vertsc, bcoords, cam_pos, mat, lights, light_amb,X):
    v1, v2, v3 = vertsp[:, 0], vertsp[:, 1], vertsp[:, 2]  #Vertix coordinates
    v1_c, v2_c, v3_c = vertsc[:, 0], vertsc[:, 1], vertsc[:, 2] #Vertix colors
    n1, n2, n3 = vertsn[:, 0], vertsn[:, 1], vertsn[:, 2] #Vertix normal vectors

    v1_light = light.light(bcoords, n1, v1_c, cam_pos, mat, lights, light_amb) # Light component of vertix
    v1_color = np.round(255*v1_light) # Final color of vertix
    v2_light = light.light(bcoords, n2, v2_c, cam_pos, mat, lights, light_amb)
    v2_color = np.round(255*v2_light)
    v3_light = light.light(bcoords, n3, v3_c, cam_pos, mat, lights, light_amb)
    v3_color = np.round(255*v3_light)

    # Shade the triangle using the gourauds shading
    pixels_in_triangle, pixels_color = render.gourauds(v1, v1_color, v2, v2_color, v3, v3_color)

    # Update the canvas with the shaded triangle
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 0] = pixels_color[:, 0]
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 1] = pixels_color[:, 1]
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 2] = pixels_color[:, 2]

    return X

def shade_phong(vertsp, vertsn, vertsc, bcoords, cam_pos, mat, lights, light_amb,X):
    v1, v2, v3 = vertsp[:, 0], vertsp[:, 1], vertsp[:, 2] #Vertix coordinates
    v1_c, v2_c, v3_c = vertsc[:, 0], vertsc[:, 1], vertsc[:, 2] #Vertix colors
    n1, n2, n3 = vertsn[:, 0], vertsn[:, 1], vertsn[:, 2] #Vertix normal vectors

    # Find the pixels on the edges
    edge1 = helpers.fill_line(v1, v2)
    edge2 = helpers.fill_line(v1, v3)
    edge3 = helpers.fill_line(v2, v3)

    if np.all(v1 == edge1[0]):
        v12_color = np.vstack((v1_c,v2_c))
        n12 = np.vstack((n1,n2))
    else:
        v12_color = np.vstack((v2_c,v1_c))
        n12 = np.vstack((n2,n1))

    if np.all(v1 == edge2[0]):
        v13_color = np.vstack((v1_c,v3_c))
        n13 = np.vstack((n1,n3))
    else:
        v13_color = np.vstack((v3_c,v1_c))
        n13 = np.vstack((n3,n1))

    if np.all(v2 == edge3[0]):
        v23_color = np.vstack((v2_c,v3_c))
        n23 = np.vstack((n2,n3))
    else:
        v23_color = np.vstack((v3_c,v2_c))
        n23 = np.vstack((n3,n2))

    edges = [edge1, edge2, edge3]
    vertex_colors = [v12_color, v13_color, v23_color]
    normals = [n12, n13, n23]

    # Remove edges with only 1 pixel
    k = 0
    for i in range(len(edges)):
        if len(edges[i-k].shape) == 1:
            del edges[i-k]
            del vertex_colors[i-k]
            del normals[i-k]
            k +=1

        # If all edges have 1 pixel, return the position of the pixel and the color of the vertex
        if len(edges) == 1:
            pixels_in_triangle = edges[0].reshape((1,2))
            v_light = light.light(edges[0], normals[0], vertex_colors[0], cam_pos, mat, lights, light_amb)
            v_color = np.round(255 * v_light).reshape((1,3))
            return pixels_in_triangle, v_color

    # Remove duplicate edges
    for i, edge in enumerate(edges):
        for j in range(i+1, len(edges)):
            if np.all(edge == edges[j]):
                del edges[i]
                del vertex_colors[i]
                del normals[i]

    # Find the color on the edges using linear interpolation
    edge_colors = []
    edge_normals = []
    edge_light = []

    # Calculate the color, the normal vector and the light component of each edge pixel
    for i in range(len(edges)):
        edge = edges[i]
        v_colors = vertex_colors[i]
        v_normals = normals[i]
        e_colors = np.array([v_colors[0, :]])
        e_normals = np.array([v_normals[0, :]])
        l = light.light(bcoords, v_normals[0,:], v_colors[0,:], cam_pos, mat, lights, light_amb)
        e_light = np.array(l)

        dx1 = max(edge[:, 0]) - min(edge[:, 0])
        dy1 = max(edge[:, 1]) - min(edge[:, 1])
        if dx1 > dy1:
            dim = 1
            xy = edge[:, 0]
        else:
            dim = 2
            xy = edge[:, 1]

        for i in range(1, len(edge)):
            r = helpers.interpolate_vectors(edge[0, :], edge[-1, :], v_colors[0, 0], v_colors[1, 0], xy[i], dim)
            g = helpers.interpolate_vectors(edge[0, :], edge[-1, :], v_colors[0, 1], v_colors[1, 1], xy[i], dim)
            b = helpers.interpolate_vectors(edge[0, :], edge[-1, :], v_colors[0, 2], v_colors[1, 2], xy[i], dim)
            n = helpers.interpolate_vectors(edge[0, :], edge[-1, :], v_normals[0,:], v_normals[1, :], xy[i], dim)
            e_colors = np.vstack((e_colors, np.array([r, g, b])))
            e_normals = np.vstack((e_normals,n))
            l = light.light(bcoords, n, np.array([r,g,b]), cam_pos, mat, lights, light_amb)
            e_light = np.vstack((e_light,l))
        edge_colors.append(e_colors)
        edge_normals.append(e_normals)
        edge_light.append(e_light)

    edges = np.vstack(edges)
    edge_colors = np.vstack(edge_colors)
    edge_normals = np.vstack(edge_normals)
    edge_light = np.vstack(edge_light)

    # Remove duplicate pixels
    edges, idx = np.unique(edges, axis=0, return_index=True)
    edge_colors = edge_colors[idx]
    edge_normals = edge_normals[idx]
    edge_light = edge_light[idx]

    # Bounding box of the triangle
    min_x = min(v1[0], v2[0], v3[0])
    max_x = max(v1[0], v2[0], v3[0])

    pixels_in_triangle = edges
    pixel_normals = edge_normals
    pixel_light = edge_light

    # Find the pixels in the inside of the triangle, the color, the normal vector and the light component for each pixel using scanlines
    for x in range(min_x, max_x):
        idx = np.where(edges[:, 0] == x)

        # Check if the line has one edge or none
        if idx[0].size == 1 or idx[0].size == 0:
            continue

        edges_in_line = np.squeeze(edges[idx, :])
        edge_color_in_line = np.squeeze(edge_colors[idx, :])
        edge_normal_in_line =  np.squeeze(edge_normals[idx, :])

        # Check if the line has more than two edges, i.e. consecutive lines
        if edges_in_line.shape[0] > 2:
            # Keep only non-conscutive edges
            idx = np.where(np.abs(np.diff(edges_in_line[:, 1])) > 1)
            idx = idx[0].tolist()
            if len(idx) == 0:
                continue
            idx.append(idx[0] + 1)
            edges_in_line = edges_in_line[idx]
            edge_color_in_line = edge_color_in_line[idx]

        # Add the pixels in between the edges to the triangle and calculate [r,g,b] values and the normal vector with linear interpolation between the vertices.
        # After finding the [r,g,b] values and the normal vector, calculate the light component
        for y in range(edges_in_line[0, 1] + 1, edges_in_line[1, 1]):
            pixels_in_triangle = np.vstack((pixels_in_triangle, np.array([x, y])))
            r = helpers.interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_color_in_line[0, 0], edge_color_in_line[1, 0], y, dim=2)
            g = helpers.interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_color_in_line[0, 1], edge_color_in_line[1, 1], y, dim=2)
            b = helpers.interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_color_in_line[0, 2], edge_color_in_line[1, 2], y, dim=2)

            n = helpers.interpolate_vectors(edges_in_line[0, :], edges_in_line[1, :], edge_normal_in_line[0, :], edge_normal_in_line[1, :], y, dim=2)
            l = light.light(bcoords, n, np.array([r, g, b]), cam_pos, mat, lights, light_amb)

            pixel_normals = np.vstack((pixel_normals, n))
            pixel_light = np.vstack((pixel_light, l))

    # Update the canvas with the shaded triangle
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 0] = np.round(255 * pixel_light[:, 0])
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 1] = np.round(255 * pixel_light[:, 1])
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 2] = np.round(255 * pixel_light[:, 2])

    return X