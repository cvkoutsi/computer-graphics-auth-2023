import numpy as np
import camera,light
import render

def calculate_normals(verts, faces):
    triangle_normals = np.zeros(faces.shape)
    normals = np.zeros(verts.shape)

    for i in range(faces.shape[1]):
        verts_idx = faces[:,i]
        v1 = verts[:,verts_idx[0]]
        v2 = verts[:,verts_idx[1]]
        v3 = verts[:,verts_idx[2]]
        edge1 = v2 - v1
        edge2 = v3 - v1
        triangle_normals[:,i] = np.cross(edge1,edge2)

    for i in range(verts.shape[1]):
        _,j = np.where(faces == i)
        t = np.mean(triangle_normals[:, j],axis=1)
        normals[:, i] = t / np.linalg.norm(t)
    return normals

def render_object(focal, eye, lookat, up, bg_color,M,N,H,W,verts, vert_colors, faces,mat, lights, light_amb):
    X = np.ones([M, N, 3]) * 255* bg_color
    Nn = calculate_normals(verts,faces)
    p2d,depth = camera.cameraLookingAt(focal, eye, lookat, up, verts)
    n2d = camera.rasterize(p2d, M, N, H, W)

    triangle_depth = np.mean(depth[faces],axis=0)
    sorted_triangle_idx = np.argsort(triangle_depth, kind='quicksort')[::-1]

    for i in range(len(sorted_triangle_idx)):
        triangle_idx = sorted_triangle_idx[i]
        vertex_idx = faces[:,triangle_idx]
        vertsp = n2d[vertex_idx].transpose()
        vertsn = Nn[:, vertex_idx]
        vertsc = vert_colors[:,vertex_idx]
        bcoords = np.sum(verts[:,vertex_idx],axis=1)/3

        pixels_in_triangle,pixels_color,X = shade_gouraud(vertsp, vertsn, vertsc, bcoords, eye, mat, lights, light_amb,X)

    return X.astype(np.uint8)

def shade_gouraud(vertsp, vertsn, vertsc, bcoords, cam_pos, mat, lights, light_amb,X):
    v1, v2, v3 = vertsp[:, 0], vertsp[:, 1], vertsp[:, 2]
    v1_c, v2_c, v3_c = vertsc[:, 0], vertsc[:, 1], vertsc[:, 2]
    n1, n2, n3 = vertsn[:, 0], vertsn[:, 1], vertsn[:, 2]

    v1_light = light.light(bcoords, n1, v1_c, cam_pos, mat, lights, light_amb)
    v1_color = np.round(255*v1_light)
    v2_light = light.light(bcoords, n2, v2_c, cam_pos, mat, lights, light_amb)
    v2_color = np.round(255*v2_light)
    v3_light = light.light(bcoords, n3, v3_c, cam_pos, mat, lights, light_amb)
    v3_color = np.round(255*v3_light)

    # Shade the triangle using the method specified by the shade_t variable
    pixels_in_triangle, pixels_color = render.shade_triangle(v1, v1_color, v2, v2_color, v3, v3_color, shade_t="gourauds")

    # Update the canvas with the shaded triangle
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 0] = pixels_color[:, 0]
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 1] = pixels_color[:, 1]
    X[pixels_in_triangle[:, 0], pixels_in_triangle[:, 1], 2] = pixels_color[:, 2]

    return pixels_in_triangle,pixels_color,X

def shade_phong(vertsp, vertsn, vertsc, bcoords, cam_pos, mat, lights, light_amb,X):
    v1, v2, v3 = vertsp[:, 0], vertsp[:, 1], vertsp[:, 2]
    v1_c, v2_c, v3_c = vertsc[:, 0], vertsc[:, 1], vertsc[:, 2]
    n1, n2, n3 = vertsn[:, 0], vertsn[:, 1], vertsn[:, 2]