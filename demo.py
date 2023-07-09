from helpers import load_data
import light
import shading
import matplotlib.pyplot as plt

import numpy as np
file = 'utils/h3.npy'
verts,vertex_colors,face_indices,cam_eye,cam_up,cam_lookat,ka,kd,ks,n_phong,light_positions, light_intensities,Ia,M,N,W,H,bg_color,focal = load_data(file)

lights = []
for i in range(len(light_intensities)):
    pos = light_positions[i]
    I = light_intensities[i]
    lights.append(light.PointLight(pos,I))


mat_ambient = light.PhongMaterial(ka,0,0,n_phong)
mat_diffuse = light.PhongMaterial(0,kd,0,n_phong)
mat_specular = light.PhongMaterial(0,0,ks,n_phong)
mat_all = light.PhongMaterial(ka,kd,ks,n_phong)



shader = 'gouraud'
# Gouraud Shading with ambient lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_ambient, lights, Ia)
plt.imshow(X)
plt.show()

# Gouraud Shading with diffused lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_diffuse, lights, Ia)
plt.imshow(X)
plt.show()

# Gouraud Shading with specular lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_specular, lights, Ia)
plt.imshow(X)
plt.show()

# Gouraud Shading with ambient, diffused and specular lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_all, lights, Ia)
plt.imshow(X)
plt.show()

shader = 'gouraud'
# Phong Shading with ambient lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_ambient, lights, Ia)
plt.imshow(X)
plt.show()

# Gouraud Shading with diffused lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_diffuse, lights, Ia)
plt.imshow(X)
plt.show()

# Gouraud Shading with specular lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_specular, lights, Ia)
plt.imshow(X)
plt.show()

# Gouraud Shading with ambient, diffused and specular lighting
X = np.ones([M, N, 3]) * 255 * bg_color
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_all, lights, Ia)
plt.imshow(X)
plt.show()
