from helpers import load_data
import matplotlib.pyplot as plt
import light,shading

file = '../utils/h3.npy'
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
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_ambient, lights, Ia)
path = 'results/'
filename = 'Gouraud_ambient.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()

# Gouraud Shading with diffused lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_diffuse, lights, Ia)
path = 'results/'
filename = 'Gouraud_diffuse.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()

# Gouraud Shading with specular lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_specular, lights, Ia)
path = 'results/'
filename = 'Gouraud_specular.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()

# Gouraud Shading with ambient, diffused and specular lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_all, lights, Ia)
path = 'results/'
filename = 'Gouraud_all.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()

shader = 'phong'

# Phong Shading with ambient lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_ambient, lights, Ia)
path = 'results/'
filename = 'Phong_ambient.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()


# Phong Shading with diffused lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_diffuse, lights, Ia)
path = 'results/'
filename = 'Phong_diffuse.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()

# Phong Shading with specular lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_specular, lights, Ia)
path = 'results/'
filename = 'Phong_specular.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()

# Phong Shading with ambient, diffused and specular lighting
X= shading.render_object(shader,focal,cam_eye,cam_lookat,cam_up,bg_color,M,N,H,W,verts, vertex_colors, face_indices,mat_all, lights, Ia)
path = 'results/'
filename = 'Phong_all.jpg'
plt.imshow(X)
# plt.savefig(path+filename)
plt.show()
