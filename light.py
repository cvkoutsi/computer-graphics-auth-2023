import numpy as np
class PhongMaterial():
    def __init__(self,ka,kd,ks,n_phong):
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.n_phong = n_phong

class PointLight():
    def __init__(self,pos_in,I0_in):
        self.pos = pos_in
        self.I0 = I0_in

def light( point, normal, vcolor, cam_pos, mat, lights, light_amb):
    I = np.zeros(3)
    for light in lights:
        L = light.pos - point  #distance of point from light source
        Ln = L / np.linalg.norm(L)
        V = cam_pos - point  #distance of point from the camera
        Vn = V / np.linalg.norm(V)
        Nn = normal

        # Calculate light intensity due to diffused light from the environment
        Ia = light_amb * mat.ka
        Ia[Ia<0] = 0

        # Calculate light intensity due to diffuse reflection
        Id = light.I0 * mat.kd * np.dot(Nn,Ln)
        Id = Id * vcolor
        Id[Id<0] = 0

        # Calculate light inensity due to specular reflection
        Is = light.I0 * mat.ks * (np.dot(2 * Nn * np.dot(Nn,Ln) - Ln, Vn)) ** mat.n_phong
        Is = Is * vcolor
        Is[Is<0] = 0

        It = Ia + Id + Is
        I = I + It

    I[I > 1] = 1
    return I