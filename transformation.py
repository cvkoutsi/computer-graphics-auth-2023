import numpy as np
from spatialmath.base import skew

def rotmat(theta, u):
    R = ((1 - np.cos(theta)) * np.dot(u,u)) + (np.cos(theta) * np.eye(3)) + (np.sin(theta) * skew(u))
    return R

def rotateTranslate(cp, theta, u, A, t):

    # Move axis of rotation
    cp = cp - np.expand_dims(A, axis=1)
    R = rotmat(theta,u)
    cq = np.dot(R,cp)
    # Move axis back
    cq = cq + np.expand_dims(A, axis=1)

    # Move by t
    cq = cq + np.expand_dims(t, axis=1)
    return cq

def changeCoordinateSystem(cp,R, c0):
    """""
    :param cp: The coordinates of a point p using the initial coordinates system
    :param R: A rotation matrix
    :param c0: The coordinates of the displacement vector v0 using the initial coordinates system
    :returns dp: The coordinates of a point p using the new coordinates system 
    """""
    R_inv = np.linalg.inv(R)
    dp = np.dot(R_inv,cp - np.expand_dims(c0, axis=1))
    return dp

