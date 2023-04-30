import numpy as np
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

if __name__ == "__main__":
    p1 = [2, 10]
    p2 = [20, 10]
    v1 = 10
    v2 = 5
    xy = 25
    dim = 1

    result = interpolate_vectors(p1, p2, v1, v2, xy, dim)
    # assert result == 4.0
    print(result)
