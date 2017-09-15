#!/usr/bin/env python2

import numpy as np

from numpy import cos, arccos, sin, arctan2, sqrt, cross, dot
from numpy.linalg import norm

import qml
from qml.fchl import generate_fchl_representation
from qml.fchl import get_local_symmetric_kernels_fchl
# from qml.fchl import get_atomic_kernels_fchl

np.set_printoptions(linewidth=99999999999)

TO_DEG = 180.0 / np.pi

def get_angle(i, a, b, X):

    b1 = X[a] - X[i]
    b2 = X[b] - X[i]

    return arccos(dot(b1, b2) / (norm(b1) * norm(b2)))


def calc_G(a, b, c, d):

    G = 3.0 / (norm(a) * norm(b) * norm(c) * norm(d))**3 * ( -4.0 + 3.0\
       * (dot(a,b)**2 + dot(a,c)**2 + dot(a,d)**2 + dot(b,c)**2 + dot(b,d)**2 + dot(c,d)**2) \
       - 9.0 * (dot(b,c) * dot(c,d) * dot(d,b)  \
              + dot(c,d) * dot(d,a) * dot(a,c)  \
              + dot(a,b) * dot(b,d) * dot(d,a)  \
              + dot(b,c) * dot(c,a) * dot(a,b)) \
        + 27.0 * (dot(a,b) * dot(b,c) * dot(c,d) * dot(d,a)))

    return G


def get_dihedral_atan2(xi, xa, xb, xc, X):

    b1 = X[xa] - X[xi]
    b2 = X[xb] - X[xa]
    b3 = X[xc] - X[xb]

    x12 = cross(b1, b2)
    x23 = cross(b2,b3)

    phi = arctan2(dot(cross(x12, x23), b2/norm(b2)), dot(x12, x23))
    # phi = arccos(cos(phi))

    A = X[xi] 
    B = X[xa] 
    C = X[xb] 
    D = X[xc] 

    a = C - B
    b = A - C
    c = B - A
    d = A - D
    e = B - D
    f = C - D

    a /= norm(a) 
    b /= norm(b) 
    c /= norm(c) 
    d /= norm(d) 
    e /= norm(e) 
    f /= norm(f) 

    E_abcd = calc_G(c, a, f,d) + calc_G(c, e, f, b) + calc_G(b, a, e, d)

    return E_abcd, phi


def get_dihedral(i, a, b, c, X):

    cos_phi= (cos(X[i,a+3,b]) - cos(X[i,a+3,c]) * cos(X[i,b+3,c])) \
            / (sin(X[i,a+3,c]) * sin(X[i,b+3,c]))

    return arccos(cos_phi)

if __name__ == "__main__":

    mol = qml.Compound(xyz="qm7/0002.xyz")
    X1 = generate_fchl_representation(mol.coordinates, mol.nuclear_charges, size=8, neighbors=8)

    mol2 = qml.Compound(xyz="NHClF.xyz")
    X2 = generate_fchl_representation(mol2.coordinates, mol2.nuclear_charges, size=8, neighbors=8)

    X = np.array([X1, X2])
    Z = [mol.nuclear_charges, mol2.nuclear_charges]

    sigmas = [25.0]

    ones_index = (X.shape[2] - 3) // 2 + 3
    #X[:,:,2,:] = 1.0
    #X[:,:,ones_index:,:] = 1.0
    #print X[0]
    #print X.shape


    # exit()
    # K1 = get_atomic_kernels_fchl(X, X, sigmas, alchemy="periodic-table")[0]

    # print K1
    K2 = get_local_symmetric_kernels_fchl(X, sigmas, alchemy="periodic-table", 
            scale_distance=1.0,
            power_distance=6.0,
            scale_angular=0.1,
            power_angular=3.0,
            scale_dihedral=0.0,
            power_dihedral=3.0,
            )[0]
    print K2
