#!/usr/bin/env python2

import numpy as np

from numpy import cos, arccos, sin, arctan2, sqrt, cross, dot
from numpy.linalg import norm

import qml
from qml.fchl import generate_fchl_representation
from qml.fchl import get_local_symmetric_kernels_fchl

if __name__ == "__main__":

    mol = qml.Compound(xyz="qm7/0002.xyz")
    X1 = generate_fchl_representation(mol.coordinates, mol.nuclear_charges, size=8, neighbors=8)

    mol2 = qml.Compound(xyz="NHClF.xyz")
    X2 = generate_fchl_representation(mol2.coordinates, mol2.nuclear_charges, size=8, neighbors=8)

    X = np.array([X1, X2])
    Z = [mol.nuclear_charges, mol2.nuclear_charges]

    sigmas = [25.0]

    K = get_local_symmetric_kernels_fchl(X, sigmas, alchemy="periodic-table", 
            scale_distance=1.0,
            power_distance=6.0,
            scale_angular=0.1,
            power_angular=3.0,
            scale_dihedral=1.0,
            power_dihedral=3.0,
            )[0]
    print K
