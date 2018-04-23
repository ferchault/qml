from __future__ import print_function



from copy import deepcopy
import os
import numpy as np
import scipy
from scipy.special import jn
from scipy.special import binom
from scipy.misc import factorial

import qml

from qml.math import cho_solve

from qml.fchl import generate_representation
from qml.fchl import get_local_symmetric_kernels
from qml.fchl import get_local_kernels
from qml.fchl import get_global_symmetric_kernels
from qml.fchl import get_global_kernels
from qml.fchl import get_atomic_kernels
from qml.fchl import get_atomic_symmetric_kernels

def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies

def test_krr_fchl_local():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.Compound() objects"
    mols = []

    keys = sorted(data.keys())

    np.random.seed(666)
    np.random.shuffle(keys)

    for xyz_file in keys[:((100//2)*3)]:

        # Initialize the qml.Compound() objects
        mol = qml.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_fchl_representation(cut_distance=1e6)
        mols.append(mol)

    # Make training and test sets
    n_test  = len(mols) // 3
    n_train = len(mols) - n_test

    training = mols[:n_train]
    test  = mols[-n_test:]

    X = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    # sigmas = [0.01 * 2**i for i in range(20)]
    sigmas = [2.5]
    llambda = 1e-7
    
    # gmas = [5.0]
    kernel_args = {
        "kernel": "cauchy",
        #"kernel": "gaussian",
        # "kernel": "inverse-multiquadratic",
        "kernel_args": {
            # "c": sigmas,
            "sigma": sigmas,
        },
        "cut_distance": 1e6,
        "alchemy": "off",
        }

    K = get_local_symmetric_kernels(X, **kernel_args)
    Ks = get_local_kernels(Xs, X, **kernel_args)

    for s, sigma in enumerate(sigmas):

        C = deepcopy(K[s])
        # Solve alpha

        C[np.diag_indices_from(C)] += llambda
        alpha = cho_solve(C,Y)

        # Calculate prediction kernel
        Yss = np.dot(Ks[s], alpha)

        mae = np.mean(np.abs(Ys - Yss))
        assert mae < 4.2, "ERROR IN FCHL KRR TEST"
        # print(sigma, mae)

if __name__ == "__main__":

    test_krr_fchl_local()
