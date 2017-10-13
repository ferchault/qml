#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append("/home/andersx/dev/qml/fchl_forceenergy/build/lib.linux-x86_64-2.7")

import matplotlib.pyplot as plt

import scipy
import scipy.stats


import numpy as np
from numpy.linalg import norm
import ast

import numpy as np
import pandas as pd

import qml
from qml.math import cho_solve
from qml.fchl import generate_fchl_representation
from qml.fchl import get_atomic_force_alphas_fchl
from qml.fchl import get_atomic_force_kernels_fchl
from qml.fchl import get_scalar_vector_alphas_fchl
from qml.fchl import get_scalar_vector_kernels_fchl
from qml.fchl import get_local_symmetric_kernels_fchl
from qml.fchl import get_local_kernels_fchl
from qml.fchl import get_atomic_kernels_fchl

np.set_printoptions(linewidth=19999999999, suppress=True, edgeitems=10)

CUT_DISTANCE = 1e6
# SIGMAS = [5.0]
# SIGMAS = [0.64] # Optimal for old cost-functions
LLAMBDA = 1e-8
# SIGMAS = [0.5, 0.25]
# SIGMAS = [0.01 * 2**i for i in range(3,12)]
SIGMAS = [0.01 * 2**i for i in range(20)]
SIZE = 19
ALCHEMY = "off"
TRAINING = int(sys.argv[1])
TEST     = 50

def csv_to_atomic_reps(csv_filename):


    df = pd.read_csv(csv_filename)

    reps = []
    y = []

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df["forces"][i]))

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        for j, atomtype in enumerate(atomtypes):

            reps.append(rep[j])
            y.append(force[j])


    return np.array(reps), -np.array(y)


def csv_to_molecular_reps(csv_filename, force_key="forces", energy_key="energy"):


    df = pd.read_csv(csv_filename)

    x = []
    f = []
    e = []
    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))
        energy = df[energy_key][i]

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
        x.append(rep)
        f.append(force)
        e.append(energy)

    return np.array(x), f, e


def test_old_forces():

    Xall, Yall = csv_to_atomic_reps("data/molecule_300.csv")

    train = SIZE*TRAINING
    test = SIZE*TEST

    X = Xall[:train]
    Y = Yall[:train]

    Xs = Xall[-test:]
    Ys = Yall[-test:]

    alphas = get_atomic_force_alphas_fchl(X, Y, SIGMAS,
                    llambda = LLAMBDA,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )


    np.save("X1_old.npy", X)
    np.save("X2_old.npy", Xs)

    np.save("alpha_old.npy", alphas)
    Ks  = get_atomic_force_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )


    for i, sigma in enumerate(SIGMAS):
        Yss = np.einsum('jkl,l->kj', Ks[i], alphas[i])

        print("RMSE FORCE COMPONENT", np.mean(np.abs(Yss - Ys)), sigma)

def test_old_energy():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")

    Eall = np.array(Eall)
    train = TRAINING
    test = TEST

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    
    Xs = Xall[-test:]
    Fs = Fall[-test:]
    Es = Eall[-test:]

    K = get_local_symmetric_kernels_fchl(X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )

    Ks = get_local_kernels_fchl(Xs, X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )
    for i, sigma in enumerate(SIGMAS):


        for j in range(train):

            K[i,j,j] += LLAMBDA

        E = np.array(E)
        alphas = cho_solve(K[i], E)

        Ess = np.dot(Ks[i], alphas)

        print("RMSE ENERGY", np.mean(np.abs(Ess - Es)), sigma)

def test_new_forces():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")
    Eall = np.array(Eall) *2
    train = TRAINING
    test = TEST

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    
    Xs = Xall[-test:]
    Fs = Fall[-test:]
    Es = Eall[-test:]

    alphas = get_scalar_vector_alphas_fchl(X, F, E, SIGMAS,
                    llambda = LLAMBDA,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )

    # np.save("alphas_new.npy", alphas)
    # exit()

    # alphas = np.load("alphas_new.npy")

    print("TEST FORCE KERNEL")
    Ks_force = get_scalar_vector_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )
    np.save("Ks_force_new.npy", Ks_force)

    # Ks_small = get_local_kernels_fchl(X, Xs, SIGMAS,
    #                 cut_distance=CUT_DISTANCE, 
    #                 alchemy=ALCHEMY,
    #             )

    X = np.reshape(X, (TRAINING*SIZE,5,SIZE))
    Xs = np.reshape(Xs, (TEST*SIZE,5,SIZE))

    print("TEST ENERGY KERNEL")
    Ks_large = get_atomic_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )
    
    # K_large = get_atomic_kernels_fchl(X, X, SIGMAS,
    #                 cut_distance=CUT_DISTANCE, 
    #                 alchemy=ALCHEMY,
    #            )

    for i, sigma in enumerate(SIGMAS):
   
        

        Fss = np.einsum('jkl,l->kj', Ks_force[i], alphas[i])

        Fs = np.array(Fs)
        Fs = np.reshape(Fs, (Fss.shape[0], Fss.shape[1]))

        print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss - Fs)), sigma)

        # print(Fs)
        # print(Fss)
        # plt.scatter(Fs, Fss, label="Force")
        # plt.legend()
        # plt.savefig("F_test.png")
        # plt.clf()
       

        
        Ks = np.zeros((TRAINING*SIZE,TEST))

        for j in range(TRAINING*SIZE):
            for k in range(TEST):

               Ks[j,k] = np.sum(Ks_large[i,j,k*SIZE:(k+1)*SIZE])

        Ess = np.dot(Ks.T, alphas[i])


        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es,Ess)
        # Ess -= intercept

        print("RMSE ENERGY", np.mean(np.abs(Ess/2 - Es/2)), sigma,
                         scipy.stats.linregress(Ess/2, Es/2),
            )


if __name__ == "__main__":

    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")
    Eall = np.array(Eall)
    print("STD ENERGY", Eall.std())

    test_old_forces()
    # test_old_energy()
    # test_new_forces()
