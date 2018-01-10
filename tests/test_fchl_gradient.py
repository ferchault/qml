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

from copy import deepcopy

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
# SIGMAS = [1.28] # Optimal for old cost-functions
LLAMBDA = 1e-8
# SIGMAS = [0.25, 0.5, 0.75, 1.0]
# SIGMAS = [0.01 * 2**i for i in range(3,12)]
SIGMAS = [0.01 * 2**i for i in range(20)]
SIZE = 19
ALCHEMY = "off"
TRAINING = int(sys.argv[1])

DX = 5.0e-4

def csv_to_molecular_reps(csv_filename, force_key="forces", energy_key="energy"):


    df = pd.read_csv(csv_filename)

    x = []
    f = []
    e = []
    c = []
    nuc = []
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
        c.append(coordinates)
        nuc.append(nuclear_charges)

    return np.array(x), f, e, np.array(c), np.array(nuc)


def retrain():
    

    Xall, Fall, Eall, Call, Nall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")
    Eall = np.array(Eall)*2
    train = TRAINING

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    
    alphas = get_scalar_vector_alphas_fchl(X, F, E, SIGMAS,
                    llambda = LLAMBDA,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )

    np.save("alphas_new.npy", alphas)

def test_force():
    
    alphas = np.load("alphas_new.npy")

    Xall, Fall, Eall, Call, Nall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")
    Eall = np.array(Eall)# * 2
    train = TRAINING

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    
    Fs = Fall[-1]
    Es = Eall[-1]

    print(Es)
    print(Fs)

    Xs = []

    coordinates = Call[-1]
    nuclear_charges = Nall[-1]
    rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
    Xs.append(rep)

    for i in range(SIZE):
        for xyz in range(3):
            
            coord = coordinates[i,xyz]

            coords = deepcopy(coordinates)

            coords[i,xyz] = coord + DX

            rep = generate_fchl_representation(coords, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
            Xs.append(rep)
            
            coords[i,xyz] = coord - DX

            rep = generate_fchl_representation(coords, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
            Xs.append(rep)
            
            coords[i,xyz] = coord

    Xs = np.array(Xs)


    Xs_force = Xs[:1]


    print("TEST FORCE KERNEL")
    Ks_force = get_scalar_vector_kernels_fchl(X, Xs_force, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )

    TEST = Xs.shape[0]

    X = np.reshape(X, (TRAINING*SIZE,5,SIZE))
    Xs = np.reshape(Xs, (TEST*SIZE,5,SIZE))

    print("TEST ENERGY KERNEL")
    Ks_large = get_atomic_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy=ALCHEMY,
                )

    for i, sigma in enumerate(SIGMAS):
   
        
        if abs (sigma - 1.28) > 0.001: continue

        Fss = np.einsum('jkl,l->kj', Ks_force[i], alphas[i])

        # print(Fss)
        

        Ks = np.zeros((TRAINING*SIZE,TEST))

        for j in range(TRAINING*SIZE):
            for k in range(TEST):

               Ks[j,k] = np.sum(Ks_large[i,j,k*SIZE:(k+1)*SIZE])

        Ess = np.dot(Ks.T, alphas[i])



        # Ess -= intercept

        # print(Ess[0]/2)
        # print(Ess[1:]/2)

        Fss_num = np.zeros((SIZE,3))

        idx = 1

        for j in range(SIZE):
            for xyz in range(3):

                Fss_num[j,xyz] += Ess[idx]

                idx += 1
                
                Fss_num[j,xyz] -= Ess[idx]

                idx += 1

        Fss_num /= (4*DX)

        print(Fs)
        print(Fss)
        print(Fss_num)
        print(Fss_num - Fss)
        print(Fss_num/Fss)

        print(scipy.stats.linregress(Fss_num.flatten(), Fss.flatten()))
        print(scipy.stats.linregress(Fss_num.flatten(), Fs.flatten()))
        print(scipy.stats.linregress(Fss.flatten(), Fs.flatten()))


        print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss_num - Fs)), sigma)
        print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss_num - Fss)), sigma)
        print("RMSE ENERGY", 
                Es/2,
                Ess[0]/2,
                Ess[0]/2 - Es/2,
                sigma,
            )


if __name__ == "__main__":

    retrain()
    test_force()
