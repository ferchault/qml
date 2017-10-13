#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append("/home/andersx/dev/qml/fchl_forceenergy/build/lib.linux-x86_64-2.7")

import matplotlib.pyplot as plt

import ast

import scipy
import scipy.stats

from copy import deepcopy

import numpy as np
from numpy.linalg import norm
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
# np.set_printoptions(suppress=True, edgeitems=10)

CUT_DISTANCE = 1e6
# SIGMAS = [5.0]
SIGMAS = [100.0] # Optimal for old cost-functions
LLAMBDA = 1e-12
# SIGMAS = [1.28]
# SIGMAS = [0.01 * 2**i for i in range(20)]
# SIGMAS = [0.01 * 2**i for i in range(11)]
SIZE = 2

TRAINING = 8
TEST     = 63

FORCE_KEY = "mopac_forces"
ENERGY_KEY = "mopac_energy"
CSV_FILE = "data/hcl/scan/scan.csv"

def csv_to_atomic_reps(csv_filename, force_key="orca_forces", energy_key="orca_energy"):


    df = pd.read_csv(csv_filename)

    reps = []
    y = []

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = -np.array(ast.literal_eval(df[force_key][i]))

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        for j in range(len(atomtypes)):

            reps.append(rep[j])
            y.append(force[j])


    return np.array(reps), np.array(y)


def csv_to_molecular_reps(csv_filename, force_key="orca_forces", energy_key="orca_energy"):


    df = pd.read_csv(csv_filename)

    x = []
    f = []
    e = []
    distance = []

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))

        dist = norm(coordinates[0] - coordinates[1])
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))
        energy = df[energy_key][i]

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        distance.append(dist)
        x.append(rep)
        f.append(force)
        e.append(energy)

    # distance = np.array(distance)[::2]

    return np.array(x), f, e, distance


def test_old_forces():

    Xall, Yall = csv_to_atomic_reps(CSV_FILE, force_key=FORCE_KEY)

    train = SIZE*TRAINING
    test = SIZE*TEST

    X = Xall[:train]
    Y = Yall[:train]

    Xs = Xall[-test:]
    Ys = Yall[-test:]

    alphas = get_atomic_force_alphas_fchl(X, Y, SIGMAS,
                    llambda = LLAMBDA,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )


    np.save("X1_old.npy", X)
    np.save("X2_old.npy", Xs)

    np.save("alpha_old.npy", alphas)
    Ks  = get_atomic_force_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )


    for i, sigma in enumerate(SIGMAS):
        Yss = np.einsum('jkl,l->kj', Ks[i], alphas[i])

        print("RMSE FORCE COMPONENT", np.mean(np.abs(Yss - Ys)))

def test_old_energy():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall, Dall = csv_to_molecular_reps(CSV_FILE, 
                                force_key=FORCE_KEY, energy_key=ENERGY_KEY)

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
                    alchemy="off"
                )

    K_test=deepcopy(K)

    Ks = get_local_kernels_fchl(Xs, X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )
    
    E = np.array(E)
    for i, sigma in enumerate(SIGMAS):


        for j in range(train):

            K[i,j,j] += LLAMBDA

        alphas = cho_solve(K[i], E)

        Ess = np.dot(Ks[i], alphas)

        print("RMSE ENERGY OLD (test) ", np.mean(np.abs(Ess - Es)), sigma)
        
        Et = np.dot(K_test[i], alphas)

        print("RMSE ENERGY OLD (train)", np.mean(np.abs(Et - E)), sigma)

def test_new_forces():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall, Dall = csv_to_molecular_reps(CSV_FILE,
                                force_key=FORCE_KEY, energy_key=ENERGY_KEY)
    Eall = np.array(Eall)
    train = TRAINING
    test = TEST

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    D = Dall[:train]
    
    Xs = Xall[-test:]
    Fs = Fall[-test:]
    Es = Eall[-test:]
    Ds = Dall[-test:]

    alphas = get_scalar_vector_alphas_fchl(X, F, E, SIGMAS,
                    llambda = LLAMBDA,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )

    Ks_force = get_scalar_vector_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )

    # Ks_small = get_local_kernels_fchl(X, Xs, SIGMAS,
    #                 cut_distance=CUT_DISTANCE, 
    #                 alchemy="off"
    #             )

    X  = np.reshape(X, (TRAINING*SIZE,5,SIZE))
    Xs = np.reshape(Xs, (TEST*SIZE,5,SIZE))
    Ks_large = get_atomic_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )
    
    K_large = get_atomic_kernels_fchl(X, X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )

    # print(Ks_large[10])
    # print(K_large[10])
    for i, sigma in enumerate(SIGMAS):
   
        

        Fss = np.einsum('jkl,l->kj', Ks_force[i], alphas[i])

        Fs = np.array(Fs)
        Fs = np.reshape(Fs, (Fss.shape[0], Fss.shape[1]))

        print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss - Fs)))
        
        Ks = np.zeros((TRAINING*SIZE,TEST*SIZE))
        for k in range(TEST*SIZE):
           for j in range(TRAINING*SIZE):

               # Ks[j,k] = np.sum(Ks_large[i,j,k*SIZE:(k+1)*SIZE]) # / SIZE
               Ks[j,k] = Ks_large[i,j,k] # / SIZE
      
        Ess = np.dot(Ks_large[i].T, alphas[i])
        
        Ks = np.zeros((TRAINING*SIZE,TRAINING))
        for k in range(TRAINING):
           for j in range(TRAINING*SIZE):

               # Ks[j,k] = np.sum(K_large[i,j,k*SIZE:(k+1)*SIZE]) # / SIZE
               Ks[j,k] = K_large[i,j,k] # / SIZE
        
        print(Ks)
        print(alphas[i])
        Et = np.dot(K_large[i].T, alphas[i])


        # Ks = np.zeros((TRAINING*SIZE,TEST*SIZE))

        # for k in range(TEST):
        #    for j in range(TRAINING):

        #        Ks[j*SIZE:(j+1)*SIZE,k*SIZE:(k+1)*SIZE] = Ks_small[i,j,k] / (SIZE**2)
        # 
        # Ess = np.dot(Ks.T, alphas[i])    

        # Ess2 = np.zeros((TEST))

        # # print(Ks)
        # print(Ess)
        # print(alphas[i])

        # for j in range(TEST):
        #     Ess2[i] = np.sum(Ess[j*SIZE:(j+1)*SIZE])
        # 
        # Ess = np.reshape(Ess, (TEST,SIZE))

        # print(alphas)
        # print(Ess)
        # print(Ess.shape)

        # Ess = np.sum(Ess, axis=1)
        # # print(Ess2)
        print(Ess)
        print(Es)
        Et = np.reshape(Et, (TRAINING,SIZE)).T
        print(Et)
        Et = np.sum(Et, axis=0)
        print(E)
        Ess = np.reshape(Ess, (TEST,SIZE)).T
        print(Et)
        Ess = np.sum(Ess, axis=0) 


        slope, intercept_train, r_value, p_value, std_err = scipy.stats.linregress(E,Et)
        Et -= intercept_train
        
        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es,Ess)
        Ess -= intercept_train


        print("RMSE ENERGY", np.mean(np.abs(Ess - Es)), sigma,
                        scipy.stats.linregress(Ess, Es),
                            )
        print("RMSE ENERGY Train", np.mean(np.abs(E - Et)), sigma,
                        scipy.stats.linregress(E,Et),
                            )
       

        plt.plot(Ds, Ess, linestyle="-", color="r", label="test/predict")
        plt.plot(D,  Et,  linestyle="-", color="b", label="training/predict" )
        plt.plot(Ds, Es,  linestyle="--", color="r", label="test/true")
        plt.plot(D,  E,   linestyle="--", color="b", label="training/true" )
        plt.legend()
        plt.savefig("E_test.png")
        plt.clf() 
        plt.scatter(Es, Ess,  color="r", label="test")
        plt.scatter(E,  Et,   color="b", label="training" )
        plt.legend()
        plt.savefig("E_scatter.png")
        plt.clf()
   


        # print(Fs)
        # print(Fss)

        Ds_numm = []
        Fs_numm = []

        for j in range(1, TEST - 1):

            if ((j % 9) == 0): continue
            if ((j % 9) == 8): continue
            numm_diff = (Ess[j-1] -Ess[j+1])/(2*0.01)

            # print(numm_diff, Fss[::2,2][j])

            Ds_numm.append(Ds[j])
            Fs_numm.append(numm_diff)

        
        
        plt.plot(Ds_numm, Fs_numm, label="Numerical")
        plt.plot(Ds, Fss[::2,2], label="Predicted")
        plt.plot(Ds, Fs[::2,2], label="True")
        plt.legend()
        plt.title("sigma = %10.2f" % sigma)
        plt.grid(True, linestyle='--')
        plt.ylim( (-200, 500) )
        plt.xlim( (1.0, 1.7) )
        plt.xlabel("H-Cl distance [angstrom]")
        plt.ylabel("Gradient component [kcal/mol/angstrom]")
        # plt.savefig("F_test_%02i.png" % i)
        plt.savefig("F_test.png")
        plt.clf()

if __name__ == "__main__":

    # Xall, Fall, Eall = csv_to_molecular_reps(CSV_FILE, 
    #                             force_key=FORCE_KEY, energy_key=ENERGY_KEY)
    # Eall = np.array(Eall)
    # print("STD ENERGY", Eall.std())

    # test_old_forces()
    test_old_energy()
    test_new_forces()

